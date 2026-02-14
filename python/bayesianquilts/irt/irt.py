import warnings
from typing import Any

import jax
import numpy as np
import jax.numpy as jnp
from flax import nnx
from bayesianquilts.model import BayesianModel
from bayesianquilts.predictors.nn.dense import Dense
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf


class IRTModel(BayesianModel):
    kappa_scale: Any = nnx.data(None)
    joint_prior_distribution: Any = nnx.data(None)
    bijectors: Any = nnx.data(None)

    def __init__(
            self,
            item_keys,
            num_people,
            num_groups=None,
            data=None,
            person_key="person",
            dim=1,
            decay=0.25,
            positive_discriminations=True,
            missing_val=-9999,
            full_rank=False,
            eta_scale=1e-2,
            kappa_scale=1e-2,
            weight_exponent=1.0,
            response_cardinality=5,
            discrimination_guess=None,
            include_independent=False,
            vi_mode='advi',
            imputation_model=None,
            dtype=tf.float64):
        super(IRTModel, self).__init__(dtype=dtype)

        self.dtype = dtype

        self.item_keys = item_keys
        self.num_items = len(item_keys)
        self.missing_val = missing_val
        self.person_key = person_key
        self.positive_discriminations = positive_discriminations
        self.eta_scale = eta_scale
        self.kappa_scale = kappa_scale
        self.weight_exponent = weight_exponent
        self.response_cardinality = response_cardinality
        self.num_people = num_people
        self.full_rank = full_rank
        self.include_independent = include_independent
        self.discrimination_guess = discrimination_guess
        self.vi_mode = vi_mode
        self.num_groups = num_groups

        self.imputation_model = imputation_model

        self.set_dimension(dim, decay)

    def set_dimension(self, dim, decay=0.25):
        self.dimensions = dim
        self.dimensional_decay = decay
        self.kappa_scale *= (decay**tf.cast(
            tf.range(dim), self.dtype)
        )[tf.newaxis, :, tf.newaxis, tf.newaxis]

    def set_params_from_samples(self, samples):
        try:
            for k in self.var_list:
                self.surrogate_sample[k] = samples[k]
        except KeyError:
            print(str(k) + " doesn't exist in your samples")
            return
        self.set_calibration_expectations()

    def create_distributions(self):
        pass

    def obtain_scoring_nn(self, hidden_layers=None):
        if self.calibrated_traits is None:
            print("Please calibrate the IRT model first")
            return
        if hidden_layers is None:
            hidden_layers = [self.num_items * 2, self.num_items * 2]
        dnn = Dense(
            self.num_items,
            [self.num_items] + hidden_layers + [self.dimensions]
        )
        ability_distribution = tfd.Independent(
            tfd.Normal(
                loc=jnp.mean(
                    self.surrogate_sample['abilities'],
                    axis=0),
                scale=jnp.std(
                    self.surrogate_sample['abilities'],
                    axis=0
                )
            ), reinterpreted_batch_ndims=2
        )
        dnn_params = dnn.weights

        def loss():
            dnn_fun = dnn.build_network(dnn_params, jnp.nn.relu)
            return -ability_distribution.log_prob(dnn_fun(self.response_data))

    def simulate_data(self, abilities=None):
        discrimination = self.calibrated_expectations['discriminations']
        if abilities is None:
            abilities = self.calibrated_expectations['abilities']
        probs = self.grm_model_prob_d(
            abilities,
            discrimination,
            self.calibrated_expectations['difficulties0'],
            self.calibrated_expectations['ddifficulties']
        )
        response_rv = tfd.Categorical(
            probs=probs
        )
        responses = response_rv.sample()
        return responses

    def project_discriminations(self, steps=1000):
        pass

    # =========================================================================
    # Imputation model validation
    # =========================================================================

    def validate_imputation_model(self):
        """Validate that the imputation model is sufficient for this IRT model.

        Performs 5 checks:
        1. Fitted check - model has been trained
        2. Item coverage - all item_keys are in the imputation model
        3. Variable type - items must be ordinal or binary (not continuous)
        4. Convergence coverage - at least one converged model per item (warning)
        5. PSIS khat diagnostic - best model khat < 0.7 (warning)

        Raises:
            ValueError: For checks 1-3 (fatal).
        """
        im = self.imputation_model
        if im is None:
            return

        # Check 1: Fitted
        if not getattr(im, 'variable_names', None) or not getattr(im, 'zero_predictor_results', None):
            raise ValueError(
                "Imputation model has not been fitted. Call fit_loo_models() first."
            )

        # Check 2: Item coverage
        missing_items = [k for k in self.item_keys if k not in im.variable_names]
        if missing_items:
            raise ValueError(
                f"Imputation model does not cover item(s): {missing_items}"
            )

        # Check 3: Variable type compatibility
        continuous_items = []
        for k in self.item_keys:
            idx = im.variable_names.index(k)
            vtype = im.variable_types.get(idx)
            if vtype == 'continuous':
                continuous_items.append(k)
        if continuous_items:
            raise ValueError(
                f"Item(s) typed as 'continuous' in imputation model "
                f"(IRT items must be ordinal/binary): {continuous_items}"
            )

        # Check 4: Convergence coverage
        items_without_models = []
        for k in self.item_keys:
            idx = im.variable_names.index(k)
            has_converged = False
            # Check zero-predictor
            zp = im.zero_predictor_results.get(idx)
            if zp is not None and zp.converged:
                has_converged = True
            # Check univariate results
            if not has_converged:
                for (target_idx, _), result in im.univariate_results.items():
                    if target_idx == idx and result.converged:
                        has_converged = True
                        break
            if not has_converged:
                items_without_models.append(k)
        if items_without_models:
            warnings.warn(
                f"Items with no converged imputation models "
                f"(will use marginal fill): {items_without_models}"
            )

        # Check 5: PSIS khat diagnostic
        high_khat_items = []
        for k in self.item_keys:
            idx = im.variable_names.index(k)
            best_khat = None
            best_elpd = -np.inf
            # Check zero-predictor
            zp = im.zero_predictor_results.get(idx)
            if zp is not None and zp.converged:
                if zp.elpd_loo_per_obs > best_elpd:
                    best_elpd = zp.elpd_loo_per_obs
                    best_khat = zp.khat_max
            # Check univariate results
            for (target_idx, _), result in im.univariate_results.items():
                if target_idx == idx and result.converged:
                    if result.elpd_loo_per_obs > best_elpd:
                        best_elpd = result.elpd_loo_per_obs
                        best_khat = result.khat_max
            if best_khat is not None and best_khat >= 0.7:
                high_khat_items.append(k)
        if high_khat_items:
            warnings.warn(
                f"Items with high PSIS khat (>= 0.7), imputation may be "
                f"unreliable: {high_khat_items}"
            )

    # =========================================================================
    # Stochastic imputation in fit()
    # =========================================================================

    def _impute_batch(self, batch, rng=None):
        """Produce a single imputed copy of a batch.

        For each item column with missing values, uses the imputation model's
        ``predict_pmf()`` to obtain a proper ordinal categorical distribution
        (stacked mixture of ordinal logistic PMFs) and samples from it.

        Args:
            batch: dict mapping keys to arrays.
            rng: numpy random Generator.

        Returns:
            A copy of the batch with missing item values filled.
        """
        if rng is None:
            rng = np.random.default_rng()

        imputed = {k: np.array(v, copy=True) for k, v in batch.items()}

        for item_key in self.item_keys:
            col = np.asarray(batch[item_key], dtype=np.float64)
            bad = np.isnan(col) | (col < 0) | (col >= self.response_cardinality)
            bad_indices = np.where(bad)[0]
            if len(bad_indices) == 0:
                continue

            for row_idx in bad_indices:
                # Build observed items dict for this row
                observed_items = {}
                for other_key in self.item_keys:
                    if other_key == item_key:
                        continue
                    val = float(batch[other_key][row_idx])
                    if not (np.isnan(val) or val < 0 or val >= self.response_cardinality):
                        observed_items[other_key] = val

                try:
                    pmf = self.imputation_model.predict_pmf(
                        observed_items,
                        target=item_key,
                        n_categories=self.response_cardinality,
                    )
                    sampled = rng.choice(self.response_cardinality, p=pmf)
                except (ValueError, KeyError, AttributeError):
                    # Fallback: sample uniformly if predict_pmf unavailable
                    sampled = rng.integers(0, self.response_cardinality)

                imputed[item_key][row_idx] = float(sampled)

        return imputed

    def _has_missing_values(self, batch):
        """Check if any item column in the batch has missing values."""
        for item_key in self.item_keys:
            col = np.asarray(batch[item_key], dtype=np.float64)
            if np.any(np.isnan(col) | (col < 0) | (col >= self.response_cardinality)):
                return True
        return False

    def fit(self, batched_data_factory, initial_values=None,
            n_imputation_samples=1, **kwargs):
        """Fit the IRT model with optional Rao-Blackwellized imputation.

        If imputation_model is set, wraps the data factory to impute missing
        values before each training step. For M>1 imputation samples, uses
        proper Rao-Blackwellization via logsumexp over imputed copies'
        log-likelihoods, rather than averaging log-likelihoods (which is
        a lower bound by Jensen's inequality).

        Args:
            batched_data_factory: Callable returning a data iterator.
            initial_values: Optional initial parameter values.
            n_imputation_samples: Number of imputed copies per batch
                with missing values (default 1).
            **kwargs: Additional args passed to _calibrate_minibatch_advi.

        Returns:
            (losses, params) tuple.
        """
        if self.imputation_model is not None:
            self.validate_imputation_model()

            model_ref = self
            n_samples = n_imputation_samples
            rng = np.random.default_rng()
            item_keys_set = set(self.item_keys)

            original_factory = batched_data_factory

            def _stack_imputed_copies(batch):
                """Create M imputed copies; stack item keys to (M, N)."""
                copies = [model_ref._impute_batch(batch, rng)
                          for _ in range(n_samples)]
                stacked = {}
                for k in batch:
                    if k in item_keys_set:
                        stacked[k] = np.stack(
                            [c[k] for c in copies], axis=0
                        )
                    else:
                        stacked[k] = batch[k]
                return stacked

            def imputing_factory():
                def imputing_iterator():
                    iterator = original_factory()
                    try:
                        for batch in iterator:
                            if not model_ref._has_missing_values(batch):
                                yield batch
                            else:
                                yield _stack_imputed_copies(batch)
                    except TypeError:
                        # Factory returned a single batch, not an iterator
                        batch = iterator
                        if not model_ref._has_missing_values(batch):
                            yield batch
                        else:
                            yield _stack_imputed_copies(batch)
                return imputing_iterator()

            def rao_blackwell_log_prob(data, prior_weight, **params):
                """Rao-Blackwellized log prob: logsumexp over M copies."""
                first_item = model_ref.item_keys[0]
                if data[first_item].ndim > 1:
                    M = data[first_item].shape[0]
                    results = []
                    for m in range(M):
                        data_m = {
                            k: (data[k][m] if k in item_keys_set
                                 else data[k])
                            for k in data
                        }
                        results.append(
                            model_ref.unormalized_log_prob(
                                data=data_m,
                                prior_weight=prior_weight,
                                **params,
                            )
                        )
                    return jax.scipy.special.logsumexp(
                        jnp.stack(results)
                    ) - jnp.log(jnp.asarray(M, dtype=model_ref.dtype))
                else:
                    return model_ref.unormalized_log_prob(
                        data=data,
                        prior_weight=prior_weight,
                        **params,
                    )

            kwargs['unormalized_log_prob_fn'] = rao_blackwell_log_prob
            return super().fit(
                imputing_factory, initial_values=initial_values, **kwargs
            )
        else:
            return super().fit(
                batched_data_factory, initial_values=initial_values, **kwargs
            )

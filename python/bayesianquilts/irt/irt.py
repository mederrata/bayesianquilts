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

    def simulate_data(self, abilities=None, seed=0):
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
        responses = response_rv.sample(seed=jax.random.PRNGKey(seed))
        return responses

    def project_discriminations(self, steps=1000):
        pass

    def validate_imputation_model(self):
        """Validate that the imputation model is suitable for this IRT model.

        Checks:
        1. Imputation model has been fitted (has variable_names).
        2. Imputation model covers all item keys.
        3. No variables are typed as 'continuous'.
        4. Warns if any items have no converged models.
        5. Warns if any items have high k-hat diagnostics.
        """
        import warnings

        if self.imputation_model is None:
            return

        im = self.imputation_model

        # Check 1: fitted
        if not getattr(im, 'variable_names', None):
            raise ValueError(
                "Imputation model has not been fitted "
                "(no variable_names found)."
            )

        # Check 2: item coverage
        covered = set(im.variable_names)
        missing = [k for k in self.item_keys if k not in covered]
        if missing:
            raise ValueError(
                f"Imputation model does not cover items: {missing}"
            )

        # Check 3: continuous variables
        var_types = getattr(im, 'variable_types', {})
        for i, name in enumerate(im.variable_names):
            vtype = var_types.get(i, 'ordinal')
            if vtype == 'continuous' and name in self.item_keys:
                raise ValueError(
                    f"Item '{name}' is typed as continuous in the "
                    f"imputation model, but IRT requires ordinal/categorical."
                )

        # Check 4: convergence
        zero_results = getattr(im, 'zero_predictor_results', {})
        for i, name in enumerate(im.variable_names):
            if name not in self.item_keys:
                continue
            result = zero_results.get(i)
            if result is not None and not getattr(result, 'converged', True):
                warnings.warn(
                    f"Item '{name}' has no converged imputation models."
                )

        # Check 5: high k-hat
        for i, name in enumerate(im.variable_names):
            if name not in self.item_keys:
                continue
            result = zero_results.get(i)
            if result is not None and getattr(result, 'khat_max', 0) > 0.7:
                warnings.warn(
                    f"Item '{name}' has high khat "
                    f"({result.khat_max:.3f} > 0.7)."
                )

    def _has_missing_values(self, batch):
        """Check if any item column in the batch has missing values."""
        for item_key in self.item_keys:
            col = np.asarray(batch[item_key], dtype=np.float64)
            if np.any(np.isnan(col) | (col < 0) | (col >= self.response_cardinality)):
                return True
        return False

    def _compute_batch_pmfs(self, batch):
        """Compute imputation PMFs for missing cells using the imputation model.

        Delegates to ``self.imputation_model.predict_pmf()`` for each
        missing cell (n, i).  Observed cells get zeros.

        Args:
            batch: dict mapping keys to arrays.

        Returns:
            np.ndarray of shape (N, I, K) with PMFs for missing cells.
        """
        N = len(batch[self.item_keys[0]])
        I = self.num_items
        K = self.response_cardinality
        pmfs = np.zeros((N, I, K), dtype=np.float64)

        for i, item_key in enumerate(self.item_keys):
            col = np.asarray(batch[item_key], dtype=np.float64)
            bad = np.isnan(col) | (col < 0) | (col >= K)
            bad_indices = np.where(bad)[0]
            if len(bad_indices) == 0:
                continue

            for row_idx in bad_indices:
                observed_items = {}
                for other_key in self.item_keys:
                    if other_key == item_key:
                        continue
                    val = float(batch[other_key][row_idx])
                    if not (np.isnan(val) or val < 0 or val >= K):
                        observed_items[other_key] = val

                try:
                    pmf = self.imputation_model.predict_pmf(
                        observed_items, target=item_key, n_categories=K,
                    )
                    pmfs[row_idx, i, :] = pmf
                except (ValueError, KeyError, AttributeError):
                    pmfs[row_idx, i, :] = 1.0 / K

        return pmfs

    def _wrap_factory_with_imputation(self, factory):
        """Wrap a data factory to attach imputation PMFs from the imputation model.

        Returns a new factory that adds ``_imputation_pmfs`` to every batch.
        """
        model_ref = self

        def imputing_factory():
            def imputing_iterator():
                iterator = factory()
                try:
                    for batch in iterator:
                        if model_ref._has_missing_values(batch):
                            batch['_imputation_pmfs'] = model_ref._compute_batch_pmfs(batch)
                        else:
                            N = len(batch[model_ref.item_keys[0]])
                            batch['_imputation_pmfs'] = np.zeros(
                                (N, model_ref.num_items, model_ref.response_cardinality),
                                dtype=np.float64,
                            )
                        yield batch
                except TypeError:
                    batch = iterator
                    if model_ref._has_missing_values(batch):
                        batch['_imputation_pmfs'] = model_ref._compute_batch_pmfs(batch)
                    else:
                        N = len(batch[model_ref.item_keys[0]])
                        batch['_imputation_pmfs'] = np.zeros(
                            (N, model_ref.num_items, model_ref.response_cardinality),
                            dtype=np.float64,
                        )
                    yield batch
            return imputing_iterator()

        return imputing_factory

    def _compute_elpd_loo(self, data_factory, n_samples=100, seed=42,
                          khat_threshold=0.7):
        """Compute PSIS-LOO after fitting, with AIS fallback for high k-hat.

        Iterates one epoch through the factory, computes per-person
        log-likelihoods under ``n_samples`` surrogate posterior draws,
        and runs PSIS-LOO.  If any k-hat > khat_threshold, uses
        AdaptiveImportanceSampler to improve those estimates.

        Results are stored as attributes on ``self`` so they persist
        via ``save_to_disk``.

        Args:
            data_factory: Callable returning an iterator of batch dicts
                (same as the training data factory).
            n_samples: Surrogate posterior draws for PSIS.
            seed: Random seed for sampling.
            khat_threshold: Use AIS for points with k-hat above this.
        """
        from bayesianquilts.metrics.nppsis import psisloo

        surrogate = self.surrogate_distribution_generator(self.params)
        key = jax.random.PRNGKey(seed)
        samples = surrogate.sample(n_samples, seed=key)

        # Reassemble factorized parameters if model has a transform method
        if hasattr(self, 'transform'):
            samples = self.transform(samples)

        log_lik_matrix = np.full(
            (n_samples, self.num_people), np.nan, dtype=np.float64
        )

        # Collect full data for potential AIS pass
        all_batches = []
        for batch in data_factory():
            people = np.asarray(batch[self.person_key], dtype=np.int32)
            pred = self.predictive_distribution(batch, **samples)
            log_lik_matrix[:, people] = np.array(pred['log_likelihood'])
            all_batches.append(batch)

        # Check coverage
        visited = ~np.isnan(log_lik_matrix[0])
        n_obs = int(np.sum(visited))
        if n_obs < self.num_people:
            print(f"  Warning: only {n_obs}/{self.num_people} people "
                  f"visited in ELPD-LOO pass")
            log_lik_matrix = log_lik_matrix[:, visited]

        loo, loos, ks = psisloo(log_lik_matrix)

        # AIS fallback if any k-hat > threshold
        bad_k = np.sum(ks > khat_threshold)
        if bad_k > 0:
            try:
                from bayesianquilts.irt.grm import GradedResponseLikelihood
                from bayesianquilts.metrics.ais import AdaptiveImportanceSampler

                print(f"  {bad_k} observations with k-hat > {khat_threshold}, "
                      f"running AIS...")

                # Build full data dict from collected batches
                full_data = {}
                for k in all_batches[0]:
                    vals = [np.asarray(b[k]) for b in all_batches]
                    full_data[k] = np.concatenate(vals, axis=0)
                full_data['n_people'] = self.num_people

                likelihood_fn = GradedResponseLikelihood(dtype=self.dtype)
                surrogate_dist = self.surrogate_distribution_generator(self.params)
                surrogate_log_prob_fn = lambda p: surrogate_dist.log_prob(p)
                prior_log_prob_fn = lambda p: self.prior_distribution.log_prob(p)

                sampler = AdaptiveImportanceSampler(
                    likelihood_fn, prior_log_prob_fn, surrogate_log_prob_fn
                )
                results = sampler.adaptive_is_loo(
                    full_data, samples,
                    variational=True,
                    khat_threshold=khat_threshold,
                )
                best = results['best']
                loos = np.array(best['ll_loo_psis'])
                ks = np.array(best['khat'])
                loo = float(np.sum(loos))
            except Exception as e:
                print(f"  AIS fallback failed ({e}), using standard PSIS-LOO")

        self.elpd_loo = float(loo)
        self.elpd_loo_se = float(np.std(loos) * np.sqrt(n_obs))
        self.elpd_loo_per_obs = self.elpd_loo / n_obs
        self.elpd_loo_se_per_obs = self.elpd_loo_se / n_obs
        self.elpd_loo_pointwise = loos
        self.elpd_loo_khat = ks
        self.elpd_loo_n_obs = n_obs

        bad_k = np.sum(ks > khat_threshold)
        print(f"  ELPD-LOO: {self.elpd_loo:.2f} "
              f"(SE: {self.elpd_loo_se:.2f})")
        print(f"  ELPD-LOO per obs: {self.elpd_loo_per_obs:.4f} "
              f"(SE: {self.elpd_loo_se_per_obs:.4f})")
        print(f"  k-hat: max={np.max(ks):.3f}, "
              f"mean={np.mean(ks):.3f}, "
              f">{khat_threshold}: {bad_k}/{n_obs}")

    def fit(self, batched_data_factory, initial_values=None,
            compute_elpd_loo=False, elpd_loo_samples=100, **kwargs):
        """Fit the IRT model with optional imputation and ELPD-LOO.

        If ``imputation_model`` is set, wraps the data factory to attach
        ``_imputation_pmfs`` to each batch for analytic Rao-Blackwellized
        marginalization over missing cells.

        After training, runs one additional pass through the data factory
        to compute PSIS-LOO diagnostics (unless ``compute_elpd_loo=False``).

        Args:
            batched_data_factory: Callable returning a data iterator.
            initial_values: Optional initial parameter values.
            compute_elpd_loo: If True, compute and store
                ELPD-LOO after fitting. Default is False.
            elpd_loo_samples: Number of surrogate posterior draws for
                the PSIS-LOO computation (default 100).
            **kwargs: Additional args passed to _calibrate_minibatch_advi.

        Returns:
            (losses, params) tuple.
        """
        kwargs.pop('n_imputation_samples', None)

        if self.imputation_model is not None:
            effective_factory = self._wrap_factory_with_imputation(
                batched_data_factory
            )
        else:
            effective_factory = batched_data_factory

        res = super().fit(
            effective_factory, initial_values=initial_values, **kwargs
        )
        losses, params = res[0], res[1]

        if compute_elpd_loo:
            print("  Computing ELPD-LOO...")
            self._compute_elpd_loo(
                effective_factory, n_samples=elpd_loo_samples
            )

        return res

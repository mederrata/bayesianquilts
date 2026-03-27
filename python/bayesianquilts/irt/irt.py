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
            parameterization="softplus",
            discrimination_prior="half_normal",
            discrimination_prior_scale=2.0,
            expected_sparsity=None,
            slab_scale=2.0,
            slab_df=4,
            rank=0,
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
        self.parameterization = parameterization
        self.discrimination_prior = discrimination_prior
        self.discrimination_prior_scale = discrimination_prior_scale
        self.expected_sparsity = expected_sparsity
        self.slab_scale = slab_scale
        self.slab_df = slab_df
        self.rank = rank

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

    def standardize_abilities(self, weights=None, reference_idx=None):
        """Rescale all model parameters so that abilities are N(0, 1) per dimension.

        The GRM response model is invariant under the affine transform
        ``theta -> (theta - mu) / sigma`` provided we also rescale:

        - ``discriminations *= sigma``
        - ``difficulties0 = (difficulties0 - mu) / sigma``
        - ``ddifficulties /= sigma``

        This modifies ``surrogate_sample`` and ``calibrated_expectations``
        in place.

        Args:
            weights: Optional (N,) array of per-person weights for computing
                the weighted mean and std.  If None, uses uniform weights
                (simple mean/std).
            reference_idx: Optional array of person indices whose abilities
                define the standardization statistics per dimension. All
                abilities (and item parameters) are shifted and scaled using
                the reference subset's mean and std. Useful when a reference
                subpopulation (e.g., the general population group) should
                anchor the ability scale.

        Returns:
            dict with ``mu`` (D,) and ``sigma`` (D,) used for rescaling.
        """
        if self.surrogate_sample is None or 'abilities' not in self.surrogate_sample:
            raise ValueError("No surrogate_sample with abilities — fit the model first")

        abilities = self.surrogate_sample['abilities']  # (S, N, D, 1, 1)
        # Compute mean/std over people (axis=1) and samples (axis=0)
        # abilities[:, :, d, 0, 0] is (S, N) for dimension d
        D = abilities.shape[2] if abilities.ndim >= 3 else 1

        mu = jnp.zeros(D, dtype=abilities.dtype)
        sigma = jnp.ones(D, dtype=abilities.dtype)

        for d in range(D):
            if abilities.ndim == 5:
                # (S, N, D, 1, 1) — posterior samples
                ab_d = abilities[:, :, d, 0, 0]  # (S, N)
            elif abilities.ndim == 4:
                # (N, D, 1, 1) — point estimate
                ab_d = abilities[:, d, 0, 0]  # (N,)
            else:
                ab_d = abilities

            # Subset to reference population for computing statistics
            if reference_idx is not None:
                ref = jnp.asarray(reference_idx)
                if ab_d.ndim == 2:
                    ab_ref = ab_d[:, ref]  # (S, N_ref)
                else:
                    ab_ref = ab_d[ref]  # (N_ref,)
                w_ref = weights[ref] if weights is not None else None
            else:
                ab_ref = ab_d
                w_ref = weights

            if w_ref is not None:
                w = jnp.asarray(w_ref, dtype=abilities.dtype)
                w = w / jnp.sum(w)
                if ab_ref.ndim == 2:
                    m = jnp.sum(ab_ref * w[jnp.newaxis, :], axis=1)
                    v = jnp.sum(w[jnp.newaxis, :] * (ab_ref - m[:, jnp.newaxis])**2, axis=1)
                    mu = mu.at[d].set(jnp.mean(m))
                    sigma = sigma.at[d].set(jnp.sqrt(jnp.mean(v)))
                else:
                    mu = mu.at[d].set(jnp.sum(w * ab_ref))
                    sigma = sigma.at[d].set(jnp.sqrt(jnp.sum(w * (ab_ref - mu[d])**2)))
            else:
                mu = mu.at[d].set(jnp.mean(ab_ref))
                sigma = sigma.at[d].set(jnp.std(ab_ref))

            # Clamp sigma
            sigma = jnp.where(sigma < 1e-8, 1.0, sigma)

        # Build rescaling arrays with proper shapes for broadcasting
        # mu_bc, sigma_bc: (1, 1, D, 1, 1) for 5D or (1, D, 1, 1) for 4D
        if abilities.ndim == 5:
            mu_bc = mu[jnp.newaxis, jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
            sigma_bc = sigma[jnp.newaxis, jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
        else:
            mu_bc = mu[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
            sigma_bc = sigma[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]

        # Discriminations shape: (S, 1, D, I, 1) or (1, D, I, 1)
        disc_ndim = self.surrogate_sample['discriminations'].ndim
        if disc_ndim == 5:
            mu_disc = mu[jnp.newaxis, jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
            sigma_disc = sigma[jnp.newaxis, jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
        else:
            mu_disc = mu[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
            sigma_disc = sigma[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]

        def rescale_dict(d):
            d['abilities'] = (d['abilities'] - mu_bc) / sigma_bc
            d['discriminations'] = d['discriminations'] * sigma_disc
            d['difficulties0'] = (d['difficulties0'] - mu_disc) / sigma_disc
            if 'ddifficulties' in d:
                d['ddifficulties'] = d['ddifficulties'] / sigma_disc
            if 'difficulties' in d:
                d['difficulties'] = (d['difficulties'] - mu_disc) / sigma_disc

        rescale_dict(self.surrogate_sample)

        if self.calibrated_expectations is not None:
            rescale_dict(self.calibrated_expectations)

        if self.calibrated_sd is not None:
            # Standard deviations scale by 1/sigma for abilities/difficulties
            # and by sigma for discriminations
            self.calibrated_sd['abilities'] = self.calibrated_sd['abilities'] / sigma_bc
            self.calibrated_sd['discriminations'] = self.calibrated_sd['discriminations'] * sigma_disc
            self.calibrated_sd['difficulties0'] = self.calibrated_sd['difficulties0'] / sigma_disc
            if 'ddifficulties' in self.calibrated_sd:
                self.calibrated_sd['ddifficulties'] = self.calibrated_sd['ddifficulties'] / sigma_disc

        return {'mu': mu, 'sigma': sigma}

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

        When the imputation model supports importance-sampling mode
        (has ``predict_mice_pmf`` and ``get_item_weight``), returns
        MICE-only PMFs and per-item stacking weights.  Otherwise falls
        back to the blended ``predict_pmf`` interface.

        Args:
            batch: dict mapping keys to arrays.

        Returns:
            tuple (pmfs, weights) where:
              - pmfs: np.ndarray (N, I, K) with PMFs for missing cells
              - weights: np.ndarray (I,) with per-item stacking weights,
                or None if the imputation model doesn't support IS mode
        """
        N = len(batch[self.item_keys[0]])
        I = self.num_items
        K = self.response_cardinality
        pmfs = np.zeros((N, I, K), dtype=np.float64)

        # Check if the imputation model supports IS mode
        use_is = (hasattr(self.imputation_model, 'predict_mice_pmf')
                  and hasattr(self.imputation_model, 'get_item_weight'))
        weights = np.zeros(I, dtype=np.float64) if use_is else None

        for i, item_key in enumerate(self.item_keys):
            col = np.asarray(batch[item_key], dtype=np.float64)
            bad = np.isnan(col) | (col < 0) | (col >= K)
            bad_indices = np.where(bad)[0]

            if use_is:
                weights[i] = self.imputation_model.get_item_weight(item_key)

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

                # Pass person index if available (for IRT baseline PMFs)
                person_idx = None
                if self.person_key in batch:
                    person_idx = int(batch[self.person_key][row_idx])

                try:
                    if use_is:
                        pmf = self.imputation_model.predict_mice_pmf(
                            observed_items, target=item_key, n_categories=K,
                        )
                    else:
                        kwargs = {}
                        if person_idx is not None:
                            kwargs['person_idx'] = person_idx
                        pmf = self.imputation_model.predict_pmf(
                            observed_items, target=item_key, n_categories=K,
                            **kwargs,
                        )
                    pmfs[row_idx, i, :] = pmf
                except (ValueError, KeyError, AttributeError, TypeError):
                    pmfs[row_idx, i, :] = 1.0 / K

        return pmfs, weights

    def _wrap_factory_with_imputation(self, factory):
        """Wrap a data factory to attach imputation PMFs from the imputation model.

        Returns a new factory that adds ``_imputation_pmfs`` and optionally
        ``_imputation_weights`` to every batch.  When weights are present,
        the GRM uses importance-sampling mode where the contribution of
        each missing cell is scaled by its stacking weight, so that
        w_mice=0 items reduce to ignorability.
        """
        model_ref = self

        def _attach_imputation(batch):
            if model_ref._has_missing_values(batch):
                pmfs, weights = model_ref._compute_batch_pmfs(batch)
                batch['_imputation_pmfs'] = pmfs
                if weights is not None:
                    batch['_imputation_weights'] = weights
            else:
                N = len(batch[model_ref.item_keys[0]])
                batch['_imputation_pmfs'] = np.zeros(
                    (N, model_ref.num_items, model_ref.response_cardinality),
                    dtype=np.float64,
                )

        def imputing_factory():
            def imputing_iterator():
                iterator = factory()
                try:
                    for batch in iterator:
                        _attach_imputation(batch)
                        yield batch
                except TypeError:
                    batch = iterator
                    _attach_imputation(batch)
                    yield batch
            return imputing_iterator()

        return imputing_factory

    def _compute_elpd_loo(self, data_factory, n_samples=100, seed=42,
                          khat_threshold=0.7, use_ais=True):
        """Compute PSIS-LOO after fitting, with adaptive IS refinement.

        Iterates one epoch through the factory, computes per-person
        log-likelihoods under ``n_samples`` surrogate posterior draws,
        and runs PSIS-LOO.  Then uses AdaptiveImportanceSampler to
        improve estimates (always when ``use_ais=True``, or only for
        high k-hat when ``use_ais=False``).

        Results are stored as attributes on ``self`` so they persist
        via ``save_to_disk``.

        Args:
            data_factory: Callable returning an iterator of batch dicts
                (same as the training data factory).
            n_samples: Surrogate posterior draws for PSIS.
            seed: Random seed for sampling.
            khat_threshold: Diagnostic threshold for k-hat.
            use_ais: If True, always run AIS for all observations.
                If False, only run AIS for observations with k-hat
                above ``khat_threshold``.
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

        # Collect full data for AIS pass
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

        # Initial PSIS-LOO (used as baseline / when AIS fails)
        loo, loos, ks = psisloo(log_lik_matrix)

        bad_k_initial = int(np.sum(ks > khat_threshold))
        print(f"  PSIS-LOO (initial): {float(loo):.2f}, "
              f"k-hat > {khat_threshold}: {bad_k_initial}/{n_obs}")

        # Use more posterior samples for better PSIS when many k-hat are bad
        if bad_k_initial > n_obs * 0.1 and n_samples < 200:
            print(f"  Many high k-hat ({bad_k_initial}/{n_obs}), "
                  f"resampling with {max(200, n_samples * 2)} draws...")
            n_samples_2 = max(200, n_samples * 2)
            key2 = jax.random.PRNGKey(seed + 999)
            samples2 = surrogate.sample(n_samples_2, seed=key2)
            if hasattr(self, 'transform'):
                samples2 = self.transform(samples2)

            log_lik_matrix2 = np.full(
                (n_samples_2, self.num_people), np.nan, dtype=np.float64
            )
            for batch in data_factory():
                people2 = np.asarray(batch[self.person_key], dtype=np.int32)
                pred2 = self.predictive_distribution(batch, **samples2)
                log_lik_matrix2[:, people2] = np.array(pred2['log_likelihood'])

            log_lik_matrix2 = log_lik_matrix2[:, visited] if n_obs < self.num_people else log_lik_matrix2
            loo2, loos2, ks2 = psisloo(log_lik_matrix2)
            bad_k2 = int(np.sum(ks2 > khat_threshold))
            print(f"  PSIS-LOO (resampled): {float(loo2):.2f}, "
                  f"k-hat > {khat_threshold}: {bad_k2}/{n_obs}")
            if not np.isnan(loo2) and (np.isnan(loo) or bad_k2 < bad_k_initial):
                loo, loos, ks = loo2, loos2, ks2

        # AIS refinement for high k-hat observations
        bad_k_count = int(np.sum(ks > khat_threshold))
        if use_ais and bad_k_count > 0 and hasattr(self, '_prepare_ais_inputs'):
            print(f"  Running AIS for {bad_k_count} high k-hat observations...")
            try:
                from bayesianquilts.metrics.ais import AdaptiveImportanceSampler
                likelihood_fn, ais_data, ais_params = self._prepare_ais_inputs(
                    all_batches, samples
                )
                sampler = AdaptiveImportanceSampler(likelihood_fn)
                ais_result = sampler.adaptive_is_loo(
                    data=ais_data,
                    params=ais_params,
                    transformations=['identity', 'mm1', 'mm2', 'pmm1', 'pmm2', 'll'],
                    khat_threshold=khat_threshold,
                    verbose=False,
                )
                ais_loos = np.array(ais_result['ll_loo_psis'])  # (N,)
                ais_khats = np.array(ais_result['khat'])  # (N,)

                # Replace PSIS estimates with AIS where AIS achieved lower k-hat
                improved = 0
                for i in range(n_obs):
                    if ks[i] > khat_threshold and ais_khats[i] < ks[i]:
                        loos[i] = ais_loos[i]
                        ks[i] = ais_khats[i]
                        improved += 1

                loo = float(np.sum(loos))
                bad_k_after = int(np.sum(ks > khat_threshold))
                print(f"  AIS improved {improved}/{bad_k_count} observations, "
                      f"k-hat > {khat_threshold}: {bad_k_after}/{n_obs}")
            except Exception as e:
                print(f"  AIS failed: {e}")
                import traceback
                traceback.print_exc()

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

    def fit_is(self, data_factory, imputation_model,
               n_samples=256, batch_size=4, seed=271828):
        """Reweight the baseline posterior via importance sampling.

        Instead of refitting with imputation, draws samples from the
        baseline (ignorability) posterior and reweights them by the
        ratio of the imputation-adjusted to the baseline likelihood.

        Since observed cells cancel and the prior cancels, the IS
        log-weight for sample s reduces to:

            log w_s = Σ_{missing (n,i)} w_mice_i * log[Σ_k q_mice(k) * p(Y=k|θ_s)]

        When w_mice=0 for all items, all log-weights are 0 and the
        result is identical to the baseline (ignorability).

        Uses PSIS (Vehtari et al. 2024) to smooth the weights and
        diagnose reliability via k-hat.

        Parameters
        ----------
        data_factory : callable
            Returns an iterator over data batches.
        imputation_model : IrtMixedImputationModel
            Fitted mixed imputation model with ``predict_mice_pmf``
            and ``get_item_weight`` methods.
        n_samples : int
            Number of posterior samples to draw.
        batch_size : int
            Chunk size over posterior samples (memory control).
        seed : int
            Random seed for sampling.

        Returns
        -------
        dict with keys:
            - ``is_weights``: normalized IS weights, shape (S,)
            - ``log_is_weights``: unnormalized log IS weights, shape (S,)
            - ``psis_weights``: PSIS-smoothed weights, shape (S,)
            - ``khat``: PSIS diagnostic (< 0.7 is reliable)
            - ``ess``: effective sample size
            - ``abilities_is``: IS-reweighted ability point estimates (N, D, 1, 1)
            - ``samples``: the posterior samples dict
        """
        from bayesianquilts.metrics.nppsis import psisloo

        item_keys = self.item_keys
        K = self.response_cardinality
        I = len(item_keys)

        # Per-item stacking weights
        item_weights = np.array(
            [imputation_model.get_item_weight(k) for k in item_keys],
            dtype=np.float64,
        )  # (I,)

        # Skip IS entirely if all weights are zero (pure ignorability)
        if np.allclose(item_weights, 0.0):
            print("  All stacking weights are 0 — IS is identical to baseline.")
            self.is_weights = None
            self.is_khat = 0.0
            return {
                'is_weights': None,
                'log_is_weights': None,
                'psis_weights': None,
                'khat': 0.0,
                'ess': float(n_samples),
                'abilities_is': self.calibrated_expectations.get('abilities'),
                'samples': self.surrogate_sample,
            }

        # Draw posterior samples from baseline
        surrogate = self.surrogate_distribution_generator(self.params)
        key = jax.random.PRNGKey(seed)
        samples = surrogate.sample(n_samples, seed=key)

        if hasattr(self, 'transform') and 'discriminations' not in samples:
            samples = self.transform(dict(samples))

        disc_all = np.asarray(samples["discriminations"])
        diff0_all = np.asarray(samples["difficulties0"])
        ddiff_all = (np.asarray(samples["ddifficulties"])
                     if "ddifficulties" in samples else None)
        abil_all = np.asarray(samples["abilities"])
        S = disc_all.shape[0]

        # Precompute MICE-only PMFs for all missing cells
        # We accumulate log IS weights across batches
        log_is_weights = np.zeros(S, dtype=np.float64)

        for batch in data_factory():
            people = np.asarray(batch[self.person_key], dtype=np.int64)
            N_batch = len(people)

            # Identify missing cells: (N_batch, I)
            responses = np.stack(
                [np.asarray(batch[k], dtype=np.float64) for k in item_keys],
                axis=-1,
            )
            bad_mask = np.isnan(responses) | (responses < 0) | (responses >= K)

            if not np.any(bad_mask):
                continue

            # Compute MICE PMFs for missing cells in this batch: (N_batch, I, K)
            mice_pmfs = np.zeros((N_batch, I, K), dtype=np.float64)
            for i, item_key in enumerate(item_keys):
                if item_weights[i] == 0.0:
                    continue
                bad_rows = np.where(bad_mask[:, i])[0]
                if len(bad_rows) == 0:
                    continue
                for row_idx in bad_rows:
                    observed_items = {}
                    for j, other_key in enumerate(item_keys):
                        if other_key == item_key:
                            continue
                        val = float(responses[row_idx, j])
                        if not (np.isnan(val) or val < 0 or val >= K):
                            observed_items[other_key] = val
                    try:
                        mice_pmfs[row_idx, i, :] = imputation_model.predict_mice_pmf(
                            observed_items, target=item_key, n_categories=K,
                        )
                    except (ValueError, KeyError, AttributeError, TypeError):
                        mice_pmfs[row_idx, i, :] = 1.0 / K

            # Compute response probs p(Y=k|θ_s) chunked over S
            for s_start in range(0, S, batch_size):
                s_end = min(s_start + batch_size, S)
                s_chunk = s_end - s_start

                disc_chunk = jnp.asarray(disc_all[s_start:s_end])
                diff0_chunk = jnp.asarray(diff0_all[s_start:s_end])
                ddiff_chunk = (jnp.asarray(ddiff_all[s_start:s_end])
                               if ddiff_all is not None else None)
                abil_chunk = jnp.asarray(abil_all[s_start:s_end])

                abil_people = abil_chunk[:, people, ...]

                # (s_chunk, N_batch, I, K)
                response_probs = np.asarray(
                    self.grm_model_prob_d(
                        abil_people, disc_chunk, diff0_chunk, ddiff_chunk,
                    )
                )

                # For each missing cell, compute:
                # w_i * log[sum_k q_mice(k) * p(Y=k|theta_s)]
                log_rp = np.log(np.maximum(response_probs, 1e-30))  # (s_chunk, N, I, K)
                log_q = np.log(np.maximum(mice_pmfs, 1e-30))  # (N, I, K)

                # log[sum_k q(k)*p(k|theta)] via logsumexp
                rb = np.asarray(jax.scipy.special.logsumexp(
                    jnp.asarray(log_rp) + jnp.asarray(log_q)[jnp.newaxis, ...],
                    axis=-1,
                ))  # (s_chunk, N_batch, I)

                # Weight by w_mice per item and mask to missing only
                weighted_rb = rb * item_weights[np.newaxis, np.newaxis, :]  # (s_chunk, N, I)
                weighted_rb = np.where(bad_mask[np.newaxis, ...], weighted_rb, 0.0)

                # Sum over people and items, accumulate per sample
                log_is_weights[s_start:s_end] += weighted_rb.sum(axis=(1, 2))

        # PSIS smoothing
        # psisloo expects (S, N) log-likelihood matrix, but we have
        # aggregate log-weights (S,). Use PSIS on the weights directly.
        # Reshape to (S, 1) for the psisloo interface.
        log_w_matrix = log_is_weights[:, np.newaxis]

        # Normalize for numerical stability
        log_w_matrix -= log_w_matrix.max()

        # Use PSIS to smooth
        try:
            _, _, ks = psisloo(log_w_matrix)
            khat = float(ks[0])
        except Exception:
            khat = np.inf

        # Compute normalized IS weights
        log_w = log_is_weights - log_is_weights.max()
        raw_weights = np.exp(log_w)
        raw_weights /= raw_weights.sum()

        # PSIS-smoothed weights (truncated Pareto smoothing)
        from bayesianquilts.metrics.nppsis import gpdfitnew as gpdfit
        try:
            # Sort and smooth the tail
            sorted_idx = np.argsort(log_is_weights)
            log_w_sorted = log_is_weights[sorted_idx]
            # Use top min(S/5, 3*sqrt(S)) for tail fitting
            M = min(S // 5, int(3 * np.sqrt(S)))
            if M > 4:
                cutoff = log_w_sorted[-(M + 1)]
                tail = log_w_sorted[-M:] - cutoff
                k_est, sigma = gpdfit(tail)
                if k_est < 0.7:
                    # Replace tail with smoothed values
                    from scipy.stats import genpareto
                    order = np.arange(1, M + 1, dtype=np.float64)
                    p = (order - 0.5) / M
                    smoothed_tail = genpareto.ppf(p, k_est, scale=sigma) + cutoff
                    psis_log_w = log_is_weights.copy()
                    psis_log_w[sorted_idx[-M:]] = smoothed_tail
                else:
                    psis_log_w = log_is_weights.copy()
            else:
                psis_log_w = log_is_weights.copy()
        except Exception:
            psis_log_w = log_is_weights.copy()

        psis_w = np.exp(psis_log_w - psis_log_w.max())
        psis_w /= psis_w.sum()

        # Effective sample size
        ess = 1.0 / np.sum(psis_w ** 2)

        # IS-reweighted ability estimates
        abilities = np.asarray(samples["abilities"])  # (S, N, D, 1, 1)
        abilities_is = np.einsum('s,s...->...', psis_w, abilities)

        print(f"  IS reweighting: k-hat={khat:.3f}, ESS={ess:.1f}/{S}, "
              f"max log-w={log_is_weights.max():.2f}, "
              f"min log-w={log_is_weights.min():.2f}")

        # Store on model for downstream use
        self.is_weights = psis_w
        self.is_log_weights = log_is_weights
        self.is_khat = khat
        self.is_ess = ess
        self.is_abilities = abilities_is
        self.is_samples = samples

        # Also update calibrated expectations with IS-reweighted values
        self.calibrated_expectations_is = {}
        for key in samples:
            arr = np.asarray(samples[key])
            self.calibrated_expectations_is[key] = np.einsum(
                's,s...->...', psis_w, arr
            )

        return {
            'is_weights': raw_weights,
            'log_is_weights': log_is_weights,
            'psis_weights': psis_w,
            'khat': khat,
            'ess': ess,
            'abilities_is': abilities_is,
            'samples': samples,
        }

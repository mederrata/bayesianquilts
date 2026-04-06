import sys
import warnings
from typing import Any, Dict, Optional

import jax
import numpy as np
import jax.numpy as jnp
from flax import nnx
from bayesianquilts.model import BayesianModel
from bayesianquilts.predictors.nn.dense import Dense
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf


def _warn_fallback(msg, exc=None):
    """Print a red warning about a fallback to degraded behavior."""
    detail = f" ({type(exc).__name__}: {exc})" if exc else ""
    sys.stderr.write(
        f"\033[91mWARNING: {msg}{detail}\033[0m\n"
    )
    sys.stderr.flush()


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
            _warn_fallback(
                f"Key '{k}' not found in samples — "
                f"surrogate_sample NOT updated")
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

    def standardize_marginal(self, data, reference_idx=None, weights=None):
        """Standardize MCMC item parameters so implied abilities are N(0,1).

        After marginal inference (fit_marginal_mcmc), abilities are not
        model parameters. This method:
        1. Computes EAP abilities from current MCMC item params
        2. Finds mu, sigma of the ability distribution (optionally on
           a reference subset)
        3. Rescales all MCMC item param samples in place

        The GRM is invariant under theta -> (theta - mu) / sigma if:
          discriminations *= sigma
          difficulties0 = (difficulties0 - mu) / sigma
          ddifficulties /= sigma

        After standardization, compute_eap_abilities will return N(0,1)
        abilities (or N(0,1) on the reference subset).

        Args:
            data: Full data dict for computing EAP abilities.
            reference_idx: Optional person indices for the reference
                subpopulation. If None, uses all people.
            weights: Optional per-person weights for computing mu/sigma.

        Returns:
            dict with ``mu`` and ``sigma`` used for rescaling.
        """
        if not hasattr(self, 'mcmc_samples') or self.mcmc_samples is None:
            raise ValueError("No mcmc_samples — run fit_marginal_mcmc first")

        # Compute EAP abilities from current (un-standardized) item params
        eap_result = self.compute_eap_abilities(data)
        eap = np.array(eap_result['eap'])  # (N,)

        # Compute mu, sigma on reference subset
        if reference_idx is not None:
            ref = np.asarray(reference_idx)
            eap_ref = eap[ref]
            w_ref = weights[ref] if weights is not None else None
        else:
            eap_ref = eap
            w_ref = weights

        if w_ref is not None:
            w = np.asarray(w_ref, dtype=np.float64)
            w = w / w.sum()
            mu = float(np.sum(w * eap_ref))
            sigma = float(np.sqrt(np.sum(w * (eap_ref - mu) ** 2)))
        else:
            mu = float(np.mean(eap_ref))
            sigma = float(np.std(eap_ref))

        if sigma < 1e-8:
            sigma = 1.0

        print(f"  Standardizing marginal: mu={mu:.4f}, sigma={sigma:.4f}")

        # Rescale MCMC samples in place
        # Shapes: (chains, samples, 1, 1, I, 1) for disc/diff0
        #         (chains, samples, 1, 1, I, K-2) for ddiff
        for key in list(self.mcmc_samples.keys()):
            if key == 'discriminations' or key.startswith('discriminations_'):
                self.mcmc_samples[key] = self.mcmc_samples[key] * sigma
            elif key == 'difficulties0' or key.startswith('difficulties0_'):
                self.mcmc_samples[key] = (self.mcmc_samples[key] - mu) / sigma
            elif key == 'ddifficulties' or key.startswith('ddifficulties_'):
                self.mcmc_samples[key] = self.mcmc_samples[key] / sigma
            elif key == 'mu' or key.startswith('mu_'):
                self.mcmc_samples[key] = (self.mcmc_samples[key] - mu) / sigma

        return {'mu': mu, 'sigma': sigma}

    def fit_surrogate_to_mcmc(self, mcmc_samples=None):
        """Fit the variational surrogate to MCMC samples via moment matching.

        For a mean-field normal surrogate, sets loc = mean, scale = std
        per coordinate. This populates ``self.params`` so that
        ``surrogate_distribution_generator`` produces a distribution
        that approximates the MCMC posterior.

        Also populates ``self.surrogate_sample`` with the MCMC samples
        (flattened across chains) so that existing downstream code
        (standardize_abilities, predictive_distribution, etc.) works.

        Args:
            mcmc_samples: Dict mapping param names to arrays of shape
                (chains, samples, ...). If None, uses self.mcmc_samples.
        """
        if mcmc_samples is None:
            mcmc_samples = self.mcmc_samples
        if mcmc_samples is None:
            raise ValueError("No MCMC samples provided or stored")

        if self.params is None:
            raise ValueError(
                "No surrogate params — call create_distributions first"
            )

        new_params = dict(self.params)

        for var_name, samples in mcmc_samples.items():
            # Flatten chains: (C, S, ...) → (C*S, ...)
            flat = samples.reshape(-1, *samples.shape[2:])
            loc = jnp.mean(flat, axis=0)
            scale = jnp.std(flat, axis=0)
            scale = jnp.maximum(scale, 1e-6)

            # Find matching surrogate params
            for pk in list(new_params.keys()):
                if pk.startswith(var_name) and pk.endswith('loc'):
                    new_params[pk] = loc
                elif pk.startswith(var_name) and pk.endswith('scale'):
                    # Surrogate scale is in unconstrained space
                    # (softplus^{-1}(scale) for softplus parameterization)
                    if self.parameterization == 'log_scale':
                        new_params[pk] = jnp.log(scale)
                    else:
                        # softplus^{-1}(x) = log(exp(x) - 1)
                        new_params[pk] = jnp.log(jnp.exp(scale) - 1.0)

        self.params = new_params

        # Populate surrogate_sample with flattened MCMC samples
        self.surrogate_sample = {}
        for var_name, samples in mcmc_samples.items():
            self.surrogate_sample[var_name] = samples.reshape(
                -1, *samples.shape[2:]
            )

        # For marginal models, abilities aren't in mcmc_samples.
        # If abilities are needed downstream, caller should run
        # compute_eap_abilities and inject them.

        print(f"  Surrogate fitted to MCMC ({len(mcmc_samples)} variables)")
        for var_name, samples in mcmc_samples.items():
            flat = samples.reshape(-1, *samples.shape[2:])
            print(f"    {var_name}: loc={float(jnp.mean(flat)):.4f}, "
                  f"scale={float(jnp.std(flat)):.4f}")

    def importance_reweight(
        self,
        data,
        mcmc_samples,
        imputation_model,
        fn=None,
        theta_grid=None,
        max_samples=None,
        seed=42,
        verbose=True,
    ):
        """Reweight baseline MCMC samples to approximate the imputed posterior.

        Given posterior samples from a baseline (ignorable-missingness) model,
        computes importance weights using the ratio of imputed to baseline
        marginal likelihoods. Returns the IS-weighted expectation and standard
        deviation of ``fn(params)`` under the imputed posterior.

        Performs PSIS smoothing and reports diagnostics (ESS, k-hat).
        When k-hat > 0.7, attempts adaptive tempering as a fallback.

        Args:
            data: Full data dict (all people, not batched). Should NOT
                already contain ``_imputation_pmfs``.
            mcmc_samples: Dict mapping param names to arrays of shape
                (chains, samples, ...) from the baseline model.
            imputation_model: A fitted imputation model (e.g.,
                PairwiseOrdinalStackingModel or IrtMixedImputationModel).
            fn: Callable mapping a param dict → scalar or array.
                Each param dict has the same keys as mcmc_samples with
                values for a single draw. If None, returns reweighted
                samples dict and diagnostics only.
            theta_grid: Quadrature grid for marginal_log_prob.
            max_samples: Cap on number of MCMC draws to use (subsampled
                randomly). None = use all.
            seed: Random seed for subsampling.
            verbose: Print progress and diagnostics.

        Returns:
            Dict with keys:
                'expectation': E_adj[fn] (or None if fn is None)
                'std': std_adj[fn] (or None if fn is None)
                'log_weights': raw log IS weights (S,)
                'psis_weights': PSIS-smoothed normalized weights (S,)
                'khat': PSIS k-hat diagnostic
                'ess': effective sample size
                'n_samples': number of samples used
                'tempered': whether adaptive tempering was applied
        """
        from bayesianquilts.metrics.nppsis import psisloo

        # --- Flatten MCMC chains ---
        flat_samples = {}
        first_key = list(mcmc_samples.keys())[0]
        n_chains, n_samp = mcmc_samples[first_key].shape[:2]
        S_total = n_chains * n_samp
        for k, v in mcmc_samples.items():
            flat_samples[k] = np.asarray(v).reshape(-1, *v.shape[2:])

        # Subsample if requested
        rng = np.random.default_rng(seed)
        if max_samples is not None and max_samples < S_total:
            idx = rng.choice(S_total, max_samples, replace=False)
            for k in flat_samples:
                flat_samples[k] = flat_samples[k][idx]
            S = max_samples
        else:
            S = S_total

        if verbose:
            print(f"  IS reweighting: {S} samples "
                  f"({n_chains} chains × {n_samp} draws)")

        # --- Prepare quadrature grid ---
        if theta_grid is None:
            tg, tlw = self._make_gauss_hermite_grid()
        else:
            tg = jnp.asarray(theta_grid)
            dtheta = tg[1] - tg[0]
            tlw = -0.5 * tg**2 - 0.5 * jnp.log(2*jnp.pi) + jnp.log(dtheta)

        # --- Prepare imputation PMFs ---
        old_imputation = getattr(self, 'imputation_model', None)
        self.imputation_model = imputation_model
        data_imputed = dict(data)
        pmfs, weights = self._compute_batch_pmfs(data_imputed)
        if pmfs is not None:
            data_imputed['_imputation_pmfs'] = pmfs
            if weights is not None:
                data_imputed['_imputation_weights'] = weights
        self.imputation_model = old_imputation

        data_baseline = {k: v for k, v in data.items()
                         if k not in ('_imputation_pmfs', '_imputation_weights')}

        # --- Compute log IS weights ---
        log_weights = np.zeros(S)
        fn_values = []

        for s in range(S):
            # Extract single-draw params
            draw_params = {}
            for k, v in flat_samples.items():
                draw_params[k] = jnp.asarray(v[s])

            # marginal_log_prob with and without imputation
            ll_baseline = float(self.marginal_log_prob(
                data_baseline, theta_grid=tg,
                theta_log_weights=tlw, prior_weight=0.0,
                **draw_params))
            ll_imputed = float(self.marginal_log_prob(
                data_imputed, theta_grid=tg,
                theta_log_weights=tlw, prior_weight=0.0,
                **draw_params))

            log_weights[s] = ll_imputed - ll_baseline

            if fn is not None:
                fn_values.append(fn(draw_params))

            if verbose and (s + 1) % 50 == 0:
                print(f"    Sample {s+1}/{S}, "
                      f"mean log w: {np.mean(log_weights[:s+1]):.4f}")
            sys.stdout.flush()

        # --- PSIS smoothing ---
        result = self._psis_smooth_and_diagnose(
            log_weights, fn_values if fn is not None else None, verbose)
        result['n_samples'] = S

        # --- Adaptive tempering if k-hat is bad ---
        if result['khat'] > 0.7 and pmfs is not None:
            if verbose:
                print(f"  k-hat={result['khat']:.3f} > 0.7, "
                      f"trying adaptive tempering...")
            tempered = self._tempered_reweight(
                data_baseline, data_imputed, flat_samples,
                tg, tlw, fn, S, verbose)
            if tempered is not None and tempered['khat'] < result['khat']:
                tempered['tempered'] = True
                if verbose:
                    print(f"  Tempering improved k-hat: "
                          f"{result['khat']:.3f} → {tempered['khat']:.3f}")
                return tempered
            elif verbose:
                print(f"  Tempering did not improve; using direct IS")

        result['tempered'] = False
        return result

    def _psis_smooth_and_diagnose(self, log_weights, fn_values, verbose):
        """Apply PSIS smoothing to log IS weights and compute diagnostics."""
        from bayesianquilts.metrics.nppsis import psisloo

        S = len(log_weights)

        # PSIS expects (S, N) — treat as single "observation"
        lw_2d = log_weights[:, None]  # (S, 1)
        try:
            _, loos, ks = psisloo(lw_2d)
            khat = float(ks[0])
        except Exception as exc:
            _warn_fallback(
                "PSIS-LOO failed, setting k-hat=inf "
                "(IS weights may be unreliable)", exc)
            khat = float('inf')

        # Normalize weights
        log_w_shifted = log_weights - np.max(log_weights)
        weights = np.exp(log_w_shifted)
        weights /= weights.sum()

        # ESS
        ess = 1.0 / np.sum(weights ** 2)

        if verbose:
            print(f"  PSIS k-hat: {khat:.3f}, ESS: {ess:.1f}/{S} "
                  f"({100*ess/S:.1f}%)")

        result = {
            'log_weights': log_weights,
            'psis_weights': weights,
            'khat': khat,
            'ess': ess,
            'expectation': None,
            'std': None,
        }

        if fn_values is not None:
            fn_arr = np.array(fn_values)
            if fn_arr.ndim == 1:
                exp_val = float(np.sum(weights * fn_arr))
                var_val = float(np.sum(weights * (fn_arr - exp_val)**2))
                result['expectation'] = exp_val
                result['std'] = float(np.sqrt(max(var_val, 0.0)))
            else:
                exp_val = np.sum(weights[:, None] * fn_arr, axis=0)
                var_val = np.sum(
                    weights[:, None] * (fn_arr - exp_val[None, :])**2,
                    axis=0)
                result['expectation'] = exp_val
                result['std'] = np.sqrt(np.maximum(var_val, 0.0))

        return result

    def _tempered_reweight(self, data_baseline, data_imputed,
                           flat_samples, tg, tlw, fn, S, verbose):
        """Bridge between baseline and imputed via geometric tempering.

        Uses a sequence of tempered distributions:
            pi_t(params) propto pi_baseline(params) * w(params)^t
        for t in [0.25, 0.5, 0.75, 1.0], performing sequential IS
        at each stage.
        """
        temps = [0.25, 0.5, 0.75, 1.0]
        current_log_weights = np.zeros(S)
        best_result = None

        for t_idx, temp in enumerate(temps):
            stage_log_weights = np.zeros(S)
            fn_values = [] if fn is not None else None

            for s in range(S):
                draw_params = {k: jnp.asarray(v[s])
                               for k, v in flat_samples.items()}

                ll_b = float(self.marginal_log_prob(
                    data_baseline, theta_grid=tg,
                    theta_log_weights=tlw, prior_weight=0.0,
                    **draw_params))
                ll_i = float(self.marginal_log_prob(
                    data_imputed, theta_grid=tg,
                    theta_log_weights=tlw, prior_weight=0.0,
                    **draw_params))

                # Tempered weight: w^t relative to previous stage
                if t_idx == 0:
                    stage_log_weights[s] = temp * (ll_i - ll_b)
                else:
                    prev_temp = temps[t_idx - 1]
                    stage_log_weights[s] = (
                        (temp - prev_temp) * (ll_i - ll_b))

                if fn is not None:
                    fn_values.append(fn(draw_params))

            current_log_weights += stage_log_weights
            result = self._psis_smooth_and_diagnose(
                current_log_weights, fn_values, verbose=False)

            if verbose:
                print(f"    Temper t={temp:.2f}: k-hat={result['khat']:.3f}, "
                      f"ESS={result['ess']:.1f}")

            if result['khat'] < 0.7:
                return result
            best_result = result

        return best_result

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
        zero_results = getattr(im, 'marginal_results', {})
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

    # Default tolerance for treating w_irt as 1 (ignorable missingness).
    # Overridden per item when adaptive thresholds are computed.
    ignorable_tol = 0.01

    # Per-item adaptive thresholds, keyed by item name.
    # Set by compute_adaptive_thresholds() after the first epoch.
    _adaptive_thresholds: Optional[Dict[str, float]] = None

    def compute_adaptive_thresholds(self, data_factory, baseline_model,
                                     sample_size=32, seed=42,
                                     inference="vi"):
        """Estimate per-item ignorability thresholds.

        Compares the pairwise imputation model's per-item ELPD against
        the baseline IRT model's per-item ELPD. Items where the pairwise
        model does not substantially outperform the baseline are treated
        as ignorable.

        Two modes depending on the inference method:

        ``inference="vi"`` (default): Uses ELBO gradient variance.
        The threshold for item i is Var[log a_i] / (S * |delta_ELPD_i|).
        Items with w_pairwise below their threshold are treated as
        ignorable. S is the MC sample size per gradient step.

        ``inference="mcmc"``: Same variance-based criterion but with
        S=1, since each MCMC sample pays the full variance cost
        without per-step averaging. The threshold for item i is
        Var[log a_i] / |delta_ELPD_i|.

        Args:
            data_factory: Callable returning a data iterator.
            baseline_model: A fitted IRT model (without imputation) whose
                per-item ELPD serves as the reference for what ignorable
                marginalization achieves.
            sample_size: MC draws per ELBO gradient step (VI mode only).
            seed: Random seed for surrogate sampling.
            inference: ``"vi"`` or ``"mcmc"``.
        """
        I = self.num_items
        K = self.response_cardinality
        S = sample_size

        use_is = (hasattr(self.imputation_model, 'predict_mice_pmf')
                  and hasattr(self.imputation_model, 'get_item_weight'))
        im = self.imputation_model

        # -- Compute delta_ELPD: pairwise model vs IRT model, per item --
        # The relevant comparison is whether the pairwise model predicts
        # item i better than the IRT model does. If not, imputing item i
        # adds noise for no gain over the IRT model's own marginalization.
        delta_elpd = np.ones(I) * 0.01

        # Get pairwise model's best per-item ELPD
        if hasattr(im, 'pairwise_model'):
            pw = im.pairwise_model
        elif hasattr(im, 'mice_model'):
            pw = im.mice_model
        else:
            pw = im

        pw_elpd = np.full(I, -np.inf)
        for i, item_key in enumerate(self.item_keys):
            if not hasattr(pw, 'variable_names') or item_key not in pw.variable_names:
                continue
            idx = pw.variable_names.index(item_key)
            for (t, p), r in pw.univariate_results.items():
                if t == idx and r.converged and r.elpd_loo_per_obs > pw_elpd[i]:
                    pw_elpd[i] = r.elpd_loo_per_obs

        # Get IRT model's per-item ELPD from the baseline model
        irt_elpd = np.full(I, -np.inf)

        if getattr(baseline_model, 'surrogate_sample', None) is None:
            raise ValueError(
                "baseline_model has no surrogate_sample — fit the baseline "
                "model and call calibrate_manually() before computing thresholds."
            )

        irt_ll_sum = np.zeros(I)
        irt_ll_count = np.zeros(I)
        for batch in data_factory():
            pred = baseline_model.predictive_distribution(
                batch, **baseline_model.surrogate_sample)
            response_probs = np.array(pred['rv'].probs_parameter())
            for i, item_key in enumerate(self.item_keys):
                col = np.asarray(batch[item_key], dtype=np.float64)
                valid = ~np.isnan(col) & (col >= 0) & (col < K)
                valid_idx = np.where(valid)[0]
                if len(valid_idx) == 0:
                    continue
                y = col[valid_idx].astype(int)
                p_obs = response_probs[:, valid_idx, i, :]
                log_p = np.log(np.maximum(
                    np.mean(p_obs[np.arange(p_obs.shape[0])[:, None],
                                  np.arange(len(y))[None, :], y[None, :]], axis=0),
                    1e-30,
                ))
                irt_ll_sum[i] += log_p.sum()
                irt_ll_count[i] += len(valid_idx)
            break
        irt_elpd = np.where(irt_ll_count > 0, irt_ll_sum / irt_ll_count, -np.inf)

        # delta_ELPD = |ELPD_pairwise - ELPD_IRT| per item
        for i in range(I):
            if np.isfinite(pw_elpd[i]) and np.isfinite(irt_elpd[i]):
                delta_elpd[i] = max(abs(pw_elpd[i] - irt_elpd[i]), 0.001)

        # -- Mode-specific threshold computation --
        if inference == "mcmc":
            # MCMC mode: same variance criterion as VI but with S=1.
            # Each MCMC draw pays the full variance cost — no per-step
            # averaging reduces the noise as in VI.
            if self.surrogate_sample is None:
                return

            log_ai_var = np.zeros(I)
            n_missing = np.zeros(I)

            for batch in data_factory():
                pmfs, _ = self._compute_batch_pmfs(batch)
                if pmfs is None:
                    continue

                pred = self.predictive_distribution(batch, **self.surrogate_sample)
                response_probs = np.array(pred['rv'].probs_parameter())

                for i in range(I):
                    col = np.asarray(batch[self.item_keys[i]], dtype=np.float64)
                    bad = np.isnan(col) | (col < 0) | (col >= K)
                    bad_idx = np.where(bad)[0]
                    if len(bad_idx) == 0:
                        continue

                    q = pmfs[bad_idx, i, :]
                    p = response_probs[:, bad_idx, i, :]
                    log_a = np.log(np.maximum(
                        np.sum(q[np.newaxis, :, :] * p, axis=-1), 1e-30
                    ))
                    var_per_person = np.var(log_a, axis=0)
                    log_ai_var[i] += var_per_person.sum()
                    n_missing[i] += len(bad_idx)
                break

            mean_var = np.where(n_missing > 0, log_ai_var / n_missing, 0.0)
            # S=1: each sample pays full variance cost
            thresholds = mean_var / delta_elpd
        else:
            # VI mode: estimate Var[log a_i(theta)] from surrogate draws
            if self.surrogate_sample is None:
                return

            log_ai_var = np.zeros(I)
            n_missing = np.zeros(I)

            for batch in data_factory():
                pmfs, _ = self._compute_batch_pmfs(batch)
                if pmfs is None:
                    continue

                pred = self.predictive_distribution(batch, **self.surrogate_sample)
                response_probs = np.array(pred['rv'].probs_parameter())

                for i in range(I):
                    col = np.asarray(batch[self.item_keys[i]], dtype=np.float64)
                    bad = np.isnan(col) | (col < 0) | (col >= K)
                    bad_idx = np.where(bad)[0]
                    if len(bad_idx) == 0:
                        continue

                    q = pmfs[bad_idx, i, :]
                    p = response_probs[:, bad_idx, i, :]
                    log_a = np.log(np.maximum(
                        np.sum(q[np.newaxis, :, :] * p, axis=-1), 1e-30
                    ))
                    var_per_person = np.var(log_a, axis=0)
                    log_ai_var[i] += var_per_person.sum()
                    n_missing[i] += len(bad_idx)
                break

            mean_var = np.where(n_missing > 0, log_ai_var / n_missing, 0.0)
            thresholds = mean_var / (S * delta_elpd)

        thresholds = np.maximum(thresholds, 0.001)

        self._adaptive_thresholds = {
            self.item_keys[i]: float(thresholds[i]) for i in range(I)
        }

        # Determine which items are ignorable
        ignorable_items = {}
        for i in range(I):
            item_key = self.item_keys[i]
            if use_is:
                w_pw = im.get_item_weight(item_key)
                ignorable_items[item_key] = w_pw <= thresholds[i]
            else:
                ignorable_items[item_key] = thresholds[i] >= 1.0
        self._ignorable_items = ignorable_items

        if hasattr(self, 'verbose') and self.verbose:
            n_ignored = sum(1 for v in ignorable_items.values() if v)
            print(f"  Adaptive thresholds ({inference}): "
                  f"{n_ignored}/{I} items treated as ignorable")
            print(f"  Threshold range: [{thresholds.min():.4f}, "
                  f"{thresholds.max():.4f}]")

    def _get_item_threshold(self, item_key: str) -> float:
        """Return the ignorability threshold for a given item (IS mode)."""
        if self._adaptive_thresholds is not None and item_key in self._adaptive_thresholds:
            return self._adaptive_thresholds[item_key]
        return self.ignorable_tol

    def _is_item_ignorable(self, item_key: str) -> bool:
        """Check if an item should be treated as ignorable (non-IS mode).

        For the pairwise-only model (no per-item w_pairwise), the implicit
        weight is 1. The item is ignorable when the adaptive threshold
        exceeds 1, meaning Var[log a_i] / (S * |delta_ELPD_i|) >= 1 —
        the variance cost of imputing exceeds the information gain.
        """
        if self._adaptive_thresholds is None:
            return False
        return self._adaptive_thresholds.get(item_key, 0.0) >= 1.0

    def _compute_batch_pmfs(self, batch):
        """Compute imputation PMFs for missing cells using the imputation model.

        When the imputation model supports importance-sampling mode
        (has ``predict_mice_pmf`` and ``get_item_weight``), returns
        MICE-only PMFs and per-item stacking weights.  Otherwise falls
        back to the blended ``predict_pmf`` interface.

        Items where ``w_pairwise`` falls below their per-item ignorability
        threshold are skipped — their missing cells are treated as
        ignorable missingness and contribute 0 to the log-likelihood.

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
                # Skip imputation for items below their ignorability threshold
                if weights[i] <= self._get_item_threshold(item_key):
                    continue
            elif self._adaptive_thresholds is not None:
                # Non-IS mode (pairwise-only): skip items where the adaptive
                # threshold indicates imputation adds more variance than signal.
                # In this mode w_pairwise is implicitly 1, so check whether
                # the item's delta-ELPD justifies the variance cost.
                if self._is_item_ignorable(item_key):
                    continue

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
                        # Only pass person_idx if the model accepts it
                        import inspect as _inspect
                        _sig = _inspect.signature(
                            self.imputation_model.predict_pmf)
                        kwargs = {}
                        if (person_idx is not None
                                and 'person_idx' in _sig.parameters):
                            kwargs['person_idx'] = person_idx
                        pmf = self.imputation_model.predict_pmf(
                            observed_items, target=item_key, n_categories=K,
                            **kwargs,
                        )
                    pmfs[row_idx, i, :] = pmf
                except (ValueError, KeyError, AttributeError) as exc:
                    _warn_fallback(
                        f"Imputation predict_pmf failed for item "
                        f"'{item_key}' person {row_idx}, "
                        f"falling back to uniform 1/{K}", exc)
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
                except TypeError as exc:
                    _warn_fallback(
                        "Data factory returned non-iterable, "
                        "treating as single batch", exc)
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

        # AIS refinement for high k-hat observations.
        # Process ONE observation at a time to minimize memory.
        # Try transformations in order of complexity, stop early per obs.
        bad_k_count = int(np.sum(ks > khat_threshold))
        if use_ais and bad_k_count > 0 and hasattr(self, '_prepare_ais_inputs'):
            print(f"  Running AIS for {bad_k_count} high k-hat observations (one at a time)...")
            try:
                from bayesianquilts.metrics.ais import AdaptiveImportanceSampler
                likelihood_fn, ais_data, ais_params = self._prepare_ais_inputs(
                    all_batches, samples
                )

                transforms_by_complexity = [
                    'identity', 'mm1', 'mm2', 'mm3',
                    'pmm1', 'pmm2', 'pmm3',
                    'kl', 'll', 'mixis',
                ]

                bad_indices = np.where(ks > khat_threshold)[0]
                total_improved = 0

                for count, obs_idx in enumerate(bad_indices):
                    # Build single-observation subset of ais_data
                    person_mask = ais_data['person_idx'] == obs_idx
                    obs_data = {
                        'person_idx': np.zeros(int(person_mask.sum()), dtype=np.int32),
                        'item_idx': ais_data['item_idx'][person_mask],
                        'y': ais_data['y'][person_mask],
                        'n_people': 1,
                    }
                    obs_params = {
                        k: v[:, obs_idx:obs_idx+1] if v.ndim >= 2 and v.shape[1] > 1 else v
                        for k, v in ais_params.items()
                    }

                    best_loo_i = loos[obs_idx]
                    best_khat_i = ks[obs_idx]
                    sampler = AdaptiveImportanceSampler(likelihood_fn)

                    print(f"    Obs {obs_idx} (k-hat={best_khat_i:.3f}, "
                          f"{count+1}/{len(bad_indices)}):", end="", flush=True)

                    for t_name in transforms_by_complexity:
                        if best_khat_i <= khat_threshold:
                            break
                        try:
                            ais_result = sampler.adaptive_is_loo(
                                data=obs_data,
                                params=obs_params,
                                transformations=[t_name],
                                khat_threshold=khat_threshold,
                                verbose=False,
                            )
                            t_khat = float(np.array(ais_result['khat'])[0])
                            t_loo = float(np.array(ais_result['ll_loo_psis'])[0])

                            if t_khat < best_khat_i:
                                best_khat_i = t_khat
                                best_loo_i = t_loo
                                print(f" {t_name}={t_khat:.3f}", end="", flush=True)
                        except Exception as exc:
                            _warn_fallback(
                                f"AIS transformation '{t_name}' failed for "
                                f"obs {obs_idx}, skipping", exc)
                            continue

                    if best_khat_i < ks[obs_idx]:
                        loos[obs_idx] = best_loo_i
                        ks[obs_idx] = best_khat_i
                        total_improved += 1
                    print(f" -> {best_khat_i:.3f}", flush=True)

                    # Free memory between observations
                    import gc
                    gc.collect()

                loo = float(np.sum(loos))
                bad_k_after = int(np.sum(ks > khat_threshold))
                print(f"  AIS improved {total_improved}/{bad_k_count} observations, "
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
                    except (ValueError, KeyError, AttributeError, TypeError) as exc:
                        _warn_fallback(
                            f"MICE predict_mice_pmf failed for item "
                            f"'{item_key}' person {row_idx}, "
                            f"falling back to uniform 1/{K}", exc)
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
        except Exception as exc:
            _warn_fallback(
                "PSIS-LOO (IS reweighting) failed, setting k-hat=inf", exc)
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
        except Exception as exc:
            _warn_fallback(
                "PSIS tail smoothing failed, using unsmoothed IS weights", exc)
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

    # ------------------------------------------------------------------
    # Marginal inference: integrate out abilities, fit item params only
    # ------------------------------------------------------------------

    def _response_probs_grid(self, theta_grid, **item_params):
        """Compute P(Y_i = k | theta_q, Xi) for each quadrature point.

        Subclasses must override this to implement their response model.

        Args:
            theta_grid: (Q,) array of ability values.
            **item_params: Item parameters (no abilities).

        Returns:
            (Q, I, K) array of response probabilities.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _response_probs_grid"
        )

    def _item_var_list(self):
        """Return list of item parameter names (everything except abilities and mu).

        The ``mu`` parameters are location shifts for difficulties that are
        partially non-identified with ``difficulties0``, creating ridges in
        the posterior that make NUTS very inefficient.  They are held fixed
        at their ADVI values during MCMC.
        """
        exclude = {'abilities'}
        for v in self.var_list:
            if v.startswith('abilities') or v.startswith('mu'):
                exclude.add(v)
        return [v for v in self.var_list if v not in exclude]

    def _make_gauss_hermite_grid(self, n_points=31):
        """Return (theta_grid, theta_log_weights) for Gauss-Hermite quadrature."""
        nodes, weights = np.polynomial.hermite.hermgauss(n_points)
        theta_grid = jnp.asarray(np.sqrt(2) * nodes, dtype=self.dtype)
        theta_log_weights = jnp.asarray(
            np.log(weights) - 0.5 * np.log(np.pi), dtype=self.dtype
        )
        return theta_grid, theta_log_weights

    def marginal_log_prob(self, data, theta_grid=None, theta_log_weights=None,
                          prior_weight=1.0, **item_params):
        """Rao-Blackwellized log posterior over item parameters only.

        Integrates out person abilities on a quadrature grid:

            log p(Xi | data) = log p(Xi)
                + sum_p log int P(x_p | theta, Xi) pi(theta) dtheta

        Uses Gauss-Hermite quadrature by default (31 nodes).
        Processes people via ``jax.lax.map`` to keep compilation tractable.

        Args:
            data: Data dict with item response columns and person key.
            theta_grid: (Q,) quadrature points. Defaults to Gauss-Hermite.
            theta_log_weights: (Q,) log quadrature weights. If None and
                theta_grid is also None, uses Gauss-Hermite weights.
            prior_weight: Weight on the log prior.
            **item_params: Item parameters (no abilities).

        Returns:
            Scalar log marginal posterior.
        """
        if theta_grid is None:
            theta_grid, theta_log_weights = self._make_gauss_hermite_grid()
        elif theta_log_weights is None:
            dtheta = theta_grid[1] - theta_grid[0]
            theta_log_weights = (
                -0.5 * theta_grid ** 2
                - 0.5 * jnp.log(2 * jnp.pi)
                + jnp.log(dtheta)
            )

        # (Q, I, K)
        response_probs = self._response_probs_grid(theta_grid, **item_params)
        log_rp = jnp.log(jnp.clip(response_probs, 1e-30, None))

        # Observed responses (N, I)
        choices = jnp.concat(
            [data[k][:, jnp.newaxis] for k in self.item_keys], axis=-1
        )
        bad = (choices < 0) | (choices >= self.response_cardinality) | jnp.isnan(choices)
        choices_int = jnp.where(bad, 0, choices).astype(jnp.int32)

        n_items = len(self.item_keys)
        n_people = choices.shape[0]

        # Vectorized over people: gather log P(observed | theta_q)
        # log_rp: (Q, I, K) → (Q, 1, I, K), choices_int: (N, I) → (1, N, I, 1)
        log_obs = jnp.take_along_axis(
            log_rp[:, None, :, :],
            choices_int[None, :, :, None],
            axis=-1
        ).squeeze(-1)  # (Q, N, I)

        # Handle missing items
        imputation_pmfs = data.get('_imputation_pmfs')
        imputation_weights = data.get('_imputation_weights')

        if imputation_pmfs is not None:
            log_q = jnp.log(jnp.maximum(imputation_pmfs, 1e-30))  # (N, I, K)
            imp_w = (jnp.asarray(imputation_weights)
                     if imputation_weights is not None
                     else jnp.ones(n_items, dtype=self.dtype))

            # RB: log sum_k q(k|x) * P(k|theta, Xi) for each (Q, N, I)
            # log_rp: (Q, I, K) → (Q, 1, I, K), log_q: (N, I, K) → (1, N, I, K)
            rb = jax.scipy.special.logsumexp(
                log_rp[:, None, :, :] + log_q[None, :, :, :], axis=-1
            )  # (Q, N, I)
            weighted_rb = imp_w[None, None, :] * rb
            log_obs = jnp.where(bad[None, :, :], weighted_rb, log_obs)
        else:
            log_obs = jnp.where(bad[None, :, :], 0.0, log_obs)

        # Sum over items → (Q, N), then logsumexp over grid → (N,)
        log_lik_per_grid = jnp.sum(log_obs, axis=-1)  # (Q, N)
        marginal_ll_per_person = jax.scipy.special.logsumexp(
            log_lik_per_grid + theta_log_weights[:, None], axis=0
        )  # (N,)

        if 'sample_weights' in data:
            sw = jnp.asarray(data['sample_weights'], dtype=marginal_ll_per_person.dtype)
            total_marginal_ll = jnp.sum(sw * marginal_ll_per_person)
        else:
            total_marginal_ll = jnp.sum(marginal_ll_per_person)

        # Log prior on item parameters
        full_params = dict(item_params)
        # Add dummy abilities so the joint prior can compute
        for v in self.var_list:
            if v not in full_params:
                shape = self.joint_prior_distribution.sample(
                    seed=jax.random.PRNGKey(0)
                )[v].shape
                full_params[v] = jnp.zeros(shape, dtype=self.dtype)
        log_prior_items = self.joint_prior_distribution.log_prob(full_params)

        return prior_weight * log_prior_items + total_marginal_ll

    def fit_marginal_advi(
        self,
        data,
        theta_grid=None,
        num_samples=25,
        num_epochs=2000,
        learning_rate=0.01,
        rank=0,
        seed=42,
        verbose=True,
        **training_kwargs,
    ):
        """Fit item parameters via ADVI with abilities Rao-Blackwellized out.

        Builds an item-only surrogate posterior (excluding abilities) and
        optimizes the ELBO against ``marginal_log_prob``.

        Args:
            data: Full data dict (all people, not batched).
            theta_grid: Quadrature grid (defaults to Gauss-Hermite).
            num_samples: Number of surrogate draws per ELBO estimate.
            num_epochs: Training epochs.
            learning_rate: Adam learning rate.
            rank: Low-rank covariance rank for the surrogate.
                0 = mean-field, >0 = low-rank + diagonal.
            seed: Random seed.
            verbose: Print progress.
            **training_kwargs: Additional kwargs for training_loop.

        Returns:
            (losses, params) tuple.
        """
        from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator
        from bayesianquilts.util import training_loop
        from tensorflow_probability.substrates.jax import bijectors as tfb

        item_var_list = self._item_var_list()

        # Build item-only prior.
        #
        # Some entries in the joint prior are callables (conditional
        # distributions) whose signature names parent variables.  When
        # those parents are not in item_var_list we must either pull
        # them into the sub-prior or freeze the callable at the current
        # ADVI point estimate.
        import inspect as _inspect
        full_model = self.joint_prior_distribution.model
        prior_dict = {}
        bijectors = {}

        # Collect point estimates for freezing conditionals
        def _point_estimate(var_name):
            """Best available point estimate for a variable."""
            if self.params is not None:
                for pk in self.params:
                    if pk.startswith(var_name) and pk.endswith('loc'):
                        return self.params[pk]
            return self.joint_prior_distribution.sample(
                seed=jax.random.PRNGKey(0))[var_name]

        # Resolve each item variable
        for v in item_var_list:
            entry = full_model[v]
            if callable(entry) and not isinstance(entry, tfd.Distribution):
                sig = _inspect.signature(entry)
                missing = [p for p in sig.parameters if p not in item_var_list]
                if missing:
                    # Check if all missing parents themselves have plain
                    # (non-callable) priors we can include
                    all_plain = all(
                        p in full_model
                        and (not callable(full_model[p])
                             or isinstance(full_model[p], tfd.Distribution))
                        for p in missing
                    )
                    if all_plain:
                        # Pull parent(s) into prior_dict
                        for p in missing:
                            if p not in prior_dict:
                                prior_dict[p] = full_model[p]
                                if hasattr(self, 'bijectors') and p in self.bijectors:
                                    bijectors[p] = self.bijectors[p]
                                else:
                                    bijectors[p] = tfb.Identity()
                        prior_dict[v] = entry
                    else:
                        # Freeze conditional at point estimates
                        parent_vals = {p: _point_estimate(p) for p in missing}
                        _warn_fallback(
                            f"Prior for '{v}' depends on {missing} which "
                            f"have conditional priors; freezing at point "
                            f"estimates for marginal ADVI")
                        # Bind only the missing parents; keep others as
                        # free parameters in the sub-prior
                        remaining = [p for p in sig.parameters
                                     if p not in missing]
                        if remaining:
                            prior_dict[v] = lambda _entry=entry, _pv=parent_vals, **kw: _entry(**{**_pv, **kw})
                        else:
                            prior_dict[v] = entry(**parent_vals)
                else:
                    # All parents are item vars — keep as-is
                    prior_dict[v] = entry
            else:
                prior_dict[v] = entry

            if v not in bijectors:
                if hasattr(self, 'bijectors') and v in self.bijectors:
                    bijectors[v] = self.bijectors[v]
                else:
                    bijectors[v] = tfb.Identity()

        item_prior = tfd.JointDistributionNamed(prior_dict)

        # Build item-only surrogate
        surrogate_gen, surrogate_init = build_factored_surrogate_posterior_generator(
            item_prior,
            bijectors=bijectors,
            dtype=self.dtype,
            rank=rank,
        )
        marginal_params = surrogate_init()

        if verbose:
            n_params = sum(np.prod(v.shape) for v in marginal_params.values())
            print(f"Marginal ADVI (Rao-Blackwellized abilities)")
            print(f"  Item vars: {item_var_list}")
            print(f"  Surrogate params: {n_params}")
            print(f"  Rank: {rank} ({'mean-field' if rank == 0 else 'low-rank'})")
            sys.stdout.flush()

        key = jax.random.PRNGKey(seed)

        # item_var_list may differ from prior_dict keys when parents
        # (e.g. 'mu', 'global_scale') were pulled in.
        surrogate_vars = list(prior_dict.keys())

        def loss_fn(params):
            surrogate = surrogate_gen(params)
            samples = surrogate.sample(num_samples, seed=key)
            # ELBO = E_q[log p(data, Xi)] - E_q[log q(Xi)]
            log_probs = []
            log_q_vals = []
            for s_idx in range(num_samples):
                sample = {v: samples[v][s_idx] for v in surrogate_vars}
                lp = self.marginal_log_prob(
                    data, theta_grid=theta_grid, **sample
                )
                log_probs.append(lp)
                log_q_vals.append(surrogate.log_prob(
                    {v: samples[v][s_idx] for v in surrogate_vars}
                ))
            log_joint = jnp.mean(jnp.stack(log_probs))
            # Estimate entropy as -E_q[log q] (works even when
            # analytical entropy is unavailable for TransformedDistributions)
            neg_entropy = jnp.mean(jnp.stack(log_q_vals))
            return -(log_joint - neg_entropy)

        # training_loop expects a data_iterator and steps_per_epoch even
        # when data is already captured in the loss_fn closure.  Provide
        # a single-step dummy iterator so the optimiser takes one gradient
        # step per epoch (the loss already averages over the full dataset).
        def _dummy_iterator():
            while True:
                yield None

        # Wrap loss_fn to match training_loop's (data, params) signature
        _orig_loss = loss_fn

        def _batched_loss(data_batch, params):
            return _orig_loss(params)

        losses, trained_params = training_loop(
            initial_values=marginal_params,
            loss_fn=_batched_loss,
            data_iterator=_dummy_iterator(),
            steps_per_epoch=1,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            **training_kwargs,
        )

        self.marginal_params = trained_params
        self.marginal_surrogate_generator = surrogate_gen

        if verbose:
            print(f"  Final ELBO: {-float(losses[-1]):.1f}")
            # Sample posterior means
            surrogate = surrogate_gen(trained_params)
            post_samples = surrogate.sample(100, seed=key)
            for v in item_var_list:
                m = float(jnp.mean(post_samples[v]))
                s = float(jnp.std(post_samples[v]))
                print(f"  {v}: mean={m:.4f}, std={s:.4f}")

        return losses, trained_params

    def fit_marginal_mcmc(
        self,
        data,
        theta_grid=None,
        num_chains=4,
        num_warmup=500,
        num_samples=500,
        target_accept_prob=0.85,
        step_size=0.01,
        seed=None,
        verbose=True,
        resume=False,
    ):
        """Run NUTS on item parameters with abilities Rao-Blackwellized out.

        Uses BlackJAX NUTS with window adaptation. Targets the marginal
        posterior over item parameters by integrating out abilities on a
        quadrature grid.

        Args:
            data: Full data dict (all people).
            theta_grid: Quadrature grid for ability integration.
            num_chains: Number of NUTS chains.
            num_warmup: Warmup steps per chain.
            num_samples: Post-warmup samples per chain.
            target_accept_prob: NUTS target acceptance.
            step_size: Initial step size.
            seed: Random seed.
            verbose: Print progress.
            resume: If True, skip warmup and continue sampling from the
                saved state of a previous call. Requires a prior call
                to fit_marginal_mcmc that populated ``self.mcmc_state_``.
                New samples are concatenated with existing
                ``self.mcmc_samples`` along the samples axis.

        Returns:
            Dict mapping item param names to posterior sample arrays of
            shape (num_chains, num_samples, ...).
        """
        import blackjax
        from jax import random

        if seed is None:
            seed = np.random.randint(0, 2**31)
        key = random.PRNGKey(seed)

        item_var_list = self._item_var_list()

        if theta_grid is not None:
            grid_desc = f"{len(theta_grid)} points (uniform)"
        else:
            grid_desc = "31-point Gauss-Hermite"

        if verbose:
            print(f"Marginal MCMC (Rao-Blackwellized abilities)")
            print(f"  Item params: {item_var_list}")
            key, tmp_key = random.split(key)
            n_params = sum(
                np.prod(self.joint_prior_distribution.sample(
                    seed=tmp_key)[v].shape)
                for v in item_var_list
            )
            print(f"  Total dimensions: {n_params}")
            print(f"  Quadrature: {grid_desc}")
            print(f"  Chains: {num_chains}, Warmup: {num_warmup}, "
                  f"Samples: {num_samples}")
            sys.stdout.flush()

        # Target log density
        def logdensity_fn(params_dict):
            return self.marginal_log_prob(
                data, theta_grid=theta_grid, **params_dict
            )

        # Initialize from ADVI solution or prior
        key, init_key = random.split(key)
        initial_positions = []
        for c in range(num_chains):
            pos = {}
            for var in item_var_list:
                loc_key = None
                if self.params is not None:
                    for pk in self.params:
                        if pk.startswith(var) and pk.endswith('loc'):
                            loc_key = pk
                            break
                if loc_key is not None:
                    val = self.params[loc_key]
                    pos[var] = val + 0.01 * random.normal(
                        random.fold_in(init_key, c), val.shape)
                else:
                    prior_sample = self.joint_prior_distribution.sample(
                        seed=random.fold_in(init_key, c))
                    pos[var] = prior_sample[var]
            initial_positions.append(pos)

        if verbose and self.params is not None:
            print("  Initializing from ADVI solution")
            sys.stdout.flush()

        from jax.flatten_util import ravel_pytree

        # Flatten position dict → 1D vector for BlackJAX
        ref_pos = initial_positions[0]
        _, unravel_fn = ravel_pytree(ref_pos)

        def logdensity_flat(x):
            return logdensity_fn(unravel_fn(x))

        n_flat = sum(np.prod(ref_pos[v].shape) for v in item_var_list)
        if verbose:
            print(f"  Flat parameter vector: {n_flat} elements")
            sys.stdout.flush()

        # Resume from previous run if requested
        if resume:
            if not hasattr(self, 'mcmc_state_') or self.mcmc_state_ is None:
                raise ValueError(
                    "resume=True but no mcmc_state_ found. "
                    "Run fit_marginal_mcmc without resume first.")
            saved = self.mcmc_state_
            num_chains = len(saved['states'])
            if verbose:
                print(f"  Resuming {num_chains} chains "
                      f"(+{num_samples} samples each)")
                sys.stdout.flush()
        else:
            object.__setattr__(self, '_mcmc_chain_states', [])
            object.__setattr__(self, '_mcmc_chain_step_sizes', [])
            object.__setattr__(self, '_mcmc_chain_inv_mass', [])
        # On resume, recover the backing lists from the saved state dict so that
        # the per-chain update at the end of the loop works even after a
        # serialize/reload cycle that stripped the private attributes.
        if resume and not hasattr(self, '_mcmc_chain_states'):
            object.__setattr__(self, '_mcmc_chain_states', list(saved['states']))
            object.__setattr__(self, '_mcmc_chain_step_sizes', list(saved['step_sizes']))
            object.__setattr__(self, '_mcmc_chain_inv_mass', list(saved['inv_mass_matrices']))

        all_samples = {var: [] for var in item_var_list}
        all_accept = []

        for chain_idx in range(num_chains):
            key, chain_key = random.split(key)
            if verbose:
                print(f"\n  Chain {chain_idx + 1}/{num_chains}:")
                sys.stdout.flush()

            if resume:
                # Skip warmup — use saved state and kernel params
                state = saved['states'][chain_idx]
                current_step_size = saved['step_sizes'][chain_idx]
                inv_mass_matrix = saved['inv_mass_matrices'][chain_idx]
                if verbose:
                    print(f"    Resuming from saved state "
                          f"(step_size={current_step_size:.6f})")
                    sys.stdout.flush()
            else:
                init_flat, _ = ravel_pytree(initial_positions[chain_idx])

                # Phase 1 uses identity mass for fast steps (even if some diverge).
                # Phase 2 uses the mass matrix estimated from phase 1 samples.
                inv_mass_matrix = jnp.ones(n_flat)
                current_step_size = step_size
                phase1_steps = num_warmup // 2
                phase2_steps = num_warmup - phase1_steps

                for phase, n_steps in [(1, phase1_steps), (2, phase2_steps)]:
                    # Single attempt per phase — no step size retries.
                    # Phase 1 may have divergences; the mass matrix from those
                    # samples still captures the right scale for phase 2.
                    if verbose:
                        print(f"    Phase {phase}: {n_steps} steps, "
                              f"step_size={current_step_size:.6f}...")
                        sys.stdout.flush()

                    kernel = blackjax.nuts(
                        logdensity_flat,
                        step_size=current_step_size,
                        inverse_mass_matrix=inv_mass_matrix,
                    )
                    if phase == 1:
                        state = kernel.init(init_flat)

                    @jax.jit
                    def warmup_step(state, step_key):
                        return kernel.step(step_key, state)

                    phase_flats = []
                    n_nondiv = 0

                    for step in range(n_steps):
                        chain_key, step_key = random.split(chain_key)
                        state, info = warmup_step(state, step_key)
                        n_nondiv += int(1 - info.is_divergent)
                        phase_flats.append(state.position)

                        if verbose and (step + 1) % 100 == 0:
                            ar = n_nondiv / (step + 1)
                            lp = float(state.logdensity)
                            print(f"      p{phase} {step + 1}/{n_steps} "
                                  f"non-div={ar:.3f} lp={lp:.1f}")
                            sys.stdout.flush()

                    rate = n_nondiv / max(n_steps, 1)

                    if verbose:
                        print(f"    Phase {phase} done: non-div={rate:.3f}")
                        sys.stdout.flush()

                    # Estimate mass matrix from second half of this phase
                    half = max(len(phase_flats) // 2, 1)
                    p_stack = jnp.stack(phase_flats[half:])
                    var_est = jnp.var(p_stack, axis=0)
                    inv_mass_matrix = jnp.maximum(var_est, 1e-6)

                    # Adjust step size based on acceptance for next phase
                    if rate > 0.9:
                        current_step_size *= 2.0
                    elif rate < 0.5:
                        current_step_size *= 0.5

            # Build sampling kernel with final adapted parameters
            sampling_kernel = blackjax.nuts(
                logdensity_flat,
                step_size=current_step_size,
                inverse_mass_matrix=inv_mass_matrix,
            )

            @jax.jit
            def sample_step(state, step_key):
                return sampling_kernel.step(step_key, state)

            # Sampling
            if verbose:
                print(f"    Sampling ({num_samples} steps, "
                      f"step_size={current_step_size:.6f})...")
                sys.stdout.flush()

            sample_flats = []
            n_accepted = 0
            for step in range(num_samples):
                chain_key, step_key = random.split(chain_key)
                state, info = sample_step(state, step_key)
                sample_flats.append(state.position)
                n_accepted += int(1 - info.is_divergent)
                if verbose and (step + 1) % 50 == 0:
                    ar = n_accepted / (step + 1)
                    lp = float(state.logdensity)
                    print(f"      step {step + 1}/{num_samples} "
                          f"non-divergent={ar:.3f} lp={lp:.1f}")
                    sys.stdout.flush()

            accept_ratio = n_accepted / num_samples
            if verbose:
                print(f"    Non-divergent ratio: {accept_ratio:.3f}")
                sys.stdout.flush()

            positions = [unravel_fn(f) for f in sample_flats]
            for var in item_var_list:
                stacked = jnp.stack(
                    [p[var] for p in positions], axis=0)
                all_samples[var].append(stacked)
            all_accept.append(accept_ratio)
            # Save per-chain state for potential resume
            if chain_idx < len(self._mcmc_chain_states):
                self._mcmc_chain_states[chain_idx] = state
                self._mcmc_chain_step_sizes[chain_idx] = current_step_size
                self._mcmc_chain_inv_mass[chain_idx] = inv_mass_matrix
            else:
                self._mcmc_chain_states.append(state)
                self._mcmc_chain_step_sizes.append(current_step_size)
                self._mcmc_chain_inv_mass.append(inv_mass_matrix)

        # Save resume state
        # Use object.__setattr__ to bypass Flax NNX pytree checks —
        # mcmc_state_ contains JAX arrays (NUTS states) that are internal
        # mutable state, not model parameters for serialization.
        object.__setattr__(self, 'mcmc_state_', {
            'states': self._mcmc_chain_states,
            'step_sizes': self._mcmc_chain_step_sizes,
            'inv_mass_matrices': self._mcmc_chain_inv_mass,
        })

        # Stack chains: (num_chains, num_samples, ...)
        result = {}
        for var in item_var_list:
            if all_samples[var]:
                result[var] = jnp.stack(all_samples[var], axis=0)

        # If resuming, concatenate with previous samples
        if resume and hasattr(self, 'mcmc_samples') and self.mcmc_samples:
            for var in item_var_list:
                if var in result and var in self.mcmc_samples:
                    result[var] = jnp.concatenate(
                        [self.mcmc_samples[var], result[var]], axis=1)

        if verbose:
            print(f"\n  Kept {len(all_accept)}/{num_chains} chains")
            if all_accept:
                print(f"  Mean non-divergent: {np.mean(all_accept):.3f}")
            for var in item_var_list:
                if var in result:
                    flat = result[var].reshape(-1, *result[var].shape[2:])
                    total_per_chain = result[var].shape[1]
                    print(f"  {var}: mean={float(jnp.mean(flat)):.4f}, "
                          f"std={float(jnp.std(flat)):.4f}"
                          f" ({total_per_chain} samples/chain)")

        self.mcmc_samples = result
        return result

    def fit_marginal_mala(
        self,
        data,
        theta_grid=None,
        num_chains=2,
        num_warmup=2000,
        num_samples=2000,
        step_size=1e-4,
        seed=None,
        verbose=True,
    ):
        """Run MALA on item parameters with abilities Rao-Blackwellized out.

        Uses BlackJAX MALA (Metropolis-Adjusted Langevin Algorithm) which
        avoids the NUTS divergence problem for stiff posteriors. Each step
        uses exactly one gradient evaluation (no tree building).

        MALA is less efficient per sample than NUTS but works reliably on
        posteriors where NUTS diverges due to extreme stiffness.

        Args:
            data: Full data dict (all people).
            theta_grid: Quadrature grid for ability integration.
            num_chains: Number of MALA chains.
            num_warmup: Warmup steps per chain.
            num_samples: Post-warmup samples per chain.
            step_size: MALA step size (typically 1e-5 to 1e-3).
            seed: Random seed.
            verbose: Print progress.

        Returns:
            Dict mapping item param names to posterior sample arrays of
            shape (num_chains, num_samples, ...).
        """
        import blackjax
        from jax import random

        if seed is None:
            seed = np.random.randint(0, 2**31)
        key = random.PRNGKey(seed)

        item_var_list = self._item_var_list()

        if verbose:
            print(f"Marginal MALA (Rao-Blackwellized abilities)")
            print(f"  Item params: {item_var_list}")
            key, tmp_key = random.split(key)
            n_params = sum(
                np.prod(self.joint_prior_distribution.sample(
                    seed=tmp_key)[v].shape)
                for v in item_var_list
            )
            print(f"  Total dimensions: {n_params}")
            print(f"  Chains: {num_chains}, Warmup: {num_warmup}, "
                  f"Samples: {num_samples}, Step size: {step_size}")
            sys.stdout.flush()

        # Target log density
        def logdensity_fn(params_dict):
            return self.marginal_log_prob(
                data, theta_grid=theta_grid, **params_dict
            )

        # Initialize from ADVI solution or prior
        key, init_key = random.split(key)
        initial_positions = []
        for c in range(num_chains):
            pos = {}
            for var in item_var_list:
                loc_key = None
                if self.params is not None:
                    for pk in self.params:
                        if pk.startswith(var) and pk.endswith('loc'):
                            loc_key = pk
                            break
                if loc_key is not None:
                    val = self.params[loc_key]
                    pos[var] = val + 0.01 * random.normal(
                        random.fold_in(init_key, c), val.shape)
                else:
                    prior_sample = self.joint_prior_distribution.sample(
                        seed=random.fold_in(init_key, c))
                    pos[var] = prior_sample[var]
            initial_positions.append(pos)

        if verbose and self.params is not None:
            print("  Initializing from ADVI solution")
            sys.stdout.flush()

        from jax.flatten_util import ravel_pytree

        ref_pos = initial_positions[0]
        _, unravel_fn = ravel_pytree(ref_pos)

        def logdensity_flat(x):
            return logdensity_fn(unravel_fn(x))

        n_flat = sum(np.prod(ref_pos[v].shape) for v in item_var_list)
        if verbose:
            print(f"  Flat parameter vector: {n_flat} elements")
            sys.stdout.flush()

        all_samples = {var: [] for var in item_var_list}
        all_accept = []

        for chain_idx in range(num_chains):
            key, chain_key = random.split(key)
            if verbose:
                print(f"\n  Chain {chain_idx + 1}/{num_chains}:")
                sys.stdout.flush()

            init_flat, _ = ravel_pytree(initial_positions[chain_idx])

            kernel = blackjax.mala(logdensity_flat, step_size=step_size)
            state = kernel.init(init_flat)

            @jax.jit
            def step_fn(state, step_key):
                return kernel.step(step_key, state)

            # Warmup
            if verbose:
                print(f"    Warmup ({num_warmup} steps)...")
                sys.stdout.flush()

            n_acc = 0
            for step in range(num_warmup):
                chain_key, step_key = random.split(chain_key)
                state, info = step_fn(state, step_key)
                n_acc += int(info.is_accepted)
                if verbose and (step + 1) % 500 == 0:
                    ar = n_acc / (step + 1)
                    lp = float(state.logdensity)
                    print(f"      warmup {step + 1}/{num_warmup} "
                          f"accept={ar:.3f} lp={lp:.1f}")
                    sys.stdout.flush()

            # Sampling
            if verbose:
                print(f"    Sampling ({num_samples} steps)...")
                sys.stdout.flush()

            sample_flats = []
            n_acc = 0
            for step in range(num_samples):
                chain_key, step_key = random.split(chain_key)
                state, info = step_fn(state, step_key)
                sample_flats.append(state.position)
                n_acc += int(info.is_accepted)
                if verbose and (step + 1) % 500 == 0:
                    ar = n_acc / (step + 1)
                    lp = float(state.logdensity)
                    print(f"      step {step + 1}/{num_samples} "
                          f"accept={ar:.3f} lp={lp:.1f}")
                    sys.stdout.flush()

            accept_ratio = n_acc / num_samples
            if verbose:
                print(f"    Accept ratio: {accept_ratio:.3f}")
                sys.stdout.flush()

            positions = [unravel_fn(f) for f in sample_flats]
            for var in item_var_list:
                stacked = jnp.stack(
                    [p[var] for p in positions], axis=0)
                all_samples[var].append(stacked)
            all_accept.append(accept_ratio)

        # Stack chains: (num_chains, num_samples, ...)
        result = {}
        for var in item_var_list:
            if all_samples[var]:
                result[var] = jnp.stack(all_samples[var], axis=0)

        if verbose:
            print(f"\n  Kept {len(all_accept)}/{num_chains} chains")
            if all_accept:
                print(f"  Mean accept: {np.mean(all_accept):.3f}")
            for var in item_var_list:
                if var in result:
                    flat = result[var].reshape(-1, *result[var].shape[2:])
                    print(f"  {var}: mean={float(jnp.mean(flat)):.4f}, "
                          f"std={float(jnp.std(flat)):.4f}")

        self.mcmc_samples = result
        return result

    def compute_eap_abilities(self, data, item_params=None, theta_grid=None,
                              theta_log_weights=None):
        """Compute Expected A Posteriori (EAP) ability estimates.

        Given fixed item parameters, computes the posterior mean ability
        for each person by numerical integration on a theta grid:

            E[theta | x_p, Xi] = sum_q theta_q * P(x_p | theta_q, Xi) * pi(theta_q)
                                 / sum_q P(x_p | theta_q, Xi) * pi(theta_q)

        If ``item_params`` is None, uses MCMC posterior means from
        ``self.mcmc_samples`` (averaging over chains and samples).

        Args:
            data: Data dict with item response columns and person key.
            item_params: Dict of item parameter arrays. If None, uses
                posterior means from ``self.mcmc_samples``.
            theta_grid: (Q,) quadrature points. Defaults to Gauss-Hermite.
            theta_log_weights: (Q,) log quadrature weights.

        Returns:
            Dict with:
                - ``eap``: (N,) posterior mean abilities
                - ``psd``: (N,) posterior standard deviations
                - ``posterior``: (N, Q) full posterior on grid
        """
        if theta_grid is None:
            theta_grid, theta_log_weights = self._make_gauss_hermite_grid(61)
        elif theta_log_weights is None:
            dtheta = theta_grid[1] - theta_grid[0]
            theta_log_weights = (
                -0.5 * theta_grid ** 2
                - 0.5 * jnp.log(2 * jnp.pi)
                + jnp.log(dtheta)
            )

        if item_params is None:
            if not hasattr(self, 'mcmc_samples') or self.mcmc_samples is None:
                raise ValueError(
                    "No item_params provided and no mcmc_samples available. "
                    "Run fit_marginal_mcmc first or pass item_params."
                )
            item_params = {}
            for var, samples in self.mcmc_samples.items():
                # (chains, samples, ...) → mean over chains and samples
                item_params[var] = jnp.mean(
                    samples.reshape(-1, *samples.shape[2:]), axis=0
                )

        # (Q, I, K)
        response_probs = self._response_probs_grid(theta_grid, **item_params)
        log_rp = jnp.log(jnp.clip(response_probs, 1e-30, None))

        # Observed responses
        choices = jnp.concat(
            [data[k][:, jnp.newaxis] for k in self.item_keys], axis=-1
        )
        bad = (choices < 0) | (choices >= self.response_cardinality) | jnp.isnan(choices)
        choices_int = jnp.where(bad, 0, choices).astype(jnp.int32)

        # Imputation
        imputation_pmfs = data.get('_imputation_pmfs')
        imputation_weights = data.get('_imputation_weights')
        has_imputation = imputation_pmfs is not None

        if has_imputation:
            log_q = jnp.log(jnp.maximum(imputation_pmfs, 1e-30))
            imp_w = (jnp.asarray(imputation_weights)
                     if imputation_weights is not None
                     else jnp.ones(len(self.item_keys), dtype=self.dtype))

        n_items = len(self.item_keys)
        n_people = choices.shape[0]

        def _compute_eap_on_grid(theta_g, theta_lw):
            """Compute EAP/PSD on a given grid."""
            rp = self._response_probs_grid(theta_g, **item_params)
            lrp = jnp.log(jnp.clip(rp, 1e-30, None))

            log_obs_g = jnp.take_along_axis(
                lrp[:, None, :, :],
                choices_int[None, :, :, None],
                axis=-1
            ).squeeze(-1)  # (Q, N, I)

            if has_imputation:
                rb = jax.scipy.special.logsumexp(
                    lrp[:, None, :, :] + log_q[None, :, :, :], axis=-1
                )
                weighted_rb = imp_w[None, None, :] * rb
                log_obs_g = jnp.where(bad[None, :, :], weighted_rb, log_obs_g)
            else:
                log_obs_g = jnp.where(bad[None, :, :], 0.0, log_obs_g)

            log_lik_g = jnp.sum(log_obs_g, axis=-1)  # (Q, N)
            log_unnorm = log_lik_g + theta_lw[:, None]
            log_norm = jax.scipy.special.logsumexp(log_unnorm, axis=0)
            post = jnp.exp(log_unnorm - log_norm[None, :])  # (Q, N)

            eap_g = jnp.sum(post * theta_g[:, None], axis=0)
            psd_g = jnp.sqrt(
                jnp.sum(post * (theta_g[:, None] - eap_g[None, :]) ** 2, axis=0)
            )
            return eap_g, psd_g, post

        # Pass 1: coarse grid to find approximate EAP
        eap_coarse, psd_coarse, _ = _compute_eap_on_grid(
            theta_grid, theta_log_weights)

        # Pass 2: refined grid centered on each person's EAP
        # Use a uniform grid of width ±4*max(psd, 0.5) around the coarse EAP
        # For efficiency, use a single grid centered on the population mean
        # with width covering the full range of abilities
        eap_mean = jnp.mean(eap_coarse)
        eap_range = jnp.maximum(jnp.std(eap_coarse), 0.5)
        fine_lo = eap_mean - 5 * eap_range
        fine_hi = eap_mean + 5 * eap_range
        n_fine = 201
        fine_grid = jnp.linspace(fine_lo, fine_hi, n_fine)
        dtheta = fine_grid[1] - fine_grid[0]
        fine_log_weights = (
            -0.5 * fine_grid ** 2
            - 0.5 * jnp.log(2 * jnp.pi)
            + jnp.log(dtheta)
        )

        eap, psd, posterior = _compute_eap_on_grid(
            fine_grid, fine_log_weights)

        return {
            'eap': eap,
            'psd': psd,
            'posterior': posterior.T,  # (N, Q)
            'theta_grid': fine_grid,
        }

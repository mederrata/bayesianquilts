import pathlib

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from typing import Any

from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd

from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator
from bayesianquilts.irt.irt import IRTModel


def _mixture_of_logits(z, nn_scales, nn_shifts, nn_logit_weights):
    """Per-item mixture of logistic sigmoids: z -> (0, 1).

    g_i(z) = sum_m w_{i,m} * sigmoid(s_{i,m} * z + c_{i,m})

    where s_{i,m} > 0 (via softplus) and w_{i,m} = softmax(raw_weights).

    Monotone by construction: sum of monotone functions with positive weights.
    When M=1, s=1, c=0: g(z) = sigmoid(z) — exactly the standard GRM.

    Args:
        z: (..., N, D, I, K-1) — scaled offsets
        nn_scales: (..., I, M) — unconstrained per-component scales (softplus -> positive)
        nn_shifts: (..., I, M) — per-component shift parameters
        nn_logit_weights: (..., I, M) — unconstrained mixture weights (softmax -> simplex)

    Returns:
        (..., N, D, I, K-1) values in (0, 1), monotone increasing in z.
    """
    # Positive scales and normalized weights
    scales = jax.nn.softplus(nn_scales)  # (..., I, M)
    weights = jax.nn.softmax(nn_logit_weights, axis=-1)  # (..., I, M)

    # Reshape for broadcasting: (..., I, M) -> (..., 1, 1, I, M)
    n_batch_dims = nn_scales.ndim - 2  # everything before (I, M)
    for _ in range(2):  # insert N, D dims
        scales = jnp.expand_dims(scales, axis=n_batch_dims)
        nn_shifts = jnp.expand_dims(nn_shifts, axis=n_batch_dims)
        weights = jnp.expand_dims(weights, axis=n_batch_dims)

    # z: (..., N, D, I, K-1) -> (..., N, D, I, K-1, 1)
    z_exp = z[..., jnp.newaxis]

    # (..., 1, 1, I, M) -> (..., 1, 1, I, 1, M) for K-1 broadcasting
    scales = scales[..., jnp.newaxis, :]
    shifts = nn_shifts[..., jnp.newaxis, :]
    weights = weights[..., jnp.newaxis, :]

    # Component sigmoids: (..., N, D, I, K-1, M)
    logits = z_exp * scales + shifts
    component_probs = jax.nn.sigmoid(logits)

    # Weighted sum over M components: (..., N, D, I, K-1)
    return jnp.sum(component_probs * weights, axis=-1)


class NeuralGRModel(IRTModel):
    """Graded Response Model with a per-item mixture-of-logits response function.

    Instead of the standard sigmoid link P(Y >= k) = sigma(a*(theta - b_k)),
    this model uses P(Y >= k) = g_i(a*(theta - b_k)) where g_i is a per-item
    mixture of logistic sigmoids:

        g_i(z) = sum_m w_{i,m} * sigma(s_{i,m} * z + c_{i,m})

    where s_{i,m} > 0 (softplus), w_{i,m} > 0 and sum_m w_{i,m} = 1 (softmax),
    and c_{i,m} are shift parameters. With M mixture components per item.

    Monotonicity: convex combination of monotone sigmoids is monotone.
    When M=1, s=1, c=0: g(z) = sigma(z) — exactly the standard GRM.
    With M > 1: flexible asymmetric ICC shapes that no single sigmoid can replicate.
    """

    response_type = "polytomous"

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
        eta_scale=0.1,
        kappa_scale=0.5,
        weight_exponent=1.0,
        response_cardinality=5,
        discrimination_guess=None,
        include_independent=False,
        vi_mode='advi',
        imputation_model=None,
        dtype=jnp.float64,
        noisy_dim=False,
        noisy_dim_eta_scale=0.1,
        noisy_dim_ability_scale=1.0,
        parameterization="softplus",
        rank=0,
        # Legacy params accepted but ignored for backward compat
        nn_hidden_sizes=None,
        per_item_nn=None,
        nn_prior_scale=0.5,
    ):
        # Monotone network hidden size per item
        self.nn_hidden_size = nn_hidden_sizes if nn_hidden_sizes is not None else 4
        self.nn_prior_scale = nn_prior_scale

        # Store noisy_dim settings before super().__init__ calls set_dimension
        self.noisy_dim = noisy_dim
        self.noisy_dim_eta_scale = noisy_dim_eta_scale
        self.noisy_dim_ability_scale = noisy_dim_ability_scale

        # If noisy_dim, the actual model dimension is dim+1
        effective_dim = dim + 1 if noisy_dim else dim

        super(NeuralGRModel, self).__init__(
            item_keys=item_keys,
            num_people=num_people,
            num_groups=num_groups,
            data=data,
            person_key=person_key,
            dim=effective_dim,
            decay=decay,
            positive_discriminations=positive_discriminations,
            missing_val=missing_val,
            full_rank=full_rank,
            eta_scale=eta_scale,
            kappa_scale=kappa_scale,
            weight_exponent=weight_exponent,
            response_cardinality=response_cardinality,
            discrimination_guess=discrimination_guess,
            include_independent=include_independent,
            vi_mode=vi_mode,
            imputation_model=imputation_model,
            parameterization=parameterization,
            rank=rank,
            dtype=dtype,
        )
        # Store the user-facing dim (before noisy augmentation)
        self._primary_dim = dim

        # Discrimination prior scale: 1.0 for primary dims, weaker for noisy dim
        disc_scale = jnp.ones(
            (1, self.dimensions, self.num_items, 1), dtype=dtype
        )
        if noisy_dim:
            # Noisy dimension gets smaller discrimination prior (weakly coupled)
            disc_scale = disc_scale.at[:, -1, :, :].set(noisy_dim_eta_scale)
        self._disc_prior_scale = disc_scale

        # Override kappa_scale for the noisy dimension to enforce strong shrinkage
        if noisy_dim:
            noisy_kappa = jnp.array(
                noisy_dim_eta_scale, dtype=dtype
            ) * jnp.ones((1, 1, 1, 1), dtype=dtype)
            self.kappa_scale = jnp.concatenate(
                [self.kappa_scale[:, :dim, :, :], noisy_kappa], axis=1
            )

        self.create_distributions()

    def _monotone_forward(self, z, nn_params):
        """Per-item mixture-of-logits response function.

        g_i(z) = sum_m w_{i,m} * sigmoid(s_{i,m} * z + c_{i,m})

        Monotone by construction: convex combination of monotone sigmoids.

        Args:
            z: (..., N, D, I, K-1) — the scaled offsets.
            nn_params: Dict with nn_scales, nn_shifts, nn_logit_weights.

        Returns:
            Array of same shape as z, with values in (0, 1).
        """
        return _mixture_of_logits(
            z, nn_params['nn_scales'], nn_params['nn_shifts'],
            nn_params['nn_logit_weights'],
        )

    def _log_monotone_forward(self, z, nn_params):
        """Log-space mixture of logits for numerical stability.

        Returns (log_cdf, log_sf) = (log(g(z)), log(1 - g(z))).
        Since g(z) is a mixture of sigmoids, we compute it and take log
        with clamping for safety.
        """
        g = self._monotone_forward(z, nn_params)
        g = jnp.clip(g, 1e-12, 1.0 - 1e-12)
        log_cdf = jnp.log(g)
        log_sf = jnp.log1p(-g)
        return log_cdf, log_sf

    def _get_nn_param_names(self):
        """Return list of mixture-of-logits parameter names."""
        return ['nn_scales', 'nn_shifts', 'nn_logit_weights']

    def _extract_nn_params(self, params):
        """Extract mixture-of-logits parameters from a params dict."""
        return {k: params[k] for k in self._get_nn_param_names() if k in params}

    def neural_grm_model_prob(self, abilities, discriminations, difficulties, nn_params):
        """Compute P(Y=k) using the Kumaraswamy CDF response function.

        Args:
            abilities: (N, D, 1, 1) or (S, N, D, 1, 1)
            discriminations: (1, D, I, 1) or (S, 1, D, I, 1)
            difficulties: (1, D, I, K-1) or (S, 1, D, I, K-1) — cumulative thresholds
            nn_params: dict with nn_log_a, nn_log_b arrays

        Returns:
            probs: (S, N, I, K) or (N, I, K) — category probabilities
        """
        if self.include_independent:
            abilities = jnp.pad(
                abilities,
                [(0, 0)] * (len(discriminations.shape) - 3) + [(1, 0)] + [(0, 0)] * 2,
            )

        offsets = difficulties - abilities  # (..., N, D, I, K-1)
        scaled = offsets * discriminations  # (..., N, D, I, K-1)

        # Apply monotone network to -scaled (matching GRM's sigmoid(-scaled) convention)
        # so that P(Y >= k) increases with ability (theta) and decreases with threshold (b_k)
        cum_probs = self._monotone_forward(-scaled, nn_params)  # P(Y >= k)

        # Pad: P(Y >= 0) = 1 on left, P(Y >= K) = 0 on right
        cum_probs = jnp.pad(
            cum_probs,
            ([(0, 0)] * (len(cum_probs.shape) - 1) + [(1, 0)]),
            mode="constant",
            constant_values=1,
        )
        cum_probs = jnp.pad(
            cum_probs,
            ([(0, 0)] * (len(cum_probs.shape) - 1) + [(0, 1)]),
            mode="constant",
            constant_values=0,
        )
        probs = cum_probs[..., :-1] - cum_probs[..., 1:]
        # Clip to avoid zero/negative probs from numerical non-monotonicity
        probs = jnp.maximum(probs, 1e-8)
        # Renormalize so probs sum to 1
        probs = probs / jnp.sum(probs, axis=-1, keepdims=True)

        # Weight by discrimination and sum over dimensions axis
        weights = (
            jnp.abs(discriminations) ** self.weight_exponent
            / jnp.sum(jnp.abs(discriminations) ** self.weight_exponent, axis=-3)[
                ..., jnp.newaxis, :, :
            ]
        )
        probs = jnp.sum(probs * weights, axis=-3)
        return probs

    def neural_grm_log_prob(self, abilities, discriminations, difficulties, nn_params):
        """Compute log P(Y=k) in log-space for numerical stability.

        Instead of computing probs and then taking log, this method works
        entirely in log-space using log_cdf and log_sf from the Kumaraswamy CDF.

        log P(Y=k) = log(F_k - F_{k+1}) where F_k = P(Y >= k)
                   = log(F_k) + log(1 - F_{k+1}/F_k)
                   = log(F_k) + log1p(-exp(log(F_{k+1}) - log(F_k)))

        Returns:
            log_probs: (..., N, I, K) — log category probabilities
            probs: (..., N, I, K) — category probabilities (for entropy etc.)
        """
        if self.include_independent:
            abilities = jnp.pad(
                abilities,
                [(0, 0)] * (len(discriminations.shape) - 3) + [(1, 0)] + [(0, 0)] * 2,
            )

        offsets = difficulties - abilities
        scaled = offsets * discriminations
        log_cdf, log_sf = self._log_monotone_forward(-scaled, nn_params)

        # log_cdf is log(P(Y >= k)) for k=1..K-1
        # Need: log(P(Y >= 0)) = 0, log(P(Y >= K)) = -inf
        neg_inf = jnp.full((*log_cdf.shape[:-1], 1), -1e10, dtype=log_cdf.dtype)
        zero_pad = jnp.zeros((*log_cdf.shape[:-1], 1), dtype=log_cdf.dtype)

        # log_F: log P(Y >= k) for k = 0, 1, ..., K
        log_F = jnp.concatenate([zero_pad, log_cdf, neg_inf], axis=-1)

        # log P(Y = k) = log(F_k - F_{k+1})
        # = log_F[k] + log1p(-exp(log_F[k+1] - log_F[k]))
        log_Fk = log_F[..., :-1]       # log F_k for k = 0..K-1
        log_Fk1 = log_F[..., 1:]       # log F_{k+1} for k = 0..K-1
        # log_ratio = log(F_{k+1}/F_k) = log_Fk1 - log_Fk, should be <= 0
        log_ratio = jnp.minimum(log_Fk1 - log_Fk, -1e-7)
        # log(1 - exp(log_ratio)) via stable log1mexp
        log_probs_per_dim = log_Fk + jnp.where(
            log_ratio < -0.6931,
            jnp.log1p(-jnp.exp(log_ratio)),
            jnp.log(-jnp.expm1(log_ratio)),
        )

        # Clamp for safety
        log_probs_per_dim = jnp.maximum(log_probs_per_dim, -20.0)

        # Also compute probs for simulation/entropy
        probs_per_dim = jnp.exp(log_probs_per_dim)
        probs_per_dim = jnp.maximum(probs_per_dim, 1e-8)
        probs_per_dim = probs_per_dim / jnp.sum(probs_per_dim, axis=-1, keepdims=True)

        # Weight by discrimination and sum over dimensions
        weights = (
            jnp.abs(discriminations) ** self.weight_exponent
            / jnp.sum(jnp.abs(discriminations) ** self.weight_exponent, axis=-3)[
                ..., jnp.newaxis, :, :
            ]
        )
        # Weighted log-probs: log(sum_d w_d * p_d(k))
        # Use logsumexp: log(sum_d exp(log(w_d) + log(p_d(k))))
        log_weights = jnp.log(jnp.maximum(weights, 1e-30))
        log_probs = jax.scipy.special.logsumexp(
            log_weights[..., :, :] + log_probs_per_dim, axis=-3
        )

        probs = jnp.sum(probs_per_dim * weights, axis=-3)
        return log_probs, probs

    def neural_grm_model_prob_d(
        self, abilities, discriminations, difficulties0, ddifficulties, nn_params
    ):
        """Convenience: construct cumulative difficulties then call neural_grm_model_prob."""
        d0 = jnp.concat([difficulties0, ddifficulties + 0.1], axis=-1)
        difficulties = jnp.cumsum(d0, axis=-1)
        return self.neural_grm_model_prob(
            abilities, discriminations, difficulties, nn_params
        )

    def predictive_distribution(
        self,
        data,
        discriminations,
        difficulties0,
        ddifficulties,
        abilities,
        **kwargs
    ):
        """Compute predictive distribution using the mixture-of-logits GRM.

        Same interface as GRModel.predictive_distribution but extracts mixture
        params from kwargs and uses neural_grm_model_prob.
        """
        nn_params = self._extract_nn_params(kwargs)

        # Shift ddifficulties to enforce minimum gap of 0.1 between thresholds
        # (ddifficulties are already positive from Softplus bijector)
        ddifficulties = ddifficulties + 0.1
        difficulties = jnp.concat([difficulties0, ddifficulties], axis=-1)
        difficulties = jnp.cumsum(difficulties, axis=-1)

        rank = len(abilities.shape)
        batch_shape = abilities.shape[: (rank - 4)]
        batch_ndims = len(batch_shape)

        people = data[self.person_key].astype(jnp.int32)
        choices = jnp.concat([data[i][:, jnp.newaxis] for i in self.item_keys], axis=-1)

        bad_choices = (choices < 0) | (choices >= self.response_cardinality) | jnp.isnan(choices)
        choices = jnp.where(bad_choices, jnp.zeros_like(choices), choices)

        for _ in range(batch_ndims):
            choices = choices[jnp.newaxis, ...]

        abilities = abilities[:, people, ...]

        # Use log-space computation to avoid NaN from near-zero probs
        log_response_probs, response_probs = self.neural_grm_log_prob(
            abilities, discriminations, difficulties, nn_params
        )

        # Log-likelihood: log P(Y=chosen_k) — index directly from log_probs
        choices_oh = jax.nn.one_hot(choices.astype(jnp.int32), self.response_cardinality)
        log_probs = jnp.sum(log_response_probs * choices_oh, axis=-1)

        # Ignorability: missing responses contribute 0 to log-likelihood
        log_probs = jnp.where(bad_choices[jnp.newaxis, ...], 0.0, log_probs)

        log_probs = jnp.sum(log_probs, axis=-1)

        rv_responses = tfd.Categorical(probs=response_probs)
        return {
            "log_likelihood": log_probs,
            "discriminations": discriminations,
            "rv": rv_responses,
        }

    def log_likelihood(
        self,
        data,
        discriminations,
        difficulties0,
        ddifficulties,
        abilities,
        *args,
        **kwargs
    ):
        prediction = self.predictive_distribution(
            data,
            discriminations,
            difficulties0,
            ddifficulties,
            abilities,
            *args,
            **kwargs
        )
        return prediction["log_likelihood"]

    def create_distributions(self, grouping_params=None):
        """Create prior and surrogate distributions.

        Same IRT priors as GRModel for abilities, mu, difficulties, discriminations.
        Additional priors for per-item Beta CDF shape parameters:
        - nn_log_a ~ Normal(0, 0.5) per item
        - nn_log_b ~ Normal(0, 0.5) per item

        At the prior mean (log_a=log_b=0), a=b=1 and betainc(1,1,x)=x,
        recovering the standard logistic GRM.
        """
        self.bijectors = {
            k: tfb.Identity() for k in ["abilities", "mu", "difficulties0"]
        }
        self.bijectors["discriminations"] = tfb.Softplus()
        self.bijectors["ddifficulties"] = tfb.Softplus()

        K = self.response_cardinality

        # --- Simplified priors for NeuralGRM (no Horseshoe hierarchy) ---
        # The NeuralGRM is a data generator, not a sparse model.
        # Use simple HalfNormal priors for discriminations instead of
        # the Horseshoe (AbsHorseshoe + SqrtInverseGamma) which causes NaN.
        grm_joint_distribution_dict = dict(
            mu=tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(
                        (1, self.dimensions, self.num_items, 1), dtype=self.dtype
                    ),
                    scale=jnp.ones(
                        (1, self.dimensions, self.num_items, 1), dtype=self.dtype
                    ),
                ),
                reinterpreted_batch_ndims=4,
            ),
            difficulties0=lambda mu: tfd.Independent(
                tfd.Normal(
                    loc=jnp.asarray(mu, dtype=self.dtype),
                    scale=jnp.ones(
                        (1, self.dimensions, self.num_items, 1), dtype=self.dtype
                    ),
                ),
                reinterpreted_batch_ndims=4,
            ),
            discriminations=tfd.Independent(
                tfd.HalfNormal(
                    scale=self._disc_prior_scale,
                ),
                reinterpreted_batch_ndims=4,
            ),
            ddifficulties=tfd.Independent(
                tfd.HalfNormal(
                    scale=jnp.ones(
                        (
                            1,
                            self.dimensions,
                            self.num_items,
                            self.response_cardinality - 2,
                        ),
                        dtype=self.dtype,
                    )
                ),
                reinterpreted_batch_ndims=4,
            ),
        )

        # --- Abilities prior ---
        if grouping_params is not None:
            raise NotImplementedError(
                "NeuralGRModel does not yet support grouping_params."
            )

        ability_dims = (
            self.dimensions
            if not self.include_independent
            else self.dimensions - 1
        )
        # Per-dimension ability prior scale: noisy dimension gets larger variance
        ability_scale = jnp.ones(
            (self.num_people, ability_dims, 1, 1), dtype=self.dtype
        )
        if self.noisy_dim:
            # Last dimension is the noisy one — wider prior
            noisy_scale = jnp.array(self.noisy_dim_ability_scale, dtype=self.dtype)
            ability_scale = ability_scale.at[:, -1, :, :].set(noisy_scale)

        grm_joint_distribution_dict["abilities"] = tfd.Independent(
            tfd.Normal(
                loc=jnp.zeros(
                    (self.num_people, ability_dims, 1, 1), dtype=self.dtype,
                ),
                scale=ability_scale,
            ),
            reinterpreted_batch_ndims=4,
        )

        # --- Per-item mixture-of-logits parameters ---
        # nn_scales: (I, M) — unconstrained, softplus -> positive scale per component
        #   Prior centered at 0.5413 so softplus(0.5413) ≈ 1.0 (unit scale = standard GRM)
        # nn_shifts: (I, M) — shift per component
        #   Prior centered at 0 (no shift = standard GRM)
        # nn_logit_weights: (I, M) — unconstrained, softmax -> mixture weights
        #   Prior centered at 0 (uniform weights)
        I = self.num_items
        M = self.nn_hidden_size  # number of mixture components
        nn_ps = self.nn_prior_scale
        grm_joint_distribution_dict['nn_scales'] = tfd.Independent(
            tfd.Normal(
                loc=0.5413 * jnp.ones((I, M), dtype=self.dtype),
                scale=nn_ps * jnp.ones((I, M), dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=2,
        )
        self.bijectors['nn_scales'] = tfb.Identity()
        grm_joint_distribution_dict['nn_shifts'] = tfd.Independent(
            tfd.Normal(
                loc=jnp.zeros((I, M), dtype=self.dtype),
                scale=nn_ps * jnp.ones((I, M), dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=2,
        )
        self.bijectors['nn_shifts'] = tfb.Identity()
        grm_joint_distribution_dict['nn_logit_weights'] = tfd.Independent(
            tfd.Normal(
                loc=jnp.zeros((I, M), dtype=self.dtype),
                scale=nn_ps * jnp.ones((I, M), dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=2,
        )
        self.bijectors['nn_logit_weights'] = tfb.Identity()

        self.joint_prior_distribution = tfd.JointDistributionNamed(
            grm_joint_distribution_dict
        )
        self.prior_distribution = self.joint_prior_distribution
        self.var_list = list(self.joint_prior_distribution.model.keys())

        self.surrogate_distribution_generator, self.surrogate_parameter_initializer = (
            build_factored_surrogate_posterior_generator(
                self.joint_prior_distribution,
                bijectors=self.bijectors,
                dtype=self.dtype,
                parameterization=self.parameterization,
                rank=self.rank,
            )
        )
        self.params = self.surrogate_parameter_initializer()

    def unormalized_log_prob(self, data, prior_weight=1., **params):
        """Compute unnormalized log probability (prior + likelihood).

        Unlike the standard GRM, the NeuralGRM omits the discrimination
        entropy term (-xlogy(w,w)/eta) because the per-item Kumaraswamy
        shape parameters already provide sufficient flexibility, and the
        1/eta factor causes NaN when eta approaches zero under the
        Horseshoe prior.
        """
        log_prior = self.joint_prior_distribution.log_prob(params)
        prediction = self.predictive_distribution(data, **params)
        log_likelihood = prediction["log_likelihood"]

        finite_portion = jnp.where(
            jnp.isfinite(log_likelihood),
            log_likelihood,
            jnp.zeros_like(log_likelihood),
        )
        min_val = jnp.min(finite_portion) - 1.0
        log_likelihood = jnp.where(
            jnp.isfinite(log_likelihood),
            log_likelihood,
            jnp.ones_like(log_likelihood) * min_val,
        )
        return jnp.astype(prior_weight, log_prior.dtype) * log_prior + jnp.sum(
            log_likelihood, axis=-1
        )

    def save_to_disk(self, path):
        """Save NeuralGRModel to disk (YAML config + HDF5 params)."""
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            '_class_name': 'NeuralGRModel',
            'item_keys': list(self.item_keys),
            'num_people': int(self.num_people),
            'dim': int(self._primary_dim),
            'decay': float(self.dimensional_decay),
            'noisy_dim': bool(self.noisy_dim),
            'noisy_dim_eta_scale': float(self.noisy_dim_eta_scale),
            'noisy_dim_ability_scale': float(self.noisy_dim_ability_scale),
            'positive_discriminations': bool(self.positive_discriminations),
            'missing_val': int(self.missing_val),
            'full_rank': bool(self.full_rank),
            'eta_scale': float(self.eta_scale),
            'kappa_scale': float(self.kappa_scale.flatten()[0]) if hasattr(self.kappa_scale, 'flatten') else float(self.kappa_scale),
            'weight_exponent': float(self.weight_exponent),
            'response_cardinality': int(self.response_cardinality),
            'include_independent': bool(self.include_independent),
            'vi_mode': str(self.vi_mode),
            'nn_hidden_sizes': int(self.nn_hidden_size),
            'nn_prior_scale': float(self.nn_prior_scale),
            'dtype': 'float64' if self.dtype == jnp.float64 else 'float32',
        }
        with open(path / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        with h5py.File(path / 'params.h5', 'w') as f:
            if self.params is not None and hasattr(self.params, 'items'):
                grp = f.create_group('params')
                for k, v in self.params.items():
                    arr = np.array(v)
                    if arr.size > 0:
                        grp.create_dataset(k, data=arr)
            if isinstance(self.point_estimate_vars, dict):
                pe_grp = f.create_group('point_estimate_vars')
                for k, v in self.point_estimate_vars.items():
                    pe_grp.create_dataset(k, data=np.array(v))

    @classmethod
    def load_from_disk(cls, path):
        """Load NeuralGRModel from disk."""
        path = pathlib.Path(path)
        with open(path / 'config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        config.pop('_class_name', None)
        dtype_str = config.pop('dtype', 'float64')
        dtype = jnp.float64 if dtype_str == 'float64' else jnp.float32
        config['dtype'] = dtype
        # Remove legacy fields that may exist in old saved models
        config.pop('per_item_nn', None)

        instance = cls(**config)

        if (path / 'params.h5').exists():
            with h5py.File(path / 'params.h5', 'r') as f:
                if 'params' in f:
                    instance.params = {k: jnp.array(v) for k, v in f['params'].items()}
                if 'point_estimate_vars' in f:
                    instance.point_estimate_vars = {
                        k: jnp.array(v) for k, v in f['point_estimate_vars'].items()
                    }

        return instance

    def simulate_data(self, abilities=None, seed=0):
        """Generate synthetic response data from the fitted model.

        Uses calibrated_expectations for Beta CDF params and item parameters.
        Returns (N, I) integer response matrix with values in [0, K-1].

        Args:
            abilities: Optional ability array of shape (N, dim, 1, 1) where
                dim is the primary dimension (excluding noisy dim). If
                noisy_dim=True, random noise abilities are appended.
                If None, uses model's calibrated abilities.
            seed: Random seed (int) for categorical sampling.
        """
        discrimination = self.calibrated_expectations['discriminations']
        if abilities is None:
            abilities = self.calibrated_expectations['abilities']
        elif self.noisy_dim:
            # abilities provided are for the primary dimensions only;
            # append random noisy-dimension abilities
            N = abilities.shape[0]
            rng = np.random.default_rng(seed + 7777)
            noisy_abilities = rng.normal(
                0, self.noisy_dim_ability_scale, size=(N, 1, 1, 1)
            )
            abilities = np.concatenate(
                [np.array(abilities), noisy_abilities], axis=1
            )

        nn_params = {
            k: self.calibrated_expectations[k]
            for k in self._get_nn_param_names()
        }

        difficulties0 = self.calibrated_expectations['difficulties0']
        if 'ddifficulties' in self.calibrated_expectations:
            ddifficulties = self.calibrated_expectations['ddifficulties']
        else:
            # K=2: no additional thresholds beyond difficulties0
            ddifficulties = jnp.zeros((*difficulties0.shape[:-1], 0))

        probs = self.neural_grm_model_prob_d(
            abilities,
            discrimination,
            difficulties0,
            ddifficulties,
            nn_params,
        )
        # Ensure valid probability simplex
        probs = jnp.clip(probs, 1e-10, None)
        probs = probs / jnp.sum(probs, axis=-1, keepdims=True)

        response_rv = tfd.Categorical(probs=probs)
        responses = response_rv.sample(seed=jax.random.PRNGKey(seed))
        # Ensure integer responses in valid range
        responses = jnp.clip(responses, 0, self.response_cardinality - 1).astype(jnp.int32)
        return np.array(responses)

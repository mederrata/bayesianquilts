import pathlib

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from typing import Any

from flax import nnx
from jax.scipy.special import xlogy
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf

from bayesianquilts.distributions import AbsHorseshoe, SqrtInverseGamma
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator
from bayesianquilts.irt.irt import IRTModel


class NeuralGRModel(IRTModel):
    """Graded Response Model with a per-item Kumaraswamy CDF response function.

    Instead of the standard sigmoid link P(Y >= k) = sigma(a*(theta - b_k)),
    this model uses P(Y >= k) = g_i(a*(theta - b_k)) where g_i is a per-item
    monotone function based on the Kumaraswamy CDF:

        g_i(z) = 1 - (1 - sigma(z)^alpha_i)^beta_i

    where sigma(z) is the standard logistic sigmoid, and alpha_i, beta_i > 0
    are per-item shape parameters.

    Monotonicity: sigma(z) is increasing -> x^a is increasing (a>0) ->
    1-x^a is decreasing -> (1-x^a)^b is decreasing (b>0) ->
    1-(1-x^a)^b is increasing. So g_i is monotone increasing.

    When alpha=beta=1: g(z) = 1-(1-sigma(z))^1 = sigma(z), recovering the
    standard logistic GRM. When alpha != beta, the ICC becomes asymmetric
    with genuinely non-logistic shapes that no standard GRM can replicate.
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
        noisy_dim_eta_scale=0.01,
        noisy_dim_ability_scale=2.0,
        # Legacy params accepted but ignored for backward compat
        nn_hidden_sizes=None,
        per_item_nn=None,
    ):
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
            dtype=dtype,
        )
        # Store the user-facing dim (before noisy augmentation)
        self._primary_dim = dim

        # Override kappa_scale for the noisy dimension to enforce strong shrinkage
        if noisy_dim:
            # kappa_scale shape is (1, D, 1, 1) after set_dimension
            # Replace the last dimension's scale with a much smaller value
            noisy_kappa = jnp.array(
                noisy_dim_eta_scale, dtype=dtype
            ) * jnp.ones((1, 1, 1, 1), dtype=dtype)
            self.kappa_scale = jnp.concatenate(
                [self.kappa_scale[:, :dim, :, :], noisy_kappa], axis=1
            )

        self.create_distributions()

    def _monotone_forward(self, z, nn_params):
        """Per-item Kumaraswamy CDF response function.

        g_i(z) = 1 - (1 - sigma(z)^a_i)^b_i

        Monotone by composition: sigma increasing, x^a increasing (a>0),
        1-x^a decreasing, (.)^b decreasing (b>0), 1-(.) increasing.

        When a=b=1: g(z) = sigma(z) — standard GRM.
        When a != b: asymmetric ICC shapes.
        When a > 1: steeper near 0 (sharper lower threshold).
        When b > 1: steeper near 1 (sharper upper threshold).

        Parameters:
        - nn_log_a: (batch..., I) — unconstrained shape param (exp -> positive)
        - nn_log_b: (batch..., I) — unconstrained shape param (exp -> positive)

        Args:
            z: (..., N, D, I, K-1) — the scaled offsets.
            nn_params: Dict with keys nn_log_a, nn_log_b.

        Returns:
            Array of same shape as z, with values in (0, 1).
        """
        log_a = nn_params['nn_log_a']  # (batch..., I)
        log_b = nn_params['nn_log_b']  # (batch..., I)

        # Positive shape params via exp, clipped for numerical stability
        a = jnp.clip(jnp.exp(log_a), 0.1, 10.0)
        b = jnp.clip(jnp.exp(log_b), 0.1, 10.0)

        # Determine batch dims: everything before the I dimension
        n_batch_dims = log_a.ndim - 1

        # Reshape a, b: (batch..., I) -> (batch..., 1, 1, I, 1)
        for _ in range(2):  # insert N, D dims
            a = jnp.expand_dims(a, axis=n_batch_dims)
            b = jnp.expand_dims(b, axis=n_batch_dims)
        a = jnp.expand_dims(a, axis=-1)  # insert K-1 dim
        b = jnp.expand_dims(b, axis=-1)

        # Apply sigmoid, then Kumaraswamy CDF
        x = jax.nn.sigmoid(z)  # (..., N, D, I, K-1) in (0, 1)
        x = jnp.clip(x, 1e-7, 1 - 1e-7)

        # Kumaraswamy CDF: 1 - (1 - x^a)^b
        x_a = jnp.power(x, a)
        return 1.0 - jnp.power(1.0 - x_a, b)

    def _get_nn_param_names(self):
        """Return list of Kumaraswamy CDF parameter names."""
        return ['nn_log_a', 'nn_log_b']

    def _extract_nn_params(self, params):
        """Extract Kumaraswamy CDF parameters from a params dict."""
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

        # Apply monotone Beta CDF to -scaled (matching GRM's sigmoid(-scaled) convention)
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

        # Weight by discrimination and sum over dimensions axis
        weights = (
            jnp.abs(discriminations) ** self.weight_exponent
            / jnp.sum(jnp.abs(discriminations) ** self.weight_exponent, axis=-3)[
                ..., jnp.newaxis, :, :
            ]
        )
        probs = jnp.sum(probs * weights, axis=-3)
        return probs

    def neural_grm_model_prob_d(
        self, abilities, discriminations, difficulties0, ddifficulties, nn_params
    ):
        """Convenience: construct cumulative difficulties then call neural_grm_model_prob."""
        d0 = jnp.concat([difficulties0, ddifficulties], axis=-1)
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
        """Compute predictive distribution using the Kumaraswamy CDF GRM.

        Same interface as GRModel.predictive_distribution but extracts Beta CDF
        params from kwargs and uses neural_grm_model_prob.
        """
        nn_params = self._extract_nn_params(kwargs)

        ddifficulties = jnp.where(
            ddifficulties < 1e-1, 1e-1 * jnp.ones_like(ddifficulties), ddifficulties
        )
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

        response_probs = self.neural_grm_model_prob(
            abilities, discriminations, difficulties, nn_params
        )
        imputed_lp = jnp.sum(xlogy(response_probs, response_probs), axis=-1)

        rv_responses = tfd.Categorical(probs=response_probs)
        log_probs = rv_responses.log_prob(choices)

        imputation_pmfs = data.get('_imputation_pmfs')
        if imputation_pmfs is not None:
            log_rp = jnp.log(jnp.maximum(response_probs, 1e-30))
            log_q = jnp.log(jnp.maximum(imputation_pmfs, 1e-30))
            rb = jax.scipy.special.logsumexp(
                log_rp + log_q[jnp.newaxis, ...], axis=-1
            )
            log_probs = jnp.where(bad_choices[jnp.newaxis, ...], rb, log_probs)
        else:
            log_probs = jnp.where(
                bad_choices[jnp.newaxis, ...], imputed_lp, log_probs
            )

        log_probs = jnp.sum(log_probs, axis=-1)

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
        self.bijectors["eta"] = tfb.Softplus()
        self.bijectors["kappa"] = tfb.Softplus()
        self.bijectors["discriminations"] = tfb.Softplus()
        self.bijectors["ddifficulties"] = tfb.Softplus()
        self.bijectors["eta_a"] = tfb.Softplus()
        self.bijectors["kappa_a"] = tfb.Softplus()
        self.bijectors["xi"] = tfb.Softplus()

        K = self.response_cardinality

        # --- Standard IRT priors (same as GRModel) ---
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
                    loc=mu,
                    scale=jnp.ones(
                        (1, self.dimensions, self.num_items, 1), dtype=self.dtype
                    ),
                ),
                reinterpreted_batch_ndims=4,
            ),
            discriminations=(
                (
                    lambda eta, kappa: tfd.Independent(
                        AbsHorseshoe(scale=eta * kappa), reinterpreted_batch_ndims=4
                    )
                )
                if self.positive_discriminations
                else (
                    lambda eta, xi, kappa: tfd.Independent(
                        tfd.Horseshoe(
                            loc=jnp.zeros(
                                (1, self.dimensions, self.num_items, 1),
                                dtype=self.dtype,
                            ),
                            scale=eta * kappa,
                        ),
                        reinterpreted_batch_ndims=4,
                    )
                )
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
            eta=tfd.Independent(
                tfd.HalfNormal(
                    scale=self.eta_scale
                    * jnp.ones((1, 1, self.num_items, 1), dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=4,
            ),
            kappa=lambda kappa_a: tfd.Independent(
                SqrtInverseGamma(
                    0.5 * jnp.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                    1.0 / kappa_a,
                ),
                reinterpreted_batch_ndims=4,
            ),
            kappa_a=tfd.Independent(
                tfd.InverseGamma(
                    0.5 * jnp.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                    jnp.ones((1, self.dimensions, 1, 1), dtype=self.dtype)
                    / self.kappa_scale**2,
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

        # --- Per-item Kumaraswamy CDF shape parameters ---
        # Prior centered at log(a)=log(b)=0, i.e. a=b=1 (standard GRM).
        # Scale 0.5 allows moderate deviations: 95% of prior mass gives
        # a, b in [exp(-1), exp(1)] ≈ [0.37, 2.72].
        I = self.num_items
        for param_name in ['nn_log_a', 'nn_log_b']:
            grm_joint_distribution_dict[param_name] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros((I,), dtype=self.dtype),
                    scale=0.5 * jnp.ones((I,), dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=1,
            )
            self.bijectors[param_name] = tfb.Identity()

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
        """Compute unnormalized log probability (prior + likelihood - entropy)."""
        log_prior = self.joint_prior_distribution.log_prob(params)
        prediction = self.predictive_distribution(data, **params)
        log_likelihood = prediction["log_likelihood"]
        weights = prediction["discriminations"]
        weights = weights / jnp.sum(weights, axis=-3, keepdims=True)
        entropy = -xlogy(weights, weights) / params["eta"]
        entropy = jnp.sum(entropy, axis=[-1, -2, -3, -4])

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
        return jnp.astype(prior_weight, log_prior.dtype) * (log_prior - entropy) + jnp.sum(
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
        config.pop('nn_hidden_sizes', None)
        config.pop('per_item_nn', None)

        instance = cls(**config)

        if (path / 'params.h5').exists():
            with h5py.File(path / 'params.h5', 'r') as f:
                if 'params' in f:
                    instance.params = {k: jnp.array(v) for k, v in f['params'].items()}

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

"""Neural Logistic Model for binary IRT items.

For binary items (K=2), the GRM's cumulative-probability machinery is unnecessary.
This model directly parameterizes:

    P(Y_i = 1 | θ) = Σ_m  w_{i,m} · σ(s_{i,m} · θ + c_{i,m})

where:
    s_{i,m} > 0  via softplus
    w_{i,m}      via softmax (simplex)
    c_{i,m}      unconstrained shifts

With M=1, s=1, c=0 this reduces to standard 1PL: P(Y=1|θ) = σ(θ).
With M>1 we get flexible asymmetric ICCs.

No separate discrimination/difficulty parameters — they're absorbed into
the mixture components' scales and shifts.
"""

import pathlib

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd

from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator
from bayesianquilts.irt.irt import IRTModel


def _mixture_of_logits_binary(theta, nn_scales, nn_shifts, nn_logit_weights):
    """Compute P(Y=1|θ) for each item via mixture of logistic sigmoids.

    Args:
        theta: (S, N, 1, 1) or (N, 1, 1) — abilities (dim=1, no item/cat axes)
        nn_scales: (S, I, M) or (I, M) — unconstrained scales
        nn_shifts: (S, I, M) or (I, M) — shift parameters
        nn_logit_weights: (S, I, M) or (I, M) — unconstrained mixture weights

    Returns:
        p: (S, N, I) or (N, I) — P(Y_i=1|θ)
    """
    scales = jax.nn.softplus(nn_scales)       # positive
    weights = jax.nn.softmax(nn_logit_weights, axis=-1)  # simplex over M

    # theta: (..., N, 1, 1) -> (..., N, 1) drop last dim
    theta = theta[..., 0]  # (..., N, 1)

    # We need (..., N, I, M)
    # theta: (..., N, 1) -> (..., N, 1, 1)  for broadcasting over I, M
    theta_exp = theta[..., jnp.newaxis]  # (..., N, 1, 1)

    # scales, shifts, weights: (..., I, M) -> (..., 1, I, M) for broadcasting over N
    n_batch = scales.ndim - 2  # number of sample dims
    for _ in range(1):  # insert N dim
        scales = jnp.expand_dims(scales, axis=n_batch)
        nn_shifts = jnp.expand_dims(nn_shifts, axis=n_batch)
        weights = jnp.expand_dims(weights, axis=n_batch)

    # (..., N, I, M)
    logits = theta_exp * scales + nn_shifts
    component_probs = jax.nn.sigmoid(logits)

    # Weighted sum over M: (..., N, I)
    p = jnp.sum(component_probs * weights, axis=-1)
    return p


class NeuralLogisticModel(IRTModel):
    """Neural logistic model for binary IRT items.

    P(Y_i = 1 | θ) = Σ_m  w_{i,m} · σ(s_{i,m} · θ + c_{i,m})

    Parameters:
        abilities: (N, 1, 1, 1) — latent trait
        nn_scales: (I, M) — per-item, per-component scale (softplus → positive)
        nn_shifts: (I, M) — per-item, per-component shift
        nn_logit_weights: (I, M) — per-item mixture weights (softmax → simplex)
    """

    response_type = "binary"

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
        response_cardinality=2,
        discrimination_guess=None,
        include_independent=False,
        vi_mode='advi',
        imputation_model=None,
        dtype=jnp.float64,
        parameterization="softplus",
        rank=0,
        nn_hidden_sizes=None,
        nn_prior_scale=0.5,
        # Ignored kwargs for pipeline compat
        noisy_dim=False,
        noisy_dim_eta_scale=0.1,
        noisy_dim_ability_scale=1.0,
        **kwargs,
    ):
        assert response_cardinality == 2, (
            f"NeuralLogisticModel is for binary items only, got K={response_cardinality}"
        )
        self.nn_hidden_size = nn_hidden_sizes if nn_hidden_sizes is not None else 4
        self.nn_prior_scale = nn_prior_scale

        super().__init__(
            item_keys=item_keys,
            num_people=num_people,
            num_groups=num_groups,
            data=data,
            person_key=person_key,
            dim=dim,
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
        self.create_distributions()

    def predictive_distribution(self, data, abilities, nn_scales, nn_shifts,
                                nn_logit_weights, **kwargs):
        """Compute log-likelihood for binary responses."""
        people = data[self.person_key].astype(jnp.int32)
        choices = jnp.concat(
            [data[i][:, jnp.newaxis] for i in self.item_keys], axis=-1
        )  # (N, I)

        bad = (choices < 0) | (choices > 1) | jnp.isnan(choices)
        choices = jnp.where(bad, jnp.zeros_like(choices), choices)

        # abilities: (S, N_all, 1, 1, 1) -> index by people
        abilities = abilities[:, people, ...]  # (S, N_batch, 1, 1, 1)

        # P(Y=1|θ): (S, N_batch, I)
        p = _mixture_of_logits_binary(
            abilities[..., 0],  # drop last dummy dim -> (S, N, 1, 1)
            nn_scales, nn_shifts, nn_logit_weights,
        )
        p = jnp.clip(p, 1e-7, 1.0 - 1e-7)

        # Binary cross-entropy
        # choices: (N, I), expand for sample dim
        y = choices.astype(self.dtype)
        for _ in range(p.ndim - y.ndim):
            y = y[jnp.newaxis, ...]
            bad = bad[jnp.newaxis, ...]

        log_probs = y * jnp.log(p) + (1.0 - y) * jnp.log1p(-p)  # (S, N, I)

        # Ignorability: missing items contribute 0
        log_probs = jnp.where(bad, 0.0, log_probs)
        log_probs = jnp.sum(log_probs, axis=-1)  # (S, N)

        # Category probs for simulation
        probs = jnp.stack([1.0 - p, p], axis=-1)  # (S, N, I, 2)
        rv = tfd.Categorical(probs=probs)

        return {
            "log_likelihood": log_probs,
            "rv": rv,
        }

    def log_likelihood(self, data, abilities, nn_scales, nn_shifts,
                       nn_logit_weights, *args, **kwargs):
        pred = self.predictive_distribution(
            data, abilities, nn_scales, nn_shifts, nn_logit_weights, **kwargs
        )
        return pred["log_likelihood"]

    def unormalized_log_prob(self, data, prior_weight=1., **params):
        log_prior = self.joint_prior_distribution.log_prob(params)
        prediction = self.predictive_distribution(data, **params)
        log_likelihood = prediction["log_likelihood"]

        finite_portion = jnp.where(
            jnp.isfinite(log_likelihood), log_likelihood,
            jnp.zeros_like(log_likelihood),
        )
        min_val = jnp.min(finite_portion) - 1.0
        log_likelihood = jnp.where(
            jnp.isfinite(log_likelihood), log_likelihood,
            jnp.ones_like(log_likelihood) * min_val,
        )
        return jnp.astype(prior_weight, log_prior.dtype) * log_prior + jnp.sum(
            log_likelihood, axis=-1
        )

    def create_distributions(self, grouping_params=None):
        """Create prior and surrogate distributions."""
        self.bijectors = {}

        I = self.num_items
        M = self.nn_hidden_size
        nn_ps = self.nn_prior_scale

        dist_dict = {}

        # Abilities: N(0, 1)
        dist_dict["abilities"] = tfd.Independent(
            tfd.Normal(
                loc=jnp.zeros((self.num_people, 1, 1, 1), dtype=self.dtype),
                scale=jnp.ones((self.num_people, 1, 1, 1), dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=4,
        )
        self.bijectors["abilities"] = tfb.Identity()

        # NN scales: centered at softplus^{-1}(1) ≈ 0.5413 so default scale ≈ 1
        dist_dict["nn_scales"] = tfd.Independent(
            tfd.Normal(
                loc=0.5413 * jnp.ones((I, M), dtype=self.dtype),
                scale=nn_ps * jnp.ones((I, M), dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=2,
        )
        self.bijectors["nn_scales"] = tfb.Identity()

        # NN shifts: centered at 0 (no shift = symmetric)
        dist_dict["nn_shifts"] = tfd.Independent(
            tfd.Normal(
                loc=jnp.zeros((I, M), dtype=self.dtype),
                scale=nn_ps * jnp.ones((I, M), dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=2,
        )
        self.bijectors["nn_shifts"] = tfb.Identity()

        # NN logit weights: centered at 0 (uniform weights)
        dist_dict["nn_logit_weights"] = tfd.Independent(
            tfd.Normal(
                loc=jnp.zeros((I, M), dtype=self.dtype),
                scale=nn_ps * jnp.ones((I, M), dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=2,
        )
        self.bijectors["nn_logit_weights"] = tfb.Identity()

        self.joint_prior_distribution = tfd.JointDistributionNamed(dist_dict)
        self.prior_distribution = self.joint_prior_distribution
        self.var_list = list(dist_dict.keys())

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

    def simulate_data(self, abilities=None, seed=0):
        """Generate binary responses from the fitted model.

        Args:
            abilities: (N, 1, 1, 1) ability array. If None, uses calibrated.
            seed: Random seed.

        Returns:
            (N, I) integer array with values in {0, 1}.
        """
        if abilities is None:
            abilities = self.calibrated_expectations['abilities']

        nn_scales = self.calibrated_expectations['nn_scales']
        nn_shifts = self.calibrated_expectations['nn_shifts']
        nn_logit_weights = self.calibrated_expectations['nn_logit_weights']

        # P(Y=1|θ): (N, I)
        p = _mixture_of_logits_binary(
            abilities[..., 0],  # (N, 1, 1)
            nn_scales, nn_shifts, nn_logit_weights,
        )
        p = jnp.clip(p, 1e-10, 1.0 - 1e-10)

        probs = jnp.stack([1.0 - p, p], axis=-1)  # (N, I, 2)
        rv = tfd.Categorical(probs=probs)
        responses = rv.sample(seed=jax.random.PRNGKey(seed))
        return np.array(responses.astype(jnp.int32))

    def save_to_disk(self, path):
        """Save to disk (YAML config + HDF5 params)."""
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            '_class_name': 'NeuralLogisticModel',
            'item_keys': list(self.item_keys),
            'num_people': int(self.num_people),
            'dim': int(self.dimensions),
            'nn_hidden_sizes': int(self.nn_hidden_size),
            'nn_prior_scale': float(self.nn_prior_scale),
            'response_cardinality': 2,
            'missing_val': int(self.missing_val),
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
            if isinstance(self.point_estimate_vars, dict):
                pe_grp = f.create_group('point_estimate_vars')
                for k, v in self.point_estimate_vars.items():
                    pe_grp.create_dataset(k, data=np.array(v))

    @classmethod
    def load_from_disk(cls, path):
        """Load from disk."""
        path = pathlib.Path(path)
        with open(path / 'config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        config.pop('_class_name', None)
        dtype_str = config.pop('dtype', 'float64')
        config['dtype'] = jnp.float64 if dtype_str == 'float64' else jnp.float32

        instance = cls(**config)

        if (path / 'params.h5').exists():
            with h5py.File(path / 'params.h5', 'r') as f:
                if 'params' in f:
                    instance.params = {
                        k: jnp.array(v) for k, v in f['params'].items()
                    }
                if 'point_estimate_vars' in f:
                    instance.point_estimate_vars = {
                        k: jnp.array(v) for k, v in f['point_estimate_vars'].items()
                    }

        return instance

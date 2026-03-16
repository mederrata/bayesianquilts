#!/usr/bin/env python3
"""Quilted Weibull survival model."""

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd

from bayesianquilts.jax.parameter import Decomposed
from bayesianquilts.metrics.ais import AutoDiffLikelihoodMixin
from bayesianquilts.model import QuiltedBayesianModel
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator

_CONC_FLOOR = 1e-6
_CONC_CEIL = 20.0
_SCALE_FLOOR = 1e-12
_SCALE_CEIL = 1e6


def _weibull_ll(concentration, scale, time, event, dtype):
    """Compute Weibull log-likelihood with censoring.

    Args:
        concentration: (S, N) shape parameter k > 0
        scale: (S, N) scale parameter lambda > 0
        time: (N,) observed times > 0
        event: (N,) event indicators (1=event, 0=censored)
        dtype: computation dtype

    Returns:
        (S, N) log-likelihood array
    """
    concentration = jnp.clip(concentration, _CONC_FLOOR, _CONC_CEIL)
    scale = jnp.clip(scale, _SCALE_FLOOR, _SCALE_CEIL)

    t = time[jnp.newaxis, :]  # (1, N)
    log_t = jnp.log(jnp.maximum(t, 1e-30))
    log_scale = jnp.log(scale)

    # z^k = (t/lambda)^k = exp(k * (log(t) - log(lambda)))
    log_z = concentration * (log_t - log_scale)
    z_k = jnp.exp(log_z)

    # log f(t) = log(k) - log(lambda) + (k-1)*(log(t) - log(lambda)) - (t/lambda)^k
    log_prob = (
        jnp.log(concentration)
        - log_scale
        + (concentration - 1) * (log_t - log_scale)
        - z_k
    )

    # log S(t) = -(t/lambda)^k
    log_sf = -z_k

    return event[jnp.newaxis, :] * log_prob + (1 - event[jnp.newaxis, :]) * log_sf


class WeibullQuilt(QuiltedBayesianModel):
    """Quilted Weibull survival model.

    Additive decomposition in log-parameter space (multiplicative on
    concentration and scale) with horseshoe priors for shrinkage of
    higher-order interactions.

    Each group gets its own Weibull(concentration, scale) parameters.
    The decomposition operates on [log_concentration, log_scale] jointly.
    """

    def __init__(
        self,
        rate_interact,
        shrinkage_scale=4e-2,
        dim_decay_factor=0.9,
        time_scale=1.0,
        dtype=jnp.float64,
        initialize_distributions=True,
    ):
        super().__init__(dtype=dtype)
        self.time_scale = float(time_scale)
        self.shrinkage_scale = shrinkage_scale
        self.dim_decay_factor = dim_decay_factor
        self.rate_interact = rate_interact

        self.param_decomposition = Decomposed(
            interactions=rate_interact,
            param_shape=[2],  # [log_concentration, log_scale]
            name="weibull",
            dtype=dtype,
            post_fn=jnp.exp,
        )

        if initialize_distributions:
            self.create_distributions()

    def create_distributions(self):
        (
            param_tensors,
            param_vars,
            param_shapes,
        ) = self.param_decomposition.generate_tensors(dtype=self.dtype)

        param_scales = {
            k: 2 * self.dim_decay_factor ** (len([d for d in v if d > 1]) - 1)
            for k, v in param_shapes.items()
        }

        self.param_vars = param_vars
        self.param_var_list = list(param_vars.keys())

        param_dict = {}
        for label, tensor in param_tensors.items():
            param_dict[label] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros_like(jnp.asarray(tensor, self.dtype)),
                    scale=param_scales[label]
                    * jnp.ones_like(jnp.asarray(tensor, self.dtype)),
                ),
                reinterpreted_batch_ndims=len(tensor.shape),
            )

        # Horseshoe hyperpriors
        interaction_prod = np.prod(self.param_decomposition._interaction_shape)
        hs_shape = [interaction_prod, 1, 2]

        param_dict["tau"] = tfd.Independent(
            tfd.HalfNormal(
                scale=self.shrinkage_scale
                * jnp.ones(hs_shape, dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=3,
        )
        param_dict["lambda_j"] = tfd.Independent(
            tfd.HalfNormal(
                scale=jnp.ones(hs_shape, dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=3,
        )

        param_tensors["tau"] = (
            self.shrinkage_scale
            * jnp.ones(hs_shape, dtype=self.dtype)
        )
        param_tensors["lambda_j"] = jnp.ones(hs_shape, dtype=self.dtype)

        param_model = tfd.JointDistributionNamed(param_dict)

        bijectors = defaultdict(tfp.bijectors.Identity)
        bijectors["tau"] = tfp.bijectors.Softplus()
        bijectors["lambda_j"] = tfp.bijectors.Softplus()

        param_surrogate_gen, param_param_init = (
            build_factored_surrogate_posterior_generator(
                param_model, bijectors=bijectors
            )
        )

        self.prior_distribution = tfd.JointDistributionNamed(
            {"param_model": param_model}
        )

        self.surrogate_distribution_generator = (
            lambda params: tfd.JointDistributionNamed(
                {**param_surrogate_gen(params).model}
            )
        )
        self.surrogate_parameter_initializer = lambda: {**param_param_init()}

        self.params = self.surrogate_parameter_initializer()
        self.var_list = list(self.params.keys())

    def predictive_distribution(self, data, **params):
        try:
            param_params = params["param_params"]
        except KeyError:
            param_params = {k: params[k] for k in self.param_var_list}

        param_indices = self.param_decomposition.retrieve_indices(data)

        # lookup returns (S, N, 1, 2) with post_fn=exp applied
        # squeeze the interaction dim to get (S, N, 2)
        weibull_params = self.param_decomposition.lookup(
            param_indices, tensors=param_params
        )
        if weibull_params.ndim == 4:
            weibull_params = weibull_params.squeeze(axis=-2)

        # weibull_params shape: (S, N, 2)
        concentration = weibull_params[..., 0]  # (S, N)
        scale = weibull_params[..., 1]  # (S, N)

        time = jnp.asarray(jnp.squeeze(data["time"]), self.dtype) / self.time_scale
        event = jnp.asarray(jnp.squeeze(data["event"]), self.dtype)

        log_likelihood = _weibull_ll(
            concentration, scale, time, event, self.dtype
        )

        return {
            "log_likelihood": log_likelihood,
            "prediction": weibull_params,
        }

    def log_likelihood(self, data, **params):
        return self.predictive_distribution(data, **params)["log_likelihood"]

    def unormalized_log_prob(self, data=None, prior_weight=1.0, **params):
        prediction = self.predictive_distribution(data, **params)
        log_likelihood = prediction["log_likelihood"]

        # Handle non-finite values
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

        param_prior_params = {
            k: jnp.asarray(params[k], self.dtype)
            for k in self.param_var_list
        }
        for k in ["tau", "lambda_j"]:
            param_prior_params[k] = jnp.asarray(params[k], self.dtype)
        prior = self.prior_distribution.log_prob(
            {"param_model": param_prior_params}
        )

        # Horseshoe effective prior: scale = tau * lambda_j
        def param_effective(lambda_j, tau):
            return tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros_like(lambda_j),
                    scale=tau * lambda_j,
                ),
                reinterpreted_batch_ndims=3,
            )

        param_coefs = self.param_decomposition.sum_parts(params)
        param_coefs = jnp.asarray(param_coefs, self.dtype)
        # Add axis for broadcasting with horseshoe: (S, 12, 2) -> (S, 12, 1, 2)
        param_coefs = param_coefs[..., jnp.newaxis, :]
        param_horseshoe = param_effective(
            params["lambda_j"], params["tau"]
        )
        prior_weight = jnp.asarray(prior_weight, self.dtype)

        energy = (
            jnp.sum(log_likelihood, axis=-1)
            + prior_weight * prior
            + prior_weight
            * jnp.asarray(param_horseshoe.log_prob(param_coefs), self.dtype)
        )

        return energy

    def expand(self, interaction):
        raise NotImplementedError("expand() not yet implemented")

    def fit(
        self,
        batched_data_factory,
        batch_size,
        dataset_size,
        warmup_max_order=6,
        sparsity_threshold=0.1,
        sparsity_method="relative_norm",
        epochs_per_stage=25,
        num_epochs=100,
        learning_rate=0.005,
        **kwargs,
    ):
        return self.staged_fit(
            batched_data_factory,
            max_order=warmup_max_order,
            sparsity_threshold=sparsity_threshold,
            sparsity_method=sparsity_method,
            epochs_per_stage=epochs_per_stage,
            final_epochs=num_epochs,
            batch_size=batch_size,
            dataset_size=dataset_size,
            learning_rate=learning_rate,
            **kwargs,
        )


class WeibullLikelihood(AutoDiffLikelihoodMixin):
    """AIS-compatible likelihood for the Weibull quilt model."""

    def __init__(self, param_decomposition, time_scale=1.0, dtype=jnp.float64):
        self.time_scale = float(time_scale)
        self.param_decomposition = param_decomposition
        self.param_var_list = list(param_decomposition._tensor_parts.keys())
        self.dtype = dtype

    def log_likelihood(self, data, params):
        """Compute per-observation log-likelihood.

        Returns (S, N) array.
        """
        param_params = {k: params[k] for k in self.param_var_list if k in params}

        param_indices = self.param_decomposition.retrieve_indices(data)
        weibull_params = self.param_decomposition.lookup(
            param_indices, tensors=param_params
        )
        if weibull_params.ndim == 4:
            weibull_params = weibull_params.squeeze(axis=-2)

        concentration = weibull_params[..., 0]
        scale = weibull_params[..., 1]

        time = jnp.asarray(jnp.squeeze(data["time"]), self.dtype) / self.time_scale
        event = jnp.asarray(jnp.squeeze(data["event"]), self.dtype)

        return _weibull_ll(concentration, scale, time, event, self.dtype)

    def extract_parameters(self, params):
        """Flatten Weibull decomposition params to (S, D)."""
        param_params = {k: params[k] for k in self.param_var_list if k in params}
        summed = self.param_decomposition.sum_parts(param_params)
        batch_shape = summed.shape[:-2]
        flat = jnp.reshape(summed, batch_shape + (-1,))
        if flat.ndim == 1:
            flat = flat[jnp.newaxis, :]
        return flat

    def reconstruct_parameters(self, flat_params, template):
        """Reconstruct from flat array -- store as global component only."""
        n_params = self.param_decomposition._param_shape[0]  # 2
        interaction_size = np.prod(
            self.param_decomposition._interaction_shape
        )

        summed = jnp.reshape(
            flat_params, flat_params.shape[:-1] + (interaction_size, n_params)
        )

        global_key = self.param_decomposition._name + "__"
        result = {}
        for k in self.param_var_list:
            if k == global_key:
                result[k] = summed
            else:
                result[k] = jnp.zeros_like(template[k])

        for k, v in template.items():
            if k not in result:
                result[k] = v

        return result

#!/usr/bin/env python3
"""Neural network piecewise exponential survival model."""

import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import tf2jax as tf

from bayesianquilts.metrics.ais import AutoDiffLikelihoodMixin
from bayesianquilts.predictors.nn.dense import DenseGaussian
from bayesianquilts.predictors.survival.piecewise_exponential_quilt import (
    _piecewise_exp_ll,
)
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator

_RATE_FLOOR = 1e-12
_RATE_CEIL = 1e4


class NeuralPiecewiseExponential(DenseGaussian):
    """ReLU neural network that outputs piecewise exponential hazard rates.

    Input: feature vector X (N, D)
    Output: n_intervals log-rates -> exp -> rates for piecewise exponential
    """

    def __init__(
        self,
        dim_regressors,
        breakpoints,
        hidden_size=16,
        depth=2,
        time_scale=1.0,
        prior_scale=1.0,
        weight_scale=0.1,
        bias_scale=0.1,
        global_rank=0,
        dtype=jnp.float64,
        **kwargs,
    ):
        self.time_scale = float(time_scale)
        self.breakpoints = jnp.asarray(breakpoints, dtype=dtype) / self.time_scale
        self.n_intervals = len(breakpoints) + 1
        self._global_rank = global_rank

        layer_sizes = [hidden_size] * depth + [self.n_intervals]

        super().__init__(
            input_size=dim_regressors,
            layer_sizes=layer_sizes,
            activation_fn=jax.nn.relu,
            weight_scale=weight_scale,
            bias_scale=bias_scale,
            prior_scale=prior_scale,
            dtype=dtype,
            **kwargs,
        )
        self.dim_regressors = dim_regressors

    def create_distributions(self):
        distribution_dict = {}
        bijectors = {}
        initial = {}
        for j, weight in enumerate(self.nn.weight_tensors[::2]):
            bijectors[f"w_{j}"] = tfb.Identity()
            distribution_dict[f"w_{j}"] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(
                        [1] * self.extra_batch_dims
                        + [self.layer_sizes[j], self.layer_sizes[j + 1]],
                        dtype=self.dtype,
                    ),
                    scale=self.prior_scale * jnp.ones(
                        [1] * self.extra_batch_dims
                        + [self.layer_sizes[j], self.layer_sizes[j + 1]],
                        dtype=self.dtype,
                    ),
                ),
                reinterpreted_batch_ndims=2 + self.extra_batch_dims,
            )
            initial[f"w_{j}"] = tf.convert_to_tensor(
                1e-3 * np.random.normal(
                    np.zeros(
                        [1] * self.extra_batch_dims
                        + [self.layer_sizes[j], self.layer_sizes[j + 1]]
                    ),
                    np.ones(
                        [1] * self.extra_batch_dims
                        + [self.layer_sizes[j], self.layer_sizes[j + 1]]
                    ),
                ),
                self.dtype,
            )

            bijectors[f"b_{j}"] = tfb.Identity()
            distribution_dict[f"b_{j}"] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(
                        [1] * self.extra_batch_dims
                        + [self.layer_sizes[j + 1]],
                        dtype=self.dtype,
                    ),
                    scale=self.prior_scale * jnp.ones(
                        [1] * self.extra_batch_dims
                        + [self.layer_sizes[j + 1]],
                        dtype=self.dtype,
                    ),
                ),
                reinterpreted_batch_ndims=1 + self.extra_batch_dims,
            )

        self.bijectors = bijectors
        self.prior_distribution = tfd.JointDistributionNamed(distribution_dict)
        self.surrogate_distribution_generator, self.surrogate_parameter_initializer = (
            build_factored_surrogate_posterior_generator(
                self.prior_distribution,
                bijectors,
                dtype=self.dtype,
                surrogate_initializers=initial,
                global_rank=self._global_rank,
            )
        )
        self.params = self.surrogate_parameter_initializer()
        self.var_list = list(self.prior_distribution.model.keys())

    def predictive_distribution(self, data, **params):
        X = jnp.asarray(data["X"], self.dtype)
        out = self.eval(X, params)  # (..., N, n_intervals)

        rates = jnp.exp(out)
        rates = jnp.clip(rates, _RATE_FLOOR, _RATE_CEIL)

        time = jnp.asarray(jnp.squeeze(data["time"]), self.dtype) / self.time_scale
        event = jnp.asarray(jnp.squeeze(data["event"]), self.dtype)

        log_likelihood = _piecewise_exp_ll(
            rates, time, event, self.breakpoints, self.dtype
        )

        return {
            "log_likelihood": log_likelihood,
            "prediction": rates,
        }

    def log_likelihood(self, data, **params):
        return self.predictive_distribution(data, **params)["log_likelihood"]

    def unormalized_log_prob(self, data=None, prior_weight=1.0, **params):
        log_lik = self.log_likelihood(data, **params)
        prior = self.prior_distribution.log_prob(params)

        if log_lik.ndim > 1:
            total_ll = jnp.sum(log_lik, axis=-1)
        else:
            total_ll = jnp.sum(log_lik)

        return total_ll + prior_weight * prior


class NeuralPiecewiseLikelihood(AutoDiffLikelihoodMixin):
    """AIS-compatible likelihood for neural piecewise exponential model."""

    def __init__(self, model):
        self.model = model
        self.dtype = model.dtype

    def log_likelihood(self, data, params):
        return self.model.log_likelihood(data, **params)

    def extract_parameters(self, params):
        flat_params = jax.vmap(
            lambda p: jax.flatten_util.ravel_pytree(p)[0]
        )(params)
        return flat_params

    def reconstruct_parameters(self, flat_params, template):
        if isinstance(template.get("w_0"), jnp.ndarray) and template["w_0"].ndim > 2:
            template = jax.tree_util.tree_map(lambda x: x[0], template)
        dummy_flat, unflatten = jax.flatten_util.ravel_pytree(template)
        K = dummy_flat.shape[0]
        input_shape = flat_params.shape
        if input_shape[-1] != K:
            raise ValueError(f"Last dimension {input_shape} != K={K}")
        batch_dims = input_shape[:-1]
        n_batch = 1
        for d in batch_dims:
            n_batch *= d
        flat_reshaped = flat_params.reshape((n_batch, K))
        unflattened_flat = jax.vmap(unflatten)(flat_reshaped)

        def reshape_leaf(leaf):
            leaf_param_shape = leaf.shape[1:]
            return leaf.reshape(batch_dims + leaf_param_shape)

        return jax.tree_util.tree_map(reshape_leaf, unflattened_flat)

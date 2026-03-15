#!/usr/bin/env python3
"""Quilted piecewise exponential survival model."""

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


def _piecewise_exp_ll(rates, time, event, breakpoints, dtype):
    """Compute piecewise exponential log-likelihood with censoring.

    Args:
        rates: (S, N, n_intervals) positive hazard rates
        time: (N,) observed times
        event: (N,) event indicators (1=event, 0=censored)
        breakpoints: (n_breaks,) interval boundaries
        dtype: computation dtype

    Returns:
        (S, N) log-likelihood array
    """
    indices = (time[:, jnp.newaxis] > breakpoints).sum(axis=-1)

    hazard = jnp.take_along_axis(
        rates, jnp.broadcast_to(
            indices[jnp.newaxis, :, jnp.newaxis],
            (rates.shape[0], indices.shape[0], 1)
        ), axis=-1
    ).squeeze(axis=-1)

    time_gaps = jnp.concatenate(
        [breakpoints[0:1], breakpoints[1:] - breakpoints[:-1]]
    )
    cum_masses = jnp.cumsum(
        rates[..., :-1] * time_gaps[jnp.newaxis, jnp.newaxis, :], axis=-1
    )
    indicator = (time[:, jnp.newaxis] > breakpoints).astype(dtype)
    cum_hazard_completed = (indicator[jnp.newaxis, ...] * cum_masses).sum(axis=-1)

    padded_breakpoints = jnp.concatenate([jnp.zeros(1, dtype=dtype), breakpoints])
    changepoint = padded_breakpoints[indices]
    cum_hazard = cum_hazard_completed + hazard * (
        time[jnp.newaxis, :] - changepoint[jnp.newaxis, :]
    )

    log_prob = jnp.log(hazard) - cum_hazard
    log_sf = -cum_hazard

    return event[jnp.newaxis, :] * log_prob + (1 - event[jnp.newaxis, :]) * log_sf


class PiecewiseExponentialQuilt(QuiltedBayesianModel):
    """Quilted piecewise exponential survival model.

    Additive decomposition in log-rate space (multiplicative on hazard)
    with horseshoe priors for shrinkage of higher-order interactions.
    """

    def __init__(
        self,
        breakpoints,
        rate_interact,
        shrinkage_scale=4e-2,
        dim_decay_factor=0.9,
        dtype=jnp.float64,
        initialize_distributions=True,
    ):
        super().__init__(dtype=dtype)
        self.breakpoints = jnp.asarray(breakpoints, dtype=dtype)
        self.n_intervals = len(breakpoints) + 1
        self.shrinkage_scale = shrinkage_scale
        self.dim_decay_factor = dim_decay_factor
        self.rate_interact = rate_interact

        self.rate_decomposition = Decomposed(
            interactions=rate_interact,
            param_shape=[self.n_intervals],
            name="log_rate",
            dtype=dtype,
            post_fn=jnp.exp,
        )

        if initialize_distributions:
            self.create_distributions()

    def create_distributions(self):
        (
            rate_tensors,
            rate_vars,
            rate_shapes,
        ) = self.rate_decomposition.generate_tensors(dtype=self.dtype)

        rate_scales = {
            k: 5 * self.dim_decay_factor ** (len([d for d in v if d > 1]) - 1)
            for k, v in rate_shapes.items()
        }

        self.rate_vars = rate_vars
        self.rate_var_list = list(rate_vars.keys())

        rate_dict = {}
        for label, tensor in rate_tensors.items():
            rate_dict[label] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros_like(jnp.asarray(tensor, self.dtype)),
                    scale=rate_scales[label]
                    * jnp.ones_like(jnp.asarray(tensor, self.dtype)),
                ),
                reinterpreted_batch_ndims=len(tensor.shape),
            )

        # Horseshoe hyperpriors
        interaction_prod = np.prod(self.rate_decomposition._interaction_shape)
        hs_shape_tau = [interaction_prod, 1, self.n_intervals]
        hs_shape_lambda = [interaction_prod, 1, self.n_intervals]

        rate_dict["c2"] = tfd.Independent(
            tfd.InverseGamma(
                jnp.ones(hs_shape_tau, dtype=self.dtype),
                jnp.ones(hs_shape_tau, dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=3,
        )
        rate_dict["tau"] = tfd.Independent(
            tfd.HalfStudentT(
                df=2,
                loc=0,
                scale=self.shrinkage_scale
                * jnp.ones(hs_shape_tau, dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=3,
        )
        rate_dict["lambda_j"] = tfd.Independent(
            tfd.HalfStudentT(
                df=5,
                loc=0,
                scale=jnp.ones(hs_shape_lambda, dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=3,
        )

        # Initial values for horseshoe params
        rate_tensors["tau"] = (
            2
            * self.shrinkage_scale
            * np.sqrt(2 / np.pi)
            * self.shrinkage_scale
            * jnp.ones(hs_shape_tau, dtype=self.dtype)
        )
        rate_tensors["c2"] = jnp.ones(hs_shape_tau, dtype=self.dtype)
        rate_tensors["lambda_j"] = jnp.ones(hs_shape_lambda, dtype=self.dtype)

        rate_model = tfd.JointDistributionNamed(rate_dict)

        bijectors = defaultdict(tfp.bijectors.Identity)
        bijectors["tau"] = tfp.bijectors.Softplus()
        bijectors["c2"] = tfp.bijectors.Softplus()
        bijectors["lambda_j"] = tfp.bijectors.Softplus()

        rate_surrogate_gen, rate_param_init = (
            build_factored_surrogate_posterior_generator(
                rate_model, bijectors=bijectors
            )
        )

        self.prior_distribution = tfd.JointDistributionNamed(
            {"rate_model": rate_model}
        )

        self.surrogate_distribution_generator = (
            lambda params: tfd.JointDistributionNamed(
                {**rate_surrogate_gen(params).model}
            )
        )
        self.surrogate_parameter_initializer = lambda: {**rate_param_init()}

        self.params = self.surrogate_parameter_initializer()
        self.var_list = list(self.params.keys())

    def predictive_distribution(self, data, **params):
        try:
            rate_params = params["rate_params"]
        except KeyError:
            rate_params = {k: params[k] for k in self.rate_var_list}

        rate_indices = self.rate_decomposition.retrieve_indices(data)

        # lookup returns (S, N, 1, n_intervals) with post_fn=exp applied
        # squeeze the interaction dim to get (S, N, n_intervals)
        rates = self.rate_decomposition.lookup(
            rate_indices, tensors=rate_params
        )
        if rates.ndim == 4:
            rates = rates.squeeze(axis=-2)

        time = jnp.asarray(jnp.squeeze(data["time"]), self.dtype)
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

        rate_prior_params = {
            k: jnp.asarray(params[k], self.dtype)
            for k in self.rate_var_list
        }
        for k in ["tau", "c2", "lambda_j"]:
            rate_prior_params[k] = jnp.asarray(params[k], self.dtype)
        prior = self.prior_distribution.log_prob(
            {"rate_model": rate_prior_params}
        )

        # Horseshoe effective prior on summed rate decomposition
        def rate_effective(lambda_j, c2, tau):
            return tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros_like(lambda_j),
                    scale=(
                        tau
                        * lambda_j
                        * jnp.sqrt(c2 / (c2 + tau**2 * lambda_j**2))
                    ),
                ),
                reinterpreted_batch_ndims=3,
            )

        rate_coefs = self.rate_decomposition.sum_parts(params)
        rate_coefs = jnp.asarray(rate_coefs, self.dtype)
        # Add axis for broadcasting with horseshoe: (S, 12, 4) -> (S, 12, 1, 4)
        rate_coefs = rate_coefs[..., jnp.newaxis, :]
        rate_horseshoe = rate_effective(
            params["lambda_j"], params["c2"], params["tau"]
        )
        prior_weight = jnp.asarray(prior_weight, self.dtype)

        energy = (
            jnp.sum(log_likelihood, axis=-1)
            + prior_weight * prior
            + prior_weight
            * jnp.asarray(rate_horseshoe.log_prob(rate_coefs), self.dtype)
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


class PiecewiseExponentialLikelihood(AutoDiffLikelihoodMixin):
    """AIS-compatible likelihood for the piecewise exponential quilt model."""

    def __init__(self, breakpoints, rate_decomposition, dtype=jnp.float64):
        self.breakpoints = jnp.asarray(breakpoints, dtype=dtype)
        self.rate_decomposition = rate_decomposition
        self.rate_var_list = list(rate_decomposition._tensor_parts.keys())
        self.dtype = dtype

    def log_likelihood(self, data, params):
        """Compute per-observation log-likelihood.

        Returns (S, N) array.
        """
        rate_params = {k: params[k] for k in self.rate_var_list if k in params}

        rate_indices = self.rate_decomposition.retrieve_indices(data)
        rates = self.rate_decomposition.lookup(
            rate_indices, tensors=rate_params
        )
        if rates.ndim == 4:
            rates = rates.squeeze(axis=-2)

        time = jnp.asarray(jnp.squeeze(data["time"]), self.dtype)
        event = jnp.asarray(jnp.squeeze(data["event"]), self.dtype)

        return _piecewise_exp_ll(rates, time, event, self.breakpoints, self.dtype)

    def extract_parameters(self, params):
        """Flatten rate decomposition params to (S, D)."""
        rate_params = {k: params[k] for k in self.rate_var_list if k in params}
        # sum_parts returns (S, prod(interaction_shape), n_intervals) before post_fn
        # We need it without post_fn for parameter space operations
        summed = self.rate_decomposition.sum_parts(rate_params)
        # Flatten last dims to get (S, D)
        batch_shape = summed.shape[:-2]
        flat = jnp.reshape(summed, batch_shape + (-1,))
        # If no batch dim, add one
        if flat.ndim == 1:
            flat = flat[jnp.newaxis, :]
        return flat

    def reconstruct_parameters(self, flat_params, template):
        """Reconstruct from flat array — store as global component only."""
        n_intervals = self.rate_decomposition._param_shape[0]
        interaction_size = np.prod(
            self.rate_decomposition._interaction_shape
        )

        # Reshape flat (S, D) -> (S, interaction_size, n_intervals)
        summed = jnp.reshape(
            flat_params, flat_params.shape[:-1] + (interaction_size, n_intervals)
        )

        # Store everything in the global component
        global_key = self.rate_decomposition._name + "__"
        result = {}
        for k in self.rate_var_list:
            if k == global_key:
                result[k] = summed
            else:
                result[k] = jnp.zeros_like(template[k])

        # Copy non-rate params from template
        for k, v in template.items():
            if k not in result:
                result[k] = v

        return result

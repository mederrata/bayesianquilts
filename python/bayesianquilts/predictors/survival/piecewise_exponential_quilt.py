#!/usr/bin/env python3
"""Quilted piecewise exponential survival model."""

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd

from bayesianquilts.distributions.gdr2 import GDR2Prior
from bayesianquilts.jax.parameter import Decomposed
from bayesianquilts.metrics.ais import AutoDiffLikelihoodMixin
from bayesianquilts.model import QuiltedBayesianModel
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator

_RATE_FLOOR = 1e-12
_RATE_CEIL = 1e4


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
    rates = jnp.clip(rates, _RATE_FLOOR, _RATE_CEIL)

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
    with configurable shrinkage priors (horseshoe or GDR2).
    """

    def __init__(
        self,
        breakpoints,
        rate_interact,
        shrinkage_scale=4e-2,
        dim_decay_factor=0.9,
        time_scale=1.0,
        dtype=jnp.float64,
        initialize_distributions=True,
        prior_type="horseshoe",
        mu_R2=0.5,
        phi_R2=2.0,
        gdr2_concentration=1.0,
    ):
        super().__init__(dtype=dtype)
        self.time_scale = float(time_scale)
        self.breakpoints = jnp.asarray(breakpoints, dtype=dtype) / self.time_scale
        self.n_intervals = len(breakpoints) + 1
        self.shrinkage_scale = shrinkage_scale
        self.dim_decay_factor = dim_decay_factor
        self.rate_interact = rate_interact
        self.prior_type = prior_type
        self.mu_R2 = mu_R2
        self.phi_R2 = phi_R2
        self.gdr2_concentration = gdr2_concentration

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

        self.rate_vars = rate_vars
        self.rate_var_list = list(rate_vars.keys())

        if self.prior_type in ("gdr2", "r2d2"):
            self._create_distributions_gdr2(rate_tensors, rate_shapes)
        else:
            self._create_distributions_horseshoe(rate_tensors, rate_shapes)

    def _create_distributions_horseshoe(self, rate_tensors, rate_shapes):
        rate_scales = {
            k: 2 * self.dim_decay_factor ** (len([d for d in v if d > 1]) - 1)
            for k, v in rate_shapes.items()
        }

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

        # Global horseshoe scale hyperprior — one tau per interval
        tau_shape = [self.n_intervals]
        rate_dict["tau"] = tfd.Independent(
            tfd.HalfNormal(
                scale=self.shrinkage_scale
                * jnp.ones(tau_shape, dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=1,
        )

        # Initial value for tau
        rate_tensors["tau"] = (
            self.shrinkage_scale
            * jnp.ones(tau_shape, dtype=self.dtype)
        )

        rate_model = tfd.JointDistributionNamed(rate_dict)

        bijectors = defaultdict(tfp.bijectors.Identity)
        bijectors["tau"] = tfp.bijectors.Softplus()

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

    def _create_distributions_gdr2(self, rate_tensors, rate_shapes):
        # Build GDR2 prior: one component per interaction term
        component_names = sorted(rate_tensors.keys())
        n_components = len(component_names)
        component_sizes = []
        component_weights = []

        for name in component_names:
            tensor = rate_tensors[name]
            component_sizes.append(int(np.prod(tensor.shape)))
            order = self.rate_decomposition.component_order(name)
            component_weights.append(
                self.dim_decay_factor ** max(order - 1, 0)
            )

        self.gdr2_prior = GDR2Prior(
            n_components=n_components,
            component_sizes=component_sizes,
            mu_R2=self.mu_R2,
            phi_R2=self.phi_R2,
            concentration=self.gdr2_concentration,
            component_weights=np.array(component_weights),
            prior_type=self.prior_type,
            dtype=self.dtype,
        )
        self._gdr2_component_names = component_names

        # Normal priors on rate components (wider — GDR2 provides shrinkage)
        rate_dict = {}
        for label, tensor in rate_tensors.items():
            rate_dict[label] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros_like(jnp.asarray(tensor, self.dtype)),
                    scale=2.0 * jnp.ones_like(jnp.asarray(tensor, self.dtype)),
                ),
                reinterpreted_batch_ndims=len(tensor.shape),
            )

        # R2 and phi in unconstrained space — sigmoid/softmax applied in log-prior
        # R2: scalar logit (sigmoid maps to (0,1))
        rate_dict["R2_logit"] = tfd.Independent(
            tfd.Normal(
                loc=jnp.zeros([1], dtype=self.dtype),
                scale=2.0 * jnp.ones([1], dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=1,
        )
        rate_tensors["R2_logit"] = jnp.zeros([1], dtype=self.dtype)

        # phi: K-dim logits (softmax maps to simplex)
        rate_dict["phi_logit"] = tfd.Independent(
            tfd.Normal(
                loc=jnp.zeros([n_components], dtype=self.dtype),
                scale=jnp.ones([n_components], dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=1,
        )
        rate_tensors["phi_logit"] = jnp.zeros(
            [n_components], dtype=self.dtype
        )

        rate_model = tfd.JointDistributionNamed(rate_dict)

        bijectors = defaultdict(tfp.bijectors.Identity)

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

        prior_weight = jnp.asarray(prior_weight, self.dtype)

        if self.prior_type in ("gdr2", "r2d2"):
            shrinkage_lp = self._gdr2_log_prior(params)
        else:
            shrinkage_lp = self._horseshoe_log_prior(params)

        # Component Normal priors from JointDistribution
        rate_prior_params = {
            k: jnp.asarray(params[k], self.dtype)
            for k in self.rate_var_list
        }
        if self.prior_type in ("gdr2", "r2d2"):
            rate_prior_params["R2_logit"] = jnp.asarray(
                params["R2_logit"], self.dtype
            )
            rate_prior_params["phi_logit"] = jnp.asarray(
                params["phi_logit"], self.dtype
            )
        else:
            rate_prior_params["tau"] = jnp.asarray(params["tau"], self.dtype)
        prior = self.prior_distribution.log_prob(
            {"rate_model": rate_prior_params}
        )

        energy = (
            jnp.sum(log_likelihood, axis=-1)
            + prior_weight * prior
            + prior_weight * shrinkage_lp
        )

        return energy

    def _horseshoe_log_prior(self, params):
        """Horseshoe effective prior on summed log-rate coefficients."""
        rate_coefs = self.rate_decomposition.sum_parts(params)
        # sum_parts applies post_fn=exp; undo to get log-rates
        rate_coefs = jnp.log(jnp.maximum(rate_coefs, _RATE_FLOOR))
        rate_coefs = jnp.asarray(rate_coefs, self.dtype)
        tau = jnp.maximum(params["tau"], 1e-8)
        if tau.ndim == 1:
            tau = tau[jnp.newaxis, :]
        tau = tau[:, jnp.newaxis, :]  # (S, 1, n_intervals)
        horseshoe = tfd.Horseshoe(scale=tau)
        return jnp.sum(horseshoe.log_prob(rate_coefs), axis=(-2, -1))

    def _gdr2_log_prior(self, params):
        """GDR2/R2D2 effective prior on rate coefficients."""
        R2_logit = jnp.asarray(params["R2_logit"], self.dtype)
        phi_logit = jnp.asarray(params["phi_logit"], self.dtype)

        # Map to constrained space
        R2 = jax.nn.sigmoid(R2_logit).squeeze(-1)
        phi = jax.nn.softmax(phi_logit, axis=-1)

        scales = self.gdr2_prior.compute_scales(R2, phi)

        # GDR2 prior on R2 and phi (includes Jacobian implicitly via log_prob)
        lp_R2 = self.gdr2_prior.log_prob_R2(R2)
        lp_phi = self.gdr2_prior.log_prob_phi(phi)

        # Collect coefficient groups in component order
        coefficients = []
        for name in self._gdr2_component_names:
            coef = jnp.asarray(params[name], self.dtype)
            coefficients.append(coef)

        lp_coef = self.gdr2_prior.log_prob_coefficients(coefficients, scales)

        return lp_R2 + lp_phi + lp_coef

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

    def __init__(self, breakpoints, rate_decomposition, time_scale=1.0,
                 dtype=jnp.float64):
        self.time_scale = float(time_scale)
        self.breakpoints = jnp.asarray(breakpoints, dtype=dtype) / self.time_scale
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

        time = jnp.asarray(jnp.squeeze(data["time"]), self.dtype) / self.time_scale
        event = jnp.asarray(jnp.squeeze(data["event"]), self.dtype)

        return _piecewise_exp_ll(rates, time, event, self.breakpoints, self.dtype)

    def extract_parameters(self, params):
        """Flatten rate decomposition params to (S, D) in log-rate space."""
        rate_params = {k: params[k] for k in self.rate_var_list if k in params}
        # sum_parts applies post_fn=exp; undo to get log-rates for parameter space ops
        summed = self.rate_decomposition.sum_parts(rate_params)
        summed = jnp.log(jnp.maximum(summed, _RATE_FLOOR))
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

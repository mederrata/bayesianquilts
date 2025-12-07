#!/usr/bin/env python3
"""Linear regression with horseshoe prior (regression version of logistic_regression.py)
"""

import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf

from bayesianquilts.model import BayesianModel
from bayesianquilts.tf.parameter import Decomposed, Interactions
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator


class LinearRegression(BayesianModel):
    def __init__(
        self,
        feature_names,
        outcome_label,
        regression_interact=None,
        dim_decay_factor=1.,
        regressor_scales=None,
        regressor_offsets=None,
        dtype=jnp.float64,
        global_horseshoe_scale=1.,
        noise_scale=1.0,
    ):
        super(LinearRegression, self).__init__(dtype=dtype)
        self.dim_decay_factor = dim_decay_factor
        self.dim_regressors = len(feature_names)
        self.feature_names = feature_names
        self.outcome_label = outcome_label
        self.noise_scale = noise_scale
        if regressor_scales is None:
            self.regressor_scales = 1
        else:
            self.regressor_scales = regressor_scales
        self.regressor_offsets = (
            regressor_offsets if regressor_offsets is not None else 0
        )

        if regression_interact is None:
            self.regression_interact = Interactions(
                [],
                exclusions=[],
            )
        else:
            self.regression_interact = regression_interact

        self.intercept_interact = Interactions(
            [],
            exclusions=[],
        )

        self.regression_decomposition = Decomposed(
            interactions=self.regression_interact,
            param_shape=[self.dim_regressors],
            name="beta",
            dtype=self.dtype,
        )

        self.intercept_decomposition = Decomposed(
            interactions=self.intercept_interact,
            param_shape=[1],
            name="intercept",
            dtype=self.dtype,
        )
        self.global_horseshoe_scale = global_horseshoe_scale
        self.create_distributions()

    def preprocessor(self):
        return lambda x: x

    def create_distributions(self):
        # distribution on regression problem

        (
            regressor_tensors,
            regression_vars,
            regression_shapes,
        ) = self.regression_decomposition.generate_tensors(dtype=self.dtype)
        regression_scales = {
            k: self.dim_decay_factor ** (len([d for d in v if d > 1]) - 1)
            for k, v in regression_shapes.items()
        }
        self.regression_var_list = list(regression_vars.keys())
        self.regression_decomposition.set_scales(regression_scales)

        regression_dict = {}

        regression_dict["beta__"] = lambda global_scale: tfd.Independent(
            tfd.Horseshoe(scale=global_scale*jnp.ones_like(regressor_tensors["beta__"])),
            reinterpreted_batch_ndims=len(regressor_tensors["beta__"].shape),
        )

        regression_dict["global_scale"] = lambda global_scale_aux: tfd.Independent(
            tfd.InverseGamma(
                concentration=0.5
                * jnp.ones(
                    [1] * len(regressor_tensors["beta__"].shape), self.dtype
                ),
                scale=1 / global_scale_aux,
            ),
            reinterpreted_batch_ndims=len(regressor_tensors["beta__"].shape),
        )
        regression_dict["global_scale_aux"] = tfd.Independent(
            tfd.InverseGamma(
                concentration=0.5
                * jnp.ones(
                    [1] * len(regressor_tensors["beta__"].shape), self.dtype
                ),
                scale=jnp.ones(
                    [1] * len(regressor_tensors["beta__"].shape), self.dtype
                )
                / self.global_horseshoe_scale**2,
            ),
            reinterpreted_batch_ndims=len(regressor_tensors["beta__"].shape),
        )

        # Add noise parameter for regression
        regression_dict["sigma"] = tfd.Independent(
            tfd.InverseGamma(
                concentration=2.0 * jnp.ones([1], self.dtype),
                scale=self.noise_scale * jnp.ones([1], self.dtype),
            ),
            reinterpreted_batch_ndims=1,
        )

        regression_model = tfd.JointDistributionNamed(regression_dict)
        regression_surrogate_generator, regression_surrogate_param_init = build_factored_surrogate_posterior_generator(
            regression_model
        )



        #  Exponential params
        (
            intercept_tensors,
            intercept_vars,
            intercept_shapes,
        ) = self.intercept_decomposition.generate_tensors(dtype=self.dtype)
        intercept_scales = {
            k: self.dim_decay_factor ** (len([d for d in v if d > 1]) - 1)
            for k, v in intercept_shapes.items()
        }
        self.intercept_decomposition.set_scales(intercept_scales)
        self.intercept_var_list = list(intercept_vars.keys())

        intercept_dict = {}
        for label, tensor in intercept_tensors.items():
            intercept_dict[label] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros_like(tf.cast(tensor, self.dtype)),
                    scale=jnp.ones_like(tf.cast(tensor, self.dtype)),
                ),
                reinterpreted_batch_ndims=len(tensor.shape),
            )

        intercept_prior = tfd.JointDistributionNamed(intercept_dict)
        intercept_surrogate_gen, intercept_param_init = build_factored_surrogate_posterior_generator(
            intercept_prior
        )

        self.prior_distribution = tfd.JointDistributionNamed(
            {"regression_model": regression_model, "intercept_model": intercept_prior}
        )
        self.surrogate_distribution_generator = lambda params: tfd.JointDistributionNamed(
            {**regression_surrogate_generator(params).model, **intercept_surrogate_gen(params).model}
        )
        self.surrogate_parameter_initializer = lambda: {
            **regression_surrogate_param_init(),
            **intercept_param_init(),
        }
        self.params = self.surrogate_parameter_initializer()

    def predictive_distribution(self, data, **params):
        try:
            regression_params = params["regression_params"]
            intercept_params = params["intercept_params"]
        except KeyError:
            regression_params = {k: params[k] for k in self.regression_var_list}
            intercept_params = {k: params[k] for k in self.intercept_var_list}

        processed = (self.preprocessor())(data)

        regression_indices = self.regression_decomposition.retrieve_indices(processed)

        intercept_indices = self.intercept_decomposition.retrieve_indices(processed)

        coef_ = self.regression_decomposition.lookup(
            regression_indices,
            tensors=regression_params,
        )

        intercept = self.intercept_decomposition.lookup(
            intercept_indices, tensors=intercept_params
        )

        X = processed["X"]
        y = processed["y"]

        # compute regression product
        X = ((X - self.regressor_offsets) / self.regressor_scales).astype(self.dtype)

        X = X[jnp.newaxis, ...]
        mu = coef_ * X
        mu = jnp.sum(mu, axis=-1) + intercept[..., 0]

        # assemble outcome random vars
        label = jnp.squeeze(y).astype(self.dtype)

        # Use Normal distribution for regression instead of Bernoulli
        sigma = params.get("sigma", jnp.array([self.noise_scale]))
        rv_outcome = tfd.Normal(loc=mu, scale=sigma)
        log_likelihood = rv_outcome.log_prob(label)

        return {
            "log_likelihood": log_likelihood,
            "prediction": rv_outcome,
            "mean": mu,
        }

    def log_likelihood(self, data, **params):
        return self.predictive_distribution(data, return_params=False, **params)[
            "log_likelihood"
        ]

    def unormalized_log_prob(self, data=None, **params):
        prediction = self.predictive_distribution(data, **params)
        log_likelihood = prediction["log_likelihood"]

        try:
            prior = self.prior_distribution.log_prob(
                {
                    "regression_model": {
                        k: params[k] for k in self.regression_var_list + ["sigma"]
                    },
                    "intercept_model": {
                        k: params[k] for k in self.intercept_var_list
                    },
                }
            )
        except ValueError:
            prior = 0

        return tf.reduce_sum(log_likelihood, axis=-1) + prior

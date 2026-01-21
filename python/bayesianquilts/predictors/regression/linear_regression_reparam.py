#!/usr/bin/env python3
"""Linear regression with reparameterized horseshoe prior (regression version of logistic_regression_reparam.py)"""
from collections import defaultdict

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.scipy.special import xlogy
from tensorflow_probability.substrates.jax import tf2jax as tf

from bayesianquilts.model import BayesianModel
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator

jax.config.update("jax_enable_x64", True)


class LinearRegression2(BayesianModel):
    def __init__(
        self,
        dim_regressors,
        scale_icept: float=1.0,
        scale_global=1.0,
        nu_global=1.0,
        nu_local=1.0,
        slab_scale=1.0,
        slab_df=1.0,
        noise_scale=1.0,
        dtype=jnp.float64,
    ):
        super(LinearRegression2, self).__init__(dtype=dtype)
        self.dim_regressors = dim_regressors
        self.scale_icept = scale_icept
        self.scale_global = scale_global
        self.nu_global = nu_global
        self.nu_local = nu_local
        self.slab_scale = slab_scale
        self.slab_df = slab_df
        self.noise_scale = noise_scale

        self.create_distributions()

    def preprocessor(self):
        return lambda x: x

    def create_distributions(self):
        # distribution on regression problem

        joint_prior_dict = {}
        joint_prior_dict["z"] = tfd.Independent(
            tfd.Normal(
                jnp.zeros((self.dim_regressors), dtype=self.dtype),
                jnp.ones((self.dim_regressors), dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=1,
        )
        joint_prior_dict["lambda"] = tfd.Independent(
            tfd.StudentT(
                self.nu_local * jnp.ones((self.dim_regressors), dtype=self.dtype),
                jnp.zeros((self.dim_regressors), dtype=self.dtype),
                jnp.ones((self.dim_regressors), dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=1,
        )
        joint_prior_dict["tau"] = tfd.Independent(
            tfb.AbsoluteValue()(
                tfd.StudentT(
                    self.nu_global * jnp.ones((1), dtype=self.dtype),
                    jnp.zeros((1), dtype=self.dtype),
                    self.scale_global * jnp.ones((1), dtype=self.dtype),
                )
            ),
            reinterpreted_batch_ndims=1,
        )
        joint_prior_dict["caux"] = tfd.Independent(
            tfd.InverseGamma(
                0.5 * self.slab_df * jnp.ones((1), dtype=self.dtype),
                0.5 * self.slab_df * jnp.ones((1), dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=1,
        )
        joint_prior_dict["beta0"] = tfd.Independent(
            tfd.Normal(
                jnp.zeros((1), dtype=self.dtype),
                self.scale_icept * jnp.ones((1), dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=1,
        )
        # Add noise parameter for regression
        joint_prior_dict["sigma"] = tfd.Independent(
            tfd.InverseGamma(
                concentration=2.0 * jnp.ones([1], self.dtype),
                scale=self.noise_scale * jnp.ones([1], self.dtype),
            ),
            reinterpreted_batch_ndims=1,
        )

        bijectors = defaultdict(lambda: tfb.Identity())
        bijectors["caux"] = tfb.Softplus()
        bijectors["tau"] = tfb.Softplus()
        bijectors["sigma"] = tfb.Softplus()

        self.prior_distribution = tfd.JointDistributionNamed(joint_prior_dict)
        self.surrogate_distribution_generator, self.surrogate_parameter_initializer = (
            build_factored_surrogate_posterior_generator(
                self.prior_distribution,
                bijectors=bijectors,
                dtype=self.dtype,
            )
        )
        self.params = self.surrogate_parameter_initializer()
        return None

    def transform(self, params):
        c = self.slab_scale * jnp.sqrt(params["caux"])
        lambda_tilde = jnp.sqrt(
            c**2
            * params["lambda"] ** 2
            / (c**2 + params["tau"] ** 2 * params["lambda"] ** 2)
        )
        beta = params["z"] * lambda_tilde * params["tau"]
        params["beta"] = beta
        return params

    def predictive_distribution(self, data, **params):

        processed = (self.preprocessor())(data)
        c = self.slab_scale * jnp.sqrt(params["caux"])
        lambda_tilde = jnp.sqrt(
            c**2
            * params["lambda"] ** 2
            / (c**2 + params["tau"] ** 2 * params["lambda"] ** 2)
        )
        beta = params["z"] * lambda_tilde * params["tau"]

        # compute regression product
        X = processed["X"].astype(self.dtype)
        mu = beta[..., jnp.newaxis, :] * X
        mu = jnp.sum(mu, -1) + params["beta0"]

        # assemble outcome random vars
        label = jnp.squeeze(data["y"]).astype(self.dtype)

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
        max_val = jnp.max(log_likelihood)

        finite_portion = jnp.where(
            jnp.isfinite(log_likelihood),
            log_likelihood,
            jnp.zeros_like(log_likelihood),
        )
        min_val = tf.reduce_min(finite_portion) - 10.0
        log_likelihood = jnp.where(
            jnp.isfinite(log_likelihood),
            log_likelihood,
            jnp.ones_like(log_likelihood) * min_val,
        )

        prior = self.prior_distribution.log_prob(params)

        return jnp.sum(log_likelihood, axis=-1) + prior

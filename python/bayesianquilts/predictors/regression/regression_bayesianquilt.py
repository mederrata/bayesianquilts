#!/usr/bin/env python3
"""Bayesian quilt model for regression

Regression version of LogisticBayesianquilt using Gaussian likelihood.
"""

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd

from bayesianquilts.jax.parameter import Decomposed
from bayesianquilts.model import BayesianModel
from bayesianquilts.util import flatten
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator

jax.config.update("jax_enable_x64", True)

class RegressionBayesianquilt(BayesianModel):
    def __init__(
        self,
        split_repr_model,
        dim_regressors,
        regression_interact,
        intercept_interact,
        split_quantiles=2,
        random_intercept_interact=None,
        dim_decay_factor=0.9,
        strategy=None,
        regressor_scales=None,
        regressor_offsets=None,
        dtype=jnp.float64,
        outcome_label="y",
        initialize_distributions=True,
        regression_shrinkage_scale=4e-2,
        noise_scale=1.0,
    ):
        super(RegressionBayesianquilt, self).__init__(dtype=dtype, strategy=strategy)
        self.split_quantiles = split_quantiles
        self.split_repr_model = split_repr_model

        self.split_repr_dim = split_repr_model.latent_dim

        self.dim_decay_factor = dim_decay_factor
        self.dim_regressors = dim_regressors
        self.random_intercept_interact = random_intercept_interact
        self.regression_shrinkage_scale = regression_shrinkage_scale
        self.outcome_label = outcome_label
        self.noise_scale = noise_scale
        if regressor_scales is None:
            self.regressor_scales = 1
        else:
            self.regressor_scales = regressor_scales
        self.regressor_offsets = (
            regressor_offsets if regressor_offsets is not None else 0
        )
        self.regression_interact = regression_interact
        self.intercept_interact = intercept_interact

        # For regression, output dimension is 1 (not outcome_classes - 1)
        self.regression_decomposition = Decomposed(
            interactions=self.regression_interact,
            param_shape=list(flatten([self.dim_regressors])) + [1],
            name="beta",
            dtype=self.dtype,
        )

        self.intercept_decomposition = Decomposed(
            interactions=self.intercept_interact,
            param_shape=[1],
            name="intercept",
            dtype=self.dtype,
        )
        if self.random_intercept_interact is not None:
            self.random_intercept_decomposition = Decomposed(
                interactions=self.random_intercept_interact,
                param_shape=[1],
                name="random_intercept",
                dtype=self.dtype,
            )
        else:
            self.random_intercept_decomposition = None

        if initialize_distributions:
            self.create_distributions()

    def preprocessor(self):
        def _preprocess(data):
            if "_preprocessed" in data.keys():
                return data
            x = data["covariates"]

            split_repr = self.split_repr_model.encode(x=x)
            split_breaks = self.split_repr_model.quantile_breaks[self.split_quantiles]
            split_indices = [
                jnp.digitize(self.split_repr_model[:, j], list(split_breaks[:, j]))[..., jnp.newaxis]
                for j in range(self.split_repr_dim)
            ]
            split_indices = {f"hx_{j}": h for j, h in enumerate(split_indices)}

            data["split_repr"] = split_repr

            return {**data, **split_indices, "_preprocessed": 1}

        return _preprocess

    def create_distributions(self):

        # distribution on regression problem

        (
            regressor_tensors,
            regression_vars,
            regression_shapes,
        ) = self.regression_decomposition.generate_tensors(dtype=self.dtype)
        regression_scales = {
            k: 5 * self.dim_decay_factor ** (len([d for d in v if d > 1]) - 1)
            for k, v in regression_shapes.items()
        }
        self.regression_vars = regression_vars
        self.regression_var_list = list(regression_vars.keys())

        regression_dict = {}
        for label, tensor in regressor_tensors.items():
            regression_dict[label] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros_like(jnp.asarray(tensor, self.dtype)),
                    scale=regression_scales[label]
                    * jnp.ones_like(jnp.asarray(tensor, self.dtype)),
                ),
                reinterpreted_batch_ndims=len(tensor.shape),
            )

        # Overall horseshoe

        regression_dict = {
            **regression_dict,
            "c2": tfd.Independent(
                tfd.InverseGamma(
                    jnp.ones(
                        [
                            np.prod(self.regression_decomposition._interaction_shape),
                            1,
                            1,
                        ],
                        dtype=self.dtype,
                    ),
                    jnp.ones(
                        [
                            np.prod(self.regression_decomposition._interaction_shape),
                            1,
                            1,
                        ],
                        dtype=self.dtype,
                    ),
                ),
                reinterpreted_batch_ndims=3,
            ),
            "tau": tfd.Independent(
                tfd.HalfStudentT(
                    df=2,
                    loc=0,
                    scale=self.regression_shrinkage_scale
                    * jnp.ones(
                        [
                            np.prod(self.regression_decomposition._interaction_shape),
                            1,
                            1,
                        ],
                        dtype=self.dtype,
                    ),
                ),
                reinterpreted_batch_ndims=3,
            ),
            "lambda_j": tfd.Independent(
                tfd.HalfStudentT(
                    df=5,
                    loc=0,
                    scale=jnp.ones(
                        [np.prod(self.regression_decomposition._interaction_shape)]
                        + list(self.regression_decomposition.shape())[-2:],
                        dtype=self.dtype,
                    ),
                ),
                reinterpreted_batch_ndims=3,
            ),
        }

        regressor_tensors["tau"] = (
            2
            * self.regression_shrinkage_scale
            * np.sqrt(2 / np.pi)
            * self.regression_shrinkage_scale
            * jnp.ones(
                [
                    np.prod(self.regression_decomposition._interaction_shape),
                    1,
                    1,
                ],
                dtype=self.dtype,
            )
        )

        regressor_tensors["c2"] = jnp.ones(
            [
                np.prod(self.regression_decomposition._interaction_shape),
                1,
                1,
            ],
            dtype=self.dtype,
        )

        regressor_tensors["lambda_j"] = jnp.ones(
            [np.prod(self.regression_decomposition._interaction_shape)]
            + list(self.regression_decomposition.shape())[-2:],
            dtype=self.dtype,
        )

        regression_model = tfd.JointDistributionNamed(regression_dict)
        bijectors = defaultdict(tfp.bijectors.Identity)
        bijectors["tau"] = tfp.bijectors.Softplus()
        bijectors["c2"] = tfp.bijectors.Softplus()
        bijectors["lambda_j"] = tfp.bijectors.Softplus()

        regression_surrogate_generator, regression_surrogate_param_init = build_factored_surrogate_posterior_generator(
            regression_model, bijectors=bijectors
        )

        # Intercept params
        (
            intercept_tensors,
            intercept_vars,
            intercept_shapes,
        ) = self.intercept_decomposition.generate_tensors(dtype=self.dtype)
        intercept_scales = {
            k: 20 * self.dim_decay_factor ** (len([d for d in v if d > 1]) - 1)
            for k, v in intercept_shapes.items()
        }
        self.intercept_vars = intercept_vars
        intercept_dict = {}
        for label, tensor in intercept_tensors.items():
            intercept_dict[label] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros_like(jnp.asarray(tensor, self.dtype)),
                    scale=intercept_scales[label]
                    * jnp.ones_like(jnp.asarray(tensor, self.dtype)),
                ),
                reinterpreted_batch_ndims=len(tensor.shape),
            )

        intercept_prior = tfd.JointDistributionNamed(intercept_dict)
        intercept_surrogate_gen, intercept_param_init = build_factored_surrogate_posterior_generator(
            intercept_prior
        )

        # Noise scale prior
        noise_dict = {
            "noise_scale": tfd.Independent(
                tfd.HalfNormal(scale=self.noise_scale * jnp.ones([1], dtype=self.dtype)),
                reinterpreted_batch_ndims=1,
            )
        }
        noise_prior = tfd.JointDistributionNamed(noise_dict)
        noise_bijectors = {"noise_scale": tfp.bijectors.Softplus()}
        noise_surrogate_gen, noise_param_init = build_factored_surrogate_posterior_generator(
            noise_prior, bijectors=noise_bijectors
        )

        if self.random_intercept_interact is not None:
            (
                random_intercept_tensors,
                random_intercept_vars,
                random_intercept_shapes,
            ) = self.random_intercept_decomposition.generate_tensors(dtype=self.dtype)
            self.random_intercept_vars = random_intercept_vars
            random_intercept_dict = {}
            for label, tensor in random_intercept_tensors.items():
                random_intercept_dict[label] = tfd.Independent(
                    tfd.Normal(
                        loc=jnp.zeros_like(jnp.asarray(tensor, self.dtype)),
                        scale=1e-1 * jnp.ones_like(jnp.asarray(tensor, self.dtype)),
                    ),
                    reinterpreted_batch_ndims=len(tensor.shape),
                )

            random_intercept_prior = tfd.JointDistributionNamed(random_intercept_dict)
            random_intercept_surrogate_gen, random_intercept_param_init = build_factored_surrogate_posterior_generator(
                random_intercept_prior
            )
            self.prior_distribution = tfd.JointDistributionNamed(
                {
                    "regression_model": regression_model,
                    "intercept_model": intercept_prior,
                    "random_intercept_model": random_intercept_prior,
                    "noise_model": noise_prior,
                }
            )
            self.surrogate_distribution_generator = lambda params: tfd.JointDistributionNamed(
                {
                    **regression_surrogate_generator(params).model,
                    **intercept_surrogate_gen(params).model,
                    **random_intercept_surrogate_gen(params).model,
                    **noise_surrogate_gen(params).model,
                }
            )
            self.surrogate_parameter_initializer = lambda: {
                **regression_surrogate_param_init(),
                **intercept_param_init(),
                **random_intercept_param_init(),
                **noise_param_init(),
            }
            self.random_intercept_var_list = list(random_intercept_vars.keys())
        else:
            self.prior_distribution = tfd.JointDistributionNamed(
                {
                    "regression_model": regression_model,
                    "intercept_model": intercept_prior,
                    "noise_model": noise_prior,
                }
            )
            self.surrogate_distribution_generator = lambda params: tfd.JointDistributionNamed(
                {
                    **regression_surrogate_generator(params).model,
                    **intercept_surrogate_gen(params).model,
                    **noise_surrogate_gen(params).model,
                }
            )
            self.surrogate_parameter_initializer = lambda: {
                **regression_surrogate_param_init(),
                **intercept_param_init(),
                **noise_param_init(),
            }

        self.regression_vars = regression_vars
        self.regression_var_list = list(regression_vars.keys())
        self.intercept_var_list = list(intercept_vars.keys())
        self.params = self.surrogate_parameter_initializer()
        self.var_list = list(self.params.keys())

    def predictive_distribution(self, data, **params):
        processed = data.copy()
        try:
            regression_params = params["regression_params"]
            intercept_params = params["intercept_params"]
            if self.random_intercept_decomposition is not None:
                random_intercept_params = params["random_intercept_params"]
        except KeyError:
            regression_params = {k: params[k] for k in self.regression_var_list}
            intercept_params = {k: params[k] for k in self.intercept_var_list}
            if self.random_intercept_decomposition is not None:
                random_intercept_params = {
                    k: params[k] for k in self.random_intercept_var_list
                }

        if "_preprocessed" not in data.keys():
            processed = (self.preprocessor())(processed)

        regression_indices = self.regression_decomposition.retrieve_indices(processed)

        # lookup_indices model coefficients

        intercept_indices = self.intercept_decomposition.retrieve_indices(processed)

        if self.random_intercept_decomposition is not None:
            random_intercept_indices = (
                self.random_intercept_decomposition.retrieve_indices(processed)
            )

        x = processed["X"]

        intercept = self.intercept_decomposition.lookup(
            intercept_indices, tensors=intercept_params
        )
        coef_ = self.regression_decomposition.lookup(
            regression_indices,
            tensors=regression_params,
        )

        x = x[jnp.newaxis, ..., jnp.newaxis]
        mu = jnp.asarray(coef_, x.dtype) * x
        mu = jnp.sum(mu, -2) + jnp.asarray(intercept[...], mu.dtype)
        mu = jnp.asarray(mu, self.dtype)

        if self.random_intercept_interact is not None:
            ranef = self.random_intercept_decomposition.lookup(
                random_intercept_indices, tensors=random_intercept_params
            )
            mu += jnp.asarray(ranef, mu.dtype)

        # Squeeze out the last dimension (was for num_classes)
        mu = jnp.squeeze(mu, axis=-1)

        # Get noise scale
        noise_scale = params.get("noise_scale", jnp.array([self.noise_scale], dtype=self.dtype))

        # assemble outcome random vars
        y = jnp.asarray(jnp.squeeze(processed[self.outcome_label]), self.dtype)
        rv_outcome = tfd.Normal(loc=mu, scale=noise_scale)
        log_likelihood = rv_outcome.log_prob(y)

        return {
            "log_likelihood": log_likelihood,
            "prediction": rv_outcome,
            "mean": mu,
        }

    def log_likelihood(self, data, **params):
        return self.predictive_distribution(data, return_params=False, **params)[
            "log_likelihood"
        ]

    def unormalized_log_prob(self, data=None, prior_weight=1.0, **params):
        prediction = self.predictive_distribution(data, **params)
        log_likelihood = prediction["log_likelihood"]
        max_val = jnp.max(log_likelihood)

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

        if self.random_intercept_interact is None:
            prior = self.prior_distribution.log_prob(
                {
                    "regression_model": {
                        k: jnp.asarray(params[k], self.dtype)
                        for k in self.regression_var_list
                    },
                    "intercept_model": {
                        k: jnp.asarray(params[k], self.dtype)
                        for k in self.intercept_var_list
                    },
                    "noise_model": {
                        "noise_scale": jnp.asarray(params["noise_scale"], self.dtype)
                    },
                }
            )
        else:
            prior = self.prior_distribution.log_prob(
                {
                    "regression_model": {
                        k: jnp.asarray(params[k], self.dtype)
                        for k in self.regression_var_list
                    },
                    "intercept_model": {
                        k: jnp.asarray(params[k], self.dtype)
                        for k in self.intercept_var_list
                    },
                    "random_intercept_model": {
                        k: jnp.asarray(params[k], self.dtype)
                        for k in self.random_intercept_var_list
                    },
                    "noise_model": {
                        "noise_scale": jnp.asarray(params["noise_scale"], self.dtype)
                    },
                }
            )

        def regression_effective(lambda_j, c2, tau):
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

        regression_coefs = self.regression_decomposition.sum_parts(params)
        regression_coefs = jnp.asarray(regression_coefs, self.dtype)
        regression_horseshoe = regression_effective(
            params["lambda_j"], params["c2"], params["tau"]
        )
        prior_weight = jnp.asarray(prior_weight, self.dtype)

        energy = (
            jnp.sum(log_likelihood, axis=-1)
            + prior_weight * prior
            + prior_weight
            * jnp.asarray(regression_horseshoe.log_prob(regression_coefs), self.dtype)
        )

        return energy

    def fit(
        self,
        batched_data_factory,
        batch_size,
        dataset_size,
        num_steps,
        warmup=25,
        warmup_max_order=6,
        clip_value=5,
        learning_rate=0.005,
        test_fn=None,
        *args,
        **kwargs,
    ):

        # train
        if warmup == 0:
            return self._calibrate_advi(num_steps=num_steps, *args, **kwargs)
        else:
            # train weibull scale first few epochs
            for max_order in range(warmup_max_order):

                # cut out higher order terms
                hot = [
                    k for k, v in self.regression_vars.items() if len(v) > max_order
                ] + [k for k, v in self.intercept_vars.items() if len(v) > max_order]
                if len(hot) == 0:
                    print("Done warming")
                    break

                print(f"Training up to {max_order} order")
                trainable_params = {k: v for k, v in self.params.items() if k not in hot}
                losses = self._calibrate_minibatch_advi(
                    batched_data_factory=batched_data_factory,
                    num_steps=warmup,
                    clip_value=clip_value,
                    batch_size=batch_size,
                    dataset_size=dataset_size,
                    trainable_variables=trainable_params,
                    learning_rate=learning_rate,
                    test_fn=test_fn,
                    **kwargs,
                )

            print(f"Training for remaining {num_steps} steps")
            losses = self._calibrate_minibatch_advi(
                batched_data_factory=batched_data_factory,
                num_steps=num_steps,
                clip_value=clip_value,
                batch_size=batch_size,
                dataset_size=dataset_size,
                trainable_variables=self.params,
                learning_rate=learning_rate,
                test_fn=test_fn,
                **kwargs,
            )
        return losses

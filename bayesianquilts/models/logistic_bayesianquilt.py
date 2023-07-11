#!/usr/bin/env python3
"""Example quilt model
"""
import argparse
import csv
import functools
import itertools
import json
import os
import sys
from collections import defaultdict
from itertools import product

import arviz as az
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow._api.v2 import data
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.math_ops import _bucketize as bucketize
from tensorflow_probability.python import distributions as tfd
from tqdm import tqdm

from bayesianquilts.metrics.classification import accuracy, auc
from bayesianquilts.model import BayesianModel
from bayesianquilts.tf.parameter import Decomposed, Interactions
from bayesianquilts.util import flatten, split_tensor
from bayesianquilts.vi.advi import build_surrogate_posterior


class LogisticBayesianquilt(BayesianModel):
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
        dtype=tf.float64,
        outcome_label="label",
        outcome_classes=2,
        initialize_distributions=True,
        regression_shrinkage_scale=4e-2,
    ):
        super(LogisticBayesianquilt, self).__init__(dtype=dtype, strategy=strategy)
        self.split_quantiles = split_quantiles
        self.split_repr_model = split_repr_model

        self.split_repr_dim = split_repr_model.latent_dim

        self.dim_decay_factor = dim_decay_factor
        self.dim_regressors = dim_regressors
        self.random_intercept_interact = random_intercept_interact
        self.regression_shrinkage_scale = regression_shrinkage_scale
        self.outcome_label = outcome_label
        self.outcome_classes = outcome_classes
        if regressor_scales is None:
            self.regressor_scales = 1
        else:
            self.regressor_scales = regressor_scales
        self.regressor_offsets = (
            regressor_offsets if regressor_offsets is not None else 0
        )
        self.regression_interact = regression_interact
        self.intercept_interact = intercept_interact

        self.regression_decomposition = Decomposed(
            interactions=self.regression_interact,
            param_shape=list(flatten([self.dim_regressors]))
            + [self.outcome_classes - 1],
            name="beta",
            dtype=self.dtype,
        )

        self.intercept_decomposition = Decomposed(
            interactions=self.intercept_interact,
            param_shape=[self.outcome_classes - 1],
            name="intercept",
            dtype=self.dtype,
        )
        if self.random_intercept_interact is not None:
            self.random_intercept_decomposition = Decomposed(
                interactions=self.random_intercept_interact,
                param_shape=[self.outcome_classes - 1],
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
            if isinstance(x, tf.RaggedTensor):
                x = x.to_tensor()

            split_repr = self.split_repr_model.encode(x=x)
            split_breaks = self.split_repr_model.quantile_breaks[self.split_quantiles]
            split_indices = [
                bucketize(self.split_repr_model[:, j], list(split_breaks[:, j]))[..., tf.newaxis]
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
        #  self.regression_decomposition.set_scales(regression_scales)

        regression_dict = {}
        for label, tensor in regressor_tensors.items():
            regression_dict[label] = tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros_like(tf.cast(tensor, self.dtype)),
                    scale=regression_scales[label]
                    * tf.ones_like(tf.cast(tensor, self.dtype)),
                ),
                reinterpreted_batch_ndims=len(tensor.shape.as_list()),
            )

        # Overall horseshoe

        regression_dict = {
            **regression_dict,
            "c2": tfd.Independent(
                tfd.InverseGamma(
                    tf.ones(
                        [
                            np.prod(self.regression_decomposition._interaction_shape),
                            1,
                            self.outcome_classes - 1,
                        ],
                        dtype=self.dtype,
                    ),
                    tf.ones(
                        [
                            np.prod(self.regression_decomposition._interaction_shape),
                            1,
                            self.outcome_classes - 1,
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
                    * tf.ones(
                        [
                            np.prod(self.regression_decomposition._interaction_shape),
                            1,
                            self.outcome_classes - 1,
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
                    scale=tf.ones(
                        [np.prod(self.regression_decomposition._interaction_shape)]
                        + self.regression_decomposition.shape()[-2:],
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
            * tf.ones(
                [
                    np.prod(self.regression_decomposition._interaction_shape),
                    1,
                    self.outcome_classes - 1,
                ],
                dtype=self.dtype,
            )
        )

        regressor_tensors["c2"] = tf.ones(
            [
                np.prod(self.regression_decomposition._interaction_shape),
                1,
                self.outcome_classes - 1,
            ],
            dtype=self.dtype,
        )

        regressor_tensors["lambda_j"] = tf.ones(
            [np.prod(self.regression_decomposition._interaction_shape)]
            + self.regression_decomposition.shape()[-2:],
            dtype=self.dtype,
        )

        regression_model = tfd.JointDistributionNamed(regression_dict)
        bijectors = defaultdict(tfp.bijectors.Identity)
        bijectors["tau"] = tfp.bijectors.Softplus()
        bijectors["c2"] = tfp.bijectors.Softplus()
        bijectors["lambda_j"] = tfp.bijectors.Softplus()

        regression_surrogate = build_surrogate_posterior(
            regression_model,
            initializers=regressor_tensors,
            bijectors=bijectors,
            gaussian_only=True,
        )

        #  Exponential params
        (
            intercept_tensors,
            intercept_vars,
            intercept_shapes,
        ) = self.intercept_decomposition.generate_tensors(dtype=self.dtype)
        intercept_scales = {
            k: 20 * self.dim_decay_factor ** (len([d for d in v if d > 1]) - 1)
            for k, v in intercept_shapes.items()
        }
        #  self.intercept_decomposition.set_scales(intercept_scales)
        self.intercept_vars = intercept_vars
        intercept_dict = {}
        for label, tensor in intercept_tensors.items():
            intercept_dict[label] = tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros_like(tf.cast(tensor, self.dtype)),
                    scale=intercept_scales[label]
                    * tf.ones_like(tf.cast(tensor, self.dtype)),
                ),
                reinterpreted_batch_ndims=len(tensor.shape.as_list()),
            )

        intercept_prior = tfd.JointDistributionNamed(intercept_dict)
        intercept_surrogate = build_surrogate_posterior(
            intercept_prior, initializers=intercept_tensors
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
                        loc=tf.zeros_like(tf.cast(tensor, self.dtype)),
                        scale=1e-1 * tf.ones_like(tf.cast(tensor, self.dtype)),
                    ),
                    reinterpreted_batch_ndims=len(tensor.shape.as_list()),
                )

            random_intercept_prior = tfd.JointDistributionNamed(random_intercept_dict)
            random_intercept_surrogate = build_surrogate_posterior(
                random_intercept_prior, initializers=random_intercept_tensors
            )
            self.prior_distribution = tfd.JointDistributionNamed(
                {
                    "regression_model": regression_model,
                    "intercept_model": intercept_prior,
                    "random_intercept_model": random_intercept_prior,
                }
            )
            self.surrogate_distribution = tfd.JointDistributionNamed(
                {
                    **regression_surrogate.model,
                    **intercept_surrogate.model,
                    **random_intercept_surrogate.model,
                }
            )
            self.random_intercept_var_list = list(
                random_intercept_surrogate.model.keys()
            )
        else:
            self.prior_distribution = tfd.JointDistributionNamed(
                {
                    "regression_model": regression_model,
                    "intercept_model": intercept_prior,
                }
            )
            self.surrogate_distribution = tfd.JointDistributionNamed(
                {**regression_surrogate.model, **intercept_surrogate.model}
            )
        self.regression_vars = regression_vars
        self.regression_var_list = list(regression_surrogate.model.keys())
        self.intercept_var_list = list(intercept_surrogate.model.keys())

        self.surrogate_vars = self.surrogate_distribution.variables
        self.var_list = list(self.surrogate_distribution.model.keys())
        return None

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
        if isinstance(regression_indices, tf.RaggedTensor):
            regression_indices = regression_indices.to_tensor()

        # lookup_indices model coefficients

        intercept_indices = self.intercept_decomposition.retrieve_indices(processed)
        if isinstance(intercept_indices, tf.RaggedTensor):
            intercept_indices = intercept_indices.to_tensor()

        if self.random_intercept_decomposition is not None:
            random_intercept_indices = (
                self.random_intercept_decomposition.retrieve_indices(processed)
            )
            if isinstance(random_intercept_indices, tf.RaggedTensor):
                random_intercept_indices = random_intercept_indices.to_tensor()

        x = processed["X"]

        intercept = self.intercept_decomposition.lookup(
            intercept_indices, tensors=intercept_params
        )
        coef_ = self.regression_decomposition.lookup(
            regression_indices,
            tensors=regression_params,
        )

        x = x[tf.newaxis, ..., tf.newaxis]
        mu = tf.cast(coef_, x.dtype) * x
        mu = tf.reduce_sum(mu, -2) + tf.cast(intercept[...], mu.dtype)
        mu = tf.cast(mu, self.dtype)

        if self.random_intercept_interact is not None:
            ranef = self.random_intercept_decomposition.lookup(
                random_intercept_indices, tensors=random_intercept_params
            )
            mu += tf.cast(ranef, mu.dtype)

        # assemble outcome random vars

        label = tf.cast(tf.squeeze(processed[self.outcome_label]), self.dtype)
        mu = tf.pad(mu, [(0, 0)] * (len(mu.shape) - 1) + [(1, 0)])
        rv_outcome = tfd.Categorical(logits=mu)
        log_likelihood = rv_outcome.log_prob(label)

        return {
            "log_likelihood": log_likelihood,
            "prediction": rv_outcome,
            "logits": mu,
        }

    def log_likelihood(self, data, **params):
        return self.predictive_distribution(data, return_params=False, **params)[
            "log_likelihood"
        ]

    def unormalized_log_prob(self, data=None, prior_weight=tf.constant(1.0), **params):
        prediction = self.predictive_distribution(data, **params)
        log_likelihood = prediction["log_likelihood"]
        max_val = tf.reduce_max(log_likelihood)

        finite_portion = tf.where(
            tf.math.is_finite(log_likelihood),
            log_likelihood,
            tf.zeros_like(log_likelihood),
        )
        min_val = tf.reduce_min(finite_portion) - 1.0
        log_likelihood = tf.where(
            tf.math.is_finite(log_likelihood),
            log_likelihood,
            tf.ones_like(log_likelihood) * min_val,
        )
        if self.random_intercept_interact is None:
            prior = self.prior_distribution.log_prob(
                {
                    "regression_model": {
                        k: tf.cast(params[k], self.dtype)
                        for k in self.regression_var_list
                    },
                    "intercept_model": {
                        k: tf.cast(params[k], self.dtype)
                        for k in self.intercept_var_list
                    },
                }
            )
        else:
            prior = self.prior_distribution.log_prob(
                {
                    "regression_model": {
                        k: tf.cast(params[k], self.dtype)
                        for k in self.regression_var_list
                    },
                    "intercept_model": {
                        k: tf.cast(params[k], self.dtype)
                        for k in self.intercept_var_list
                    },
                    "random_intercept_model": {
                        k: tf.cast(params[k], self.dtype)
                        for k in self.random_intercept_var_list
                    },
                }
            )

        def regression_effective(lambda_j, c2, tau):
            return tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros_like(lambda_j),
                    scale=(
                        tau
                        * lambda_j
                        * tf.math.sqrt(c2 / (c2 + tau**2 * lambda_j**2))
                    ),
                ),
                reinterpreted_batch_ndims=3,
            )

        regression_coefs = self.regression_decomposition.sum_parts(params)
        regression_coefs = tf.cast(regression_coefs, self.dtype)
        regression_horseshoe = regression_effective(
            params["lambda_j"], params["c2"], params["tau"]
        )
        prior_weight = tf.cast(prior_weight, self.dtype)

        energy = (
            tf.reduce_sum(log_likelihood, axis=-1)
            + prior_weight * prior
            + prior_weight
            * tf.cast(regression_horseshoe.log_prob(regression_coefs), self.dtype)
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
                trainable_variables = [
                    t
                    for t in self.surrogate_distribution.variables
                    if t._shared_name.split("/")[0] not in hot
                ]

                """

                batch = next(iter(data_factory()))
                test = self.predictive_distribution(data=batch, **self.sample(1))
                n_infinite = tf.reduce_sum(
                    tf.cast(tf.math.is_inf(test["log_likelihood"]), tf.int32), axis=0
                )
                variational_loss_fn = functools.partial(
                    csiszar_divergence.monte_carlo_variational_loss,
                    discrepancy_fn=tfp.vi.kl_reverse,
                    importance_sample_size=4,
                    # Silent fallback to score-function gradients leads to
                    # difficult-to-debug failures, so force reparameterization gradients by
                    # default.
                    gradient_estimator=(
                        csiszar_divergence.GradientEstimators.REPARAMETERIZATION
                    ),
                )

                def complete_variational_loss_fn(seed=None):
                    return variational_loss_fn(
                        functools.partial(self.unormalized_log_prob, data=batch),
                        self.surrogate_distribution,
                        sample_size=1,
                        seed=seed,
                    )
                
                with tf.GradientTape(
                    watch_accessed_variables=trainable_variables is None
                ) as tape:
                    for v in trainable_variables or []:
                        tape.watch(v)
                    loss = complete_variational_loss_fn()

                    grad = tape.gradient(loss, trainable_variables)

                with tf.GradientTape(
                    watch_accessed_variables=trainable_variables is None
                ) as tape:
                    for v in trainable_variables or []:
                        tape.watch(v)
                    test_energy = self.unormalized_log_prob(data=batch, **self.sample(1))
                    grad_e = tape.gradient(test_energy, trainable_variables)
                """
                print(f"Training up to {max_order} order")
                losses = self._calibrate_minibatch_advi(
                    batched_data_factory=batched_data_factory,
                    num_steps=warmup,
                    clip_value=clip_value,
                    batch_size=batch_size,
                    dataset_size=dataset_size,
                    trainable_variables=trainable_variables,
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
                trainable_variables=self.surrogate_distribution.variables,
                learning_rate=learning_rate,
                test_fn=test_fn,
                **kwargs,
            )
        return losses

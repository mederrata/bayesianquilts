#!/usr/bin/env python3
"""Example quilt model
"""
import argparse
import sys
import os
import csv
import itertools
from itertools import product
import arviz as az
from tqdm import tqdm
import json

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

from bayesianquilts.model import BayesianModel
from mederrata_spmf import PoissonFactorization
from bayesianquilts.tf.parameter import Interactions, Decomposed
from bayesianquilts.vi.advi import build_surrogate_posterior


tfd = tfp.distributions


def psis_smoothing(weights, threshold=0.7):
    log_weights = tf.math.log(weights)
    psis_weights = []

    for log_weight in log_weights:
        sorted_indices = tf.argsort(log_weight, axis=-1, direction="DESCENDING")
        sorted_log_weight = tf.gather(log_weight, sorted_indices, axis=-1)
        cumsum_log_weight = tf.cumsum(sorted_log_weight, axis=-1)
        threshold_value = (1.0 - threshold) * tf.reduce_max(cumsum_log_weight, axis=-1, keepdims=True)
        psis_weight = tf.exp(tf.math.minimum(sorted_log_weight - threshold_value, 0.0))
        original_order_indices = tf.argsort(sorted_indices, axis=-1)
        psis_weight = tf.gather(psis_weight, original_order_indices, axis=-1)
        psis_weights.append(psis_weight)

    return tf.stack(psis_weights)


class LogisticRegression(BayesianModel):
    def __init__(
        self,
        dim_regressors,
        regression_interact=None,
        dim_decay_factor=0.5,
        regressor_scales=None,
        regressor_offsets=None,
        dtype=tf.float64,
    ):
        super(LogisticRegression, self).__init__(dtype=dtype)
        self.dim_decay_factor = dim_decay_factor
        self.dim_regressors = dim_regressors
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
        self.regression_decomposition.set_scales(regression_scales)

        regression_dict = {}
        for label, tensor in regressor_tensors.items():
            regression_dict[label] = tfd.Independent(
                tfd.Horseshoe(scale=tf.ones_like(tf.cast(tensor, self.dtype))),
                reinterpreted_batch_ndims=len(tensor.shape.as_list()),
            )

        regression_model = tfd.JointDistributionNamed(regression_dict)
        regression_surrogate = build_surrogate_posterior(
            regression_model, initializers=regressor_tensors
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

        intercept_dict = {}
        for label, tensor in intercept_tensors.items():
            intercept_dict[label] = tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros_like(tf.cast(tensor, self.dtype)),
                    scale=tf.ones_like(tf.cast(tensor, self.dtype)),
                ),
                reinterpreted_batch_ndims=len(tensor.shape.as_list()),
            )

        intercept_prior = tfd.JointDistributionNamed(intercept_dict)
        intercept_surrogate = build_surrogate_posterior(
            intercept_prior, initializers=intercept_tensors
        )

        self.prior_distribution = tfd.JointDistributionNamed(
            {"regression_model": regression_model, "intercept_model": intercept_prior}
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
        try:
            regression_params = params["regression_params"]
            intercept_params = params["intercept_params"]
        except KeyError:
            regression_params = {k: params[k] for k in self.regression_var_list}
            intercept_params = {k: params[k] for k in self.intercept_var_list}
            
        processed = (self.preprocessor())(data)

        regression_indices = self.regression_decomposition.retrieve_indices(processed)
        if isinstance(regression_indices, tf.RaggedTensor):
            regression_indices = regression_indices.to_tensor()
            
        intercept_indices = self.intercept_decomposition.retrieve_indices(processed)
        if isinstance(intercept_indices, tf.RaggedTensor):
            intercept_indices = intercept_indices.to_tensor()

        coef_ = self.regression_decomposition.lookup(
            regression_indices,
            tensors=regression_params,
        )

        intercept = self.intercept_decomposition.lookup(
            intercept_indices, tensors=intercept_params
        )

        # compute regression product
        X = tf.cast(
            (data["X"] - self.regressor_offsets) / self.regressor_scales,
            self.dtype,
        )

        X = X[tf.newaxis, ...]
        mu = coef_ * X
        mu = tf.reduce_sum(mu, -1) + intercept[..., 0]

        # assemble outcome random vars

        label = tf.cast(tf.squeeze(data["y"]), self.dtype)

        rv_outcome = tfd.Bernoulli(logits=mu)
        log_likelihood = rv_outcome.log_prob(label)

        # add on the breakpoint model for hx

        return {
            "log_likelihood": log_likelihood,
            "prediction": rv_outcome,
            "logits": mu,
        }

    def log_likelihood(self, data, **params):
        return self.predictive_distribution(data, return_params=False, **params)[
            "log_likelihood"
        ]

    def unormalized_log_prob(self, data=None, **params):
        prediction = self.predictive_distribution(data, **params)
        log_likelihood = prediction["log_likelihood"]
        max_val = tf.reduce_max(log_likelihood)

        finite_portion = tf.where(
            tf.math.is_finite(log_likelihood),
            log_likelihood,
            tf.zeros_like(log_likelihood),
        )
        min_val = tf.reduce_min(finite_portion) - 10.0
        log_likelihood = tf.where(
            tf.math.is_finite(log_likelihood),
            log_likelihood,
            tf.ones_like(log_likelihood) * min_val,
        )

        prior = self.prior_distribution.log_prob(
            {
                "regression_model": {
                    k: tf.cast(params[k], self.dtype) for k in self.regression_var_list
                },
                "intercept_model": {
                    k: tf.cast(params[k], self.dtype) for k in self.intercept_var_list
                },
            }
        )

        return tf.reduce_sum(log_likelihood, axis=-1) + prior

    def stepaway_loo(self, data, params, step_size):
        """Compute step-away transformation for LOO
        
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        
        y = tf.cast(tf.squeeze(data['y']), tf.float64)
        X = tf.cast(data['X'], tf.float64)
        mu = tf.reduce_sum(params['beta__']*data['X'], axis=-1) + params['intercept__'][..., 0]
        sigma = tf.math.sigmoid(mu)
        nu = (y*(sigma)+(1-y)*(1-sigma))**-1
        nu_weights = nu/tf.reduce_sum(nu, axis=0, keepdims=True)
        
        grad_pre = tf.cast(y, tf.float64)*tf.cast(1-sigma, tf.float64)-tf.cast(1-y, tf.float64)*tf.cast(sigma, tf.float64)
        grad_beta = grad_pre[..., tf.newaxis] * X
        grad_intercept = grad_pre[..., tf.newaxis]
        
        beta_new = params['beta__'] - step_size*grad_beta
        intercept_new = params['intercept__'] - step_size*grad_intercept
        mu_new = tf.reduce_sum(beta_new*data['X'], axis=-1) + intercept_new[..., 0]
        sigma_new = tf.math.sigmoid(mu_new)
        nu_new = (y*(sigma_new)+(1-y)*(1-sigma_new))**-1
        J = 1 + step_size*(1 + tf.math.reduce_sum(X*X, -1))*sigma*(1-sigma)

        log_beta_new = self.surrogate_distribution.model['beta__'].log_prob(beta_new[..., tf.newaxis, :])
        log_beta = self.surrogate_distribution.model['beta__'].log_prob(params['beta__'])
        log_intercept_new = self.surrogate_distribution.model['intercept__'].log_prob(intercept_new[..., tf.newaxis, :])
        log_intercept= self.surrogate_distribution.model['intercept__'].log_prob(params['intercept__'])
        eta = nu_new*J*tf.math.exp(log_beta_new + log_intercept_new - log_beta[:, tf.newaxis] - log_intercept[:, tf.newaxis])
        eta_weights = eta/tf.reduce_sum(eta, axis=0, keepdims=True)
        
        p_loo_stepaway = tf.reduce_sum(sigma_new * eta_weights, axis=0)
        p_loo = tf.reduce_sum(sigma*nu_weights, axis=0)
        ll_loo = tf.reduce_sum(nu_weights/nu, axis=0)
        ll_loo_stepaway = tf.reduce_sum(eta_weights/nu_new, axis=0)

        return {
            'eta_weights': eta_weights, 'p': tf.reduce_mean(sigma, axis=0), 'nu_weights': nu_weights, 'p_loo': p_loo, 'p_loo_stepaway': p_loo_stepaway, 'll_loo_stepaway': ll_loo_stepaway, 'll_loo': ll_loo}

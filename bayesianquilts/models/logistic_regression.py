#!/usr/bin/env python3
"""Example quilt model
"""

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import tf2jax as tf

from bayesianquilts.model import BayesianModel
from bayesianquilts.tf.parameter import Decomposed, Interactions
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator

tfd = tfp.distributions


def psis_smoothing(weights, threshold=0.7):
    log_weights = tf.math.log(weights)
    psis_weights = []

    for log_weight in log_weights:
        sorted_indices = tf.argsort(log_weight, axis=-1, direction="DESCENDING")
        sorted_log_weight = tf.gather(log_weight, sorted_indices, axis=-1)
        cumsum_log_weight = tf.cumsum(sorted_log_weight, axis=-1)
        threshold_value = (1.0 - threshold) * tf.reduce_max(
            cumsum_log_weight, axis=-1, keepdims=True
        )
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
        global_horseshoe_scale=None,
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
        self.regression_decomposition.set_scales(regression_scales)

        regression_dict = {}

        regression_dict["beta__"] = lambda global_scale: tfd.Independent(
            tfd.Horseshoe(scale=global_scale),
            reinterpreted_batch_ndims=len(regressor_tensors["beta__"].shape.as_list()),
        )

        regression_dict["global_scale"] = lambda global_scale_aux: tfd.Independent(
            tfd.InverseGamma(
                concentration=0.5
                * jnp.ones(
                    [1] * len(regressor_tensors["beta__"].shape.as_list()), self.dtype
                ),
                scale=1 / global_scale_aux,
            ),
            reinterpreted_batch_ndims=len(regressor_tensors["beta__"].shape.as_list()),
        )
        regression_dict["global_scale_aux"] = tfd.Independent(
            tfd.InverseGamma(
                concentration=0.5
                * jnp.ones(
                    [1] * len(regressor_tensors["beta__"].shape.as_list()), self.dtype
                ),
                scale=jnp.ones(
                    [1] * len(regressor_tensors["beta__"].shape.as_list()), self.dtype
                )
                / self.global_horseshoe_scale**2,
            ),
            reinterpreted_batch_ndims=len(regressor_tensors["beta__"].shape.as_list()),
        )

        regression_model = tfd.JointDistributionNamed(regression_dict)
        regression_surrogate_generator, regression_surrogate_param_init = build_factored_surrogate_posterior_generator(
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
                    loc=jnp.zeros_like(tf.cast(tensor, self.dtype)),
                    scale=jnp.ones_like(tf.cast(tensor, self.dtype)),
                ),
                reinterpreted_batch_ndims=len(tensor.shape.as_list()),
            )

        intercept_prior = tfd.JointDistributionNamed(intercept_dict)
        intercept_surrogate_gen, intercept_param_init = build_factored_surrogate_posterior_generator(
            intercept_prior, initializers=intercept_tensors
        )

        self.prior_distribution = tfd.JointDistributionNamed(
            {"regression_model": regression_model, "intercept_model": intercept_prior}
        )
        self.surrogate_distribution_gen = lambda params: tfd.JointDistributionNamed(
            {**regression_surrogate_generator(params).model, **intercept_surrogate_gen(params).model}
        )
        self.surrogate_parameter_initializer = lambda: {
            **regression_surrogate_param_init(),
            **intercept_param_init(),
        }

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
            jnp.zeros_like(log_likelihood),
        )
        min_val = tf.reduce_min(finite_portion) - 10.0
        log_likelihood = tf.where(
            tf.math.is_finite(log_likelihood),
            log_likelihood,
            jnp.ones_like(log_likelihood) * min_val,
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

    def entropy(self, probs):
        return -tf.math.xlogy(probs, probs)

    def adaptive_is_loo(self, data, params, hbar=1.0, variational=True):
        """_summary_

        Args:
            data (_type_): _description_
            params (_type_): _description_
            hbar (float, optional): _description_. Defaults to 1.0.
            variational (bool, optional):
                Should we trust the variational approximation?
                If False, assumes that one is passing in all the data at once in a single batch.
                Defaults to True.

        Returns:
            _type_: _description_
        """

        # scaled (theta - bar(theta))/Sigma
        beta = params["beta__"]
        intercept = params["intercept__"]
        X = tf.cast(data["X"], self.dtype)
        y = tf.squeeze(tf.cast(data["y"], self.dtype))
        mu = tf.reduce_sum(beta * X, axis=-1) + intercept[..., 0]
        sigma = tf.math.sigmoid(mu)
        ell = y * (sigma) + (1 - y) * (1 - sigma)
        log_ell = tf.math.xlogy(y, sigma) + tf.math.xlogy(1 - y, 1 - sigma)
        log_ell_prime = y * (1 - sigma) - (1 - y) * sigma
        log_ell_doubleprime = -sigma * (1 - sigma)

        """
        sigma.shape is samples x datapoints
        """

        # compute # \nabla\log\pi(\btheta|\calD)
        if variational:
            # \nabla\log\pi = -\Sigma^{-1}(theta - \bar{\theta})
            grad_log_pi = tf.concat(
                [
                    -(intercept - self.surrogate_distribution.model["intercept__"].mean())
                    / self.surrogate_distribution.model["intercept__"].variance(),
                    -(beta - self.surrogate_distribution.model["beta__"].mean())
                    / self.surrogate_distribution.model["beta__"].variance(),
                ],
                axis=-1,
            )
            intercept_sd = (
                self.surrogate_distribution.model["intercept__"].variance() ** 0.5
            )
            beta_sd = self.surrogate_distribution.model["beta__"].variance() ** 0.5

            log_pi = self.surrogate_distribution.model["beta__"].log_prob(
                params["beta__"]
            ) + self.surrogate_distribution.model["intercept__"].log_prob(
                params["intercept__"]
            )
            log_pi -= tf.reduce_max(log_pi, axis=0)
        else:
            """
            Recall Bayes rule:
            \log pi(\btheta|\calD) = \sum_i\log ell_i(\btheta) + \log\pi(\btheta) + const

            so
            \nabla\log\pi(\btheta|\calD) = \sum_i (ell_i)'x + grad\log\pi(\btheta)

            """
            log_pi = tf.reduce_sum(log_ell, axis=1, keepdims=True)[:, 0]
            log_pi += self.prior_distribution.log_prob(
                {
                    "regression_model": {
                        k: tf.cast(params[k], self.dtype) for k in self.regression_var_list
                    },
                    "intercept_model": {
                        k: tf.cast(params[k], self.dtype) for k in self.intercept_var_list
                    },
                }
            )

            # pi \propto
            # log_pi.shape: [samples]

            # log_ell.shape  [samples, data]
            # X.shape [data, features]
            log_ell_tot = tf.reduce_sum(log_ell[..., tf.newaxis], axis=1, keepdims=True)
            grad_log_pi = tf.concat(
                [
                    log_ell_tot,
                    tf.reduce_sum(log_ell[..., tf.newaxis] * X, axis=1, keepdims=True),
                ],
                axis=-1,
            )
            # TODO NEED PRIOR TERM

            # grad_log_pi.shape [samples, 1, parameters]

            prior_intercept_sd = (
                self.prior_distribution.model["intercept_model"]
                .model["intercept__"]
                .variance()
                ** 0.5
            )
            prior_beta_sd = params["global_scale"]

            intercept_sd = tf.math.reduce_std(intercept, 0, keepdims=True)
            beta_sd = tf.math.reduce_std(beta, 0, keepdims=True)

        # log-likelihood descent

        def T_ll():
            Q_beta = -log_ell_prime[..., tf.newaxis] * X
            Q_intercept = -log_ell_prime[..., tf.newaxis]

            standardized = tf.concat(
                [Q_beta / beta_sd, Q_intercept / intercept_sd], axis=-1
            )
            standardized = tf.reduce_max(tf.math.abs(standardized), axis=-1)
            standardized = tf.reduce_max(standardized, axis=0, keepdims=True)[
                ..., tf.newaxis
            ]

            h = hbar / standardized
            logJ = tf.math.log1p(
                tf.math.abs(
                    h
                    * (1 + tf.math.reduce_sum(X**2, -1, keepdims=True))[tf.newaxis, :, :]
                    * (sigma * (1 - sigma))[..., tf.newaxis]
                )[..., 0]
            )
            beta_ll = beta + h * Q_beta
            intercept_ll = intercept + h * Q_intercept
            return beta_ll, intercept_ll, logJ

        def T_kl():
            log_pi_ = log_pi - tf.reduce_max(log_pi, axis=0)

            Q_beta = ((-1) ** y * tf.math.exp(log_pi_[..., tf.newaxis] + mu * (1 - 2 * y)))[
                ..., tf.newaxis
            ] * data["X"]
            Q_intercept = (
                (-1) ** y * tf.math.exp(log_pi_[..., tf.newaxis] + mu * (1 - 2 * y))
            )[..., tf.newaxis]

            # log_pi.shape: [samples]
            # mu.shape: [samples, data]
            # y.shape [data]

            dQ = (-1) ** y[tf.newaxis, :] * tf.math.exp(
                log_pi_[..., tf.newaxis] + mu * (1 - 2 * y[tf.newaxis, :])
            )
            dQ *= (
                grad_log_pi[..., 0]
                + (1 - 2 * y)[tf.newaxis, :]
                + tf.reduce_sum(
                    X
                    * (
                        grad_log_pi[..., 1:]
                        + ((1 - 2 * y)[:, tf.newaxis] * X)[tf.newaxis, :, :]
                    ),
                    axis=-1,
                )
            )

            standardized = tf.concat(
                [Q_beta / beta_sd, Q_intercept / intercept_sd], axis=-1
            )
            standardized = tf.reduce_max(tf.math.abs(standardized), axis=-1)
            standardized = tf.reduce_max(standardized, axis=0, keepdims=True)[
                ..., tf.newaxis
            ]

            h = hbar / standardized

            intercept_kl = intercept + h * Q_intercept
            beta_kl = beta + h * Q_beta

            logJ = tf.math.log1p(tf.math.abs(h[..., 0] * dQ))
            return beta_kl, intercept_kl, logJ

        # variance descent -(log ell)'/l

        def T_I():
            Q = jnp.zeros_like(log_ell)
            return (
                beta + Q[..., tf.newaxis],
                intercept + Q[..., tf.newaxis],
                jnp.zeros_like(Q),
            )

        def T_var():
            log_pi_ = log_pi - tf.reduce_max(log_pi, axis=0)

            Q_beta = (
                (-1) ** y * tf.math.exp(log_pi_[..., tf.newaxis] + mu * 2 * (1 - 2 * y))
            )[..., tf.newaxis] * data["X"]
            Q_intercept = (
                (-1) ** y * tf.math.exp(log_pi_[..., tf.newaxis] + mu * 2 * (1 - 2 * y))
            )[..., tf.newaxis]

            # log_pi.shape: [samples]
            # mu.shape: [samples, data]
            # y.shape [data]

            dQ = (-1) ** y[tf.newaxis, :] * tf.math.exp(
                log_pi_[..., tf.newaxis] + mu * 2 * (1 - 2 * y[tf.newaxis, :])
            )
            dQ *= (
                grad_log_pi[..., 0]
                + 2 * (1 - 2 * y)[tf.newaxis, :]
                + tf.reduce_sum(
                    X
                    * (
                        grad_log_pi[..., 1:]
                        + (2 * (1 - 2 * y)[:, tf.newaxis] * X)[tf.newaxis, :, :]
                    ),
                    axis=-1,
                )
            )

            standardized = tf.concat(
                [Q_beta / beta_sd, Q_intercept / intercept_sd], axis=-1
            )
            standardized = tf.reduce_max(tf.math.abs(standardized), axis=-1)
            standardized = tf.reduce_max(standardized, axis=0, keepdims=True)[
                ..., tf.newaxis
            ]

            h = hbar / standardized

            intercept_kl = intercept + h * Q_intercept
            beta_kl = beta + h * Q_beta

            logJ = tf.math.log1p(tf.math.abs(h[..., 0] * dQ))
            return beta_kl, intercept_kl, logJ

        def IS(Q):
            beta_new, intercept_new, logJ = Q()
            mu_new = tf.reduce_sum(beta_new * X, axis=-1) + intercept_new[..., 0]
            sigma_new = tf.math.sigmoid(mu_new)
            ell_new = y * (sigma_new) + (1 - y) * (1 - sigma_new)
            log_ell_new = tf.math.xlogy(y, sigma_new) + tf.math.xlogy(1 - y, 1 - sigma_new)
            transformed = params.copy()
            transformed["beta__"] = beta_new[..., tf.newaxis, :]
            transformed["intercept__"] = intercept_new[..., tf.newaxis, :]
            transformed["global_scale"] = transformed["global_scale"][..., tf.newaxis, :]
            transformed["global_scale_aux"] = transformed["global_scale_aux"][
                ..., tf.newaxis, :
            ]

            if variational:
                # We trust the variational approximation, so \hat{pi} = pi
                # N_samples x N_data
                delta_log_pi = (
                    self.surrogate_distribution.log_prob(transformed)
                    - log_pi[:, tf.newaxis]
                )
                delta_log_pi = delta_log_pi - tf.reduce_max(
                    delta_log_pi, axis=0, keepdims=True
                )
                pass
            else:
                # we don't trust the variational approximation
                # Need to compute log_pi directly by summing over the likelihood

                ell_cross = tf.math.sigmoid(
                    tf.reduce_sum(beta_new[..., tf.newaxis, :] * X, -1)
                    + intercept_new
                )
                ell_cross = tf.math.xlogy(y, ell_cross) + tf.math.xlogy(
                    1 - y, 1 - ell_cross
                )
                ell_cross = tf.math.reduce_sum(ell_cross, axis=-1)

                log_pi_new = self.prior_distribution.log_prob(
                    {
                        "regression_model": {
                            k: tf.cast(transformed[k], self.dtype)
                            for k in self.regression_var_list
                        },
                        "intercept_model": {
                            k: tf.cast(transformed[k], self.dtype)
                            for k in self.intercept_var_list
                        },
                    }
                )
                log_pi_new += ell_cross
                # Incorporate the prior
                delta_log_pi = log_pi_new - log_pi[:, tf.newaxis]

            log_eta_weights = delta_log_pi - log_ell_new + logJ
            log_eta_weights -= tf.reduce_max(log_eta_weights, axis=0, keepdims=True)
            psis_weights, khat = nppsis.psislw(log_eta_weights)

            eta_weights = tf.math.exp(log_eta_weights)
            eta_weights = eta_weights / tf.reduce_sum(eta_weights, axis=0, keepdims=True)

            psis_weights = tf.math.exp(psis_weights)
            psis_weights = psis_weights / tf.math.reduce_sum(
                psis_weights, axis=0, keepdims=True
            )

            weight_entropy = self.entropy(eta_weights)
            psis_entropy = self.entropy(psis_weights)

            p_loo_new = tf.reduce_sum(sigma_new * eta_weights, axis=0)
            p_loo_psis = tf.reduce_sum(sigma_new * psis_weights, axis=0)
            p_loo_sd = tf.math.reduce_std(sigma_new * eta_weights, axis=0)
            ll_loo_new = tf.reduce_sum(eta_weights * ell_new, axis=0)
            ll_loo_psis = tf.reduce_sum(psis_weights * ell_new, axis=0)
            ll_loo_sd = tf.math.reduce_std(eta_weights * ell_new, axis=0)
            return (
                eta_weights,
                psis_weights,
                p_loo_new,
                p_loo_sd,
                ll_loo_new,
                ll_loo_sd,
                weight_entropy,
                khat,
                p_loo_psis,
                ll_loo_psis,
            )

        (
            eta_I,
            eta_I_psis,
            p_loo_I,
            p_loo_I_sd,
            ll_loo_I,
            ll_loo_I_sd,
            S_I,
            k_I,
            p_psis_I,
            ll_psis_I,
        ) = IS(T_I)
        (
            eta_kl,
            eta_kl_psis,
            p_loo_kl,
            p_loo_kl_sd,
            ll_loo_kl,
            ll_loo_kl_sd,
            S_kl,
            k_kl,
            p_psis_kl,
            ll_psis_kl,
        ) = IS(T_kl)

        (
            eta_ll,
            eta_ll_psis,
            p_loo_ll,
            p_loo_ll_sd,
            ll_loo_ll,
            ll_loo_ll_sd,
            S_ll,
            k_ll,
            p_psis_ll,
            ll_psis_ll,
        ) = IS(T_ll)

        (
            eta_var,
            eta_var_psis,
            p_loo_var,
            p_loo_var_sd,
            ll_loo_var,
            ll_loo_var_sd,
            S_var,
            k_var,
            p_psis_var,
            ll_psis_var,
        ) = IS(T_var)

        # kl descent

        return {
            "I": {
                "p_loo": p_loo_I,
                "p_loo_sd": p_loo_I_sd,
                "ll_loo": ll_loo_I,
                "ll_loo_sd": ll_loo_I_sd,
                "S": S_I,
                "khat": k_I,
                "p_psis": p_psis_I,
                "ll_psis": ll_psis_I,
            },
            "KL": {
                "p_loo": p_loo_kl,
                "p_loo_sd": p_loo_kl_sd,
                "ll_loo": ll_loo_kl,
                "ll_loo_sd": ll_loo_kl_sd,
                "S": S_kl,
                "khat": k_kl,
                "p_psis": p_psis_kl,
                "ll_psis": ll_psis_kl,
            },
            "LL": {
                "p_loo": p_loo_kl,
                "p_loo_sd": p_loo_kl_sd,
                "ll_loo": ll_loo_kl,
                "ll_loo_sd": ll_loo_kl_sd,
                "S": S_ll,
                "khat": k_ll,
                "p_psis": p_psis_ll,
                "ll_psis": ll_psis_ll,
            },
            "Var": {
                "p_loo": p_loo_var,
                "p_loo_sd": p_loo_var_sd,
                "ll_loo": ll_loo_var,
                "ll_loo_sd": ll_loo_var_sd,
                "S": S_var,
                "khat": k_var,
                "p_psis": p_psis_var,
                "ll_psis": ll_psis_var,
            },
        }
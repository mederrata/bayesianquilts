#!/usr/bin/env python3
"""Example quilt model
"""
from collections import defaultdict

import tensorflow as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd

from bayesianquilts.model import BayesianModel
from bayesianquilts.vi.advi import build_surrogate_posterior


class LogisticRegression2(BayesianModel):
    def __init__(
        self,
        dim_regressors,
        scale_icept=1.0,
        scale_global=1.0,
        nu_global=1.0,
        nu_local=1.0,
        slab_scale=1.0,
        slab_df=1.0,
        dtype=tf.float64,
    ):
        super(LogisticRegression2, self).__init__(dtype=dtype)
        self.dim_regressors = dim_regressors
        self.scale_icept = scale_icept
        self.scale_global = scale_global
        self.nu_global = nu_global
        self.nu_local = nu_local
        self.slab_scale = slab_scale
        self.slab_df = slab_df

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
        bijectors = defaultdict(lambda: tfb.Identity())
        bijectors["caux"] = tfb.Softplus()
        bijectors["tau"] = tfb.Softplus()
        self.prior_distribution = tfd.JointDistributionNamed(joint_prior_dict)
        self.surrogate_distribution = build_surrogate_posterior(
            self.prior_distribution, bijectors=bijectors
        )
        self.surrogate_vars = self.surrogate_distribution.variables
        self.var_list = list(self.surrogate_distribution.model.keys())
        return None
    
    def transform(self, params):
        c = self.slab_scale * tf.math.sqrt(params["caux"])
        lambda_tilde = tf.math.sqrt(
            c**2
            * params["lambda"] ** 2
            / (c**2 + params["tau"] ** 2 * params["lambda"] ** 2)
        )
        beta = params["z"] * lambda_tilde * params["tau"]
        params['beta'] = beta
        return params

    def predictive_distribution(self, data, **params):

        processed = (self.preprocessor())(data)
        c = self.slab_scale * tf.math.sqrt(params["caux"])
        lambda_tilde = tf.math.sqrt(
            c**2
            * params["lambda"] ** 2
            / (c**2 + params["tau"] ** 2 * params["lambda"] ** 2)
        )
        beta = params["z"] * lambda_tilde * params["tau"]
        # compute regression product
        X = tf.cast(
            processed["X"],
            self.dtype,
        )
        mu = beta[..., tf.newaxis, :] * X
        mu = tf.reduce_sum(mu, -1) + params["beta0"]

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

        prior = self.prior_distribution.log_prob(params)

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
        c = self.slab_scale * tf.math.sqrt(params["caux"])
        lambda_tilde = tf.math.sqrt(
            c**2
            * params["lambda"] ** 2
            / (c**2 + params["tau"] ** 2 * params["lambda"] ** 2)
        )
        beta = params["z"] * lambda_tilde * params["tau"]
        intercept = params["beta0"]
        X = tf.cast(data["X"], self.dtype)
        y = tf.cast(data["y"], self.dtype)[:, 0]
        mu = beta[..., tf.newaxis, :] * X
        mu = tf.reduce_sum(mu, -1) + params["beta0"]
        sigma = tf.math.sigmoid(mu)
        ell = y * (sigma) + (1 - y) * (1 - sigma)
        log_ell = tf.math.xlogy(y, sigma) + tf.math.xlogy(1 - y, 1 - sigma)
        log_ell_prime = y * (1 - sigma) - (1 - y) * sigma
        log_ell_doubleprime = -sigma * (1 - sigma)
        _, khat0 = nppsis.psislw(-log_ell)

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
            # log_pi.shape: [samples]
        else:
            """
            Recall Bayes rule:
            \log pi(\btheta|\calD) = \sum_i\log ell_i(\btheta) + \log\pi(\btheta) + const

            so
            \nabla\log\pi(\btheta|\calD) = \sum_i (ell_i)'x + grad\log\pi(\btheta)

            """
            log_prior = self.prior_distribution.log_prob_parts(params)
            log_prior = log_prior["z"] + log_prior["beta0"]

            log_pi = tf.reduce_sum(log_ell, axis=1, keepdims=True)[:, 0]

            # pi \propto
            grad_log_pi = tf.concat(
                [
                    tf.reduce_sum(log_ell_prime[..., tf.newaxis], axis=1, keepdims=True),
                    tf.reduce_sum(
                        log_ell_prime[..., tf.newaxis] * X, axis=1, keepdims=True
                    ),
                ],
                axis=-1,
            )

            grad_log_prior = -0.5 * tf.concat(
                [(params["beta0"] / self.scale_icept) ** 2, (params["z"]) ** 2],
                axis=-1,
            )
            grad_log_pi += grad_log_prior[:, tf.newaxis, :]

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
            beta_ll = beta[..., tf.newaxis, :] + h * Q_beta
            intercept_ll = intercept[..., tf.newaxis, :] + h * Q_intercept
            return beta_ll, intercept_ll, logJ

        def T_kl():
            log_pi_ = log_pi - tf.reduce_max(log_pi, axis=0, keepdims=True)
            Q_beta = ((-1) ** y * tf.math.exp(log_pi_[..., tf.newaxis] + mu * (1 - 2 * y)))[
                ..., tf.newaxis
            ] * X
            Q_intercept = (
                ((-1) ** y) * tf.math.exp(log_pi_[..., tf.newaxis] + mu * (1 - 2 * y))
            )[..., tf.newaxis]

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
                        + (1 - 2 * y)[:, tf.newaxis] * X[tf.newaxis, ...]
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

            intercept_kl = intercept[..., tf.newaxis] + h * Q_intercept
            beta_kl = beta[..., tf.newaxis, :] + h * Q_beta

            logJ = tf.math.log1p(tf.math.abs(h[..., 0] * dQ))
            return beta_kl, intercept_kl, logJ

        # variance descent -(log ell)'/l

        def T_I():
            Q = jnp.zeros_like(log_ell)
            return (
                beta[:, tf.newaxis, :] + Q[..., tf.newaxis],
                intercept[..., tf.newaxis] + Q[..., tf.newaxis],
                jnp.zeros_like(Q),
            )

        def T_var():
            log_pi_ = log_pi - tf.reduce_max(log_pi, axis=0, keepdims=True)

            Q_beta = (
                (-1) ** y * tf.math.exp(log_pi_[..., tf.newaxis] + 2 * mu * (1 - 2 * y))
            )[..., tf.newaxis] * X
            Q_intercept = (
                (-1) ** y * tf.math.exp(log_pi_[..., tf.newaxis] + 2 * mu * (1 - 2 * y))
            )[..., tf.newaxis]

            dQ = (
                (-1) ** y[tf.newaxis, :]
                * tf.math.exp(
                    log_pi_[..., tf.newaxis] + 2 * mu * (1 - 2 * y[tf.newaxis, :])
                )
                * (
                    grad_log_pi[..., 0]
                    + (1 - 2 * y)[tf.newaxis, :]
                    + tf.reduce_sum(
                        X * (grad_log_pi[..., 1:] + 2 * (1 - 2 * y)[:, tf.newaxis] * X),
                        axis=-1,
                    )
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

            intercept_kl = intercept[..., tf.newaxis, :] + h * Q_intercept
            beta_kl = beta[..., tf.newaxis, :] + h * Q_beta

            logJ = tf.math.log1p(tf.math.abs(h[..., 0] * dQ))
            return beta_kl, intercept_kl, logJ

        def IS(Q):
            beta_new, intercept_new, logJ = Q()
            mu_new = tf.reduce_sum(beta_new * X, axis=-1) + intercept_new[..., 0]
            sigma_new = tf.math.sigmoid(mu_new)
            ell_new = y * (sigma_new) + (1 - y) * (1 - sigma_new)
            log_ell_new = tf.math.xlogy(y, sigma_new) + tf.math.xlogy(1 - y, 1 - sigma_new)
            c = self.slab_scale * tf.math.sqrt(params["caux"])
            lambda_tilde = tf.math.sqrt(
                c**2
                * params["lambda"] ** 2
                / (c**2 + params["tau"] ** 2 * params["lambda"] ** 2)
            )
            transformed = params.copy()
            transformed["z"] = beta_new / (
                lambda_tilde[:, tf.newaxis, :] * params["tau"][..., tf.newaxis]
            )
            transformed["beta0"] = intercept_new

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
                    tf.reduce_sum(beta_new[..., tf.newaxis, :] * X, -1) + intercept_new
                )
                ell_cross = tf.math.xlogy(y, ell_cross) + tf.math.xlogy(
                    1 - y, 1 - ell_cross
                )
                ell_cross = tf.math.reduce_sum(ell_cross, axis=-1)

                log_prior_new = self.prior_distribution.log_prob_parts(transformed)
                log_prior_new = log_prior_new["z"] + log_prior_new["beta0"]
                log_pi_new = ell_cross
                delta_log_prior = log_prior_new - log_prior[:, tf.newaxis]
                # Incorporate the prior
                delta_log_pi = log_pi_new - log_pi[:, tf.newaxis] + delta_log_prior
            log_eta_weights = delta_log_pi - log_ell_new + logJ
            log_eta_weights = log_eta_weights - tf.reduce_max(log_eta_weights, axis=0)
            psis_weights, khat = nppsis.psislw(log_eta_weights)
            _, khat_test = nppsis.psislw(-log_ell_new - tf.reduce_max(-log_ell_new, axis=0))

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



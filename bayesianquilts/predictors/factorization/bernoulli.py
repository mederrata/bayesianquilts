#!/usr/bin/env python3
"""Sparse probabilistic bernoulli matrix factorization using the horseshoe
"""


import jax.numpy as jnp
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd

from bayesianquilts.distributions import AbsHorseshoe, SqrtInverseGamma
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator
from bayesianquilts.predictors.factorization.poisson import FactorizationModel


class BernoulliFactorization(FactorizationModel):
    """Sparse (horseshoe) bernoulli matrix factorization
    Arguments:
        object {[type]} -- [description]
    """

    def log_likelihood_components(self, s, u, v, w, data, *args, **kwargs):
        """Returns the log likelihood without summing along axes
        Arguments:
            s {jnp.ndarray} -- Samples of s
            u {jnp.ndarray} -- Samples of u
            v {jnp.ndarray} -- Samples of v
            w {jnp.ndarray} -- Samples of w
        Keyword Arguments:
            data {jnp.ndarray} -- Count matrix (default: {None})
        Returns:
            [jnp.ndarray] -- log likelihood in broadcasted shape
        """

        theta_u = self.encode(data[self.count_key], u, s)
        phi = self.intercept_matrix(w, s)
        B = self.decoding_matrix(v)

        theta_beta = jnp.matmul(theta_u, B)
        theta_beta = self.decoder_function(theta_beta)

        rate = theta_beta + phi
        rv_bernoulli = tfd.Bernoulli(logits=rate)

        return {
            "log_likelihood": rv_bernoulli.log_prob(
                data[self.count_key].astype(self.dtype)
            ),
            "rate": rate,
        }

    def create_distributions(self):
        """Create distribution objects"""
        self.bijectors = {
            "u": tfb.Softplus(),
            "v": tfb.Identity(),
            "u_eta": tfb.Softplus(),
            "u_tau": tfb.Softplus(),
            "s": tfb.Softplus(),
            "s_eta": tfb.Softplus(),
            "s_tau": tfb.Softplus(),
            "w": tfb.Identity(),
        }
        symmetry_breaking_decay = (
            self.symmetry_breaking_decay
            ** jnp.arange(self.latent_dim, dtype=self.dtype)[jnp.newaxis, ...]
        )

        distribution_dict = {
            "v": tfd.Independent(
                tfd.Normal(
                    loc=0.1
                    * jnp.zeros((self.latent_dim, self.feature_dim), dtype=self.dtype),
                    scale=0.1
                    * jnp.ones((self.latent_dim, self.feature_dim), dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=2,
            ),
            "w": tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros((1, self.feature_dim), dtype=self.dtype),
                    scale=jnp.ones((1, self.feature_dim), dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=2,
            ),
        }
        if self.horseshoe_plus:
            distribution_dict = {
                **distribution_dict,
                "u": lambda u_eta, u_tau: tfd.Independent(
                    tfd.HalfNormal(scale=u_eta * u_tau * symmetry_breaking_decay),
                    reinterpreted_batch_ndims=2,
                ),
                "u_eta": tfd.Independent(
                    tfd.HalfCauchy(
                        loc=jnp.zeros(
                            (self.feature_dim, self.latent_dim), dtype=self.dtype
                        ),
                        scale=jnp.ones(
                            (self.feature_dim, self.latent_dim), dtype=self.dtype
                        ),
                    ),
                    reinterpreted_batch_ndims=2,
                ),
                "u_tau": tfd.Independent(
                    tfd.HalfCauchy(
                        loc=jnp.zeros((1, self.latent_dim), dtype=self.dtype),
                        scale=jnp.ones((1, self.latent_dim), dtype=self.dtype)
                        * self.u_tau_scale,
                    ),
                    reinterpreted_batch_ndims=2,
                ),
            }
            distribution_dict["s"] = lambda s_eta, s_tau: tfd.Independent(
                tfd.HalfNormal(scale=s_eta * s_tau), reinterpreted_batch_ndims=2
            )
            distribution_dict["s_eta"] = tfd.Independent(
                tfd.HalfCauchy(
                    loc=jnp.zeros((2, self.feature_dim), dtype=self.dtype),
                    scale=jnp.ones((2, self.feature_dim), dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=2,
            )
            distribution_dict["s_tau"] = tfd.Independent(
                tfd.HalfCauchy(
                    loc=jnp.zeros((1, self.feature_dim), dtype=self.dtype),
                    scale=jnp.ones((1, self.feature_dim), dtype=self.dtype)
                    * self.s_tau_scale,
                ),
                reinterpreted_batch_ndims=2,
            )

            self.bijectors["u_eta_a"] = tfb.Softplus()
            self.bijectors["u_tau_a"] = tfb.Softplus()

            self.bijectors["s_eta_a"] = tfb.Softplus()
            self.bijectors["s_tau_a"] = tfb.Softplus()

            distribution_dict["u_eta"] = lambda u_eta_a: tfd.Independent(
                SqrtInverseGamma(
                    concentration=0.5
                    * jnp.ones((self.feature_dim, self.latent_dim), dtype=self.dtype),
                    scale=1.0 / u_eta_a,
                ),
                reinterpreted_batch_ndims=2,
            )
            distribution_dict["u_eta_a"] = tfd.Independent(
                tfd.InverseGamma(
                    concentration=0.5
                    * jnp.ones((self.feature_dim, self.latent_dim), dtype=self.dtype),
                    scale=jnp.ones(
                        (self.feature_dim, self.latent_dim), dtype=self.dtype
                    ),
                ),
                reinterpreted_batch_ndims=2,
            )
            distribution_dict["u_tau"] = lambda u_tau_a: tfd.Independent(
                SqrtInverseGamma(
                    concentration=0.5
                    * jnp.ones((1, self.latent_dim), dtype=self.dtype),
                    scale=1.0 / u_tau_a,
                ),
                reinterpreted_batch_ndims=2,
            )
            distribution_dict["u_tau_a"] = tfd.Independent(
                tfd.InverseGamma(
                    concentration=0.5
                    * jnp.ones((1, self.latent_dim), dtype=self.dtype),
                    scale=jnp.ones((1, self.latent_dim), dtype=self.dtype)
                    / self.u_tau_scale**2,
                ),
                reinterpreted_batch_ndims=2,
            )

            distribution_dict["s_eta"] = lambda s_eta_a: tfd.Independent(
                SqrtInverseGamma(
                    concentration=0.5
                    * jnp.ones((2, self.feature_dim), dtype=self.dtype),
                    scale=1.0 / s_eta_a,
                ),
                reinterpreted_batch_ndims=2,
            )
            distribution_dict["s_eta_a"] = tfd.Independent(
                tfd.InverseGamma(
                    concentration=0.5
                    * jnp.ones((2, self.feature_dim), dtype=self.dtype),
                    scale=jnp.ones((2, self.feature_dim), dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=2,
            )
            distribution_dict["s_tau"] = lambda s_tau_a: tfd.Independent(
                SqrtInverseGamma(
                    concentration=0.5
                    * jnp.ones((1, self.feature_dim), dtype=self.dtype),
                    scale=1.0 / s_tau_a,
                ),
                reinterpreted_batch_ndims=2,
            )
            distribution_dict["s_tau_a"] = tfd.Independent(
                tfd.InverseGamma(
                    concentration=0.5
                    * jnp.ones((1, self.feature_dim), dtype=self.dtype),
                    scale=jnp.ones((1, self.feature_dim), dtype=self.dtype)
                    / self.s_tau_scale**2,
                ),
                reinterpreted_batch_ndims=2,
            )
        else:
            distribution_dict = {
                **distribution_dict,
                "u": tfd.Independent(
                    AbsHorseshoe(
                        scale=(
                            self.u_tau_scale
                            * symmetry_breaking_decay
                            * jnp.ones(
                                (self.feature_dim, self.latent_dim), dtype=self.dtype
                            )
                        ),
                        reinterpreted_batch_ndims=2,
                    )
                ),
                "s": tfd.Independent(
                    AbsHorseshoe(
                        scale=self.s_tau_scale
                        * jnp.ones((1, self.feature_dim), dtype=self.dtype)
                    ),
                    reinterpreted_batch_ndims=2,
                ),
            }

        self.prior_distribution = tfd.JointDistributionNamed(distribution_dict)

        self.surrogate_distribution_generator, self.surrogate_parameter_initializer = (
            build_factored_surrogate_posterior_generator(
                prior_distribution=self.prior_distribution,
                bijectors=self.bijectors,
                dtype=self.dtype,
            )
        )
        self.params = self.surrogate_parameter_initializer()

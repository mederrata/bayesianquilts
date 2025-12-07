#!/usr/bin/env python3
"""Mixed-type matrix factorization
"""


import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd

from bayesianquilts.distributions import AbsHorseshoe, SqrtInverseGamma
from bayesianquilts.model import BayesianModel
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator
from bayesianquilts.predictors.factorization.poisson import FactorizationModel


class MixedFactorization(FactorizationModel):
    """Sparse (horseshoe) mixed-type matrix factorization
    Arguments:
        object {[type]} -- [description]
    """

    def __init__(
        self,
        likelihood_types,
        latent_dim=None,
        feature_dim=None,
        u_tau_scale=0.01,
        s_tau_scale=1.0,
        sigma_scale=1.0,
        symmetry_breaking_decay=0.99,
        encoder_function=None,
        decoder_function=None,
        log_transform=False,
        horshoe_plus=True,
        column_norms=None,
        count_key="counts",
        dtype=jnp.float64,
        **kwargs,
    ):
        """Instantiate PMF object
        Keyword Arguments:
            likelihood_types {list} -- List of likelihood types for each feature
            latent_dim {int]} -- P (default: {None})
            u_tau_scale {float} -- Global shrinkage scale on u (default: {1.})
            s_tau_scale {int} -- Global shrinkage scale on s (default: {1})
            sigma_scale {float} -- Global shrinkage scale on sigma (default: {1.})
            symmetry_breaking_decay {float} -- Decay factor along dimensions
                                                on u (default: {0.5})
            decoder_function {function} -- f(x) (default: {lambda x: x/scale})
            encoder_function {function} -- g(x) (default: {lambda x: x/scale})
            horseshe_plus {bool} -- Whether to use hierarchical horseshoe plus (default : {True})
            dtype {[type]} -- [description] (default: {jnp.float64})
        """

        super(MixedFactorization, self).__init__(
            latent_dim=latent_dim,
            feature_dim=feature_dim,
            u_tau_scale=u_tau_scale,
            s_tau_scale=s_tau_scale,
            symmetry_breaking_decay=symmetry_breaking_decay,
            encoder_function=encoder_function,
            decoder_function=decoder_function,
            log_transform=log_transform,
            horshoe_plus=horshoe_plus,
            column_norms=column_norms,
            count_key=count_key,
            dtype=dtype,
            **kwargs,
        )
        self.likelihood_types = likelihood_types
        self.sigma_scale = sigma_scale
        
    def create_distributions(self):
        """Create distribution objects"""
        self.bijectors = {
            "u": tfb.Identity(),
            "v": tfb.Identity(),
            "u_eta": tfb.Softplus(),
            "u_tau": tfb.Softplus(),
            "s": tfb.Softplus(),
            "s_eta": tfb.Softplus(),
            "s_tau": tfb.Softplus(),
            "w": tfb.Identity(),
        }
        if 'gaussian' in self.likelihood_types:
            self.bijectors['sigma'] = tfb.Softplus()

        symmetry_breaking_decay = (
            self.symmetry_breaking_decay
            ** jnp.arange(self.latent_dim, dtype=self.dtype)[jnp.newaxis, ...]
        )

        distribution_dict = {
            "v": tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(
                        (self.latent_dim, self.feature_dim),
                        dtype=self.dtype),
                    scale=0.1*jnp.ones(
                        (self.latent_dim, self.feature_dim),
                        dtype=self.dtype)
                ), reinterpreted_batch_ndims=2
            ),
            "w": tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(
                        (1, self.feature_dim),
                        dtype=self.dtype),
                    scale=jnp.ones(
                        (1, self.feature_dim), dtype=self.dtype)
                ),
                reinterpreted_batch_ndims=2
            ),
        }
        if 'gaussian' in self.likelihood_types:
            distribution_dict['sigma'] = tfd.Independent(
                tfd.HalfCauchy(
                    loc=jnp.zeros(self.feature_dim, dtype=self.dtype),
                    scale=jnp.ones(
                        self.feature_dim,
                        dtype=self.dtype)*self.sigma_scale
                ), reinterpreted_batch_ndims=1
            )

        if self.horseshoe_plus:
            distribution_dict = {
                **distribution_dict,
                'u': lambda u_eta, u_tau: tfd.Independent(
                    tfd.Normal(
                        loc=jnp.zeros(
                            (self.feature_dim, self.latent_dim),
                            dtype=self.dtype),
                        scale=u_eta*u_tau*symmetry_breaking_decay
                    ), reinterpreted_batch_ndims=2
                ),
                'u_eta': tfd.Independent(
                    tfd.HalfCauchy(
                        loc=jnp.zeros(
                            (self.feature_dim, self.latent_dim),
                            dtype=self.dtype),
                        scale=jnp.ones(
                            (self.feature_dim, self.latent_dim),
                            dtype=self.dtype)
                    ), reinterpreted_batch_ndims=2
                ),
                'u_tau': tfd.Independent(
                    tfd.HalfCauchy(
                        loc=jnp.zeros(
                            (1, self.latent_dim),
                            dtype=self.dtype),
                        scale=jnp.ones(
                            (1, self.latent_dim),
                            dtype=self.dtype)*self.u_tau_scale
                    ), reinterpreted_batch_ndims=2
                ),
            }
            distribution_dict['s'] = lambda s_eta, s_tau: tfd.Independent(
                tfd.HalfNormal(
                    scale=s_eta*s_tau
                ), reinterpreted_batch_ndims=2
            )
            distribution_dict['s_eta'] = tfd.Independent(
                tfd.HalfCauchy(
                    loc=jnp.zeros(
                        (2, self.feature_dim),
                        dtype=self.dtype),
                    scale=jnp.ones(
                        (2, self.feature_dim),
                        dtype=self.dtype)
                ), reinterpreted_batch_ndims=2
            )
            distribution_dict['s_tau'] = tfd.Independent(
                tfd.HalfCauchy(
                    loc=jnp.zeros((1, self.feature_dim), dtype=self.dtype),
                    scale=jnp.ones(
                        (1, self.feature_dim),
                        dtype=self.dtype)*self.s_tau_scale
                ), reinterpreted_batch_ndims=2
            )

            self.bijectors['u_eta_a'] = tfb.Softplus()
            self.bijectors['u_tau_a'] = tfb.Softplus()

            self.bijectors['s_eta_a'] = tfb.Softplus()
            self.bijectors['s_tau_a'] = tfb.Softplus()

            distribution_dict['u_eta'] = lambda u_eta_a: tfd.Independent(
                SqrtInverseGamma(
                    concentration=0.5*jnp.ones(
                        (self.feature_dim, self.latent_dim),
                        dtype=self.dtype
                    ),
                    scale=1.0/u_eta_a
                ), reinterpreted_batch_ndims=2
            )
            distribution_dict['u_eta_a'] = tfd.Independent(
                tfd.InverseGamma(
                    concentration=0.5*jnp.ones(
                        (self.feature_dim, self.latent_dim),
                        dtype=self.dtype
                    ),
                    scale=jnp.ones(
                        (self.feature_dim, self.latent_dim),
                        dtype=self.dtype)
                ), reinterpreted_batch_ndims=2
            )
            distribution_dict['u_tau'] = lambda u_tau_a: tfd.Independent(
                SqrtInverseGamma(
                    concentration=0.5*jnp.ones(
                        (1, self.latent_dim),
                        dtype=self.dtype
                    ),
                    scale=1.0/u_tau_a
                ), reinterpreted_batch_ndims=2
            )
            distribution_dict['u_tau_a'] = tfd.Independent(
                tfd.InverseGamma(
                    concentration=0.5*jnp.ones(
                        (1, self.latent_dim), dtype=self.dtype
                    ),
                    scale=jnp.ones(
                        (1, self.latent_dim), dtype=self.dtype
                    )/self.u_tau_scale**2
                ), reinterpreted_batch_ndims=2
            )

            distribution_dict['s_eta'] = lambda s_eta_a: tfd.Independent(
                SqrtInverseGamma(
                    concentration=0.5*jnp.ones(
                        (2, self.feature_dim),
                        dtype=self.dtype
                    ),
                    scale=1.0/s_eta_a
                ), reinterpreted_batch_ndims=2
            )
            distribution_dict['s_eta_a'] = tfd.Independent(
                tfd.InverseGamma(
                    concentration=0.5*jnp.ones(
                        (2, self.feature_dim), dtype=self.dtype
                    ),
                    scale=jnp.ones((2, self.feature_dim), dtype=self.dtype)
                ), reinterpreted_batch_ndims=2
            )
            distribution_dict['s_tau'] = lambda s_tau_a: tfd.Independent(
                SqrtInverseGamma(
                    concentration=0.5*jnp.ones(
                        (1, self.feature_dim), dtype=self.dtype
                    ),
                    scale=1.0/s_tau_a
                ), reinterpreted_batch_ndims=2
            )
            distribution_dict['s_tau_a'] = tfd.Independent(
                tfd.InverseGamma(
                    concentration=0.5*jnp.ones(
                        (1, self.feature_dim), dtype=self.dtype
                    ),
                    scale=jnp.ones(
                        (1, self.feature_dim),
                        dtype=self.dtype)/self.s_tau_scale**2
                ), reinterpreted_batch_ndims=2
            )
        else:
            distribution_dict = {
                **distribution_dict,
                'u': tfd.Independent(
                    tfd.Horseshoe(
                        loc=jnp.zeros(
                                (self.feature_dim, self.latent_dim),
                                dtype=self.dtype
                            ),
                        scale=(
                            self.u_tau_scale*symmetry_breaking_decay*jnp.ones(
                                (self.feature_dim, self.latent_dim),
                                dtype=self.dtype
                            )
                        ), reinterpreted_batch_ndims=2
                    )
                ),
                's': tfd.Independent(
                    AbsHorseshoe(
                        scale=self.s_tau_scale*jnp.ones(
                            (1, self.feature_dim), dtype=self.dtype
                        )
                    ), reinterpreted_batch_ndims=2
                )
            }

        self.prior_distribution = tfd.JointDistributionNamed(
            distribution_dict)

        self.surrogate_distribution_generator, self.surrogate_parameter_initializer = (
            build_factored_surrogate_posterior_generator(
                prior_distribution=self.prior_distribution,
                bijectors=self.bijectors,
                dtype=self.dtype,
            )
        )
        self.params = self.surrogate_parameter_initializer()

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

        loc = theta_beta + phi
        
        log_likelihoods = []
        for i, likelihood_type in enumerate(self.likelihood_types):
            if likelihood_type == 'poisson':
                rv = tfd.Poisson(rate=loc[..., i])
                ll = rv.log_prob(data[self.count_key][..., i].astype(self.dtype))
            elif likelihood_type == 'bernoulli':
                rv = tfd.Bernoulli(logits=loc[..., i])
                ll = rv.log_prob(data[self.count_key][..., i].astype(self.dtype))
            elif likelihood_type == 'gaussian':
                sigma = kwargs['sigma']
                rv = tfd.Normal(loc=loc[..., i], scale=sigma[..., i])
                ll = rv.log_prob(data[self.count_key][..., i].astype(self.dtype))
            else:
                raise ValueError(f"Unknown likelihood type: {likelihood_type}")
            
            log_likelihoods.append(ll)

        return {
            "log_likelihood": jnp.stack(log_likelihoods, axis=-1),
            "loc": loc,
        }
#!/usr/bin/env python3
"""Sparse probabilistic poisson matrix factorization using the horseshoe
"""

import jax.numpy as jnp
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd

from bayesianquilts.distributions import AbsHorseshoe, SqrtInverseGamma
from bayesianquilts.model import BayesianModel
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator


class FactorizationModel(BayesianModel):
    """Base class for factorization models
    """

    bijectors = None
    var_list = []
    s_tau_scale = 1

    def encoder_function(self, x):
        """Encoder function (g)
        Args:
            x ([type]): [description]
        Returns:
            [type]: [description]
        """
        if self.log_transform:
            return jnp.log(x / self.eta_i + 1.0)
        return jnp.array(x).astype(self.dtype) / self.eta_i

    def decoder_function(self, x):
        """Decoder function (f)
        Args:
            x ([type]): [description]
        Returns:
            [type]: [description]
        """
        if self.log_transform:
            return jnp.exp(x * self.eta_i) - 1.0
        return jnp.array(x).astype(self.dtype) * self.eta_i

    def __init__(
        self,
        latent_dim=None,
        feature_dim=None,
        u_tau_scale=0.01,
        s_tau_scale=1.0,
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
            latent_dim {int]} -- P (default: {None})
            u_tau_scale {float} -- Global shrinkage scale on u (default: {1.})
            s_tau_scale {int} -- Global shrinkage scale on s (default: {1})
            symmetry_breaking_decay {float} -- Decay factor along dimensions
                                                on u (default: {0.5})
            decoder_function {function} -- f(x) (default: {lambda x: x/scale})
            encoder_function {function} -- g(x) (default: {lambda x: x/scale})
            horseshe_plus {bool} -- Whether to use hierarchical horseshoe plus (default : {True})
            dtype {[type]} -- [description] (default: {jnp.float64})
        """

        super(FactorizationModel, self).__init__(
            data_transform_fn=None,
            dtype=dtype,
        )

        self.horseshoe_plus = horshoe_plus
        self.eta_i = 1.0
        self.xi_u_global = 1.0
        if column_norms is not None:
            self.eta_i = column_norms
        self.count_key = count_key

        if encoder_function is not None:
            self.encoder_function = encoder_function
        if decoder_function is not None:
            self.decoder_function = decoder_function
        self.dtype = dtype
        self.symmetry_breaking_decay = symmetry_breaking_decay
        self.log_transform = log_transform

        self.feature_dim = feature_dim
        self.latent_dim = self.feature_dim if (latent_dim) is None else latent_dim

        self.u_tau_scale = u_tau_scale
        self.s_tau_scale = s_tau_scale
        self.create_distributions()
        self.set_calibration_expectations()
        print(f"Feature dim: {self.feature_dim} -> Latent dim {self.latent_dim}")

    def create_distributions(self):
        """Create distribution objects"""
        raise NotImplementedError

    def unormalized_log_prob_parts(self, data, prior_weight=1.0, **params):
        """Energy function
        Keyword Arguments:
            data {dict} -- Should be a single batch (default: {None})
        Returns:
            jnp.ndarray -- Energy of broadcasted shape
        """

        prior_parts = self.prior_distribution.log_prob_parts(params)
        prior_parts = {k: v * prior_weight for k, v in prior_parts.items()}
        log_likelihood = self.log_likelihood_components(data=data, **params)[
            "log_likelihood"
        ]

        s = params["s"]
        theta = self.encode(x=data[self.count_key], u=params["u"], s=s)
        rv_theta = tfd.Independent(
            tfd.HalfNormal(scale=jnp.ones_like(theta, dtype=self.dtype)),
            reinterpreted_batch_ndims=2,
        )

        prior_parts["z"] = rv_theta.log_prob(theta)

        log_likelihood = jnp.sum(log_likelihood, axis=[-1, -2])
        prior_parts["x"] = log_likelihood

        return prior_parts

    def encode(self, x, u=None, s=None):
        """Returns theta given x
        Args:
            x ([type]): [description]
        Returns:
            [type]: [description]
        """
        u = self.calibrated_expectations["u"] if u is None else u
        s = self.calibrated_expectations["s"] if s is None else s

        encoding = self.encoding_matrix(u, s)  # A = (\alpha_{ik})

        z = jnp.matmul(self.encoder_function(x.astype(self.dtype)), encoding)
        return z

    def encoding_matrix(self, u=None, s=None):
        """Output A = (\alpha_{ik})

        Returns:
            jnp.ndarray: batch_shape x I x K
        """
        u = self.calibrated_expectations["u"] if u is None else u

        s = self.calibrated_expectations["s"] if s is None else s
        weights = s / jnp.sum(s, axis=-2, keepdims=True)
        weights_1 = jnp.expand_dims(weights[..., 0, :], -1)

        encoding = weights_1 * u
        return encoding

    def decoding_matrix(self, v=None):
        """Output $B=(\beta_{ki})$

        Args:
            v (jnp.ndarray): default:None

        Returns:
            [type]: [description]
        """
        v = self.calibrated_expectations["v"] if v is None else v
        return v

    def intercept_matrix(self, w=None, s=None):
        """export phi

        Args:
            w ([type], optional): [description]. Defaults to None.
            s ([type], optional): [description]. Defaults to None.

        Returns:
            jnp.ndarray: batch_shape x 1 x I
        """

        w = self.calibrated_expectations["w"] if w is None else w

        s = self.calibrated_expectations["s"] if s is None else s
        weights = s / jnp.sum(s, axis=-2)[..., jnp.newaxis, :]
        weights_2 = jnp.expand_dims(weights[..., 1, :], -1)
        L = len(weights_2.shape)
        trans = tuple(list(range(L - 2)) + [L - 1, L - 2])
        weights_2 = jnp.transpose(weights_2, trans)
        return self.eta_i * weights_2 * w


class PoissonFactorization(FactorizationModel):
    """Sparse (horseshoe) poisson matrix factorization
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
        rv_poisson = tfd.Poisson(rate=rate)

        return {
            "log_likelihood": rv_poisson.log_prob(
                data[self.count_key].astype(self.dtype)
            ),
            "rate": rate,
        }
    def create_distributions(self):
        """Create distribution objects"""
        self.bijectors = {
            "u": tfb.Softplus(),
            "v": tfb.Softplus(),
            "u_eta": tfb.Softplus(),
            "u_tau": tfb.Softplus(),
            "s": tfb.Softplus(),
            "s_eta": tfb.Softplus(),
            "s_tau": tfb.Softplus(),
            "w": tfb.Softplus(),
        }
        symmetry_breaking_decay = (
            self.symmetry_breaking_decay
            ** jnp.arange(self.latent_dim, dtype=self.dtype)[jnp.newaxis, ...]
        )

        distribution_dict = {
            "v": tfd.Independent(
                tfd.HalfNormal(
                    scale=0.1
                    * jnp.ones((self.latent_dim, self.feature_dim), dtype=self.dtype)
                ),
                reinterpreted_batch_ndims=2,
            ),
            "w": tfd.Independent(
                tfd.HalfNormal(scale=jnp.ones((1, self.feature_dim), dtype=self.dtype)),
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
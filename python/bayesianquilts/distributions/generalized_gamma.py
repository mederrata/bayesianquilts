# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""The GeneralizedGamma distribution class."""

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax.distributions import distribution
from tensorflow_probability.substrates.jax.bijectors import softplus as softplus_bijector
from tensorflow_probability.substrates.jax.distributions import kullback_leibler
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.substrates.jax.internal import (
    assert_util, dtype_util, parameter_properties, prefer_static, tensor_util)
from jax import random
from jax.scipy import special as jax_special
__all__ = [
    'GeneralizedGamma',
    '_kl_ggamma_ggamma'
]


class GeneralizedGamma(distribution.Distribution):
    """Generalized Gamma distribution

    The Generalized Gamma generalizes the Gamma
    distribution with an additional exponent parameter. It is parameterized by
    location `loc`, scale `scale` and shape `power`.

    #### Mathematical details

    Following the wikipedia parameterization 
    https://en.wikipedia.org/wiki/Generalized_gamma_distribution
    f(x; a=scale, d=concentration, p=exponent) = 
      \frac{(p/a^d) x^{d-1} e^{-(x/a)^p}}{\Gamma(d/p)}
    """

    def __init__(self,
                 scale, concentration, exponent,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='GeneralizedGamma'):
        parameters = dict(locals())
        with tfp.util.name_scope(name) as name:
            super(GeneralizedGamma, self).__init__(
                dtype=dtype_util.common_dtype(
                    [scale, concentration, exponent], dtype_hint=jnp.float32),
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                reparameterization_type=(
                    reparameterization.FULLY_REPARAMETERIZED
                ),
                parameters=parameters,
                name=name)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            concentration=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
            scale=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
            exponent=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
        # pylint: enable=g-long-lambda

    def _mean(self):
        return self.scale * jnp.exp(
            jax_special.gammaln((self.concentration + 1.)/self.exponent)
            - jax_special.gammaln(self.concentration/self.exponent)
        )

    def _variance(self):
        return self.scale**2 * (
            jnp.exp(
                jax_special.gammaln((self.concentration+2.)/self.exponent)
                - jax_special.gammaln(self.concentration/self.exponent)
            )
            - jnp.exp(
                2*(
                    jax_special.gammaln((self.concentration+1.)/self.exponent)
                    - jax_special.gammaln(self.concentration/self.exponent)
                )

            )
        )

    def _cdf(self, x):
        return jax_special.gammainc(self.concentration/self.exponent,
                              (x/self.scale)**self.exponent) * jnp.exp(
            -jax_special.gammaln(self.concentration/self.exponent)
        )

    def _log_prob(self, x):
        log_unnormalized_prob = (
            jax_special.xlogy(self.concentration-1., x) - (x/self.scale)**self.exponent)
        log_prefactor = (
            jnp.log(self.exponent) - jax_special.xlogy(self.concentration, self.scale)
            - jax_special.gammaln(self.concentration/self.exponent))
        return log_unnormalized_prob + log_prefactor

    def _entropy(self):
        scale = jnp.asarray(self.scale)
        concentration = jnp.asarray(self.concentration)
        exponent = jnp.asarray(self.exponent)
        return (
            jnp.log(scale) + jax_special.gammaln(concentration/exponent)
            - jnp.log(exponent) + concentration/exponent
            + (1.0 - concentration)/exponent *
            jax_special.digamma(concentration/exponent)
        )

    def _stddev(self):
        return jnp.sqrt(self._variance())

    def _mode(self):
        concentration = jnp.asarray(self.concentration)
        exponent = jnp.asarray(self.exponent)
        scale = jnp.asarray(self.scale)
        mode = scale*jnp.power(
            (concentration - 1.)/exponent,
            1./exponent
        )
        mode = jnp.where(
            concentration > 1.,
            mode,
            jnp.zeros_like(mode)
        )
        return mode

    def _default_event_space_bijector(self):
        return softplus_bijector.Softplus(validate_args=self.validate_args)

    @property
    def scale(self):
        return self.parameters['scale']

    @property
    def concentration(self):
        return self.parameters['concentration']

    @property
    def exponent(self):
        return self.parameters['exponent']

    def _batch_shape(self):
        return jnp.broadcast_shapes(
            self.scale.shape,
            self.concentration.shape,
            self.exponent.shape
        )

    def _event_shape_tensor(self):
        return jnp.array([], dtype=jnp.int32)

    def _sample_n(self, n, seed=None):
        """Sample based on transforming Gamma RVs
        Arguments:
          n {int} -- [description]
        Keyword Arguments:
          seed {int} -- [description] (default: {None})
        Returns:
          [type] -- [description]
        """
        gamma_samples = random.gamma(
            key=seed,
            a=self.concentration/self.exponent,
            shape=(n,),
            dtype=self.dtype
        )
        ggamma_samples = (
            self.scale*jnp.exp(jnp.log(gamma_samples)/self.exponent)
        )
        return ggamma_samples

    def _event_shape(self):
        return ()

    def _sample_control_dependencies(self, x):
        assertions = []
        if not self.validate_args:
            return assertions
        assertions.append(assert_util.assert_non_negative(
            x, message='Sample must be non-negative.'))
        return assertions

    def _parameter_control_dependencies(self, is_init):
        if not self.validate_args:
            return []
        assertions = []
        if is_init != tensor_util.is_ref(self.scale):
            assertions.append(assert_util.assert_positive(
                self.scale,
                message='Argument `scale` must be positive.'))
        if is_init != tensor_util.is_ref(self.concentration):
            assertions.append(assert_util.assert_positive(
                self.concentration,
                message='Argument `concentration` must be positive.'))
        if is_init != tensor_util.is_ref(self.exponent):
            assertions.append(assert_util.assert_positive(
                self.exponent,
                message='Argument `exponent` must be positive.'))
        return assertions


@kullback_leibler.RegisterKL(GeneralizedGamma, GeneralizedGamma)
def _kl_ggamma_ggamma(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b GeneralizedGamma.

  Args:
    a: instance of a GeneralizedGamma distribution object.
    b: instance of a GeneralizedGamma distribution object.
    name: (optional) Name to use for created operations.
      Default is '_kl_ggamma_ggamma'.

  Returns:
    Batchwise KL(a || b)

  Raises:
    TypeError: If `a` and `b` are not `GeneralizedGamma` distributions.

  #### References
  [1] C. Bauckhage and A. B. T. H. (2013). The Blitzstein-Diaconis-GAN-Estimator
      for the KL-Divergence between two-parameter Gamma-Distributions.
      arXiv:1310.3713 [cs.IT]
  """
  with tfp.util.name_scope(name or '_kl_ggamma_ggamma'):
        # Result from https://arxiv.org/pdf/1310.3713.pdf
        a_concentration = jnp.asarray(a.concentration)
        b_concentration = jnp.asarray(b.concentration)
        a_scale = jnp.asarray(a.scale)
        b_scale = jnp.asarray(b.scale)
        a_exponent = jnp.asarray(a.exponent)
        b_exponent = jnp.asarray(b.exponent)

        return (
            jnp.log(a_exponent) - jnp.log(b_exponent)
            + b_concentration*jnp.log(b_scale) -
            a_concentration*jnp.log(a_scale)
            + jax_special.gammaln(b_concentration/b_exponent) -
            jax_special.gammaln(a_concentration/a_exponent)
            + (a_concentration - b_concentration)*(jax_special.digamma(a_concentration /
                                                                   a_exponent)/a_exponent + jnp.log(a_scale))
            + jnp.exp(jax_special.gammaln(
                (a_concentration + b_exponent)/a_exponent
            )-jax_special.gammaln(a_concentration/a_exponent) * jnp.power(a_scale/b_scale, b_exponent))
            - a_concentration/a_exponent
        )


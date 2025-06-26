import tensorflow_probability as tfp
from tensorflow_probability.python.bijectors import \
    identity as identity_bijector
from tensorflow_probability.python.bijectors import \
    softplus as softplus_bijector
from tensorflow_probability.python.distributions import (
    JointDistributionNamed, TransformedDistribution, distribution,
    kullback_leibler)
from tensorflow_probability.python.internal import (assert_util,
                                                    distribution_util,
                                                    dtype_util,
                                                    parameter_properties,
                                                    prefer_static,
                                                    reparameterization,
                                                    tensor_util)
from tensorflow_probability.substrates.jax import tf2jax as tf

tfd = tfp.distributions
tfb = tfp.bijectors

convert_nonref_to_tensor = tensor_util.convert_nonref_to_tensor


class AbsHorseshoe(TransformedDistribution):
    def __init__(
        self, scale, validate_args=False, allow_nan_stats=True, name="AbsHorseshoe"
    ):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super(AbsHorseshoe, self).__init__(
                distribution=tfd.Horseshoe(scale=scale),
                bijector=tfb.AbsoluteValue(),
                validate_args=validate_args,
                parameters=parameters,
                name=name,
            )

    @classmethod
    def _params_event_ndims(cls):
        return dict(scale=0)

    @property
    def scale(self):
        """Distribution parameter for the
        pre-transformed standard deviation."""
        return self.distribution.scale

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            scale=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(
                        low=dtype_util.eps(dtype))
                )
            )
        )


class SoftplusHorseshoe(TransformedDistribution):
    def __init__(
        self, scale, validate_args=False, allow_nan_stats=True, name="SoftplusHorseshoe"
    ):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super(SoftplusHorseshoe, self).__init__(
                distribution=tfd.Horseshoe(scale=scale),
                bijector=tfb.Softplus(),
                validate_args=validate_args,
                parameters=parameters,
                name=name,
            )

    @classmethod
    def _params_event_ndims(cls):
        return dict(scale=0)

    @property
    def scale(self):
        """Distribution parameter for the
        pre-transformed standard deviation."""
        return self.distribution.scale

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            scale=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(
                        low=dtype_util.eps(dtype))
                )
            )
        )

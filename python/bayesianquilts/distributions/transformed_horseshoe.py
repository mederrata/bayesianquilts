import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.bijectors import softplus as softplus_bijector
from tensorflow_probability.substrates.jax.internal import (
    dtype_util, parameter_properties)



class AbsHorseshoe(tfd.TransformedDistribution):
    def __init__(
        self, scale, validate_args=False, allow_nan_stats=True, name="AbsHorseshoe"
    ):
        parameters = dict(locals())
        super(AbsHorseshoe, self).__init__(
            distribution=tfd.Horseshoe(scale=scale),
            bijector=tfb.AbsoluteValue(),
            validate_args=validate_args,
            parameters=parameters,
            name=name,
        )

    @property
    def scale(self):
        """Distribution parameter for the
        pre-transformed standard deviation."""
        return self.parameters["scale"]

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


class SoftplusHorseshoe(tfd.TransformedDistribution):
    def __init__(
        self, scale, validate_args=False, allow_nan_stats=True, name="SoftplusHorseshoe"
    ):
        parameters = dict(locals())
        super(SoftplusHorseshoe, self).__init__(
            distribution=tfd.Horseshoe(scale=scale),
            bijector=tfb.Softplus(),
            validate_args=validate_args,
            parameters=parameters,
            name=name,
        )

    @property
    def scale(self):
        """Distribution parameter for the
        pre-transformed standard deviation."""
        return self.parameters["scale"]

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


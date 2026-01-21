import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.bijectors import softplus as softplus_bijector
from tensorflow_probability.substrates.jax.internal import (
    dtype_util, parameter_properties)



class SqrtInverseGamma(tfd.TransformedDistribution):
    def __init__(
        self,
        concentration,
        scale,
        validate_args=False,
        allow_nan_stats=True,
        name="SqrtInverseGamma",
    ):
        parameters = dict(locals())
        super(SqrtInverseGamma, self).__init__(
            distribution=tfd.InverseGamma(
                concentration=concentration, scale=scale),
            bijector=tfb.Power(power=0.5),
            validate_args=validate_args,
            parameters=parameters,
            name=name,
        )

    @property
    def concentration(self):
        """Distribution parameter for the pre-transformed concentration."""
        return self.parameters["concentration"]

    @property
    def scale(self):
        """Distribution parameter for the
        pre-transformed standard deviation."""
        return self.parameters["scale"]

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            concentration=parameter_properties.ParameterProperties(),
            scale=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(
                        low=dtype_util.eps(dtype))
                )
            ),
        )


import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.bijectors import softplus as softplus_bijector
from tensorflow_probability.substrates.jax.internal import (
    dtype_util, parameter_properties)



class SqrtCauchy(tfd.TransformedDistribution):
    def __init__(
        self, loc, scale, validate_args=False, allow_nan_stats=True, name="SqrtCauchy"
    ):
        parameters = dict(locals())
        super(SqrtCauchy, self).__init__(
            distribution=tfd.HalfCauchy(loc=loc, scale=scale),
            bijector=tfb.Sqrt(),
            validate_args=validate_args,
            parameters=parameters,
            name=name,
        )

    @property
    def loc(self):
        """Distribution parameter for the pre-transformed mean."""
        return self.parameters["loc"]

    @property
    def scale(self):
        """Distribution parameter for the
        pre-transformed standard deviation."""
        return self.parameters["scale"]

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            loc=parameter_properties.ParameterProperties(),
            scale=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(
                        low=dtype_util.eps(dtype))
                )
            ),
        )


class LogHalfCauchy(tfd.TransformedDistribution):
    """Exponent of RV follows a HalfCauchy distribution

    log(x) ~ HalfCauchy

    Arguments:
        TransformedDistribution {[type]} -- [description]

    Returns:
        tfp.distribution -- [description]
    """

    def __init__(
        self,
        loc,
        scale,
        validate_args=False,
        allow_nan_stats=True,
        name="LogHalfCauchy",
    ):
        parameters = dict(locals())
        super(LogHalfCauchy, self).__init__(
            distribution=tfd.HalfCauchy(loc=loc, scale=scale),
            bijector=tfb.Exp(),
            validate_args=validate_args,
            parameters=parameters,
            name=name,
        )

    @property
    def loc(self):
        """Distribution parameter for the pre-transformed mean."""
        return self.parameters["loc"]

    @property
    def scale(self):
        """Distribution parameter for the pre-transformed
        standard deviation."""
        return self.parameters["scale"]

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            loc=parameter_properties.ParameterProperties(),
            scale=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(
                        low=dtype_util.eps(dtype))
                )
            ),
        )


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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions import distribution

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.internal import parameter_properties

from tensorflow_probability.python.distributions import kullback_leibler

import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.bijectors as tfb

from tensorflow.python.ops.math_ops import bucketize


class PiecewiseExponential(tfd.Distribution):
    def __init__(
        self,
        rates,
        breakpoints,
        validate_args=False,
        allow_nan_stats=True,
        name="PiecewiseExponential",
    ):

        parameters = dict(locals())
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype(
                [rates, breakpoints], dtype_hint=tf.float32)
            self._rates = tensor_util.convert_nonref_to_tensor(
                rates, dtype=dtype, name='rates')
            self._breakpoints = tensor_util.convert_nonref_to_tensor(
                breakpoints, dtype=dtype, name='breakpoints')
            super(PiecewiseExponential, self).__init__(
                dtype=dtype,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                reparameterization_type=(
                    reparameterization.FULLY_REPARAMETERIZED
                ),
                parameters=parameters,
                name=name)

        # compute cumulative masses
        time_gaps = (
            self._breakpoints[..., 1:] - self._breakpoints[..., :-1]
        )
        time_gaps = tf.concat(
            [self._breakpoints[..., 0][..., tf.newaxis], time_gaps],
            axis=-1)
        masses = self._rates[..., :-1] * time_gaps
        self.cum_hazards = tf.cumsum(masses, axis=-1)

    @property
    def rates(self):
        return self._rates

    @property
    def breakpoints(self):
        return self._breakpoints

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            rates=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
            breakpoints=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        )

    def hazard(self, value, **kwargs):
        """Query self.rates to determine the hazard. This function
        takes care of broadcasting between

        self._breakpoints is of shape param_batch_shape x n_breaks
        self._rates is of shape param_batch_shape x (n_breaks - 1)

        Args:
            value (tf.Tensor): Tensor of shape data_batch_shape

        Returns:
            hazard (tf.Tensor): Tensor of shape
                param_batch_shape x data_batch_shape


        """
        value = tf.convert_to_tensor(value)
        value_batch_shape = value.shape.as_list()
        if len(value_batch_shape) == 0:
            value = value[tf.newaxis]
            value_batch_shape = value.shape.as_list()

        breakpoint_batch_shape = self._breakpoints.shape.as_list()[:-1]
        rate_batch_shape = self.rates.shape.as_list()[:-1]

        if len(breakpoint_batch_shape) > len(rate_batch_shape):
            breakpoints = self._breakpoints
            rates = self.rates[tf.newaxis, ...]
        elif len(breakpoint_batch_shape) < len(rate_batch_shape):
            breakpoints = self._breakpoints[tf.newaxis, ...]
            rates = self.rates
        else:
            breakpoints = self._breakpoints
            rates = self.rates

        if len(value_batch_shape) > 0:
            rates = rates[..., tf.newaxis, :]
            breakpoints = breakpoints[..., tf.newaxis, :]

        if len(rate_batch_shape) + len(breakpoint_batch_shape) > 0:
            value = value[tf.newaxis, ..., tf.newaxis]

        indices = tf.reduce_sum(
            tf.cast(value >= breakpoints, tf.int32),
            axis=-1)
        hazards = tf.gather_nd(
            tf.tile(rates, [1, value_batch_shape[0], 1]),
            indices[..., tf.newaxis], batch_dims=2
        )
        return hazards

    def cumulative_hazard(self, value, ret_hazard=False, **kwargs):
        """Get the cumulative hazard. This function
        takes care of broadcasting between

        self._breakpoints is of shape param_batch_shape x n_breaks
        self._rates is of shape param_batch_shape x (n_breaks - 1)

        Args:
            value (tf.Tensor): Tensor of shape data_batch_shape

        Returns:
            hazard (tf.Tensor): Tensor of shape
                param_batch_shape x data_batch_shape


        """
        value = tf.convert_to_tensor(value)
        try:
            value_batch_shape = value.shape.as_list()
        except ValueError:
            value_batch_shape = []
        if len(value_batch_shape) == 0:
            value = value[tf.newaxis]
            value_batch_shape = [1]

        breakpoint_batch_shape = self._breakpoints.shape.as_list()[:-1]
        rate_batch_shape = self.rates.shape.as_list()[:-1]

        if len(breakpoint_batch_shape) > len(rate_batch_shape):
            breakpoints = self._breakpoints
            rank_diff = len(breakpoint_batch_shape) - len(rate_batch_shape)
            rates = rates[rank_diff * (tf.newaxis,) + (...,)]
            # rates = self.rates[tf.newaxis, ...]
        elif len(breakpoint_batch_shape) < len(rate_batch_shape):
            rates = self.rates
            rank_diff = len(self.rates.shape.as_list()) - \
                len(self._breakpoints.shape.as_list())
            breakpoints = self._breakpoints[rank_diff * (tf.newaxis,) + (...,)]
            # breakpoints = self._breakpoints[tf.newaxis, ...]

        else:
            breakpoints = self._breakpoints
            rates = self.rates

        cum_hazard = self.cum_hazards
        if (len(value_batch_shape) > 0) and not (rate_batch_shape[-1] == value_batch_shape[-1]):
            rates = rates[..., tf.newaxis, :]
            breakpoints = breakpoints[..., tf.newaxis, :]
            cum_hazard = cum_hazard[..., tf.newaxis, :]

        if len(rate_batch_shape) + len(breakpoint_batch_shape) > 0:
            value = value[tf.newaxis, ..., tf.newaxis]

        indicator = tf.cast(value >= breakpoints, tf.int32)
        indices = tf.reduce_sum(
            indicator,
            axis=-1)
        try:
            middle_dims = [value_batch_shape[0]
                           ] if value_batch_shape[0] is not None else [1]
        except IndexError:
            middle_dims = [1]
        hazards = tf.gather_nd(
            tf.tile(rates, [1] + middle_dims + [1]),
            indices[..., tf.newaxis], batch_dims=2
        )

        breakpoints = tf.tile(breakpoints, [1] + middle_dims + [1])
        breakpoints = tf.concat(
            [tf.zeros((breakpoints.shape[0], value_batch_shape[0], 1),
                      dtype=breakpoints.dtype), breakpoints],
            axis=-1)
        changepoints = tf.gather_nd(
            breakpoints,
            indices[..., tf.newaxis], batch_dims=2
        )

        cum_hazard = tf.reduce_sum(
            tf.cast(indicator, cum_hazard.dtype)
            * cum_hazard,
            axis=-1)

        cum_hazard += hazards*(value[..., 0]-changepoints)
        if ret_hazard:
            return cum_hazard, hazards
        return cum_hazard

    def _log_survival_function(self, value, **kwargs):
        return -self.cumulative_hazard(value, **kwargs)

    def _log_prob(self, value, **kwargs):
        cum, haz = self.cumulative_hazard(value, ret_hazard=True)

        return tf.math.log(haz) - cum

    def _cdf(self, value, name='cdf', **kwargs):
        return 1.0 - tf.math.exp(self._log_survival_function(value))


def demo():
    pe = PiecewiseExponential(
        rates=[[1, 2, 1], [1, 3, 1]], breakpoints=[[2, 8], [3, 8]])
    h = pe.hazard([1, 2.5, 10, 4.5])
    print(h)
    ch = pe.cumulative_hazard([1, 2.5, 10, 4.5])
    print(ch)
    pr = pe.log_prob([1, 2.5, 10, 4.5])
    print(pr)

    batched_rates = [[[1, 2, 1], [1, 3, 1]], [[1, 2, 1], [1, 3, 1]]]
    # @TODO make this class batch-safe
    pass


if __name__ == "__main__":
    demo()

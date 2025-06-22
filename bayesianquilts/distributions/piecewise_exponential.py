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

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd
from tensorflow.python.ops.math_ops import bucketize
from tensorflow_probability.python.bijectors import \
    softplus as softplus_bijector
from tensorflow_probability.python.distributions import (distribution,
                                                         kullback_leibler)
from tensorflow_probability.python.internal import (assert_util,
                                                    distribution_util,
                                                    dtype_util,
                                                    parameter_properties,
                                                    prefer_static,
                                                    reparameterization,
                                                    tensor_util)


class PiecewiseExponential(tfd.Distribution):
    def __init__(
        self,
        rates=None,
        breakpoints=None,
        hazard_fn=None,
        validate_args=False,
        allow_nan_stats=True,
        name="PiecewiseExponential",
    ):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([rates, breakpoints], dtype_hint=tf.float32)
            self._rates = tensor_util.convert_nonref_to_tensor(
                rates, dtype=dtype, name="rates"
            )
            self._breakpoints = tensor_util.convert_nonref_to_tensor(
                breakpoints, dtype=dtype, name="breakpoints"
            )
            super(PiecewiseExponential, self).__init__(
                dtype=dtype,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                reparameterization_type=(reparameterization.FULLY_REPARAMETERIZED),
                parameters=parameters,
                name=name,
            )

        params = self.broadcast_to_params(tile=True)
        rates = params["rates"]
        breakpoints = params["breakpoints"]
        # compute cumulative masses
        time_gaps = breakpoints[..., 1:] - breakpoints[..., :-1]
        time_gaps = tf.concat(
            [breakpoints[..., 0][..., tf.newaxis], time_gaps], axis=-1
        )
        masses = rates[..., :-1] * time_gaps
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
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))
                )
            ),
            breakpoints=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))
                )
            ),
        )

    def broadcast_to_params(self, value=None, tile=False):
        """Reshape/broadcast different variations of the internal parameters
        and the value

        self._breakpoints is of shape param_batch_shape x n_breaks
        self._rates is of shape param_batch_shape x (n_breaks - 1)
            param_batch_shape might also have an axis common with value

        Args:
            value (tf.Tensor): Tensor of shape data_batch_shape

        Returns:
            {
                'value': Tensor of shape data_batch_shape
                'rates': Tensor of shape sample_shape x data_batch_shape x n_breaks
                'breakpoints': Tensor of shape sample_shape x data_batch_shape x (n_breaks - 1)
            }
        """

        breakpoint_batch_shape = self._breakpoints.shape.as_list()[:-1]
        rate_batch_shape = self.rates.shape.as_list()[:-1]

        if len(breakpoint_batch_shape) > len(rate_batch_shape):
            breakpoints = self._breakpoints
            rank_diff = len(breakpoint_batch_shape) - len(rate_batch_shape)
            rates = rates[rank_diff * (tf.newaxis,) + (...,)]
        elif len(breakpoint_batch_shape) < len(rate_batch_shape):
            rates = self.rates
            rank_diff = len(self.rates.shape.as_list()) - len(
                self._breakpoints.shape.as_list()
            )
            breakpoints = self._breakpoints[rank_diff * (tf.newaxis,) + (...,)]

        else:
            breakpoints = self._breakpoints
            rates = self.rates

        rate_batch_shape = rates.shape.as_list()[:-1]
        breakpoint_batch_shape = breakpoints.shape.as_list()[:-1]
        # tile rates and breakpoints so they have the same batch shape

        if tile:
            rate_tile = []
            breakpoint_tile = []
            for r, d in zip(rate_batch_shape, breakpoint_batch_shape):
                if r is None or d is None:
                    rate_tile += [1]
                    breakpoint_tile += [1]
                else:
                    if r >= d:
                        breakpoint_tile += [int(r / d)]
                        rate_tile += [1]
                    else:
                        breakpoint_tile += [1]
                        rate_tile += [int(d / r)]
            rate_tile += [1]
            breakpoint_tile += [1]
            rates = tf.tile(rates, rate_tile)
            breakpoints = tf.tile(breakpoints, breakpoint_tile)

        if value is None:
            return {"rates": rates, "breakpoints": breakpoints}

        rate_batch_shape = rates.shape.as_list()[:-1]
        breakpoint_batch_shape = breakpoints.shape.as_list()[:-1]
        value = tf.squeeze(tf.convert_to_tensor(value))
        try:
            data_batch_shape = value.shape.as_list()
        except ValueError:
            data_batch_shape = [1]

        cond1 = data_batch_shape == breakpoint_batch_shape[-(len(data_batch_shape)) :]
        cond2 = data_batch_shape == rate_batch_shape[-(len(data_batch_shape)) :]
        if cond1 or cond2:
            # we are possibly overlapping already so we'll broadcast as is
            pass
        else:
            # need to insert dimensions for broadcasting
            rates = rates[(...,) + len(data_batch_shape) * (tf.newaxis,)]
            permutation = list(range(len(rates.shape.as_list())))
            permutation.remove(len(rate_batch_shape))
            permutation.append(len(rate_batch_shape))
            rates = tf.transpose(rates, permutation)
            breakpoints = breakpoints[(...,) + len(data_batch_shape) * (tf.newaxis,)]
            breakpoints = tf.transpose(breakpoints, permutation)

            if tile:
                rates = tf.tile(
                    rates, [1] * len(rate_batch_shape) + data_batch_shape + [1]
                )
                breakpoints = tf.tile(
                    breakpoints, [1] * len(rate_batch_shape) + data_batch_shape + [1]
                )

        return {"value": value, "rates": rates, "breakpoints": breakpoints}

    def hazard(self, value, **kwargs):
        """Query self.rates to determine the hazard. This function
        takes care of broadcasting between

        self._breakpoints is of shape param_batch_shape x n_breaks
        self._rates is of shape param_batch_shape x (n_breaks - 1)
            param_batch_shape might also have an axis common with value

        Args:
            value (tf.Tensor): Tensor of shape data_batch_shape

        Returns:
            hazard (tf.Tensor): Tensor of shape
                param_batch_shape x data_batch_shape


        """
        reshaped = self.broadcast_to_params(value, tile=True)
        value = reshaped["value"]
        rates = reshaped["rates"]
        breakpoints = reshaped["breakpoints"]

        indices = tf.reduce_sum(
            tf.cast(value[..., tf.newaxis] >= breakpoints, tf.int32), axis=-1
        )
        hazards = tf.gather_nd(
            rates, indices[..., tf.newaxis], batch_dims=(len(rates.shape.as_list()) - 1)
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
        reshaped = self.broadcast_to_params(value, tile=True)
        value = reshaped["value"]
        rates = reshaped["rates"]
        breakpoints = reshaped["breakpoints"]
        indicator = tf.cast(value[..., tf.newaxis] > breakpoints, tf.int32)
        indices = tf.reduce_sum(indicator, axis=-1)
        hazards = tf.gather_nd(
            rates, indices[..., tf.newaxis], batch_dims=(len(rates.shape.as_list()) - 1)
        )
        # @TODO Fix this

        breakpoints = tf.pad(
            breakpoints, [(0, 0)] * (len(breakpoints.shape.as_list()) - 1) + [(1, 0)]
        )
        changepoints = tf.gather_nd(
            breakpoints,
            indices[..., tf.newaxis],
            batch_dims=(len(breakpoints.shape.as_list()) - 1),
        )
        try:
            if len(self.cum_hazards.shape.as_list()) < len(indicator.shape.as_list()):
                cum_hazard = self.cum_hazards[..., tf.newaxis, :]
            else:
                cum_hazard = self.cum_hazards
        except ValueError:
            cum_hazard = self.cum_hazards
        cum_hazard = tf.reduce_sum(
            tf.cast(indicator, cum_hazard.dtype)
            * (cum_hazard + jnp.zeros_like(tf.cast(indicator, cum_hazard.dtype))),
            axis=-1,
        )

        cum_hazard += hazards * (value - changepoints)
        if ret_hazard:
            return cum_hazard, hazards
        return cum_hazard

    def _log_survival_function(self, value, **kwargs):
        return -self.cumulative_hazard(value, **kwargs)

    def _log_prob(self, value, **kwargs):
        cum, haz = self.cumulative_hazard(value, ret_hazard=True)

        return tf.math.log(haz) - cum

    def _cdf(self, value, name="cdf", **kwargs):
        return 1.0 - tf.math.exp(self._log_survival_function(value))


class PiecewiseExponentialRaggedBreaks(PiecewiseExponential):
    def __init__(
        self,
        rates,
        breakpoints,
        validate_args=False,
        allow_nan_stats=True,
        name="PiecewiseExponentialRaggedBreaks",
    ):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            try:
                dtype = dtype_util.common_dtype(
                    [rates, breakpoints], dtype_hint=tf.float32
                )
            except TypeError:
                dtype = rates.dtype
            if isinstance(rates, tf.RaggedTensor):
                self._rates = rates
            else:
                self._rates = tf.ragged.constant(rates, dtype=dtype, name="rates")
            if isinstance(breakpoints, tf.RaggedTensor):
                self._breakpoints = breakpoints
            else:
                self._breakpoints = tf.ragged.constant(
                    breakpoints, dtype=dtype, name="breakpoints"
                )
            super(PiecewiseExponential, self).__init__(
                dtype=dtype,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                reparameterization_type=(reparameterization.FULLY_REPARAMETERIZED),
                parameters=parameters,
                name=name,
            )
        try:
            params = self.broadcast_to_params(tile=True)
            rates = params["rates"]
        except ValueError:
            pass
        breakpoints = params["breakpoints"]

        def get_cumhazards(x):
            b = x[0]
            if isinstance(b, tf.RaggedTensor):
                b = b.to_tensor()
            r = x[1]
            if isinstance(r, tf.RaggedTensor):
                r = r.to_tensor()
            b = tf.concat(
                [b[..., 0][..., tf.newaxis], b[..., 1:] - b[..., :-1]], axis=-1
            )
            masses = r[..., :-1] * tf.cast(b, rates.dtype)
            hazards = tf.cumsum(masses, axis=-1)
            hazards = tf.RaggedTensor.from_tensor(hazards, ragged_rank=1)
            return hazards

        # compute cumulative masses
        self.cum_hazards = tf.map_fn(
            get_cumhazards,
            (breakpoints, rates),
            dtype=(breakpoints.dtype, rates.dtype),
            fn_output_signature=tf.RaggedTensorSpec(
                shape=breakpoints.shape.as_list()[1:], dtype=rates.dtype
            ),
        )

        return

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            rates=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))
                )
            ),
            breakpoints=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))
                )
            ),
        )

    def hazard(self, value, **kwargs):
        """Query self.rates to determine the hazard. This function
        takes care of broadcasting between

        self._breakpoints is of shape param_batch_shape x n_breaks
        self._rates is of shape param_batch_shape x (n_breaks - 1)
            param_batch_shape might also have an axis common with value

        Args:
            value (tf.Tensor): Tensor of shape data_batch_shape

        Returns:
            hazard (tf.Tensor): Tensor of shape
                param_batch_shape x data_batch_shape


        """

        rates = self.rates
        breakpoints = self.breakpoints

        batch_shape = rates.shape.as_list()[1:-1]

        # x = (value[0], breakpoints[0], rates[0])
        def get_hazard(x):
            v, b, r = x
            if isinstance(v, tf.RaggedTensor):
                v = v.to_tensor()
            if isinstance(b, tf.RaggedTensor):
                b = b.to_tensor()
            indices = tf.reduce_sum(tf.cast(v[..., tf.newaxis] >= b, tf.int32), axis=-1)
            hazards = tf.gather(tf.transpose(r.to_tensor()), indices)[0, ...]
            return hazards

        hazards = tf.map_fn(
            get_hazard,
            (value, breakpoints, rates),
            dtype=(value.dtype, breakpoints.dtype, rates.dtype),
            fn_output_signature=tf.TensorSpec(shape=batch_shape, dtype=rates.dtype),
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
        reshaped = self.broadcast_to_params(value, tile=True)
        rates = reshaped["rates"]
        breakpoints = reshaped["breakpoints"]

        indicator = tf.map_fn(
            lambda x: tf.cast(x[0][..., tf.newaxis] >= x[1], tf.int32),
            (value, breakpoints),
            dtype=(value.dtype, breakpoints.dtype),
            fn_output_signature=tf.RaggedTensorSpec(
                shape=breakpoints.shape.as_list()[1:], dtype=tf.int32
            ),
        )
        indices = tf.reduce_sum(indicator, axis=-1)
        hazards = tf.gather(rates, indices, batch_dims=(len(rates.shape.as_list()) - 1))
        hazards = hazards.to_tensor()

        flat = tf.RaggedTensor.from_row_starts(
            breakpoints.flat_values, breakpoints.values.row_starts()
        )
        flat = tf.map_fn(
            lambda x: tf.pad(x, [[1, 0]]),
            flat,
            flat.dtype,
            fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=flat.dtype),
        )
        changepoints = tf.gather(
            flat,
            indices.flat_values,
            batch_dims=(len(flat.shape.as_list()) - 1),
        )

        changepoints = tf.reshape(changepoints, indices.shape)
        try:
            if len(self.cum_hazards.shape.as_list()) < len(indicator.shape.as_list()):
                cum_hazard = self.cum_hazards[tf.newaxis, ...]
            else:
                cum_hazard = self.cum_hazards
        except ValueError:
            cum_hazard = self.cum_hazards
        cum_hazard = tf.reduce_sum(
            tf.cast(indicator, cum_hazard.dtype)
            * (cum_hazard + jnp.zeros_like(tf.cast(indicator, cum_hazard.dtype))),
            axis=-1,
        )
        cum_hazard = cum_hazard.to_tensor()

        cum_hazard += hazards * (value[..., tf.newaxis] - changepoints)
        if ret_hazard:
            return cum_hazard, hazards
        return cum_hazard

    def broadcast_to_params(self, value=None, tile=False):
        breakpoint_batch_shape = self._breakpoints.shape.as_list()[:-1]
        rate_batch_shape = self.rates.shape.as_list()[:-1]
        if len(breakpoint_batch_shape) > len(rate_batch_shape):
            breakpoints = self._breakpoints
            rank_diff = len(breakpoint_batch_shape) - len(rate_batch_shape)
            rates = tf.map_fn(
                lambda x: x[rank_diff * (tf.newaxis,) + (...,)],
                rates,
                dtype=rates.dtype,
                fn_output_signature=tf.RaggedTensorSpec(
                    shape=[None], dtype=rates.dtype
                ),
            )
        elif len(breakpoint_batch_shape) < len(rate_batch_shape):
            rates = self.rates
            rank_diff = len(self.rates.shape.as_list()) - len(
                self._breakpoints.shape.as_list()
            )
            breakpoints = self._breakpoints[rank_diff * (tf.newaxis,) + (...,)]

        else:
            breakpoints = self._breakpoints
            rates = self.rates

        rate_batch_shape = rates.shape.as_list()[:-1]
        breakpoint_batch_shape = breakpoints.shape.as_list()[:-1]

        # tile rates and breakpoints so they have the same batch shape

        if tile:
            rate_tile = []
            breakpoint_tile = []
            for r, d in zip(rate_batch_shape, breakpoint_batch_shape):
                if r is None or d is None:
                    rate_tile += [1]
                    breakpoint_tile += [1]
                else:
                    if r >= d:
                        breakpoint_tile += [int(r / d)]
                        rate_tile += [1]
                    else:
                        breakpoint_tile += [1]
                        rate_tile += [int(d / r)]
            if not (np.prod(rate_tile) == 1 and np.prod(breakpoint_tile) == 1):
                rate_tile += [1]
                breakpoint_tile += [1]
                if np.prod(rate_tile) != 1:
                    rates = tf.map_fn(
                        lambda x: tf.tile(x, rate_tile),
                        rates,
                        dtype=rates.dtype,
                        fn_output_signature=tf.RaggedTensorSpec(
                            shape=[None], dtype=rates.dtype
                        ),
                    )
                if np.prod(breakpoint_tile) != 1:
                    breakpoints = tf.map_fn(
                        lambda x: tf.tile(x, breakpoint_tile[1:]),
                        breakpoints,
                        dtype=breakpoints.dtype,
                        fn_output_signature=tf.RaggedTensorSpec(
                            shape=breakpoint_tile[1:2] + [None], dtype=breakpoints.dtype
                        ),
                    )

        # if value is None:
        return {"rates": rates, "breakpoints": breakpoints}

        rate_batch_shape = rates.shape.as_list()[:-1]
        breakpoint_batch_shape = breakpoints.shape.as_list()[:-1]
        value = tf.squeeze(tf.convert_to_tensor(value))
        try:
            data_batch_shape = value.shape.as_list()
        except ValueError:
            data_batch_shape = [1]

        cond1 = data_batch_shape == breakpoint_batch_shape[-(len(data_batch_shape)) :]
        cond2 = data_batch_shape == rate_batch_shape[-(len(data_batch_shape)) :]
        if cond1 or cond2:
            # we are possibly overlapping already so we'll broadcast as is
            pass
        else:
            # need to insert dimensions for broadcasting
            # rates = rates[(...,) + len(data_batch_shape) * (tf.newaxis,)]
            # rates = rates[len(data_batch_shape) * (tf.newaxis,) + (...,)]
            # permutation = list(range(len(rates.shape.as_list())))
            # permutation.remove(len(rate_batch_shape))
            # permutation.append(len(rate_batch_shape))
            # rates = tf.transpose(rates, permutation)
            # breakpoints = breakpoints[(...,) + len(data_batch_shape) * (tf.newaxis,)]
            # breakpoints = breakpoints[len(data_batch_shape) * (tf.newaxis,) + (...,)]
            # breakpoints = tf.transpose(breakpoints, permutation)

            if tile:
                rates = tf.map_fn(
                    lambda x: rates,
                    value,
                    dtype=value.dtype,
                    fn_output_signature=tf.RaggedTensorSpec(
                        shape=rates.shape, dtype=value.dtype
                    ),
                )
                breakpoints = tf.map_fn(
                    lambda x: breakpoints,
                    value,
                    dtype=value.dtype,
                    fn_output_signature=tf.RaggedTensorSpec(
                        shape=breakpoints.shape, dtype=value.dtype
                    ),
                )
        return {"value": value, "rates": rates, "breakpoints": breakpoints}


def demo():
    print("Demo: one set of breakpoints to broadcast")
    pe = PiecewiseExponential(rates=[[1, 2, 1], [1, 3, 1]], breakpoints=[[2, 8]])
    h = pe.hazard([1, 2.5, 10, 4.5, 0.5, 3, 2])
    print(h)
    ch = pe.cumulative_hazard([1, 2.5, 10, 11, 12])
    print(ch)
    pr = pe.log_prob([1, 2.5, 10, 4.5])
    print(pr)

    print("Demo: one set of breakpoints for each rate")
    pe = PiecewiseExponentialRaggedBreaks(
        rates=[[1, 2, 1, 3], [1, 3, 1]], breakpoints=[[2, 8, 10], [3, 8]]
    )
    h = pe.hazard([1, 2.5, 10, 4.5, 0.5, 3, 2])
    print(h)
    ch = pe.cumulative_hazard([1, 2.5, 10, 4.5])
    print(ch)
    pr = pe.log_prob([1, 2.5, 10, 4.5])
    print(pr)

    print("Demo: one set of breakpoints for each rate")
    pe = PiecewiseExponential(
        rates=[[1, 2, 1], [1, 3, 1]], breakpoints=[[2, 8], [3, 8]]
    )
    h = pe.hazard([1, 2.5, 10, 4.5, 0.5, 3, 2])
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

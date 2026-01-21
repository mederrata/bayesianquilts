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
import numpy as np
from tensorflow_probability.substrates.jax.distributions import distribution
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.substrates.jax.bijectors import softplus as softplus_bijector
from tensorflow_probability.substrates.jax.internal import (
    dtype_util, parameter_properties, tensor_util)
from tensorflow_probability.substrates import jax as tfp

class PiecewiseExponential(distribution.Distribution):
    def __init__(
        self,
        rates=None,
        breakpoints=None,
        validate_args=False,
        allow_nan_stats=True,
        name="PiecewiseExponential",
    ):
        parameters = dict(locals())
        with tfp.util.name_scope(name) as name:
            super(PiecewiseExponential, self).__init__(
                dtype=dtype_util.common_dtype([rates, breakpoints], dtype_hint=jnp.float32),
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                reparameterization_type=(reparameterization.FULLY_REPARAMETERIZED),
                parameters=parameters,
                name=name,
            )

        rates = jnp.asarray(self.rates)
        breakpoints = jnp.asarray(self.breakpoints)
        # compute cumulative masses
        time_gaps = breakpoints[..., 1:] - breakpoints[..., :-1]
        time_gaps = jnp.concatenate(
            [breakpoints[..., 0][..., jnp.newaxis], time_gaps], axis=-1
        )
        masses = rates[..., :-1] * time_gaps
        self.cum_hazards = jnp.cumsum(masses, axis=-1)

    @property
    def rates(self):
        return self.parameters['rates']

    @property
    def breakpoints(self):
        return self.parameters['breakpoints']

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
            value (jnp.ndarray): Tensor of shape data_batch_shape

        Returns:
            hazard (jnp.ndarray): Tensor of shape
                param_batch_shape x data_batch_shape


        """
        value = jnp.asarray(value)
        rates = jnp.asarray(self.rates)
        breakpoints = jnp.asarray(self.breakpoints)

        indices = (value[..., jnp.newaxis] >= breakpoints).sum(axis=-1)
        hazards = jnp.take_along_axis(rates, indices[..., jnp.newaxis], axis=-1).squeeze(axis=-1)
        return hazards

    def cumulative_hazard(self, value, ret_hazard=False, **kwargs):
        """Get the cumulative hazard. This function
        takes care of broadcasting between

        self._breakpoints is of shape param_batch_shape x n_breaks
        self._rates is of shape param_batch_shape x (n_breaks - 1)

        Args:
            value (jnp.ndarray): Tensor of shape data_batch_shape

        Returns:
            hazard (jnp.ndarray): Tensor of shape
                param_batch_shape x data_batch_shape


        """
        value = jnp.asarray(value)
        rates = jnp.asarray(self.rates)
        breakpoints = jnp.asarray(self.breakpoints)
        indicator = (value[..., jnp.newaxis] > breakpoints).astype(jnp.int32)
        indices = indicator.sum(axis=-1)
        hazards = jnp.take_along_axis(rates, indices[..., jnp.newaxis], axis=-1).squeeze(axis=-1)

        padded_breakpoints = jnp.pad(
            breakpoints, [(0, 0)] * (len(breakpoints.shape) - 1) + [(1, 0)]
        )
        changepoints = jnp.take_along_axis(padded_breakpoints, indices[..., jnp.newaxis], axis=-1).squeeze(axis=-1)
        
        cum_hazard = (indicator.astype(self.cum_hazards.dtype) * self.cum_hazards).sum(axis=-1)

        cum_hazard += hazards * (value - changepoints)
        if ret_hazard:
            return cum_hazard, hazards
        return cum_hazard

    def _log_survival_function(self, value, **kwargs):
        return -self.cumulative_hazard(value, **kwargs)

    def _log_prob(self, value, **kwargs):
        cum, haz = self.cumulative_hazard(value, ret_hazard=True)

        return jnp.log(haz) - cum

    def _cdf(self, value, name="cdf", **kwargs):
        return 1.0 - jnp.exp(self._log_survival_function(value))




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
        self.hazard_sums = (
            self._breakpoints[..., 1:] - self._breakpoints[..., :-1]
        )
        pass

    @property
    def rates(self):
        return self._rates

    @property
    def breakpoints(self):
        self._breakpoints

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


def demo():
    pe = PiecewiseExponential(
        rates=[1, 2, 1], breakpoints=[2, 8])
    pass


if __name__ == "__main__":
    demo()

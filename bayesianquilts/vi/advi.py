import functools
import inspect
import os
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.data.ops.dataset_ops import BatchDataset
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops, math_ops, state_ops
from tensorflow.python.tools.inspect_checkpoint import \
    print_tensors_in_checkpoint_file
from tensorflow.python.training import optimizer
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import softplus as softplus_lib
from tensorflow_probability.python.distributions.transformed_distribution import \
    TransformedDistribution
from tensorflow_probability.python.internal import (dtype_util, prefer_static,
                                                    tensorshape_util)
from tensorflow_probability.python.vi import csiszar_divergence
from tqdm import tqdm

from tensorflow_probability.python.vi import GradientEstimators

from bayesianquilts.distributions import SqrtInverseGamma

from bayesianquilts.util import TransformedVariable

tfd = tfp.distributions
tfb = tfp.bijectors


def build_trainable_concentration_scale_distribution(
    initial_concentration,
    initial_scale,
    event_ndims,
    distribution_fn=tfd.InverseGamma,
    validate_args=False,
    strategy=None,
    name=None,
    surrogate_params=None,
):
    """Builds a variational distribution from a location-scale family.
    Args:
      initial_concentration: Float `Tensor` initial concentration.
      initial_scale: Float `Tensor` initial scale.
      event_ndims: Integer `Tensor` number of event dimensions
        in `initial_concentration`.
      distribution_fn: Optional constructor for a `tfd.Distribution` instance
        in a location-scale family. This should have signature `dist =
        distribution_fn(loc, scale, validate_args)`.
        Default value: `tfd.Normal`.
      validate_args: Python `bool`. Whether to validate input with asserts.
        This imposes a runtime cost. If `validate_args` is `False`, and the
        inputs are invalid, correct behavior is not guaranteed.
        Default value: `False`.
      name: Python `str` name prefixed to ops created by this function.
        Default value: `None` (i.e.,
          'build_trainable_location_scale_distribution').
    Returns:
      posterior_dist: A `tfd.Distribution` instance.
    """
    scope = (
        strategy.scope()
        if strategy is not None
        else tf.name_scope(
            name or "build_trainable_concentration_scale_distribution")
    )
    with scope:
        dtype = dtype_util.common_dtype(
            [initial_concentration, initial_scale], dtype_hint=tf.float64
        )
        initial_concentration = tf.cast(initial_concentration, dtype=dtype)
        initial_scale = tf.cast(initial_scale, dtype=dtype)

        loc = TransformedVariable(
            initial_concentration,
            softplus_lib.Softplus(),
            scope=scope,
            name="concentration",
        )
        scale = TransformedVariable(
            initial_scale, softplus_lib.Softplus(), scope=scope, name="scale"
        )
        posterior_dist = distribution_fn(
            concentration=loc, scale=scale, validate_args=validate_args
        )

        # Ensure the distribution has the desired number of event dimensions.
        static_event_ndims = tf.get_static_value(event_ndims)
        if static_event_ndims is None or static_event_ndims > 0:
            posterior_dist = tfd.Independent(
                posterior_dist,
                reinterpreted_batch_ndims=event_ndims,
                validate_args=validate_args,
            )

    return posterior_dist


def build_trainable_location_scale_distribution(
    initial_loc,
    initial_scale,
    event_ndims,
    distribution_fn=tfd.Normal,
    validate_args=False,
    strategy=None,
    name=None,
):
    """Builds a variational distribution from a location-scale family.
    Args:
      initial_loc: Float `Tensor` initial location.
      initial_scale: Float `Tensor` initial scale.
      event_ndims: Integer `Tensor` number of event dimensions
                    in `initial_loc`.
      distribution_fn: Optional constructor for a `tfd.Distribution` instance
        in a location-scale family. This should have signature `dist =
        distribution_fn(loc, scale, validate_args)`.
        Default value: `tfd.Normal`.
      validate_args: Python `bool`. Whether to validate input with asserts.
        This
        imposes a runtime cost. If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
        Default value: `False`.
      name: Python `str` name prefixed to ops created by this function.
        Default value: `None` (i.e.,
          'build_trainable_location_scale_distribution').
    Returns:
      posterior_dist: A `tfd.Distribution` instance.
    """
    scope = (
        strategy.scope()
        if strategy is not None
        else tf.name_scope(
            name or "build_trainable_location_scale_distribution")
    )
    with scope:
        dtype = dtype_util.common_dtype(
            [initial_loc, initial_scale], dtype_hint=tf.float32
        )
        initial_loc = tf.convert_to_tensor(initial_loc, dtype=dtype)
        initial_scale = tf.convert_to_tensor(initial_scale, dtype=dtype)

        loc = tf.Variable(initial_value=initial_loc, name="loc")
        scale = TransformedVariable(
            initial_scale, softplus_lib.Softplus(), scope=scope, name="scale"
        )
        posterior_dist = distribution_fn(
            loc=loc, scale=scale, validate_args=validate_args
        )

        # Ensure the distribution has the desired number of event dimensions.
        static_event_ndims = tf.get_static_value(event_ndims)
        if static_event_ndims is None or static_event_ndims > 0:
            posterior_dist = tfd.Independent(
                posterior_dist,
                reinterpreted_batch_ndims=event_ndims,
                validate_args=validate_args,
            )

    return posterior_dist


build_trainable_InverseGamma_dist = functools.partial(
    build_trainable_concentration_scale_distribution, distribution_fn=tfd.InverseGamma
)

build_trainable_normal_dist = functools.partial(
    build_trainable_location_scale_distribution, distribution_fn=tfd.Normal
)


def build_surrogate_posterior(
    joint_distribution_named,
    bijectors=None,
    exclude=[],
    num_samples=25,
    initializers={},
    strategy=None,
    name=None,
    gaussian_only=False,
    dtype=tf.float64,
    surrogate_params=None,
):
    if surrogate_params is None:
        surrogate_params = {}
    prior_sample = joint_distribution_named.sample(int(num_samples))
    surrogate_dict = {}
    means = {
        k: tf.cast(tf.reduce_mean(v, axis=0), dtype=dtype)
        for k, v in prior_sample.items()
    }

    prior_sample = joint_distribution_named.sample()
    bijectors = defaultdict(tfb.Identity) if bijectors is None else bijectors
    for k, v in joint_distribution_named.model.items():
        if name is not None:
            label = f"{name}/{k}"
        else:
            label = k
        if k in exclude:
            continue
        if callable(v):
            test_input = {a: prior_sample[a]
                          for a in inspect.getfullargspec(v).args}
            test_distribution = v(**test_input)
        else:
            test_distribution = v
        if (isinstance(
            test_distribution.distribution, tfd.InverseGamma
        ) or isinstance(
            test_distribution.distribution, SqrtInverseGamma
        )) and not gaussian_only:
            surrogate_dict[k] = bijectors[k](
                build_trainable_InverseGamma_dist(
                    2.0 * tf.ones(test_distribution.event_shape, dtype=dtype),
                    tf.ones(test_distribution.event_shape, dtype=dtype),
                    len(test_distribution.event_shape),
                    strategy=strategy,
                    name=label,
                )
            )
        else:
            if k in initializers.keys():
                loc = initializers[k]
            else:
                loc = means[k]
            surrogate_dict[k] = bijectors[k](
                build_trainable_normal_dist(
                    tfb.Invert(bijectors[k])(loc),
                    1e-2 * tf.ones(test_distribution.event_shape, dtype=dtype),
                    len(test_distribution.event_shape),
                    strategy=strategy,
                    name=label,
                )
            )
    surrogate = tfd.JointDistributionNamed(surrogate_dict)
    for j, var in enumerate(surrogate.trainable_variables):
        if var.name in surrogate_params.keys():
            surrogate.trainable_variables[j].assign(surrogate_params[var.name])
    return surrogate


def build_trainable_concentration_distribution(
    initial_concentration,
    event_ndims,
    distribution_fn=tfd.Dirichlet,
    validate_args=False,
    strategy=None,
    name=None,
):
    """Builds a variational distribution from a location-scale family.
    Args:
      initial_concentration: Float `Tensor` initial concentration.
      event_ndims: Integer `Tensor` number of event dimensions
        in `initial_concentration`.
      distribution_fn: Optional constructor for a `tfd.Distribution` instance
        in a location-scale family. This should have signature `dist =
        distribution_fn(loc, scale, validate_args)`.
        Default value: `tfd.Normal`.
      validate_args: Python `bool`. Whether to validate input with asserts.
        This imposes a runtime cost. If `validate_args` is `False`, and the
        inputs are invalid, correct behavior is not guaranteed.
        Default value: `False`.
      name: Python `str` name prefixed to ops created by this function.
        Default value: `None` (i.e.,
          'build_trainable_location_scale_distribution').
    Returns:
      posterior_dist: A `tfd.Distribution` instance.
    """
    scope = (
        strategy.scope()
        if strategy is not None
        else tf.name_scope(name or "build_trainable_concentration_distribution")
    )
    with scope:
        dtype = dtype_util.common_dtype(
            [initial_concentration], dtype_hint=tf.float32)

        loc = TransformedVariable(
            initial_concentration,
            softplus_lib.Softplus(),
            scope=scope,
            name="concentration",
        )

        posterior_dist = distribution_fn(
            concentration=loc, validate_args=validate_args)

        # Ensure the distribution has the desired number of event dimensions.
        static_event_ndims = tf.get_static_value(event_ndims)
        if static_event_ndims is None or static_event_ndims > 0:
            posterior_dist = tfd.Independent(
                posterior_dist,
                reinterpreted_batch_ndims=event_ndims,
                validate_args=validate_args,
            )

    return posterior_dist

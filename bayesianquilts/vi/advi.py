import functools
import inspect
import typing
from collections import defaultdict

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.util as tfp_util
from jax import random
from tensorflow_probability.substrates.jax import tf2jax as tf
from tensorflow_probability.substrates.jax.bijectors import \
    softplus as softplus_lib
from tensorflow_probability.substrates.jax.internal import dtype_util

from bayesianquilts.distributions import SqrtInverseGamma


def build_trainable_concentration_scale_distribution(
    initial_concentration: jax.typing.ArrayLike,
    initial_scale: jax.typing.ArrayLike,
    event_ndims: int,
    distribution_fn: tfd.Distribution = tfd.InverseGamma,
    validate_args: bool = False,
    name: str | None = None,
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
    """
    scope = (
        strategy.scope()
        if strategy is not None
        else tf.name_scope(
            name or "build_trainable_concentration_scale_distribution")
    )
    with scope:
    """
    name = "" if None else name

    dtype = dtype_util.common_dtype(
        [initial_concentration, initial_scale], dtype_hint=jnp.float64
    )
    initial_concentration = initial_concentration.astype(dtype)
    initial_scale = initial_scale.astype(dtype)

    loc = tfp_util.TransformedVariable(
        initial_concentration,
        softplus_lib.Softplus(),
        dtype=dtype,
        name=f"{name}__concentration",
    )
    scale = tfp_util.TransformedVariable(
        initial_scale, softplus_lib.Softplus(), name=f"{name}__scale", dtype=dtype
    )
    posterior_dist = distribution_fn(
        concentration=loc, scale=scale, validate_args=validate_args
    )

    # Ensure the distribution has the desired number of event dimensions.
    static_event_ndims = event_ndims
    if static_event_ndims is None or static_event_ndims > 0:
        posterior_dist = tfd.Independent(
            posterior_dist,
            reinterpreted_batch_ndims=event_ndims,
            validate_args=validate_args,
        )

    return posterior_dist


def build_trainable_location_scale_distribution(
    initial_loc: jax.typing.ArrayLike,
    initial_scale: jax.typing.ArrayLike,
    event_ndims: int,
    distribution_fn: tfd.Distribution = tfd.Normal,
    validate_args: bool = False,
    name: str | None = None,
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

    """
    scope = (
        strategy.scope()
        if strategy is not None
        else tf.name_scope(
            name or "build_trainable_location_scale_distribution")
    )
    with scope:
    """

    name = "" if None else name
    dtype = dtype_util.common_dtype(
        [initial_loc, initial_scale], dtype_hint=jnp.float32
    )

    loc = tfp_util.TransformedVariable(
        initial_value=initial_loc, name=f"{name}__loc", bijector=tfb.Identity()
    )
    scale = tfp_util.TransformedVariable(
        initial_scale, softplus_lib.Softplus(), name=f"{name}__scale"
    )
    posterior_dist = distribution_fn(loc=loc, scale=scale, validate_args=validate_args)

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


def _make_factorized_distribution_from_labeled_params(params: dict[str, jnp.ndarray]):
    pass


def build_factored_surrogate_posterior_generator(
    joint_distribution_named,
    bijectors=None,
    exclude: typing.List[str] = [],
    num_samples: int = 25,
    surrogate_initializers: dict[str, jax.typing.ArrayLike] = None,
    prefix: str = None,
    gaussian_only: bool = False,
    dtype=jnp.float32,
):
    """Builds a stateless surrogate posterior from a joint distribution named.

    Args:
        joint_distribution_named (_type_): _description_
        bijectors (_type_, optional): _description_. Defaults to None.
        exclude (list, optional): _description_. Defaults to [].
        num_samples (int, optional): _description_. Defaults to 25.
        initializers (dict, optional): _description_. Defaults to {}.
        name (_type_, optional): _description_. Defaults to None.
        gaussian_only (bool, optional): _description_. Defaults to False.
        dtype (_type_, optional): _description_. Defaults to jnp.float32.
        surrogate_params (_type_, optional): Values for the underlying surrogate parameters. Defaults to None. We'll use these values for initialization.

    Returns:
        _type_: _description_
    """
    if surrogate_initializers is None:
        surrogate_initializers = {}
    _, sample_key = random.split(random.PRNGKey(0))
    prior_sample = joint_distribution_named.sample(seed=random.PRNGKey(0))
    # create the parameters for the surrogate posterior
    var_labels = []
    if bijectors is None: bijectors = defaultdict(tfb.Identity)
    for k, v in joint_distribution_named.model.items():
        if k in exclude:
            continue
        # variable name is prefix\name\distribution\param
        # for example, prefix\intercept\normal\loc
        label = k if prefix is None else f"{prefix}\\{k}"
        if callable(v):
            test_input = {a: prior_sample[a] for a in inspect.getfullargspec(v).args}
            test_distribution = v(**test_input)
        else:
            test_distribution = v
        if (
            isinstance(test_distribution.distribution, tfd.InverseGamma)
            or isinstance(test_distribution.distribution, SqrtInverseGamma)
        ) and not gaussian_only:
            if k in bijectors.keys():
                label += f"\\{bijectors[k].name}"
            else:
                label += f"\\{bijectors.get(k, tfb.Identity()).name}"
            label += "\\igamma"
            var_labels += [label + "\\concentration", label + "\\scale"]
        else:
            if k in bijectors.keys():
                label += f"\\{bijectors[k].name}"
            else:
                label += f"\\{bijectors.get(k, tfb.Identity()).name}"
            label += "\\normal"
            var_labels += [label + "\\loc", label + "\\scale"]


    target_sample = joint_distribution_named.sample(num_samples, seed=random.PRNGKey(0))
    means = {k: jnp.mean(v, axis=0).astype(dtype) for k, v in target_sample.items()}

    bijectors = defaultdict(tfb.Identity) if bijectors is None else bijectors

    def _init_params_fn():
        _params = {}
        for label in var_labels:
            if label in surrogate_initializers.keys():
                _params[label] = surrogate_initializers[label]
                continue
            _config = label.split("\\")
            k = _config[-4]
            if _config[-1] == "loc":
                _params[label] = jnp.zeros_like(means[k])
            elif _config[-1] == "scale":
                _params[label] = 5e-3*jnp.ones_like(means[k])
            elif _config[-1] == "concentration":
                _params[label] = jnp.ones_like(means[k]) * 2.0
            else:
                raise ValueError(f"Unknown parameter type: {_config[-1]}")
        return _params
    

    def _make_dist_fn(surrogate_params: dict[str, jax.typing.ArrayLike]):
        surrogate_dict = {}

        for k, v in joint_distribution_named.model.items():
            if k in exclude:
                continue
            label = k if prefix is None else f"{prefix}\\{k}"
            if callable(v):
                test_input = {
                    a: prior_sample[a] for a in inspect.getfullargspec(v).args
                }
                test_distribution = v(**test_input)
            else:
                test_distribution = v
            if (
                isinstance(test_distribution.distribution, tfd.InverseGamma)
                or isinstance(test_distribution.distribution, SqrtInverseGamma)
            ) and not gaussian_only:
                if k in bijectors.keys():
                    label += bijectors[k].name
                else:
                    label += f"\\{bijectors.get(k, tfb.Identity()).name}"
                label += f"\\igamma"
                # Inverse Gamma distribution
                surrogate_dict[k] = bijectors.get(k, tfb.Identity())(
                    build_trainable_InverseGamma_dist(
                        surrogate_params[label+"\\concentration"], # 2.0 * jnp.ones(test_distribution.event_shape, dtype=dtype),
                        surrogate_params[label+"\\scale"],# jnp.ones(test_distribution.event_shape, dtype=dtype),
                        len(test_distribution.event_shape),
                        name=label,
                    )
                )
            else:
                label += f"\\{bijectors.get(k, tfb.Identity()).name}\\normal"
                surrogate_dict[k] = bijectors.get(k, tfb.Identity())(
                    build_trainable_normal_dist(
                        tfb.Invert(bijectors.get(k, tfb.Identity()))(surrogate_params[label+"\\loc"]),
                        surrogate_params[label+"\\scale"],
                        len(test_distribution.event_shape),
                        name=label,
                    )
                )
        surrogate = tfd.JointDistributionNamed(surrogate_dict)
        return surrogate

    return _make_dist_fn, _init_params_fn


def build_trainable_concentration_distribution(
    initial_concentration: jax.typing.ArrayLike,
    event_ndims: int,
    distribution_fn: tfd.Distribution=tfd.Dirichlet,
    validate_args: bool=False,
    name: str | None=None,
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

    """
    scope = (
        strategy.scope()
        if strategy is not None
        else tf.name_scope(name or "build_trainable_concentration_distribution")
    )
    with scope:
    """

    scope = None
    name = "" if None else name

    loc = tfp_util.TransformedVariable(
        initial_concentration,
        softplus_lib.Softplus(),
        scope=scope,
        name=f"{name}__concentration",
    )

    posterior_dist = distribution_fn(concentration=loc, validate_args=validate_args)

    # Ensure the distribution has the desired number of event dimensions.
    static_event_ndims = tf.get_static_value(event_ndims)
    if static_event_ndims is None or static_event_ndims > 0:
        posterior_dist = tfd.Independent(
            posterior_dist,
            reinterpreted_batch_ndims=event_ndims,
            validate_args=validate_args,
        )

    return posterior_dist

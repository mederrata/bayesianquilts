import functools
import inspect
import typing
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.util as tfp_util
from jax import random
from tensorflow_probability.substrates.jax import tf2jax as tf
from tensorflow_probability.substrates.jax.bijectors import \
    softplus as softplus_lib
from tensorflow_probability.substrates.jax.internal import dtype_util

from bayesianquilts.distributions import SqrtInverseGamma


# ---------------------------------------------------------------------------
# Low-rank multivariate normal helpers
# ---------------------------------------------------------------------------

def _build_lowrank_mvn(loc_flat, log_diag_scale, factor, event_shape=None):
    """Build MVN with diagonal + low-rank covariance using MultivariateNormalTriL.

    Parameterizes q(z) = N(loc, diag(exp(2*log_diag_scale)) + factor @ factor^T)

    Computes the Cholesky of the full covariance matrix and delegates to
    TFP's MultivariateNormalTriL, ensuring full compatibility with
    JointDistributionNamed and experimental_sample_and_log_prob.

    Cost: O(d^3) for Cholesky at construction (traced once under JIT),
    O(d^2) per sample/log_prob evaluation.

    Args:
        loc_flat: Mean vector, shape (d,).
        log_diag_scale: Log of diagonal scale, shape (d,).
            Actual diagonal variance = exp(2 * log_diag_scale).
        factor: Low-rank factor matrix, shape (d, r).
        event_shape: If provided, a Reshape bijector is applied so that
            samples have this shape instead of (d,). Useful when the
            original parameter has shape (D, K) but is flattened internally.

    Returns:
        A TFP distribution with event_shape = event_shape or (d,).
    """
    d = loc_flat.shape[-1]
    diag_scale = jnp.exp(log_diag_scale)

    # Full covariance: diag(s^2) + F @ F^T
    cov = jnp.diag(diag_scale ** 2) + factor @ factor.T
    scale_tril = jnp.linalg.cholesky(cov)

    dist = tfd.MultivariateNormalTriL(loc=loc_flat, scale_tril=scale_tril)

    if event_shape is not None and event_shape != (d,):
        dist = tfd.TransformedDistribution(
            dist,
            tfb.Reshape(event_shape_out=event_shape, event_shape_in=(d,)),
        )

    return dist


# ---------------------------------------------------------------------------
# Distribution builders
# ---------------------------------------------------------------------------

def build_trainable_concentration_scale_distribution(
    initial_concentration: jax.typing.ArrayLike,
    initial_scale: jax.typing.ArrayLike,
    event_ndims: int,
    distribution_fn: tfd.Distribution = tfd.InverseGamma,
    validate_args: bool = False,
    name: str | None = None,
):
    """Builds a variational distribution from a concentration-scale family."""
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
    """Builds a variational distribution from a location-scale family."""
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

    static_event_ndims = tf.get_static_value(event_ndims)
    if static_event_ndims is None or static_event_ndims > 0:
        posterior_dist = tfd.Independent(
            posterior_dist,
            reinterpreted_batch_ndims=event_ndims,
            validate_args=validate_args,
        )

    return posterior_dist


def _build_normal_from_log_scale(loc, log_scale, event_ndims, name=""):
    """Build Normal distribution using exp(log_scale) parameterization.

    This provides better gradient flow than softplus for small scales and
    better numerical behavior in reduced precision (float16/bfloat16).
    """
    scale = jnp.exp(log_scale)
    dist = tfd.Normal(loc=loc, scale=scale)
    if event_ndims is not None and event_ndims > 0:
        dist = tfd.Independent(dist, reinterpreted_batch_ndims=event_ndims)
    return dist


def _build_normal_from_natural_params(eta1, neg_half_precision, event_ndims, name=""):
    """Build Normal distribution from natural parameters.

    Natural parameterization for Gaussian:
        eta1 = mu / sigma^2     (precision-weighted mean)
        eta2 = -1 / (2*sigma^2) (negative half-precision)

    Standard Adam on (eta1, eta2) implicitly approximates natural gradient
    descent on (mu, sigma^2), typically giving faster convergence.

    Args:
        eta1: Precision-weighted mean parameter.
        neg_half_precision: Raw parameter; actual eta2 = -softplus(neg_half_precision)
            to enforce eta2 < 0 (ensuring sigma^2 > 0).
    """
    # eta2 = -softplus(raw) ensures eta2 < 0
    eta2 = -jax.nn.softplus(neg_half_precision)
    # sigma^2 = -1/(2*eta2) = 1/(2*softplus(raw))
    variance = -0.5 / eta2
    sigma = jnp.sqrt(variance)
    # mu = eta1 * sigma^2 = -eta1 / (2*eta2)
    mu = eta1 * variance

    dist = tfd.Normal(loc=mu, scale=sigma)
    if event_ndims is not None and event_ndims > 0:
        dist = tfd.Independent(dist, reinterpreted_batch_ndims=event_ndims)
    return dist


build_trainable_InverseGamma_dist = functools.partial(
    build_trainable_concentration_scale_distribution, distribution_fn=tfd.InverseGamma
)

build_trainable_normal_dist = functools.partial(
    build_trainable_location_scale_distribution, distribution_fn=tfd.Normal
)


def _make_factorized_distribution_from_labeled_params(params: dict[str, jnp.ndarray]):
    pass


# ---------------------------------------------------------------------------
# Main surrogate posterior builder
# ---------------------------------------------------------------------------

def build_factored_surrogate_posterior_generator(
    joint_distribution_named,
    bijectors=None,
    exclude: typing.List[str] = [],
    num_samples: int = 25,
    surrogate_initializers: dict[str, jax.typing.ArrayLike] = None,
    prefix: str = None,
    gaussian_only: bool = False,
    noise: float = 1e-2,
    dtype=jnp.float32,
    parameterization: str = "softplus",
    rank: int = 0,
):
    """Builds a stateless surrogate posterior generator from a prior.

    Args:
        joint_distribution_named: Prior JointDistributionNamed.
        bijectors: Dict mapping variable names to bijectors for constrained
            parameters.
        exclude: Variable names to exclude from the surrogate.
        num_samples: Number of prior samples for initialization.
        surrogate_initializers: Dict of custom initial values.
        prefix: Label prefix for parameter names.
        gaussian_only: If True, use Normal surrogates even for InverseGamma
            priors.
        noise: Initial scale for surrogate standard deviations.
        dtype: Parameter dtype.
        parameterization: Scale parameterization for Normal surrogates.
            - "softplus": scale = softplus(raw_scale) (original, default)
            - "log_scale": scale = exp(log_scale). Better gradient flow for
              small scales and better float16/bfloat16 support.
            - "natural": Natural parameterization (eta1, eta2). Standard Adam
              on natural parameters implicitly approximates natural gradient
              descent, giving faster convergence.
        rank: Rank of low-rank covariance correction (per variable).
            0 = mean-field (default), >0 = diagonal + low-rank covariance.
            Each variable gets a factor matrix of shape (prod(event_shape), rank)
            to capture the top correlations within that variable.

    Returns:
        Tuple of (distribution_generator_fn, parameter_initializer_fn).
    """
    if surrogate_initializers is None:
        surrogate_initializers = {}
    _, sample_key = random.split(random.PRNGKey(np.random.randint(0, 1e8)))
    prior_sample = joint_distribution_named.sample(seed=sample_key)
    # create the parameters for the surrogate posterior
    var_labels = []
    if bijectors is None: bijectors = defaultdict(tfb.Identity)

    # Track event shapes for low-rank initialization
    var_event_shapes = {}

    for k, v in joint_distribution_named.model.items():
        if k in exclude:
            continue
        label = k if prefix is None else f"{prefix}\\{k}"
        if callable(v):
            test_input = {a: prior_sample[a] for a in inspect.getfullargspec(v).args}
            test_distribution = v(**test_input)
        else:
            test_distribution = v

        var_event_shapes[k] = tuple(test_distribution.event_shape)

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

            if rank > 0:
                label += "\\lowrank"
                var_labels += [label + "\\loc", label + "\\log_diag_scale", label + "\\factor"]
            elif parameterization == "natural":
                label += "\\natural"
                var_labels += [label + "\\eta1", label + "\\neg_half_prec"]
            elif parameterization == "log_scale":
                label += "\\normal"
                var_labels += [label + "\\loc", label + "\\log_scale"]
            else:
                label += "\\normal"
                var_labels += [label + "\\loc", label + "\\scale"]

    target_sample = joint_distribution_named.sample(
        num_samples, seed=random.PRNGKey(np.random.randint(0, 1e8))
    )
    means = {k: jnp.mean(v, axis=0).astype(dtype) for k, v in target_sample.items()}
    surrogate_initializers = surrogate_initializers if not None else {}
    for k, v in surrogate_initializers.items():
        means[k] = v
    bijectors = defaultdict(tfb.Identity) if bijectors is None else bijectors

    def _init_params_fn():
        _params = {}
        for label in var_labels:
            if label in surrogate_initializers.keys():
                _params[label] = surrogate_initializers[label]
                continue
            _config = label.split("\\")
            # Variable name is at position -4 for standard labels
            # For lowrank: prefix\varname\bijector\lowrank\param -> -4
            # For natural: prefix\varname\bijector\natural\param -> -4
            k = _config[-4] if len(_config) >= 4 else _config[0]
            param_type = _config[-1]

            if param_type == "loc":
                _params[label] = means[k]
            elif param_type == "scale":
                _params[label] = noise * jnp.ones_like(means[k])
            elif param_type == "log_scale":
                # Initialize so that exp(log_scale) = noise
                _params[label] = jnp.log(noise) * jnp.ones_like(means[k])
            elif param_type == "log_diag_scale":
                # Low-rank diagonal scale initialization
                _params[label] = jnp.log(noise) * jnp.ones(
                    (int(np.prod(var_event_shapes[k])),), dtype=dtype
                )
            elif param_type == "factor":
                # Low-rank factor: small random initialization
                d = int(np.prod(var_event_shapes[k]))
                _params[label] = noise * 0.1 * random.normal(
                    random.PRNGKey(hash(label) % (2**31)),
                    (d, rank), dtype=dtype
                )
            elif param_type == "eta1":
                # Natural param: eta1 = mu / sigma^2
                # With initial mu = means[k] and sigma^2 = noise^2:
                sigma_sq = noise ** 2
                _params[label] = (means[k] / sigma_sq).astype(dtype)
            elif param_type == "neg_half_prec":
                # Natural param: neg_half_precision raw value
                # We want eta2 = -softplus(raw) = -1/(2*sigma^2)
                # So softplus(raw) = 1/(2*noise^2)
                # raw = softplus_inverse(1/(2*noise^2))
                target = 1.0 / (2.0 * noise ** 2)
                # softplus_inverse(x) = log(exp(x) - 1) ≈ x for large x
                raw = np.where(
                    target > 20.0,
                    target,
                    np.log(np.expm1(min(target, 20.0))),
                )
                _params[label] = jnp.asarray(raw, dtype=dtype) * jnp.ones_like(means[k])
            elif param_type == "concentration":
                _params[label] = jnp.ones_like(means[k]) * 2.0
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
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

            event_ndims = len(test_distribution.event_shape)
            event_shape = tuple(test_distribution.event_shape)

            if (
                isinstance(test_distribution.distribution, tfd.InverseGamma)
                or isinstance(test_distribution.distribution, SqrtInverseGamma)
            ) and not gaussian_only:
                if k in bijectors.keys():
                    label += f"\\{bijectors[k].name}"
                else:
                    label += f"\\{bijectors.get(k, tfb.Identity()).name}"
                label += f"\\igamma"
                surrogate_dict[k] = bijectors.get(k, tfb.Identity())(
                    build_trainable_InverseGamma_dist(
                        surrogate_params[label + "\\concentration"],
                        surrogate_params[label + "\\scale"],
                        event_ndims,
                        name=label,
                    )
                )
            else:
                bij = bijectors.get(k, tfb.Identity())
                bij_name = bij.name if k in bijectors.keys() else bijectors.get(k, tfb.Identity()).name
                label += f"\\{bij_name}"

                if rank > 0:
                    label += "\\lowrank"
                    log_diag = surrogate_params[label + "\\log_diag_scale"]
                    factor = surrogate_params[label + "\\factor"]

                    # Apply inverse bijector to loc for constrained params
                    loc_for_dist = tfb.Invert(bij)(
                        surrogate_params[label + "\\loc"]
                    ).reshape(-1)

                    dist = _build_lowrank_mvn(
                        loc_flat=loc_for_dist,
                        log_diag_scale=log_diag,
                        factor=factor,
                        event_shape=event_shape if event_ndims > 0 else None,
                    )
                    surrogate_dict[k] = bij(dist)

                elif parameterization == "natural":
                    label += "\\natural"
                    eta1 = surrogate_params[label + "\\eta1"]
                    neg_half_prec = surrogate_params[label + "\\neg_half_prec"]
                    dist = _build_normal_from_natural_params(
                        eta1, neg_half_prec, event_ndims, name=label
                    )
                    surrogate_dict[k] = bij(dist)

                elif parameterization == "log_scale":
                    label += "\\normal"
                    loc = tfb.Invert(bij)(surrogate_params[label + "\\loc"])
                    log_scale = surrogate_params[label + "\\log_scale"]
                    dist = _build_normal_from_log_scale(
                        loc, log_scale, event_ndims, name=label
                    )
                    surrogate_dict[k] = bij(dist)

                else:
                    # Default softplus parameterization
                    label += "\\normal"
                    surrogate_dict[k] = bij(
                        build_trainable_normal_dist(
                            tfb.Invert(bij)(surrogate_params[label + "\\loc"]),
                            surrogate_params[label + "\\scale"],
                            event_ndims,
                            name=label,
                        )
                    )

        surrogate = tfd.JointDistributionNamed(surrogate_dict)
        return surrogate

    return _make_dist_fn, _init_params_fn


# ---------------------------------------------------------------------------
# Pathfinder initialization
# ---------------------------------------------------------------------------

def pathfinder_initialize(
    log_prob_fn,
    surrogate_initializer,
    data,
    dataset_size=None,
    batch_size=None,
    num_samples=200,
    maxiter=100,
    seed=0,
):
    """Initialize ADVI surrogate parameters using the Pathfinder algorithm.

    Runs a few L-BFGS iterations on the log joint density to find a good
    (loc, scale) starting point, dramatically reducing the number of ADVI
    epochs needed for convergence.

    Args:
        log_prob_fn: Model's unnormalized log probability function.
            Signature: log_prob_fn(data=..., prior_weight=..., **params) -> scalar.
        surrogate_initializer: Callable returning default surrogate params dict
            with labeled keys (from build_factored_surrogate_posterior_generator).
        data: Representative data batch for evaluating the log density.
        dataset_size: Total number of data points N. Used with batch_size to
            set prior_weight and scale the density so Pathfinder sees
            full-data curvature.
        batch_size: Size B of the provided data batch.
        num_samples: Number of samples to draw from the Pathfinder approximation.
        maxiter: Maximum L-BFGS iterations for Pathfinder.
        seed: Random seed.

    Returns:
        Dictionary of initialized surrogate parameters (same structure as
        surrogate_initializer() output).
    """
    try:
        from blackjax.vi import pathfinder
    except ImportError:
        raise ImportError(
            "blackjax is required for pathfinder_initialize. "
            "Install it with: pip install blackjax"
        )

    default_params = surrogate_initializer()

    # Parse labels to build {var_name: {param_type: label}} mapping
    var_param_map = {}  # var_name -> {param_type -> label}
    for label in default_params:
        parts = label.split("\\")
        var_name = parts[-4] if len(parts) >= 4 else parts[0]
        param_type = parts[-1]
        if var_name not in var_param_map:
            var_param_map[var_name] = {}
        var_param_map[var_name][param_type] = label

    # Extract initial position from loc params
    var_names = list(var_param_map.keys())
    initial_dict = {}
    for var_name, pmap in var_param_map.items():
        if "loc" in pmap:
            initial_dict[var_name] = default_params[pmap["loc"]]
        elif "eta1" in pmap:
            # Natural params: recover loc from eta1 and neg_half_prec
            eta1 = default_params[pmap["eta1"]]
            neg_half_prec = default_params[pmap["neg_half_prec"]]
            eta2 = -jax.nn.softplus(neg_half_prec)
            variance = -0.5 / eta2
            initial_dict[var_name] = eta1 * variance
        else:
            # Concentration/scale params (InverseGamma) — skip for Pathfinder
            continue

    flat_init, unflatten_fn = jax.flatten_util.ravel_pytree(initial_dict)

    # Compute prior_weight and density scaling
    if dataset_size is not None and batch_size is not None:
        prior_weight = batch_size / dataset_size
        density_scale = dataset_size / batch_size
    else:
        prior_weight = 1.0
        density_scale = 1.0

    def logdensity_fn(params_flat):
        params_dict = unflatten_fn(params_flat)
        lp = log_prob_fn(data=data, prior_weight=prior_weight, **params_dict)
        return jnp.squeeze(density_scale * lp)

    key = jax.random.PRNGKey(seed)

    # Run Pathfinder
    state, _ = pathfinder.approximate(
        rng_key=jax.random.fold_in(key, 0),
        logdensity_fn=logdensity_fn,
        initial_position=flat_init,
        num_samples=num_samples,
        maxiter=maxiter,
    )

    # Sample from the Pathfinder approximation
    sample_key = jax.random.fold_in(key, 1)
    samples_flat = pathfinder.sample(sample_key, state, num_samples=num_samples)
    if isinstance(samples_flat, tuple):
        samples_flat = samples_flat[0]

    # Unflatten and compute per-variable mean and std
    samples_dicts = jax.vmap(unflatten_fn)(samples_flat)
    var_mean = {}
    var_std = {}
    for var_name in initial_dict:
        var_samples = samples_dicts[var_name]
        var_mean[var_name] = jnp.mean(var_samples, axis=0)
        var_std[var_name] = jnp.maximum(jnp.std(var_samples, axis=0), 1e-6)

    # Map back to surrogate labels
    init_params = dict(default_params)  # copy defaults
    for var_name, pmap in var_param_map.items():
        if var_name not in var_mean:
            # InverseGamma or other non-loc params — keep defaults
            continue

        mean = var_mean[var_name]
        std = var_std[var_name]

        for param_type, label in pmap.items():
            if param_type == "loc":
                init_params[label] = mean
            elif param_type == "scale":
                init_params[label] = std
            elif param_type == "log_scale":
                init_params[label] = jnp.log(std)
            elif param_type == "log_diag_scale":
                init_params[label] = jnp.log(std.reshape(-1))
            elif param_type == "eta1":
                init_params[label] = mean / (std ** 2)
            elif param_type == "neg_half_prec":
                # softplus_inverse(1 / (2 * std^2))
                target = 1.0 / (2.0 * std ** 2)
                init_params[label] = jnp.where(
                    target > 20.0,
                    target,
                    jnp.log(jnp.expm1(jnp.clip(target, a_max=20.0))),
                )
            elif param_type == "factor":
                pass  # leave at default (small random)
            elif param_type in ("concentration",):
                pass  # leave at default

    return init_params


# ---------------------------------------------------------------------------
# Concentration-only distribution builder (e.g. Dirichlet)
# ---------------------------------------------------------------------------

def build_trainable_concentration_distribution(
    initial_concentration: jax.typing.ArrayLike,
    event_ndims: int,
    distribution_fn: tfd.Distribution = tfd.Dirichlet,
    validate_args: bool = False,
    name: str | None = None,
):
    """Builds a variational distribution from a concentration family."""
    scope = None
    name = "" if None else name

    loc = tfp_util.TransformedVariable(
        initial_concentration,
        softplus_lib.Softplus(),
        scope=scope,
        name=f"{name}__concentration",
    )

    posterior_dist = distribution_fn(concentration=loc, validate_args=validate_args)

    static_event_ndims = tf.get_static_value(event_ndims)
    if static_event_ndims is None or static_event_ndims > 0:
        posterior_dist = tfd.Independent(
            posterior_dist,
            reinterpreted_batch_ndims=event_ndims,
            validate_args=validate_args,
        )

    return posterior_dist

"""Noise-regularized transformations for one Bayesian LOO fold.

This module is the correctness-first implementation of PMM1--PMM3 and
user-supplied LL/KL/Var vector fields.  It uses three explicit,
non-overlapping roles for posterior draws:

* ``fit_draws`` estimate a weighted moment map or normalize a vector field;
* ``tune_draws`` select the partial step ``h``;
* ``evaluation_draws`` produce the reported estimate and diagnostics.

The transformed importance ratio is evaluated from an explicit full-posterior
log density.  Consequently the prior ratio cannot be silently omitted:

    log w(theta)
      = log pi(T(theta) | D)
        - log likelihood_i(T(theta))
        - log pi(theta | D)
        + log |det J_T|.

All coordinates must be unconstrained.  The public functions operate on a
matrix of flattened draws with shape ``(samples, parameters)``; model-specific
code is responsible for flattening and unflattening parameter pytrees inside
the two log-density callbacks.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

Array = jax.Array
PMMMethod = Literal["pmm1", "pmm2", "pmm3"]
LogDensity = Callable[[Array], Array]
VectorField = Callable[[Array], Array]
VectorFieldJacobian = Callable[[Array], Array]


def _validate_draws(draws: Array, name: str) -> Array:
    draws = jnp.asarray(draws)
    if draws.ndim != 2:
        raise ValueError(f"{name} must have shape (samples, parameters)")
    if draws.shape[0] < 2 or draws.shape[1] < 1:
        raise ValueError(f"{name} must contain at least two nonempty draws")
    return draws


def _validate_log_values(values: Array, sample_size: int, name: str) -> Array:
    values = jnp.asarray(values)
    if values.shape != (sample_size,):
        raise ValueError(f"{name} must return shape ({sample_size},)")
    return values


def _weighted_moments(draws: Array, log_weights: Array) -> tuple[Array, Array]:
    weights = jax.nn.softmax(log_weights)
    mean = jnp.sum(weights[:, None] * draws, axis=0)
    centered = draws - mean
    covariance = (centered.T * weights) @ centered
    return mean, covariance


def _ordinary_moments(draws: Array) -> tuple[Array, Array]:
    mean = jnp.mean(draws, axis=0)
    centered = draws - mean
    covariance = centered.T @ centered / draws.shape[0]
    return mean, covariance


@dataclass(frozen=True)
class AffineMap:
    """An affine map ``theta -> linear @ theta + offset``."""

    linear: Array
    offset: Array
    h: float

    @property
    def log_abs_determinant(self) -> Array:
        sign, log_abs_determinant = jnp.linalg.slogdet(self.linear)
        if float(sign) <= 0.0:
            raise ValueError("The affine map is not orientation-preserving.")
        return log_abs_determinant

    def __call__(self, draws: Array) -> Array:
        draws = _validate_draws(draws, "draws")
        if draws.shape[1] != self.linear.shape[0]:
            raise ValueError("draw dimension does not match affine map")
        return draws @ self.linear.T + self.offset

    def log_abs_determinants(self, draws: Array) -> Array:
        """Return one exact log determinant per input draw."""

        draws = _validate_draws(draws, "draws")
        return jnp.full(
            (draws.shape[0],),
            self.log_abs_determinant,
            dtype=draws.dtype,
        )


@dataclass(frozen=True)
class VectorFieldMap:
    """A partial vector-field map.

    vector_field must return shape (samples, parameters), and
    vector_field_jacobian must return
    (samples, parameters, parameters).  The caller supplies a global
    Lipschitz bound when fitting the field; this is checked before a
    non-identity map is constructed.
    """

    vector_field: VectorField
    vector_field_jacobian: VectorFieldJacobian
    coefficient: float
    h: float
    dimension: int

    def __call__(self, draws: Array) -> Array:
        draws = _validate_draws(draws, "draws")
        if draws.shape[1] != self.dimension:
            raise ValueError("draw dimension does not match vector-field map")
        field = jnp.asarray(self.vector_field(draws))
        if field.shape != draws.shape:
            raise ValueError("vector_field must return the same shape as draws")
        return draws + self.coefficient * field

    def log_abs_determinants(self, draws: Array) -> Array:
        """Return exact pointwise log absolute Jacobian determinants."""

        draws = _validate_draws(draws, "draws")
        if draws.shape[1] != self.dimension:
            raise ValueError("draw dimension does not match vector-field map")
        jacobians = jnp.asarray(self.vector_field_jacobian(draws))
        expected_shape = (draws.shape[0], self.dimension, self.dimension)
        if jacobians.shape != expected_shape:
            raise ValueError(
                "vector_field_jacobian must return shape "
                f"{expected_shape}"
            )
        identity = jnp.eye(self.dimension, dtype=draws.dtype)
        signs, log_abs_determinants = jnp.linalg.slogdet(
            identity[None, :, :] + self.coefficient * jacobians
        )
        if bool(jnp.any(signs <= 0.0)):
            raise ValueError("the vector-field map has a non-positive Jacobian")
        return log_abs_determinants


@dataclass(frozen=True)
class FittedVectorField:
    """A sample-normalized vector field before step regularization."""

    vector_field: VectorField
    vector_field_jacobian: VectorFieldJacobian
    normalization: float
    posterior_scale: Array
    dimension: int
    global_lipschitz_bound: float
    max_h: float

    def at(self, h: float) -> VectorFieldMap:
        h = float(h)
        if not 0.0 <= h <= self.max_h:
            raise ValueError(f"flow step h must lie in [0, {self.max_h}]")
        coefficient = (
            0.0 if math.isinf(self.normalization) else h / self.normalization
        )
        if (
            h > 0.0
            and coefficient * self.global_lipschitz_bound >= 1.0
        ):
            raise ValueError(
                "the requested step violates the supplied global "
                "Lipschitz certificate"
            )
        return VectorFieldMap(
            vector_field=self.vector_field,
            vector_field_jacobian=self.vector_field_jacobian,
            coefficient=float(coefficient),
            h=h,
            dimension=self.dimension,
        )


@dataclass(frozen=True)
class FittedPMM:
    """Full empirical PMM map before partial-step regularization."""

    method: PMMMethod
    full_linear: Array
    full_offset: Array
    proposal_mean: Array
    weighted_mean: Array
    ridge: float

    def at(self, h: float) -> AffineMap:
        """Interpolate between identity (``h=0``) and full PMM (``h=1``)."""

        h = float(h)
        if not 0.0 <= h <= 1.0:
            raise ValueError("PMM partial step h must lie in [0, 1]")
        dimension = self.full_linear.shape[0]
        linear = (1.0 - h) * jnp.eye(
            dimension, dtype=self.full_linear.dtype
        ) + h * self.full_linear
        offset = h * self.full_offset
        return AffineMap(linear=linear, offset=offset, h=h)


@dataclass(frozen=True)
class RegularizedLOOResult:
    """Result of fitting, tuning, and evaluating one regularized PMM map."""

    fitted_map: FittedPMM
    selected_map: AffineMap
    h_grid: Array
    tuning_scores: Array
    transformed_evaluation_draws: Array
    evaluation_log_weights: Array
    raw_effective_sample_size: Array


@dataclass(frozen=True)
class CrossFittedLOOResult:
    """Three-fold rotation of fit, tune, and evaluation roles."""

    fold_results: tuple[RegularizedLOOResult, ...]
    transformed_evaluation_draws: Array
    evaluation_log_weights: Array
    selected_steps: Array
    raw_effective_sample_size: Array


@dataclass(frozen=True)
class RegularizedFlowLOOResult:
    """Result of fitting, tuning, and evaluating one vector-field map."""

    fitted_map: FittedVectorField
    selected_map: VectorFieldMap
    h_grid: Array
    tuning_scores: Array
    transformed_evaluation_draws: Array
    evaluation_log_weights: Array
    raw_effective_sample_size: Array


@dataclass(frozen=True)
class CrossFittedFlowLOOResult:
    """Three-fold rotation for one sample-normalized vector field."""

    fold_results: tuple[RegularizedFlowLOOResult, ...]
    transformed_evaluation_draws: Array
    evaluation_log_weights: Array
    selected_steps: Array
    raw_effective_sample_size: Array


def fit_pmm(
    draws: Array,
    identity_log_weights: Array,
    method: PMMMethod = "pmm3",
    ridge: float = 1e-8,
) -> FittedPMM:
    """Fit a full PMM map from one draw split.

    ``identity_log_weights`` are the target-to-full-posterior log ratios on
    ``draws``.  For a LOO fold they are simply the negative held-out
    log-likelihood, up to a constant.
    """

    draws = _validate_draws(draws, "draws")
    identity_log_weights = _validate_log_values(
        identity_log_weights, draws.shape[0], "identity_log_weights"
    )
    if method not in ("pmm1", "pmm2", "pmm3"):
        raise ValueError("method must be one of 'pmm1', 'pmm2', or 'pmm3'")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")

    proposal_mean, proposal_covariance = _ordinary_moments(draws)
    weighted_mean, weighted_covariance = _weighted_moments(
        draws, identity_log_weights
    )
    dimension = draws.shape[1]

    if method == "pmm1":
        full_linear = jnp.eye(dimension, dtype=draws.dtype)
    elif method == "pmm2":
        proposal_variance = jnp.diag(proposal_covariance)
        weighted_variance = jnp.diag(weighted_covariance)
        scale = jnp.sqrt(
            (weighted_variance + ridge) / (proposal_variance + ridge)
        )
        full_linear = jnp.diag(scale)
    else:
        if draws.shape[0] <= dimension and ridge == 0.0:
            raise ValueError(
                "PMM3 without a ridge requires more fit draws than parameters"
            )
        identity = jnp.eye(dimension, dtype=draws.dtype)
        proposal_cholesky = jnp.linalg.cholesky(
            proposal_covariance + ridge * identity
        )
        weighted_cholesky = jnp.linalg.cholesky(
            weighted_covariance + ridge * identity
        )
        # L_w L^{-1}, computed without explicitly forming the inverse.
        full_linear = jnp.linalg.solve(
            proposal_cholesky.T, weighted_cholesky.T
        ).T

    full_offset = weighted_mean - full_linear @ proposal_mean
    return FittedPMM(
        method=method,
        full_linear=full_linear,
        full_offset=full_offset,
        proposal_mean=proposal_mean,
        weighted_mean=weighted_mean,
        ridge=float(ridge),
    )


def fit_vector_field(
    draws: Array,
    vector_field: VectorField,
    vector_field_jacobian: VectorFieldJacobian,
    posterior_scale: Array,
    global_lipschitz_bound: float,
    max_h: float = 1.0,
) -> FittedVectorField:
    """Fit the sample displacement normalization for LL, KL, or Var.

    The field and its Jacobian are model-specific callbacks.  A finite global
    Lipschitz bound is required so every non-identity candidate has an explicit
    bijectivity certificate.  posterior_scale contains positive marginal
    posterior standard deviations in the coordinates of draws.
    """

    draws = _validate_draws(draws, "draws")
    posterior_scale = jnp.asarray(posterior_scale)
    if posterior_scale.shape != (draws.shape[1],):
        raise ValueError(
            "posterior_scale must have shape (parameters,)"
        )
    if bool(jnp.any(~jnp.isfinite(posterior_scale))) or bool(
        jnp.any(posterior_scale <= 0.0)
    ):
        raise ValueError("posterior_scale must contain finite positive values")
    global_lipschitz_bound = float(global_lipschitz_bound)
    if (
        not math.isfinite(global_lipschitz_bound)
        or global_lipschitz_bound < 0.0
    ):
        raise ValueError(
            "global_lipschitz_bound must be finite and nonnegative"
        )
    max_h = float(max_h)
    if not math.isfinite(max_h) or max_h <= 0.0:
        raise ValueError("max_h must be finite and positive")

    field = jnp.asarray(vector_field(draws))
    if field.shape != draws.shape:
        raise ValueError("vector_field must return the same shape as draws")
    if bool(jnp.any(~jnp.isfinite(field))):
        raise ValueError("vector_field returned a non-finite value")
    normalization = float(
        jnp.max(jnp.abs(field) / posterior_scale[None, :])
    )
    if normalization == 0.0:
        normalization = float("inf")
    fitted = FittedVectorField(
        vector_field=vector_field,
        vector_field_jacobian=vector_field_jacobian,
        normalization=normalization,
        posterior_scale=posterior_scale,
        dimension=draws.shape[1],
        global_lipschitz_bound=global_lipschitz_bound,
        max_h=max_h,
    )
    fitted.at(max_h)
    return fitted


def transformed_log_weights(
    base_draws: Array,
    transformation: AffineMap | VectorFieldMap,
    log_full_posterior: LogDensity,
    log_heldout_likelihood: LogDensity,
) -> tuple[Array, Array]:
    """Apply a map and return its exact, unnormalized LOO log weights.

    ``log_full_posterior`` must evaluate the full-data posterior density in the
    same unconstrained coordinates as ``base_draws`` and must include the
    prior.  Normalizing constants may be omitted because the resulting weights
    are self-normalized.
    """

    base_draws = _validate_draws(base_draws, "base_draws")
    transformed_draws = transformation(base_draws)
    log_full_base = _validate_log_values(
        log_full_posterior(base_draws), base_draws.shape[0], "log_full_posterior"
    )
    log_full_transformed = _validate_log_values(
        log_full_posterior(transformed_draws),
        base_draws.shape[0],
        "log_full_posterior",
    )
    log_heldout_transformed = _validate_log_values(
        log_heldout_likelihood(transformed_draws),
        base_draws.shape[0],
        "log_heldout_likelihood",
    )
    log_weights = (
        log_full_transformed
        - log_heldout_transformed
        - log_full_base
        + transformation.log_abs_determinants(base_draws)
    )
    return transformed_draws, log_weights


def log_weight_dispersion(log_weights: Array, clip: float | None = 30.0) -> Array:
    """Return empirical ``log(1 + CV^2)`` for unnormalized weights.

    When ``clip`` is finite, log weights are clipped relative to their maximum
    before evaluation.  This bounds the influence of a single tuning draw; the
    final returned importance weights are never clipped.
    """

    log_weights = jnp.asarray(log_weights)
    if log_weights.ndim != 1:
        raise ValueError("log_weights must be one-dimensional")
    centered = log_weights - jnp.max(log_weights)
    if clip is not None:
        if clip <= 0.0:
            raise ValueError("clip must be positive or None")
        centered = jnp.clip(centered, -float(clip), 0.0)
    log_sample_size = jnp.log(log_weights.size)
    log_mean = jax.scipy.special.logsumexp(centered) - log_sample_size
    log_second_mean = (
        jax.scipy.special.logsumexp(2.0 * centered) - log_sample_size
    )
    return log_second_mean - 2.0 * log_mean


def clipped_log_weight_loss(
    log_weights: Array,
    center: float | Array = 0.0,
    clip: float = 5.0,
) -> Array:
    """Return a bounded empirical log-weight dispersion loss.

    The loss is ``mean(min((log_weight - center)**2, clip**2))``.  Estimate a
    separate center for each candidate on the map-fitting split, then hold it
    fixed while scoring tuning draws.  Conditional on the fitted maps and
    centers, this is an average of independent losses bounded by ``clip**2``.
    """

    log_weights = jnp.asarray(log_weights)
    if log_weights.ndim != 1:
        raise ValueError("log_weights must be one-dimensional")
    if clip <= 0.0:
        raise ValueError("clip must be positive")
    squared_deviation = jnp.square(log_weights - jnp.asarray(center))
    return jnp.mean(jnp.minimum(squared_deviation, float(clip) ** 2))


def raw_effective_sample_size(log_weights: Array) -> Array:
    """Return the usual raw importance-sampling effective sample size."""

    log_weights = jnp.asarray(log_weights)
    normalized = jax.nn.softmax(log_weights)
    return 1.0 / jnp.sum(jnp.square(normalized))


def self_normalized_expectation(values: Array, log_weights: Array) -> Array:
    """Compute a self-normalized importance-sampling expectation."""

    values = jnp.asarray(values)
    log_weights = jnp.asarray(log_weights)
    if values.shape[0] != log_weights.shape[0]:
        raise ValueError("values and log_weights must share their sample axis")
    weights = jax.nn.softmax(log_weights)
    reshape = (weights.shape[0],) + (1,) * (values.ndim - 1)
    return jnp.sum(weights.reshape(reshape) * values, axis=0)


def regularized_pmm_loo_fold(
    fit_draws: Array,
    tune_draws: Array,
    evaluation_draws: Array,
    log_full_posterior: LogDensity,
    log_heldout_likelihood: LogDensity,
    method: PMMMethod = "pmm3",
    h_grid: Sequence[float] = tuple(i / 20.0 for i in range(21)),
    ridge: float = 1e-8,
    tuning_clip: float = 5.0,
) -> RegularizedLOOResult:
    """Fit, independently tune, and evaluate regularized PMM for one LOO fold."""

    fit_draws = _validate_draws(fit_draws, "fit_draws")
    tune_draws = _validate_draws(tune_draws, "tune_draws")
    evaluation_draws = _validate_draws(evaluation_draws, "evaluation_draws")
    dimension = fit_draws.shape[1]
    if tune_draws.shape[1] != dimension or evaluation_draws.shape[1] != dimension:
        raise ValueError("all three draw splits must have the same parameter dimension")

    grid = jnp.asarray(tuple(float(h) for h in h_grid), dtype=fit_draws.dtype)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("h_grid must be a nonempty one-dimensional sequence")
    if bool(jnp.any((grid < 0.0) | (grid > 1.0))):
        raise ValueError("every h_grid value must lie in [0, 1]")

    fit_identity_log_weights = -_validate_log_values(
        log_heldout_likelihood(fit_draws),
        fit_draws.shape[0],
        "log_heldout_likelihood",
    )
    fitted_map = fit_pmm(
        fit_draws,
        fit_identity_log_weights,
        method=method,
        ridge=ridge,
    )
    tuning_scores = []
    for h in grid.tolist():
        candidate = fitted_map.at(h)
        _, fit_candidate_log_weights = transformed_log_weights(
            fit_draws,
            candidate,
            log_full_posterior,
            log_heldout_likelihood,
        )
        fit_center = jnp.mean(fit_candidate_log_weights)
        _, candidate_log_weights = transformed_log_weights(
            tune_draws,
            candidate,
            log_full_posterior,
            log_heldout_likelihood,
        )
        tuning_scores.append(
            clipped_log_weight_loss(
                candidate_log_weights,
                center=fit_center,
                clip=tuning_clip,
            )
        )
    tuning_scores_array = jnp.stack(tuning_scores)
    selected_index = int(jnp.argmin(tuning_scores_array))
    selected_map = fitted_map.at(float(grid[selected_index]))
    transformed_evaluation_draws, evaluation_log_weights = transformed_log_weights(
        evaluation_draws,
        selected_map,
        log_full_posterior,
        log_heldout_likelihood,
    )
    return RegularizedLOOResult(
        fitted_map=fitted_map,
        selected_map=selected_map,
        h_grid=grid,
        tuning_scores=tuning_scores_array,
        transformed_evaluation_draws=transformed_evaluation_draws,
        evaluation_log_weights=evaluation_log_weights,
        raw_effective_sample_size=raw_effective_sample_size(
            evaluation_log_weights
        ),
    )


def regularized_flow_loo_fold(
    fit_draws: Array,
    tune_draws: Array,
    evaluation_draws: Array,
    log_full_posterior: LogDensity,
    log_heldout_likelihood: LogDensity,
    vector_field: VectorField,
    vector_field_jacobian: VectorFieldJacobian,
    posterior_scale: Array,
    global_lipschitz_bound: float,
    h_grid: Sequence[float] = (
        0.0,
        1.0 / 256.0,
        1.0 / 128.0,
        1.0 / 64.0,
        1.0 / 32.0,
        1.0 / 16.0,
        1.0 / 8.0,
        1.0 / 4.0,
        1.0 / 2.0,
    ),
    tuning_clip: float = 5.0,
    log_target_function: LogDensity | None = None,
) -> RegularizedFlowLOOResult:
    """Fit, independently tune, and evaluate one LL/KL/Var field.

    Pass log_target_function for a positive, target-specific Var integrand.
    Its transformed log contribution is then used for tuning.  LL and KL use
    log-weight dispersion by leaving this callback as None.
    """

    fit_draws = _validate_draws(fit_draws, "fit_draws")
    tune_draws = _validate_draws(tune_draws, "tune_draws")
    evaluation_draws = _validate_draws(
        evaluation_draws, "evaluation_draws"
    )
    dimension = fit_draws.shape[1]
    if (
        tune_draws.shape[1] != dimension
        or evaluation_draws.shape[1] != dimension
    ):
        raise ValueError(
            "all three draw splits must have the same parameter dimension"
        )

    grid = jnp.asarray(
        tuple(float(h) for h in h_grid), dtype=fit_draws.dtype
    )
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("h_grid must be a nonempty one-dimensional sequence")
    if bool(jnp.any(grid < 0.0)):
        raise ValueError("every h_grid value must be nonnegative")
    max_h = float(jnp.max(grid))
    if max_h <= 0.0:
        raise ValueError("h_grid must contain at least one positive step")

    fitted_map = fit_vector_field(
        fit_draws,
        vector_field=vector_field,
        vector_field_jacobian=vector_field_jacobian,
        posterior_scale=posterior_scale,
        global_lipschitz_bound=global_lipschitz_bound,
        max_h=max_h,
    )

    def score_values(
        transformed_draws: Array, log_weights: Array
    ) -> Array:
        if log_target_function is None:
            return log_weights
        log_target = _validate_log_values(
            log_target_function(transformed_draws),
            transformed_draws.shape[0],
            "log_target_function",
        )
        return log_weights + log_target

    tuning_scores = []
    for h in grid.tolist():
        candidate = fitted_map.at(h)
        fit_transformed, fit_log_weights = transformed_log_weights(
            fit_draws,
            candidate,
            log_full_posterior,
            log_heldout_likelihood,
        )
        fit_center = jnp.mean(
            score_values(fit_transformed, fit_log_weights)
        )
        tune_transformed, tune_log_weights = transformed_log_weights(
            tune_draws,
            candidate,
            log_full_posterior,
            log_heldout_likelihood,
        )
        tuning_scores.append(
            clipped_log_weight_loss(
                score_values(tune_transformed, tune_log_weights),
                center=fit_center,
                clip=tuning_clip,
            )
        )
    tuning_scores_array = jnp.stack(tuning_scores)
    selected_index = int(jnp.argmin(tuning_scores_array))
    selected_map = fitted_map.at(float(grid[selected_index]))
    transformed_evaluation_draws, evaluation_log_weights = (
        transformed_log_weights(
            evaluation_draws,
            selected_map,
            log_full_posterior,
            log_heldout_likelihood,
        )
    )
    return RegularizedFlowLOOResult(
        fitted_map=fitted_map,
        selected_map=selected_map,
        h_grid=grid,
        tuning_scores=tuning_scores_array,
        transformed_evaluation_draws=transformed_evaluation_draws,
        evaluation_log_weights=evaluation_log_weights,
        raw_effective_sample_size=raw_effective_sample_size(
            evaluation_log_weights
        ),
    )


def three_fold_split(draws: Array, key: Array) -> tuple[Array, Array, Array]:
    """Randomly partition posterior draws into three disjoint folds.

    Reproducibility is controlled entirely by the caller-supplied JAX key.
    """

    draws = _validate_draws(draws, "draws")
    if draws.shape[0] < 6:
        raise ValueError("at least six draws are required for a three-fold split")
    permutation = jax.random.permutation(key, draws.shape[0])
    boundaries = (draws.shape[0] // 3, 2 * draws.shape[0] // 3)
    indices = (
        permutation[: boundaries[0]],
        permutation[boundaries[0] : boundaries[1]],
        permutation[boundaries[1] :],
    )
    return tuple(draws[index] for index in indices)


def crossfit_regularized_pmm_loo_fold(
    draw_folds: Sequence[Array],
    log_full_posterior: LogDensity,
    log_heldout_likelihood: LogDensity,
    method: PMMMethod = "pmm3",
    h_grid: Sequence[float] = tuple(i / 20.0 for i in range(21)),
    ridge: float = 1e-8,
    tuning_clip: float = 5.0,
) -> CrossFittedLOOResult:
    """Cross-fit one LOO fold by rotating three posterior-draw roles.

    With input folds ``(A, B, C)``, the rotations are ``(fit=A, tune=B,
    evaluate=C)``, ``(fit=B, tune=C, evaluate=A)``, and ``(fit=C, tune=A,
    evaluate=B)``.  Thus every draw is used once for final estimation without
    ever evaluating a map on draws that fitted or selected it.
    """

    if len(draw_folds) != 3:
        raise ValueError("cross-fitting requires exactly three draw folds")
    folds = tuple(
        _validate_draws(draws, f"draw_folds[{index}]")
        for index, draws in enumerate(draw_folds)
    )
    dimensions = {draws.shape[1] for draws in folds}
    if len(dimensions) != 1:
        raise ValueError("all draw folds must have the same parameter dimension")

    rotations = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    results = tuple(
        regularized_pmm_loo_fold(
            folds[fit_index],
            folds[tune_index],
            folds[evaluation_index],
            log_full_posterior=log_full_posterior,
            log_heldout_likelihood=log_heldout_likelihood,
            method=method,
            h_grid=h_grid,
            ridge=ridge,
            tuning_clip=tuning_clip,
        )
        for fit_index, tune_index, evaluation_index in rotations
    )
    transformed = jnp.concatenate(
        [result.transformed_evaluation_draws for result in results], axis=0
    )
    log_weights = jnp.concatenate(
        [result.evaluation_log_weights for result in results], axis=0
    )
    selected_steps = jnp.asarray(
        [result.selected_map.h for result in results], dtype=transformed.dtype
    )
    return CrossFittedLOOResult(
        fold_results=results,
        transformed_evaluation_draws=transformed,
        evaluation_log_weights=log_weights,
        selected_steps=selected_steps,
        raw_effective_sample_size=raw_effective_sample_size(log_weights),
    )


def crossfit_regularized_flow_loo_fold(
    draw_folds: Sequence[Array],
    log_full_posterior: LogDensity,
    log_heldout_likelihood: LogDensity,
    vector_field: VectorField,
    vector_field_jacobian: VectorFieldJacobian,
    posterior_scale: Array,
    global_lipschitz_bound: float,
    h_grid: Sequence[float] = (
        0.0,
        1.0 / 256.0,
        1.0 / 128.0,
        1.0 / 64.0,
        1.0 / 32.0,
        1.0 / 16.0,
        1.0 / 8.0,
        1.0 / 4.0,
        1.0 / 2.0,
    ),
    tuning_clip: float = 5.0,
    log_target_function: LogDensity | None = None,
) -> CrossFittedFlowLOOResult:
    """Cross-fit one LL/KL/Var field by rotating three draw roles."""

    if len(draw_folds) != 3:
        raise ValueError("cross-fitting requires exactly three draw folds")
    folds = tuple(
        _validate_draws(draws, f"draw_folds[{index}]")
        for index, draws in enumerate(draw_folds)
    )
    dimensions = {draws.shape[1] for draws in folds}
    if len(dimensions) != 1:
        raise ValueError("all draw folds must have the same parameter dimension")

    rotations = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    results = tuple(
        regularized_flow_loo_fold(
            folds[fit_index],
            folds[tune_index],
            folds[evaluation_index],
            log_full_posterior=log_full_posterior,
            log_heldout_likelihood=log_heldout_likelihood,
            vector_field=vector_field,
            vector_field_jacobian=vector_field_jacobian,
            posterior_scale=posterior_scale,
            global_lipschitz_bound=global_lipschitz_bound,
            h_grid=h_grid,
            tuning_clip=tuning_clip,
            log_target_function=log_target_function,
        )
        for fit_index, tune_index, evaluation_index in rotations
    )
    transformed = jnp.concatenate(
        [result.transformed_evaluation_draws for result in results], axis=0
    )
    log_weights = jnp.concatenate(
        [result.evaluation_log_weights for result in results], axis=0
    )
    selected_steps = jnp.asarray(
        [result.selected_map.h for result in results],
        dtype=transformed.dtype,
    )
    return CrossFittedFlowLOOResult(
        fold_results=results,
        transformed_evaluation_draws=transformed,
        evaluation_log_weights=log_weights,
        selected_steps=selected_steps,
        raw_effective_sample_size=raw_effective_sample_size(log_weights),
    )


__all__ = [
    "AffineMap",
    "CrossFittedFlowLOOResult",
    "CrossFittedLOOResult",
    "FittedPMM",
    "FittedVectorField",
    "RegularizedFlowLOOResult",
    "RegularizedLOOResult",
    "VectorFieldMap",
    "clipped_log_weight_loss",
    "crossfit_regularized_flow_loo_fold",
    "crossfit_regularized_pmm_loo_fold",
    "fit_pmm",
    "fit_vector_field",
    "log_weight_dispersion",
    "raw_effective_sample_size",
    "regularized_flow_loo_fold",
    "regularized_pmm_loo_fold",
    "self_normalized_expectation",
    "three_fold_split",
    "transformed_log_weights",
]

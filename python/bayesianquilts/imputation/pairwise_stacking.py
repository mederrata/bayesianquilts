"""
Pairwise Ordinal Stacking Model

Combines regression-based univariate models with Dirichlet-multinomial
contingency table models for predicting ordinal/categorical variables.

The contingency table approach uses the empirical joint distribution of two
variables to predict one from the other via row-conditional probabilities
with a Dirichlet prior for smoothing.

For each (target, predictor) pair where both are ordinal/categorical, this
module fits two candidate models:
  1. A regression model (ordinal logistic / logistic / linear)
  2. A Dirichlet-multinomial model based on the contingency table

Stacking weights are determined by LOO-ELPD.

This module is self-contained and does NOT depend on MICEBayesianLOO.
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict

import tensorflow_probability.substrates.jax as tfp

from bayesianquilts.imputation.univariate_models import (
    UnivariateModelResult,
    SimpleLinearRegression,
    SimpleLogisticRegression,
    SimpleOrdinalLogisticRegression,
    ordinal_one_hot_encode,
    infer_variable_type,
    run_inference_with_fallback,
    compute_loo_elpd,
)

tfd = tfp.distributions


# ---------------------------------------------------------------------------
# Frozen config for serialization-safe parallel workers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FitConfig:
    """Immutable config snapshot passed to module-level worker functions.

    By extracting these from ``self`` before the parallel loop, we avoid
    serializing the mutable ``PairwiseOrdinalStackingModel`` instance (whose
    result dicts change size as results are collected).
    """
    variable_types: tuple  # tuple of (idx, type_str) pairs
    prior_scale: float
    noise_scale: float
    pathfinder_num_samples: int
    pathfinder_maxiter: int
    min_obs: int
    batch_size: Optional[int]
    inference_method: str
    verbose: bool
    dtype: Any  # jnp dtype
    global_ordinal_values: Optional[tuple]  # None or tuple of floats
    n_global_classes: int

    def get_var_type(self, idx: int) -> Optional[str]:
        for i, t in self.variable_types:
            if i == idx:
                return t
        return None


def _make_fit_config(model: "PairwiseOrdinalStackingModel") -> _FitConfig:
    return _FitConfig(
        variable_types=tuple(model.variable_types.items()),
        prior_scale=model.prior_scale,
        noise_scale=model.noise_scale,
        pathfinder_num_samples=model.pathfinder_num_samples,
        pathfinder_maxiter=model.pathfinder_maxiter,
        min_obs=model.min_obs,
        batch_size=model.batch_size,
        inference_method=model.inference_method,
        verbose=model.verbose,
        dtype=model.dtype,
        global_ordinal_values=(
            tuple(model.global_ordinal_values.tolist())
            if model.global_ordinal_values is not None else None
        ),
        n_global_classes=model.n_global_classes,
    )


# ---------------------------------------------------------------------------
# Module-level worker functions (no reference to self)
# ---------------------------------------------------------------------------

def _worker_fit_zero_predictor(
    data: np.ndarray,
    target_idx: int,
    cfg: _FitConfig,
    seed: int,
    sample_weights: Optional[np.ndarray] = None,
) -> UnivariateModelResult:
    """Fit a zero-predictor (intercept-only) regression model."""
    mask = ~np.isnan(data[:, target_idx])
    y = data[mask, target_idx]
    n_obs = len(y)
    obs_weights = sample_weights[mask] if sample_weights is not None else None

    if n_obs < cfg.min_obs:
        return UnivariateModelResult(
            n_obs=n_obs, elpd_loo=float("-inf"),
            elpd_loo_per_obs=float("-inf"), elpd_loo_per_obs_se=float("inf"),
            khat_max=float("inf"), khat_mean=float("inf"),
            predictor_idx=None, target_idx=target_idx, converged=False,
        )

    var_type = cfg.get_var_type(target_idx)
    if var_type is None:
        var_type = infer_variable_type(y)

    scale_factor = 1.0
    if cfg.batch_size is not None and n_obs > cfg.batch_size:
        rng = np.random.RandomState(seed)
        subsample_idx = rng.choice(n_obs, size=cfg.batch_size, replace=False)
        y_batch = y[subsample_idx]
        scale_factor = n_obs / cfg.batch_size
        obs_weights_batch = obs_weights[subsample_idx] if obs_weights is not None else None
    else:
        y_batch = y
        obs_weights_batch = obs_weights

    X = np.zeros((len(y_batch), 1), dtype=np.float32)
    data_dict = {"X": X, "y": y_batch.astype(np.float32)}
    if obs_weights_batch is not None:
        data_dict["weights"] = obs_weights_batch.astype(np.float32)

    global_ov = np.array(cfg.global_ordinal_values) if cfg.global_ordinal_values is not None else None

    if var_type == "binary":
        model = SimpleLogisticRegression(
            n_predictors=1, prior_scale=cfg.prior_scale,
            dtype=cfg.dtype, n_obs=n_obs,
        )
    elif var_type == "ordinal":
        if global_ov is not None:
            unique_vals = global_ov
            n_classes = cfg.n_global_classes
        else:
            unique_vals = np.unique(y_batch)
            n_classes = len(unique_vals)
        val_map = {val: i for i, val in enumerate(sorted(unique_vals))}
        data_dict["y"] = np.array([val_map[val] for val in y_batch], dtype=np.float32)
        model = SimpleOrdinalLogisticRegression(
            n_classes=n_classes, n_predictors=1,
            prior_scale=cfg.prior_scale, dtype=cfg.dtype, n_obs=n_obs,
        )
    else:
        model = SimpleLinearRegression(
            n_predictors=1, prior_scale=cfg.prior_scale,
            noise_scale=cfg.noise_scale, dtype=cfg.dtype, n_obs=n_obs,
        )

    params, elbo, converged, _, surrogate_fn = run_inference_with_fallback(
        model, data_dict, scale_factor=scale_factor, seed=seed,
        current_dtype=cfg.dtype, inference_method=cfg.inference_method,
        num_samples=cfg.pathfinder_num_samples, maxiter=cfg.pathfinder_maxiter,
        verbose=cfg.verbose,
    )

    if not converged or params is None:
        return UnivariateModelResult(
            n_obs=n_obs, elpd_loo=float("-inf"),
            elpd_loo_per_obs=float("-inf"), elpd_loo_per_obs_se=float("inf"),
            khat_max=float("inf"), khat_mean=float("inf"),
            predictor_idx=None, target_idx=target_idx, converged=False,
        )

    elpd_loo, elpd_se, khat_max, khat_mean = compute_loo_elpd(model, data_dict, params)

    if khat_max > 0.7:
        return UnivariateModelResult(
            n_obs=n_obs, elpd_loo=float("-inf"),
            elpd_loo_per_obs=float("-inf"), elpd_loo_per_obs_se=float("inf"),
            khat_max=khat_max, khat_mean=khat_mean,
            predictor_idx=None, target_idx=target_idx, converged=False,
        )

    beta_eff = model._get_beta(params)
    beta_mean = np.array(np.mean(beta_eff, axis=0))
    intercept_mean = float(np.mean(params["intercept"])) if "intercept" in params else None
    cutpoints_mean = None
    if "cutpoints_raw" in params:
        transformed = jax.vmap(model._transform_cutpoints)(params["cutpoints_raw"])
        cutpoints_mean = np.array(np.mean(transformed, axis=0))

    return UnivariateModelResult(
        n_obs=n_obs, elpd_loo=elpd_loo,
        elpd_loo_per_obs=elpd_loo / n_obs if n_obs > 0 else float("-inf"),
        elpd_loo_per_obs_se=elpd_se / n_obs if n_obs > 0 else float("inf"),
        khat_max=khat_max, khat_mean=khat_mean,
        predictor_idx=None, target_idx=target_idx, converged=converged,
        params=None, beta_mean=beta_mean,
        intercept_mean=intercept_mean, cutpoints_mean=cutpoints_mean,
    )


def _worker_fit_univariate(
    data: np.ndarray,
    target_idx: int,
    predictor_idx: int,
    cfg: _FitConfig,
    seed: int,
    sample_weights: Optional[np.ndarray] = None,
) -> UnivariateModelResult:
    """Fit a one-predictor univariate regression model."""
    mask = ~np.isnan(data[:, target_idx]) & ~np.isnan(data[:, predictor_idx])
    n_obs = int(np.sum(mask))
    obs_weights = sample_weights[mask] if sample_weights is not None else None

    if n_obs < cfg.min_obs:
        return UnivariateModelResult(
            n_obs=n_obs, elpd_loo=float("-inf"),
            elpd_loo_per_obs=float("-inf"), elpd_loo_per_obs_se=float("inf"),
            khat_max=float("inf"), khat_mean=float("inf"),
            predictor_idx=predictor_idx, target_idx=target_idx, converged=False,
        )

    X_raw = data[mask, predictor_idx : predictor_idx + 1].astype(np.float32)
    y = data[mask, target_idx].astype(np.float32)

    target_var_type = cfg.get_var_type(target_idx)
    if target_var_type is None:
        target_var_type = infer_variable_type(y)

    predictor_var_type = cfg.get_var_type(predictor_idx)
    if predictor_var_type is None:
        predictor_var_type = infer_variable_type(data[mask, predictor_idx])

    scale_factor = 1.0
    if cfg.batch_size is not None and n_obs > cfg.batch_size:
        rng = np.random.RandomState(seed)
        subsample_idx = rng.choice(n_obs, size=cfg.batch_size, replace=False)
        X_raw_batch = X_raw[subsample_idx]
        y_batch = y[subsample_idx]
        scale_factor = n_obs / cfg.batch_size
        obs_weights_batch = obs_weights[subsample_idx] if obs_weights is not None else None
    else:
        X_raw_batch = X_raw
        y_batch = y
        obs_weights_batch = obs_weights

    global_ov = np.array(cfg.global_ordinal_values) if cfg.global_ordinal_values is not None else None

    if predictor_var_type == "ordinal":
        if global_ov is not None:
            max_val = int(np.max(global_ov))
        else:
            max_val = int(np.max(X_raw_batch.flatten()))
        X = ordinal_one_hot_encode(X_raw_batch.astype(int), max_val).astype(np.float32)
        X_mean = 0.0
        X_std = 1.0
        n_predictors = X.shape[1]
    else:
        X = X_raw_batch
        X_mean = float(np.mean(X))
        X_std = float(np.std(X))
        if X_std > 1e-6:
            X = (X - X_mean) / X_std
        else:
            X_std = 1.0
        n_predictors = 1

    data_dict = {"X": X, "y": y_batch}
    if obs_weights_batch is not None:
        data_dict["weights"] = obs_weights_batch.astype(np.float32)

    if target_var_type == "binary":
        model = SimpleLogisticRegression(
            n_predictors=n_predictors, prior_scale=cfg.prior_scale,
            dtype=cfg.dtype, n_obs=n_obs,
        )
    elif target_var_type == "ordinal":
        if global_ov is not None:
            unique_vals = global_ov
            n_classes = cfg.n_global_classes
        else:
            unique_vals = np.unique(y_batch)
            n_classes = len(unique_vals)
        val_map = {val: i for i, val in enumerate(sorted(unique_vals))}
        data_dict["y"] = np.array([val_map[val] for val in y_batch], dtype=np.float32)
        model = SimpleOrdinalLogisticRegression(
            n_classes=n_classes, n_predictors=n_predictors,
            prior_scale=cfg.prior_scale, dtype=cfg.dtype, n_obs=n_obs,
        )
    else:
        model = SimpleLinearRegression(
            n_predictors=n_predictors, prior_scale=cfg.prior_scale,
            noise_scale=cfg.noise_scale, dtype=cfg.dtype, n_obs=n_obs,
        )

    params, elbo, converged, _, surrogate_fn = run_inference_with_fallback(
        model, data_dict, scale_factor=scale_factor, seed=seed,
        current_dtype=cfg.dtype, inference_method=cfg.inference_method,
        num_samples=cfg.pathfinder_num_samples, maxiter=cfg.pathfinder_maxiter,
        verbose=cfg.verbose,
    )

    if not converged or params is None:
        return UnivariateModelResult(
            n_obs=n_obs, elpd_loo=float("-inf"),
            elpd_loo_per_obs=float("-inf"), elpd_loo_per_obs_se=float("inf"),
            khat_max=float("inf"), khat_mean=float("inf"),
            predictor_idx=predictor_idx, target_idx=target_idx, converged=False,
            predictor_mean=X_mean, predictor_std=X_std,
        )

    elpd_loo, elpd_se, khat_max, khat_mean = compute_loo_elpd(model, data_dict, params)

    if khat_max > 0.7:
        return UnivariateModelResult(
            n_obs=n_obs, elpd_loo=float("-inf"),
            elpd_loo_per_obs=float("-inf"), elpd_loo_per_obs_se=float("inf"),
            khat_max=khat_max, khat_mean=khat_mean,
            predictor_idx=predictor_idx, target_idx=target_idx, converged=False,
            predictor_mean=X_mean, predictor_std=X_std,
        )

    beta_eff = model._get_beta(params)
    beta_mean = np.array(np.mean(beta_eff, axis=0))
    intercept_mean = float(np.mean(params["intercept"])) if "intercept" in params else None
    cutpoints_mean = None
    if "cutpoints_raw" in params:
        transformed = jax.vmap(model._transform_cutpoints)(params["cutpoints_raw"])
        cutpoints_mean = np.array(np.mean(transformed, axis=0))

    return UnivariateModelResult(
        n_obs=n_obs, elpd_loo=elpd_loo,
        elpd_loo_per_obs=elpd_loo / n_obs if n_obs > 0 else float("-inf"),
        elpd_loo_per_obs_se=elpd_se / n_obs if n_obs > 0 else float("inf"),
        khat_max=khat_max, khat_mean=khat_mean,
        predictor_idx=predictor_idx, target_idx=target_idx,
        converged=converged, params=None,
        predictor_mean=X_mean, predictor_std=X_std,
        beta_mean=beta_mean, intercept_mean=intercept_mean,
        cutpoints_mean=cutpoints_mean,
    )


# ---------------------------------------------------------------------------
# Dirichlet-multinomial contingency table model
# ---------------------------------------------------------------------------

@dataclass
class DirichletMultinomialResult:
    """Results from fitting a Dirichlet-multinomial contingency table model."""

    n_obs: int
    elpd_loo: float
    elpd_loo_per_obs: float
    elpd_loo_per_obs_se: float
    predictor_idx: Optional[int]
    target_idx: int
    converged: bool
    # Row-conditional parameters: alpha[k, :] are Dirichlet concentrations
    # for target categories given predictor category k.
    alpha_posterior: Optional[np.ndarray] = None  # (K_pred, K_target)
    predictor_categories: Optional[np.ndarray] = None
    target_categories: Optional[np.ndarray] = None


class DirichletMultinomialContingency:
    """Dirichlet-multinomial model for the contingency table between two
    categorical/ordinal variables.

    For predictor category k, the target distribution is modelled as::

        theta_k ~ Dirichlet(alpha_k)
        y | x=k  ~ Categorical(theta_k)

    The posterior after observing the contingency table row n_k is::

        theta_k | n_k ~ Dirichlet(alpha_k + n_k)

    LOO-ELPD is computed analytically via the leave-one-out predictive
    probability of a Dirichlet-multinomial (Polya distribution).

    The default prior uses ``alpha = 0.5`` per cell (Jeffreys prior for the
    multinomial), which is the standard reference prior. It ensures that
    unobserved cells get a small but nonzero predicted probability without
    the oversmoothing that ``alpha = 1`` (Laplace/uniform) would cause in
    sparse contingency tables.
    """

    def __init__(
        self,
        alpha_prior: float = 0.5,
        min_obs: int = 5,
    ):
        """
        Args:
            alpha_prior: Symmetric Dirichlet concentration per cell.
                         0.5 = Jeffreys prior (default, standard reference prior).
                         1.0 = Laplace smoothing (uniform Dirichlet).
                         Values < 1 give sparser posteriors; values > 1
                         pull more toward uniform.
            min_obs: Minimum total observations required.
        """
        self.alpha_prior = alpha_prior
        self.min_obs = min_obs

    def fit_and_loo(
        self,
        x: np.ndarray,
        y: np.ndarray,
        predictor_categories: np.ndarray,
        target_categories: np.ndarray,
        predictor_idx: int,
        target_idx: int,
        weights: Optional[np.ndarray] = None,
    ) -> DirichletMultinomialResult:
        """Fit the contingency table model and compute LOO-ELPD.

        Args:
            x: Predictor values (integer-coded), shape (N,).
            y: Target values (integer-coded), shape (N,).
            predictor_categories: Sorted unique predictor category values.
            target_categories: Sorted unique target category values.
            predictor_idx: Index of predictor variable.
            target_idx: Index of target variable.
            weights: Per-observation sampling weights, shape (N,). When
                     provided, weighted counts are normalized to the Kish
                     effective sample size to maintain correct posterior
                     precision.

        Returns:
            DirichletMultinomialResult with LOO-ELPD and posterior concentrations.
        """
        n_obs = len(x)
        K_pred = len(predictor_categories)
        K_target = len(target_categories)

        if n_obs < self.min_obs:
            return DirichletMultinomialResult(
                n_obs=n_obs,
                elpd_loo=float("-inf"),
                elpd_loo_per_obs=float("-inf"),
                elpd_loo_per_obs_se=float("inf"),
                predictor_idx=predictor_idx,
                target_idx=target_idx,
                converged=False,
            )

        # Map raw values to 0-based indices
        pred_map = {v: i for i, v in enumerate(predictor_categories)}
        tgt_map = {v: i for i, v in enumerate(target_categories)}

        x_idx = np.array([pred_map[v] for v in x], dtype=int)
        y_idx = np.array([tgt_map[v] for v in y], dtype=int)

        # Build contingency table with optional weighting
        counts = np.zeros((K_pred, K_target), dtype=np.float64)
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            # Normalize weights to Kish effective sample size
            n_eff = (w.sum()) ** 2 / (w ** 2).sum()
            w_normalized = w * (n_eff / w.sum())
            for xi, yi, wi in zip(x_idx, y_idx, w_normalized):
                counts[xi, yi] += wi
        else:
            for xi, yi in zip(x_idx, y_idx):
                counts[xi, yi] += 1.0
            n_eff = float(n_obs)

        # Posterior concentrations per row: alpha_post[k, :] = alpha_prior + counts[k, :]
        alpha0 = self.alpha_prior
        alpha_post = alpha0 + counts  # (K_pred, K_target)

        # LOO predictive log-probability for each observation.
        # For observation i with x_i = k, y_i = j, the LOO contribution
        # is the weight of observation i.  Under weighting, the analytic
        # LOO formula is approximate (exact only for unit weights).
        loos = np.zeros(n_obs)
        for i in range(n_obs):
            k = x_idx[i]
            j = y_idx[i]
            wi = w_normalized[i] if weights is not None else 1.0
            row_sum = alpha_post[k, :].sum()
            numer = alpha_post[k, j] - wi
            denom = row_sum - wi
            if denom > 0 and numer > 0:
                loos[i] = np.log(numer / denom)
            else:
                loos[i] = -np.inf

        elpd_loo = float(np.sum(loos))
        elpd_se = float(np.sqrt(n_eff * np.var(loos)))

        return DirichletMultinomialResult(
            n_obs=n_obs,
            elpd_loo=elpd_loo,
            elpd_loo_per_obs=elpd_loo / n_obs if n_obs > 0 else float("-inf"),
            elpd_loo_per_obs_se=elpd_se / n_obs if n_obs > 0 else float("inf"),
            predictor_idx=predictor_idx,
            target_idx=target_idx,
            converged=True,
            alpha_posterior=alpha_post,
            predictor_categories=predictor_categories,
            target_categories=target_categories,
        )

    def predict_pmf(
        self,
        result: DirichletMultinomialResult,
        predictor_value: float,
    ) -> np.ndarray:
        """Predict target PMF given a predictor value.

        Args:
            result: Fitted DirichletMultinomialResult.
            predictor_value: Observed value of the predictor variable.

        Returns:
            Array of shape (K_target,) with predicted probabilities.
        """
        if result.alpha_posterior is None or result.predictor_categories is None:
            K = len(result.target_categories) if result.target_categories is not None else 2
            return np.ones(K) / K

        pred_cats = result.predictor_categories
        # Find the closest category
        dists = np.abs(pred_cats - predictor_value)
        k = int(np.argmin(dists))

        # Posterior predictive = normalised Dirichlet concentration
        alpha_row = result.alpha_posterior[k, :]
        total = alpha_row.sum()
        if total > 0:
            return alpha_row / total
        else:
            K = len(alpha_row)
            return np.ones(K) / K

    def predict_expected(
        self,
        result: DirichletMultinomialResult,
        predictor_value: float,
    ) -> float:
        """Predict expected target value given a predictor value.

        Computes E[Y | X = predictor_value] using the posterior predictive PMF
        and the target category values.
        """
        pmf = self.predict_pmf(result, predictor_value)
        if result.target_categories is not None:
            return float(np.sum(result.target_categories * pmf))
        else:
            return float(np.sum(np.arange(len(pmf)) * pmf))


# ---------------------------------------------------------------------------
# Main stacking model
# ---------------------------------------------------------------------------

class PairwiseOrdinalStackingModel:
    """Stacking model that combines regression models and Dirichlet-multinomial
    contingency table models for pairwise prediction between variables.

    This model:
    1. Fits regression-based univariate models (linear, logistic, ordinal
       logistic) with horseshoe priors for automatic variable selection.
    2. Fits Dirichlet-multinomial contingency table models for pairs where
       both variables are ordinal or categorical.
    3. Retains zero-predictor (intercept-only) models as a baseline for
       both regression and DM model families.
    4. Computes stacking weights via LOO-ELPD.

    This class is self-contained and does not depend on MICEBayesianLOO.
    """

    def __init__(
        self,
        alpha_prior: float = 0.5,
        prior_scale: float = 1.0,
        noise_scale: float = 1.0,
        pathfinder_num_samples: int = 200,
        pathfinder_maxiter: int = 100,
        min_obs: int = 5,
        batch_size: Optional[int] = None,
        inference_method: str = "pathfinder",
        verbose: bool = True,
    ):
        """
        Args:
            alpha_prior: Symmetric Dirichlet concentration for DM models.
                         0.5 = Jeffreys prior (default).
            prior_scale: Prior scale for regression coefficients.
            noise_scale: Prior scale for noise (continuous variables).
            pathfinder_num_samples: Number of samples for Pathfinder.
            pathfinder_maxiter: Maximum iterations for Pathfinder.
            min_obs: Minimum observations required to fit a model.
            batch_size: Maximum observations per regression model fit
                       (None = use all data).
            inference_method: 'pathfinder' or 'advi'.
            verbose: Print progress information.
        """
        self.alpha_prior = alpha_prior
        self.prior_scale = prior_scale
        self.noise_scale = noise_scale
        self.pathfinder_num_samples = pathfinder_num_samples
        self.pathfinder_maxiter = pathfinder_maxiter
        self.min_obs = min_obs
        self.batch_size = batch_size
        self.inference_method = inference_method
        self.verbose = verbose
        self.dtype = jnp.float32

        self.dm_model = DirichletMultinomialContingency(
            alpha_prior=alpha_prior, min_obs=min_obs
        )

        # Fitted state
        self.variable_names: List[str] = []
        self.variable_types: Dict[int, str] = {}
        self.n_obs_total: int = 0
        self.global_ordinal_values: Optional[np.ndarray] = None
        self.n_global_classes: int = 0

        # Regression results
        self.zero_predictor_results: Dict[int, UnivariateModelResult] = {}
        self.univariate_results: Dict[Tuple[int, int], UnivariateModelResult] = {}

        # Dirichlet-multinomial results
        self.dm_results: Dict[Tuple[int, int], DirichletMultinomialResult] = {}
        self.dm_zero_results: Dict[int, DirichletMultinomialResult] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_overlapping_mask(self, data: np.ndarray, idx1: int, idx2: int) -> np.ndarray:
        return ~np.isnan(data[:, idx1]) & ~np.isnan(data[:, idx2])

    # ------------------------------------------------------------------
    # Prediction helpers (ported from MICEBayesianLOO)
    # ------------------------------------------------------------------

    def _predict_single_univariate(
        self,
        uni_result: UnivariateModelResult,
        predictor_value: float,
        target_var_type: str,
    ) -> float:
        """Predict a single value using a univariate regression model."""
        X_mean = uni_result.predictor_mean if uni_result.predictor_mean is not None else 0.0
        X_std = uni_result.predictor_std if uni_result.predictor_std is not None else 1.0
        x_standardized = (predictor_value - X_mean) / X_std

        beta = uni_result.beta_mean
        intercept = uni_result.intercept_mean if uni_result.intercept_mean is not None else 0.0
        cutpoints = uni_result.cutpoints_mean

        if beta is None:
            raise ValueError("No parameters available for prediction")

        beta_val = beta[0] if isinstance(beta, (np.ndarray, list)) else beta
        eta = float(x_standardized * beta_val + intercept)

        if target_var_type == "binary":
            return float(1.0 / (1.0 + np.exp(-eta)))
        elif target_var_type == "ordinal" and cutpoints is not None:
            def sigmoid(x):
                return 1.0 / (1.0 + np.exp(-x))

            p_le = sigmoid(cutpoints - eta)
            p_le = np.concatenate([[0.0], p_le, [1.0]])
            p = np.diff(p_le)
            categories = np.arange(len(p))
            return float(np.sum(categories * p))
        else:
            return float(eta)

    def _ordinal_pmf_from_regression(
        self,
        result: UnivariateModelResult,
        eta: float,
        n_categories: int,
    ) -> np.ndarray:
        """Compute category PMF from a regression model."""
        cutpoints = result.cutpoints_mean
        if cutpoints is None:
            if n_categories == 2:
                def _sigmoid(x):
                    return 1.0 / (1.0 + np.exp(-x))
                p1 = _sigmoid(eta)
                return np.array([1.0 - p1, p1])
            return np.ones(n_categories) / n_categories

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        p_le = sigmoid(cutpoints - eta)
        p_le = np.concatenate([[0.0], p_le, [1.0]])
        p = np.diff(p_le)
        p = np.clip(p, 0, None)
        total = p.sum()
        if total > 0:
            p /= total
        else:
            p = np.ones(n_categories) / n_categories
        return p

    def _regression_zero_predict(
        self, zr: UnivariateModelResult, var_type: str,
    ) -> float:
        """Predict from a zero-predictor regression model."""
        intercept = (
            zr.intercept_mean
            if zr.intercept_mean is not None
            else (
                float(np.mean(zr.params["intercept"]))
                if zr.params and "intercept" in zr.params
                else 0.0
            )
        )
        if var_type == "binary":
            return float(1.0 / (1.0 + np.exp(-intercept)))
        elif var_type == "ordinal" and zr.cutpoints_mean is not None:
            def sigmoid(x):
                return 1.0 / (1.0 + np.exp(-x))
            p_le = sigmoid(zr.cutpoints_mean - intercept)
            p_le = np.concatenate([[0.0], p_le, [1.0]])
            p = np.diff(p_le)
            return float(np.sum(np.arange(len(p)) * p))
        else:
            return float(intercept)

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------

    def fit(
        self,
        X_df: pd.DataFrame,
        seed: int = 42,
        n_jobs: int = -1,
        n_top_features: int = 50,
        save_dir: Optional[Union[str, Path]] = None,
        groups: Optional[np.ndarray] = None,
        group_weights: Optional[Dict[Any, float]] = None,
    ) -> "PairwiseOrdinalStackingModel":
        """Fit all regression and Dirichlet-multinomial models.

        Args:
            X_df: DataFrame with potentially missing values (NaN).
            seed: Random seed.
            n_jobs: Number of parallel jobs for regression models.
                    -1 uses all cores.
            n_top_features: Top correlated features per target for regression.
            save_dir: Directory to save incremental results.
            groups: Group label for each respondent, shape (n_respondents,).
                    Used with ``group_weights`` to compute IPW sampling weights
                    that correct for stratified or oversampled calibration data.
                    Group 0 is treated as the general population, which may
                    contain unlabeled members of other groups at their natural
                    population rate.
            group_weights: Target population proportion for each group label,
                    e.g. ``{0: 0.95, 1: 0.05}``. Must sum to 1. Required when
                    ``groups`` is provided.

        Returns:
            self
        """
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs):
                return iterable

        try:
            from joblib import Parallel, delayed
        except ImportError:
            print("Warning: joblib not installed. Falling back to sequential.")
            n_jobs = 1

            def Parallel(n_jobs=1, **kwargs):
                return lambda x: list(x)

            def delayed(func):
                return func

        data = X_df.values
        self.variable_names = list(X_df.columns)
        n_variables = data.shape[1]
        self.n_obs_total = data.shape[0]

        # Compute per-respondent sampling weights from group labels
        sample_weights = None
        if groups is not None:
            if group_weights is None:
                raise ValueError("group_weights must be provided when groups is specified")
            groups = np.asarray(groups)
            if len(groups) != self.n_obs_total:
                raise ValueError(
                    f"groups length ({len(groups)}) must match number of "
                    f"respondents ({self.n_obs_total})"
                )
            n = self.n_obs_total
            sample_weights = np.ones(n, dtype=np.float64)
            for g, W_g in group_weights.items():
                g_mask = groups == g
                n_g = int(g_mask.sum())
                if n_g > 0:
                    sample_weights[g_mask] = W_g * n / n_g
            if self.verbose:
                print(f"  IPW sampling weights: min={sample_weights.min():.3f}, "
                      f"max={sample_weights.max():.3f}, "
                      f"n_eff={sample_weights.sum()**2 / (sample_weights**2).sum():.0f}")

        # 1. Infer variable types and global ordinal values
        all_ordinal_values = set()
        for i in range(n_variables):
            var_type = self.variable_types.get(i)
            if var_type is None:
                y_obs = data[~np.isnan(data[:, i]), i]
                var_type = infer_variable_type(y_obs)
                self.variable_types[i] = var_type
            if var_type == "ordinal":
                y_obs = data[~np.isnan(data[:, i]), i]
                all_ordinal_values.update(np.unique(y_obs).tolist())

        if all_ordinal_values:
            self.global_ordinal_values = np.array(sorted(list(all_ordinal_values)))
            self.n_global_classes = len(self.global_ordinal_values)
        else:
            self.global_ordinal_values = None
            self.n_global_classes = 0

        # Compute Spearman correlation matrix for feature selection
        if self.verbose:
            print("Computing feature correlations...")
        corr_matrix = np.array(X_df.corr(method="spearman").abs().values)
        np.fill_diagonal(corr_matrix, -1.0)

        if self.verbose:
            print(f"Fitting PairwiseOrdinalStackingModel")
            print(f"  Variables: {n_variables}")
            print(f"  Observations: {self.n_obs_total}")
            print(f"  Min obs per model: {self.min_obs}")
            print(f"  Parallel jobs: {n_jobs}")
            print(f"  Top features per target: {n_top_features}")
            print(f"  DM alpha prior: {self.alpha_prior}")
            if self.n_global_classes > 0:
                print(f"  Global ordinal values: {self.global_ordinal_values} (n={self.n_global_classes})")

        # Snapshot config for parallel workers (avoids serializing self)
        cfg = _make_fit_config(self)

        # ----------------------------------------------------------
        # Phase 1: Fit zero-predictor regression models
        # ----------------------------------------------------------
        if self.verbose:
            print("\nFitting zero-predictor regression models...")

        try:
            results_gen = Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(_worker_fit_zero_predictor)(data, i, cfg, seed + i, sample_weights)
                for i in range(n_variables)
            )
        except Exception as e:
            if n_jobs != 1:
                if self.verbose:
                    print(f"  Parallel fitting failed ({e}), falling back to sequential")
                n_jobs = 1
                results_gen = (
                    _worker_fit_zero_predictor(data, i, cfg, seed + i, sample_weights)
                    for i in range(n_variables)
                )

        if not self.verbose:
            results_gen = tqdm(results_gen, total=n_variables, desc="Zero-Predictor Regression")

        for i, result in enumerate(results_gen):
            self.zero_predictor_results[i] = result
            if self.verbose and result.converged:
                print(f"  Var {i} ({self.variable_names[i]}): elpd/n={result.elpd_loo_per_obs:.4f}")

        # ----------------------------------------------------------
        # Phase 2: Fit one-predictor regression models
        # ----------------------------------------------------------
        if self.verbose:
            print("\nFitting one-predictor regression models...")

        for i in range(n_variables):
            target_name = self.variable_names[i]

            # Identify top correlated predictors
            target_corrs = corr_matrix[i, :]
            valid_mask = np.isfinite(target_corrs)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                continue

            sorted_indices = valid_indices[np.argsort(target_corrs[valid_indices])[::-1]]
            sorted_indices = [idx for idx in sorted_indices if idx != i]
            top_features = sorted_indices[:n_top_features]

            valid_predictors = []
            for j in top_features:
                mask = self._get_overlapping_mask(data, i, j)
                if np.sum(mask) >= self.min_obs:
                    valid_predictors.append(j)

            if not valid_predictors:
                continue

            if self.verbose:
                print(f"  Processing {target_name} ({len(valid_predictors)} predictors)")

            try:
                results_gen = Parallel(n_jobs=n_jobs, return_as="generator")(
                    delayed(_worker_fit_univariate)(
                        data, i, j, cfg, seed + n_variables + (i * n_variables + j),
                        sample_weights,
                    )
                    for j in valid_predictors
                )
            except Exception as e:
                if n_jobs != 1:
                    if self.verbose:
                        print(f"  Parallel fitting failed ({e}), falling back to sequential")
                    n_jobs = 1
                results_gen = (
                    _worker_fit_univariate(
                        data, i, j, cfg, seed + n_variables + (i * n_variables + j),
                        sample_weights,
                    )
                    for j in valid_predictors
                )

            if not self.verbose:
                results_gen = tqdm(
                    results_gen, total=len(valid_predictors),
                    desc=f"Predictors for {target_name[:20]}...", leave=False,
                )

            for j, result in zip(valid_predictors, results_gen):
                self.univariate_results[(i, j)] = result

        # ----------------------------------------------------------
        # Phase 3: Fit Dirichlet-multinomial models
        # ----------------------------------------------------------
        if self.verbose:
            print("\nFitting Dirichlet-multinomial contingency table models...")

        categorical_indices = [
            i for i in range(n_variables)
            if self.variable_types.get(i) in ("ordinal", "binary")
        ]

        if self.verbose:
            print(f"  Categorical/ordinal variables: {len(categorical_indices)} of {n_variables}")

        # Category values per variable
        var_categories: Dict[int, np.ndarray] = {}
        for i in categorical_indices:
            obs = data[~np.isnan(data[:, i]), i]
            var_categories[i] = np.sort(np.unique(obs))

        # Zero-predictor DM models (marginal)
        for i in categorical_indices:
            obs_mask = ~np.isnan(data[:, i])
            y_obs = data[obs_mask, i]
            cats = var_categories[i]
            if len(y_obs) < self.min_obs:
                continue
            dummy_x = np.zeros(len(y_obs), dtype=int)
            dm_w = sample_weights[obs_mask] if sample_weights is not None else None
            result = self.dm_model.fit_and_loo(
                x=dummy_x, y=y_obs,
                predictor_categories=np.array([0]),
                target_categories=cats,
                predictor_idx=-1, target_idx=i,
                weights=dm_w,
            )
            self.dm_zero_results[i] = result
            if self.verbose and result.converged:
                print(f"  Zero-predictor DM var {i} ({self.variable_names[i]}): elpd/n={result.elpd_loo_per_obs:.4f}")

        # Pairwise DM models
        n_dm_fitted = 0
        for i in categorical_indices:
            target_cats = var_categories[i]
            for j in categorical_indices:
                if i == j:
                    continue
                pred_cats = var_categories[j]
                mask = ~np.isnan(data[:, i]) & ~np.isnan(data[:, j])
                n_overlap = int(np.sum(mask))
                if n_overlap < self.min_obs:
                    continue

                dm_w = sample_weights[mask] if sample_weights is not None else None
                result = self.dm_model.fit_and_loo(
                    x=data[mask, j], y=data[mask, i],
                    predictor_categories=pred_cats,
                    target_categories=target_cats,
                    predictor_idx=j, target_idx=i,
                    weights=dm_w,
                )
                self.dm_results[(i, j)] = result
                n_dm_fitted += 1

                if self.verbose and result.converged:
                    print(
                        f"  DM ({self.variable_names[j]}->{self.variable_names[i]}): "
                        f"elpd/n={result.elpd_loo_per_obs:.4f}, n={result.n_obs}"
                    )

        if self.verbose:
            n_reg_zero = sum(1 for r in self.zero_predictor_results.values() if r.converged)
            n_reg_uni = sum(1 for r in self.univariate_results.values() if r.converged)
            n_dm_conv = sum(1 for r in self.dm_results.values() if r.converged)
            n_dm_zero = sum(1 for r in self.dm_zero_results.values() if r.converged)
            print(f"\nCompleted:")
            print(f"  Regression zero-predictor: {n_reg_zero}/{len(self.zero_predictor_results)}")
            print(f"  Regression univariate: {n_reg_uni}/{len(self.univariate_results)}")
            print(f"  DM zero-predictor: {n_dm_zero}/{len(self.dm_zero_results)}")
            print(f"  DM pairwise: {n_dm_conv}/{n_dm_fitted}")

        return self

    # ------------------------------------------------------------------
    # predict()
    # ------------------------------------------------------------------

    def predict(
        self,
        items: Dict[str, float],
        target: str,
        return_details: bool = False,
        uncertainty_penalty: float = 1.0,
    ) -> Union[float, Dict[str, Any]]:
        """Predict a target variable using stacked regression + DM models.

        Collects predictions from:
          - Zero-predictor regression model (intercept-only)
          - Zero-predictor DM model (marginal)
          - Univariate regression models for each available predictor
          - DM contingency models for each available categorical predictor

        Stacking weights are computed from exp(elpd - penalty * se).

        Args:
            items: Dict mapping variable names to observed values.
            target: Name of the target variable to predict.
            return_details: If True, return detailed breakdown.
            uncertainty_penalty: Penalty factor for ELPD uncertainty.

        Returns:
            Stacked prediction (float), or dict with details if requested.
        """
        if target not in self.variable_names:
            raise ValueError(f"Target '{target}' not in variable_names: {self.variable_names}")

        target_idx = self.variable_names.index(target)
        var_type = self.variable_types.get(target_idx, "continuous")

        # Collect (name, elpd_per_obs, se_per_obs, prediction)
        models_info: List[Tuple[str, float, float, float]] = []

        # 1. Zero-predictor regression model
        if target_idx in self.zero_predictor_results:
            zr = self.zero_predictor_results[target_idx]
            if zr.converged:
                pred = self._regression_zero_predict(zr, var_type)
                models_info.append(("reg:intercept", zr.elpd_loo_per_obs, zr.elpd_loo_per_obs_se, pred))

        # 2. Zero-predictor DM model (marginal)
        if target_idx in self.dm_zero_results:
            dmz = self.dm_zero_results[target_idx]
            if dmz.converged:
                pred = self.dm_model.predict_expected(dmz, 0)
                models_info.append(("dm:marginal", dmz.elpd_loo_per_obs, dmz.elpd_loo_per_obs_se, pred))

        # 3. Per-predictor models
        for predictor_name, predictor_value in items.items():
            if predictor_name not in self.variable_names or predictor_name == target:
                continue

            predictor_idx = self.variable_names.index(predictor_name)
            key = (target_idx, predictor_idx)

            # Regression model
            if key in self.univariate_results:
                ur = self.univariate_results[key]
                if ur.converged:
                    pred = self._predict_single_univariate(ur, predictor_value, var_type)
                    models_info.append((f"reg:{predictor_name}", ur.elpd_loo_per_obs, ur.elpd_loo_per_obs_se, float(pred)))

            # DM model
            if key in self.dm_results:
                dmr = self.dm_results[key]
                if dmr.converged:
                    pred = self.dm_model.predict_expected(dmr, predictor_value)
                    models_info.append((f"dm:{predictor_name}", dmr.elpd_loo_per_obs, dmr.elpd_loo_per_obs_se, float(pred)))

        if not models_info:
            raise ValueError(f"No converged models available for target '{target}'")

        stacked_pred, weights = self._compute_stacking_weights(models_info, uncertainty_penalty)

        if return_details:
            return {
                "prediction": stacked_pred,
                "weights": {m[0]: float(w) for m, w in zip(models_info, weights)},
                "elpd_loo_per_obs": {m[0]: float(m[1]) for m in models_info},
                "elpd_loo_per_obs_se": {m[0]: float(m[2]) for m in models_info},
                "predictions": {m[0]: float(m[3]) for m in models_info},
                "n_obs_total": self.n_obs_total,
            }
        return stacked_pred

    # ------------------------------------------------------------------
    # predict_pmf()
    # ------------------------------------------------------------------

    def predict_pmf(
        self,
        items: Dict[str, float],
        target: str,
        n_categories: Optional[int] = None,
        uncertainty_penalty: float = 1.0,
    ) -> np.ndarray:
        """Predict a full categorical PMF for a target variable.

        Mixes regression-based ordinal PMFs with DM-based PMFs using stacking
        weights.

        Args:
            items: Observed variable name -> value mapping.
            target: Target variable name.
            n_categories: Number of response categories. If None, inferred.
            uncertainty_penalty: Penalty for ELPD uncertainty.

        Returns:
            Array of shape (n_categories,) summing to 1.
        """
        if target not in self.variable_names:
            raise ValueError(f"Target '{target}' not in variable_names: {self.variable_names}")

        target_idx = self.variable_names.index(target)

        # Infer n_categories
        if n_categories is None:
            for dmr in list(self.dm_zero_results.values()) + list(self.dm_results.values()):
                if dmr.target_idx == target_idx and dmr.target_categories is not None:
                    n_categories = len(dmr.target_categories)
                    break
            if n_categories is None and self.n_global_classes > 0:
                n_categories = self.n_global_classes
            if n_categories is None:
                raise ValueError("Cannot infer n_categories; please provide explicitly.")

        # Collect (name, elpd_per_obs, se_per_obs, pmf)
        models_info: List[Tuple[str, float, float, np.ndarray]] = []

        # Zero-predictor regression
        if target_idx in self.zero_predictor_results:
            zr = self.zero_predictor_results[target_idx]
            if zr.converged:
                intercept = zr.intercept_mean if zr.intercept_mean is not None else 0.0
                pmf = self._ordinal_pmf_from_regression(zr, intercept, n_categories)
                models_info.append(("reg:intercept", zr.elpd_loo_per_obs, zr.elpd_loo_per_obs_se, pmf))

        # Zero-predictor DM
        if target_idx in self.dm_zero_results:
            dmz = self.dm_zero_results[target_idx]
            if dmz.converged:
                pmf = self.dm_model.predict_pmf(dmz, 0)
                pmf = self._align_pmf(pmf, n_categories)
                models_info.append(("dm:marginal", dmz.elpd_loo_per_obs, dmz.elpd_loo_per_obs_se, pmf))

        # Per-predictor models
        for predictor_name, predictor_value in items.items():
            if predictor_name not in self.variable_names or predictor_name == target:
                continue

            predictor_idx = self.variable_names.index(predictor_name)
            key = (target_idx, predictor_idx)

            # Regression PMF
            if key in self.univariate_results:
                ur = self.univariate_results[key]
                if ur.converged:
                    X_mean = ur.predictor_mean if ur.predictor_mean is not None else 0.0
                    X_std = ur.predictor_std if ur.predictor_std is not None else 1.0
                    x_std = (predictor_value - X_mean) / X_std
                    beta_val = ur.beta_mean[0] if isinstance(ur.beta_mean, (np.ndarray, list)) else ur.beta_mean
                    intercept = ur.intercept_mean if ur.intercept_mean is not None else 0.0
                    eta = float(x_std * beta_val + intercept)
                    pmf = self._ordinal_pmf_from_regression(ur, eta, n_categories)
                    models_info.append((f"reg:{predictor_name}", ur.elpd_loo_per_obs, ur.elpd_loo_per_obs_se, pmf))

            # DM PMF
            if key in self.dm_results:
                dmr = self.dm_results[key]
                if dmr.converged:
                    pmf = self.dm_model.predict_pmf(dmr, predictor_value)
                    pmf = self._align_pmf(pmf, n_categories)
                    models_info.append((f"dm:{predictor_name}", dmr.elpd_loo_per_obs, dmr.elpd_loo_per_obs_se, pmf))

        if not models_info:
            return np.ones(n_categories) / n_categories

        elpd_values = np.array([m[1] for m in models_info])
        se_values = np.array([m[2] for m in models_info])
        weights = self._elpd_weights(elpd_values, se_values, uncertainty_penalty)

        pmf_stack = np.stack([m[3] for m in models_info], axis=0)
        stacked_pmf = weights @ pmf_stack
        stacked_pmf = np.clip(stacked_pmf, 0, None)
        total = stacked_pmf.sum()
        if total > 0:
            stacked_pmf /= total
        else:
            stacked_pmf = np.ones(n_categories) / n_categories
        return stacked_pmf

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_elpd_matrix(self) -> Dict[str, np.ndarray]:
        """Get ELPD matrices for both regression and DM models.

        Returns:
            Dict with keys 'regression' and 'dirichlet_multinomial', each
            containing an (n_variables, n_variables) array of per-obs ELPD.
        """
        n = len(self.variable_names)

        reg_matrix = np.full((n, n), np.nan)
        for i, result in self.zero_predictor_results.items():
            if result.converged:
                reg_matrix[i, i] = result.elpd_loo_per_obs
        for (ti, pi), result in self.univariate_results.items():
            if result.converged:
                reg_matrix[ti, pi] = result.elpd_loo_per_obs

        dm_matrix = np.full((n, n), np.nan)
        for i, result in self.dm_zero_results.items():
            if result.converged:
                dm_matrix[i, i] = result.elpd_loo_per_obs
        for (ti, pi), result in self.dm_results.items():
            if result.converged:
                dm_matrix[ti, pi] = result.elpd_loo_per_obs

        return {"regression": reg_matrix, "dirichlet_multinomial": dm_matrix}

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics for all model types."""
        n_reg_zero = sum(1 for r in self.zero_predictor_results.values() if r.converged)
        n_reg_uni = sum(1 for r in self.univariate_results.values() if r.converged)
        n_dm_conv = sum(1 for r in self.dm_results.values() if r.converged)
        n_dm_zero = sum(1 for r in self.dm_zero_results.values() if r.converged)

        return {
            "n_variables": len(self.variable_names),
            "variable_names": self.variable_names,
            "variable_types": self.variable_types,
            "n_obs_total": self.n_obs_total,
            "n_reg_zero_converged": n_reg_zero,
            "n_reg_zero_total": len(self.zero_predictor_results),
            "n_reg_univariate_converged": n_reg_uni,
            "n_reg_univariate_total": len(self.univariate_results),
            "n_dm_zero_converged": n_dm_zero,
            "n_dm_zero_total": len(self.dm_zero_results),
            "n_dm_pairwise_converged": n_dm_conv,
            "n_dm_pairwise_total": len(self.dm_results),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_to_disk(self, path: Union[str, Path], backend: str = "hdf5") -> None:
        """Save model to a directory with config.yaml + numerical arrays.

        Matches the BayesianModel directory format: metadata in config.yaml,
        numerical arrays in params.h5 (or tensors.safetensors).

        Args:
            path: Directory to write to (created if it does not exist).
            backend: ``"hdf5"`` (default). Safetensors not yet supported.
        """
        import h5py

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        def _scalar(x):
            if x is None:
                return None
            if isinstance(x, (np.number,)):
                return x.item()
            if isinstance(x, (np.ndarray, jnp.ndarray)):
                return np.array(x).tolist()
            return x

        # -- config.yaml: metadata only (no large arrays) --
        config = {
            "version": "2.0",
            "_backend": backend,
            "config": {
                "alpha_prior": self.alpha_prior,
                "prior_scale": self.prior_scale,
                "noise_scale": self.noise_scale,
                "pathfinder_num_samples": self.pathfinder_num_samples,
                "pathfinder_maxiter": self.pathfinder_maxiter,
                "min_obs": self.min_obs,
                "inference_method": self.inference_method,
            },
            "data": {
                "variable_names": self.variable_names,
                "variable_types": {int(k): v for k, v in self.variable_types.items()},
                "n_obs_total": self.n_obs_total,
                "global_ordinal_values": _scalar(self.global_ordinal_values),
                "n_global_classes": self.n_global_classes,
            },
            "zero_predictor_meta": {},
            "univariate_meta": [],
            "dm_zero_meta": {},
            "dm_meta": [],
        }

        for k, v in self.zero_predictor_results.items():
            config["zero_predictor_meta"][int(k)] = {
                "n_obs": int(v.n_obs), "elpd_loo": float(v.elpd_loo),
                "elpd_loo_per_obs": float(v.elpd_loo_per_obs),
                "elpd_loo_per_obs_se": float(v.elpd_loo_per_obs_se),
                "khat_max": float(v.khat_max), "khat_mean": float(v.khat_mean),
                "predictor_idx": None if v.predictor_idx is None else int(v.predictor_idx),
                "target_idx": int(v.target_idx), "converged": bool(v.converged),
                "predictor_mean": _scalar(v.predictor_mean),
                "predictor_std": _scalar(v.predictor_std),
            }

        for (t, p), v in self.univariate_results.items():
            config["univariate_meta"].append({
                "target_idx": int(t), "predictor_idx": int(p),
                "n_obs": int(v.n_obs), "elpd_loo": float(v.elpd_loo),
                "elpd_loo_per_obs": float(v.elpd_loo_per_obs),
                "elpd_loo_per_obs_se": float(v.elpd_loo_per_obs_se),
                "khat_max": float(v.khat_max), "khat_mean": float(v.khat_mean),
                "converged": bool(v.converged),
                "predictor_mean": _scalar(v.predictor_mean),
                "predictor_std": _scalar(v.predictor_std),
            })

        for k, v in self.dm_zero_results.items():
            config["dm_zero_meta"][int(k)] = {
                "n_obs": int(v.n_obs), "elpd_loo": float(v.elpd_loo),
                "elpd_loo_per_obs": float(v.elpd_loo_per_obs),
                "elpd_loo_per_obs_se": float(v.elpd_loo_per_obs_se),
                "predictor_idx": v.predictor_idx,
                "target_idx": int(v.target_idx), "converged": bool(v.converged),
            }

        for (t, p), v in self.dm_results.items():
            config["dm_meta"].append({
                "target_idx": int(t), "predictor_idx": int(p),
                "n_obs": int(v.n_obs), "elpd_loo": float(v.elpd_loo),
                "elpd_loo_per_obs": float(v.elpd_loo_per_obs),
                "elpd_loo_per_obs_se": float(v.elpd_loo_per_obs_se),
                "converged": bool(v.converged),
            })

        def repr_float(dumper, data):
            return dumper.represent_float(float(data))
        def repr_int(dumper, data):
            return dumper.represent_int(int(data))
        yaml.add_representer(np.float32, repr_float)
        yaml.add_representer(np.float64, repr_float)
        yaml.add_representer(np.int32, repr_int)
        yaml.add_representer(np.int64, repr_int)

        with open(path / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        # -- params.h5: numerical arrays --
        with h5py.File(path / "params.h5", "w") as f:
            for k, v in self.zero_predictor_results.items():
                grp = f.create_group(f"zero_predictor/{int(k)}")
                if v.beta_mean is not None:
                    grp.create_dataset("beta_mean", data=np.atleast_1d(np.asarray(v.beta_mean)))
                if v.intercept_mean is not None:
                    grp.create_dataset("intercept_mean", data=np.atleast_1d(np.asarray(v.intercept_mean)))
                if v.cutpoints_mean is not None:
                    grp.create_dataset("cutpoints_mean", data=np.asarray(v.cutpoints_mean))

            for (t, p), v in self.univariate_results.items():
                grp = f.create_group(f"univariate/{int(t)}_{int(p)}")
                if v.beta_mean is not None:
                    grp.create_dataset("beta_mean", data=np.atleast_1d(np.asarray(v.beta_mean)))
                if v.intercept_mean is not None:
                    grp.create_dataset("intercept_mean", data=np.atleast_1d(np.asarray(v.intercept_mean)))
                if v.cutpoints_mean is not None:
                    grp.create_dataset("cutpoints_mean", data=np.asarray(v.cutpoints_mean))

            for k, v in self.dm_zero_results.items():
                grp = f.create_group(f"dm_zero/{int(k)}")
                if v.alpha_posterior is not None:
                    grp.create_dataset("alpha_posterior", data=np.asarray(v.alpha_posterior))
                if v.predictor_categories is not None:
                    grp.create_dataset("predictor_categories", data=np.asarray(v.predictor_categories))
                if v.target_categories is not None:
                    grp.create_dataset("target_categories", data=np.asarray(v.target_categories))

            for (t, p), v in self.dm_results.items():
                grp = f.create_group(f"dm/{int(t)}_{int(p)}")
                if v.alpha_posterior is not None:
                    grp.create_dataset("alpha_posterior", data=np.asarray(v.alpha_posterior))
                if v.predictor_categories is not None:
                    grp.create_dataset("predictor_categories", data=np.asarray(v.predictor_categories))
                if v.target_categories is not None:
                    grp.create_dataset("target_categories", data=np.asarray(v.target_categories))

    @classmethod
    def load_from_disk(cls, path: Union[str, Path]) -> "PairwiseOrdinalStackingModel":
        """Load model from a directory with config.yaml + params.h5.

        Args:
            path: Directory containing config.yaml and params.h5.

        Returns:
            Loaded PairwiseOrdinalStackingModel instance.
        """
        import h5py

        path = Path(path)
        with open(path / "config.yaml", "r") as f:
            config = yaml.safe_load(f)

        cfg = config["config"]
        instance = cls(
            alpha_prior=cfg.get("alpha_prior", 0.5),
            prior_scale=cfg.get("prior_scale", 1.0),
            noise_scale=cfg.get("noise_scale", 1.0),
            pathfinder_num_samples=cfg.get("pathfinder_num_samples", 200),
            pathfinder_maxiter=cfg.get("pathfinder_maxiter", 100),
            min_obs=cfg.get("min_obs", 5),
            inference_method=cfg.get("inference_method", "pathfinder"),
            verbose=False,
        )

        data = config["data"]
        instance.variable_names = data["variable_names"]
        instance.variable_types = {int(k): v for k, v in data["variable_types"].items()}
        instance.n_obs_total = data["n_obs_total"]
        gv = data.get("global_ordinal_values")
        instance.global_ordinal_values = np.array(gv) if gv is not None else None
        instance.n_global_classes = data.get("n_global_classes", 0)

        # Load numerical arrays
        with h5py.File(path / "params.h5", "r") as f:
            # Regression zero-predictor
            for k_str, meta in config.get("zero_predictor_meta", {}).items():
                k = int(k_str)
                prefix = f"zero_predictor/{k}"
                beta_mean = np.array(f[f"{prefix}/beta_mean"]) if f"{prefix}/beta_mean" in f else None
                if beta_mean is not None and beta_mean.ndim == 1 and beta_mean.shape[0] == 1:
                    beta_mean = float(beta_mean[0])
                intercept_mean = float(f[f"{prefix}/intercept_mean"][0]) if f"{prefix}/intercept_mean" in f else None
                cutpoints_mean = np.array(f[f"{prefix}/cutpoints_mean"]) if f"{prefix}/cutpoints_mean" in f else None
                instance.zero_predictor_results[k] = UnivariateModelResult(
                    **meta, beta_mean=beta_mean, intercept_mean=intercept_mean,
                    cutpoints_mean=cutpoints_mean,
                )

            # Regression univariate
            for item in config.get("univariate_meta", []):
                t, p = int(item["target_idx"]), int(item["predictor_idx"])
                prefix = f"univariate/{t}_{p}"
                beta_mean = np.array(f[f"{prefix}/beta_mean"]) if f"{prefix}/beta_mean" in f else None
                if beta_mean is not None and beta_mean.ndim == 1 and beta_mean.shape[0] == 1:
                    beta_mean = float(beta_mean[0])
                intercept_mean = float(f[f"{prefix}/intercept_mean"][0]) if f"{prefix}/intercept_mean" in f else None
                cutpoints_mean = np.array(f[f"{prefix}/cutpoints_mean"]) if f"{prefix}/cutpoints_mean" in f else None
                meta_copy = {k2: v2 for k2, v2 in item.items() if k2 not in ("target_idx", "predictor_idx")}
                meta_copy["target_idx"] = t
                meta_copy["predictor_idx"] = p
                instance.univariate_results[(t, p)] = UnivariateModelResult(
                    **meta_copy, beta_mean=beta_mean, intercept_mean=intercept_mean,
                    cutpoints_mean=cutpoints_mean,
                )

            # DM zero-predictor
            for k_str, meta in config.get("dm_zero_meta", {}).items():
                k = int(k_str)
                prefix = f"dm_zero/{k}"
                alpha_post = np.array(f[f"{prefix}/alpha_posterior"]) if f"{prefix}/alpha_posterior" in f else None
                pred_cats = np.array(f[f"{prefix}/predictor_categories"]) if f"{prefix}/predictor_categories" in f else None
                tgt_cats = np.array(f[f"{prefix}/target_categories"]) if f"{prefix}/target_categories" in f else None
                instance.dm_zero_results[k] = DirichletMultinomialResult(
                    **meta, alpha_posterior=alpha_post,
                    predictor_categories=pred_cats, target_categories=tgt_cats,
                )

            # DM pairwise
            for item in config.get("dm_meta", []):
                t, p = int(item["target_idx"]), int(item["predictor_idx"])
                prefix = f"dm/{t}_{p}"
                alpha_post = np.array(f[f"{prefix}/alpha_posterior"]) if f"{prefix}/alpha_posterior" in f else None
                pred_cats = np.array(f[f"{prefix}/predictor_categories"]) if f"{prefix}/predictor_categories" in f else None
                tgt_cats = np.array(f[f"{prefix}/target_categories"]) if f"{prefix}/target_categories" in f else None
                meta_copy = {k2: v2 for k2, v2 in item.items() if k2 not in ("target_idx", "predictor_idx")}
                meta_copy["target_idx"] = t
                meta_copy["predictor_idx"] = p
                instance.dm_results[(t, p)] = DirichletMultinomialResult(
                    **meta_copy, alpha_posterior=alpha_post,
                    predictor_categories=pred_cats, target_categories=tgt_cats,
                )

        return instance

    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted model to a YAML file."""
        path = Path(path)

        def to_python(x):
            if x is None:
                return None
            if isinstance(x, (np.ndarray, jnp.ndarray)):
                return np.array(x).tolist()
            if isinstance(x, (np.number,)):
                return x.item()
            return x

        def result_to_dict(r: UnivariateModelResult) -> Dict[str, Any]:
            d = asdict(r)
            if d["params"] is not None:
                d["params"] = {k: to_python(v) for k, v in d["params"].items()}
            for field in ["predictor_mean", "predictor_std", "beta_mean", "intercept_mean", "cutpoints_mean"]:
                if field in d:
                    d[field] = to_python(d[field])
            return d

        def dm_result_to_dict(r: DirichletMultinomialResult) -> Dict[str, Any]:
            return {
                "n_obs": r.n_obs,
                "elpd_loo": float(r.elpd_loo),
                "elpd_loo_per_obs": float(r.elpd_loo_per_obs),
                "elpd_loo_per_obs_se": float(r.elpd_loo_per_obs_se),
                "predictor_idx": r.predictor_idx,
                "target_idx": r.target_idx,
                "converged": r.converged,
                "alpha_posterior": to_python(r.alpha_posterior),
                "predictor_categories": to_python(r.predictor_categories),
                "target_categories": to_python(r.target_categories),
            }

        state = {
            "version": "1.0",
            "config": {
                "alpha_prior": self.alpha_prior,
                "prior_scale": self.prior_scale,
                "noise_scale": self.noise_scale,
                "pathfinder_num_samples": self.pathfinder_num_samples,
                "pathfinder_maxiter": self.pathfinder_maxiter,
                "min_obs": self.min_obs,
                "inference_method": self.inference_method,
            },
            "data": {
                "variable_names": self.variable_names,
                "variable_types": {int(k): v for k, v in self.variable_types.items()},
                "n_obs_total": self.n_obs_total,
                "global_ordinal_values": to_python(self.global_ordinal_values),
                "n_global_classes": self.n_global_classes,
            },
            "zero_predictor_results": {
                int(k): result_to_dict(v) for k, v in self.zero_predictor_results.items()
            },
            "univariate_results": [
                {"target_idx": int(k[0]), "predictor_idx": int(k[1]), "result": result_to_dict(v)}
                for k, v in self.univariate_results.items()
            ],
            "dm_zero_results": {
                int(k): dm_result_to_dict(v) for k, v in self.dm_zero_results.items()
            },
            "dm_results": [
                {"target_idx": int(k[0]), "predictor_idx": int(k[1]), "result": dm_result_to_dict(v)}
                for k, v in self.dm_results.items()
            ],
        }

        def repr_float(dumper, data):
            return dumper.represent_float(float(data))

        def repr_int(dumper, data):
            return dumper.represent_int(int(data))

        yaml.add_representer(np.float32, repr_float)
        yaml.add_representer(np.float64, repr_float)
        yaml.add_representer(np.int32, repr_int)
        yaml.add_representer(np.int64, repr_int)

        with open(path, "w") as f:
            yaml.dump(state, f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PairwiseOrdinalStackingModel":
        """Load a fitted model from a YAML file."""
        path = Path(path)
        with open(path, "r") as f:
            state = yaml.safe_load(f)

        cfg = state["config"]
        instance = cls(
            alpha_prior=cfg.get("alpha_prior", 0.5),
            prior_scale=cfg.get("prior_scale", 1.0),
            noise_scale=cfg.get("noise_scale", 1.0),
            pathfinder_num_samples=cfg.get("pathfinder_num_samples", 200),
            pathfinder_maxiter=cfg.get("pathfinder_maxiter", 100),
            min_obs=cfg.get("min_obs", 5),
            inference_method=cfg.get("inference_method", "pathfinder"),
            verbose=False,
        )

        data = state["data"]
        instance.variable_names = data["variable_names"]
        instance.variable_types = {int(k): v for k, v in data["variable_types"].items()}
        instance.n_obs_total = data["n_obs_total"]
        gv = data.get("global_ordinal_values")
        instance.global_ordinal_values = np.array(gv) if gv is not None else None
        instance.n_global_classes = data.get("n_global_classes", 0)

        # Restore regression results
        for k_str, v in state.get("zero_predictor_results", {}).items():
            k = int(k_str)
            if v.get("beta_mean") is not None and isinstance(v["beta_mean"], list):
                v["beta_mean"] = np.array(v["beta_mean"])
            if v.get("cutpoints_mean") is not None and isinstance(v["cutpoints_mean"], list):
                v["cutpoints_mean"] = np.array(v["cutpoints_mean"])
            instance.zero_predictor_results[k] = UnivariateModelResult(**v)

        for item in state.get("univariate_results", []):
            key = (int(item["target_idx"]), int(item["predictor_idx"]))
            v = item["result"]
            if v.get("beta_mean") is not None and isinstance(v["beta_mean"], list):
                v["beta_mean"] = np.array(v["beta_mean"])
            if v.get("cutpoints_mean") is not None and isinstance(v["cutpoints_mean"], list):
                v["cutpoints_mean"] = np.array(v["cutpoints_mean"])
            instance.univariate_results[key] = UnivariateModelResult(**v)

        # Restore DM results
        for k_str, v in state.get("dm_zero_results", {}).items():
            k = int(k_str)
            if v.get("alpha_posterior") is not None:
                v["alpha_posterior"] = np.array(v["alpha_posterior"])
            if v.get("predictor_categories") is not None:
                v["predictor_categories"] = np.array(v["predictor_categories"])
            if v.get("target_categories") is not None:
                v["target_categories"] = np.array(v["target_categories"])
            instance.dm_zero_results[k] = DirichletMultinomialResult(**v)

        for item in state.get("dm_results", []):
            key = (int(item["target_idx"]), int(item["predictor_idx"]))
            v = item["result"]
            if v.get("alpha_posterior") is not None:
                v["alpha_posterior"] = np.array(v["alpha_posterior"])
            if v.get("predictor_categories") is not None:
                v["predictor_categories"] = np.array(v["predictor_categories"])
            if v.get("target_categories") is not None:
                v["target_categories"] = np.array(v["target_categories"])
            instance.dm_results[key] = DirichletMultinomialResult(**v)

        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_stacking_weights(
        self,
        models_info: List[Tuple[str, float, float, float]],
        uncertainty_penalty: float,
    ) -> Tuple[float, np.ndarray]:
        elpd_values = np.array([m[1] for m in models_info])
        se_values = np.array([m[2] for m in models_info])
        predictions = np.array([m[3] for m in models_info])
        weights = self._elpd_weights(elpd_values, se_values, uncertainty_penalty)
        stacked_pred = float(np.sum(weights * predictions))
        return stacked_pred, weights

    @staticmethod
    def _elpd_weights(
        elpd_values: np.ndarray,
        se_values: np.ndarray,
        uncertainty_penalty: float,
    ) -> np.ndarray:
        se_safe = np.where(np.isfinite(se_values), se_values, 1e6)
        adjusted = elpd_values - uncertainty_penalty * se_safe

        finite_mask = np.isfinite(adjusted)
        if not np.any(finite_mask):
            return np.ones(len(elpd_values)) / len(elpd_values)

        max_adj = np.max(adjusted[finite_mask])
        log_w = np.where(finite_mask, adjusted - max_adj, -np.inf)
        weights = np.exp(log_w)
        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            weights = np.ones(len(elpd_values)) / len(elpd_values)
        return weights

    @staticmethod
    def _align_pmf(pmf: np.ndarray, n_categories: int) -> np.ndarray:
        if len(pmf) == n_categories:
            return pmf
        if len(pmf) < n_categories:
            padded = np.zeros(n_categories)
            padded[: len(pmf)] = pmf
            return padded
        truncated = pmf[:n_categories]
        total = truncated.sum()
        if total > 0:
            truncated /= total
        return truncated

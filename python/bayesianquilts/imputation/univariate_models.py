"""
Shared univariate Bayesian regression models and helpers.

This module contains the building blocks used by both MICEBayesianLOO and
PairwiseOrdinalStackingModel:

- Regularized horseshoe prior helpers
- UnivariateModelResult dataclass
- SimpleLinearRegression, SimpleLogisticRegression, SimpleOrdinalLogisticRegression
- SimpleModelLikelihood (AutoDiff wrapper)
- Inference runners (Pathfinder, ADVI) with float64 fallback
- LOO-ELPD computation via PSIS
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
import jax.flatten_util


def _warn_fallback(msg, exc=None):
    """Print a red warning about a fallback to degraded behavior."""
    detail = f" ({type(exc).__name__}: {exc})" if exc else ""
    sys.stderr.write(
        f"\033[91mWARNING: {msg}{detail}\033[0m\n"
    )
    sys.stderr.flush()

jax.config.update("jax_enable_x64", True)

from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

import tensorflow_probability.substrates.jax as tfp
import blackjax.vi.pathfinder as pathfinder

from bayesianquilts.metrics import nppsis
from bayesianquilts.metrics.ais import (
    LikelihoodFunction,
    AutoDiffLikelihoodMixin,
)

tfd = tfp.distributions


# ---------------------------------------------------------------------------
# Horseshoe prior helpers
# ---------------------------------------------------------------------------

def _horseshoe_log_prior(log_tau, log_local_scales, log_c, beta_raw,
                          tau0, slab_scale=2.0, slab_df=4.0):
    """Compute log prior for regularized horseshoe (Piironen & Vehtari 2017).

    All scale parameters are on the log scale (unconstrained).
    Returns scalar log-density including Jacobian corrections.
    """
    tau = jnp.exp(log_tau)
    local_scales = jnp.exp(log_local_scales)
    c2 = jnp.exp(2.0 * log_c)

    lp_tau = (jnp.log(2.0) - jnp.log(jnp.pi) - jnp.log(tau0)
              - jnp.log1p((tau / tau0) ** 2) + log_tau)

    lp_local = jnp.sum(
        jnp.log(2.0) - jnp.log(jnp.pi)
        - jnp.log1p(local_scales ** 2) + log_local_scales
    )

    alpha = slab_df / 2.0
    beta_ig = slab_df * slab_scale ** 2 / 2.0
    from jax.scipy.special import gammaln
    lp_c = (alpha * jnp.log(beta_ig) - gammaln(alpha)
            + (-alpha - 1) * jnp.log(c2) - beta_ig / c2
            + jnp.log(2.0) + 2.0 * log_c)

    lp_beta_raw = -0.5 * jnp.sum(beta_raw ** 2) - 0.5 * beta_raw.shape[0] * jnp.log(2.0 * jnp.pi)

    return lp_tau + lp_local + lp_c + lp_beta_raw


def _horseshoe_reconstruct_beta(beta_raw, log_tau, log_local_scales, log_c):
    """Reconstruct beta from the non-centered horseshoe parameterization."""
    tau = jnp.exp(log_tau)
    local_scales = jnp.exp(log_local_scales)
    c = jnp.exp(log_c)
    tilde_lambda = c * local_scales / jnp.sqrt(c ** 2 + tau ** 2 * local_scales ** 2)
    return beta_raw * tau * tilde_lambda


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class UnivariateModelResult:
    """Results from fitting a univariate model."""
    n_obs: int
    elpd_loo: float
    elpd_loo_per_obs: float
    elpd_loo_per_obs_se: float
    khat_max: float
    khat_mean: float
    predictor_idx: Optional[int]
    target_idx: int
    converged: bool
    params: Optional[Dict[str, jnp.ndarray]] = None
    predictor_mean: Optional[float] = None
    predictor_std: Optional[float] = None
    beta_mean: Optional[Union[float, np.ndarray]] = None
    intercept_mean: Optional[float] = None
    cutpoints_mean: Optional[np.ndarray] = None
    loo_values: Optional[np.ndarray] = None  # per-observation LOO log-predictive densities


# ---------------------------------------------------------------------------
# Simple regression models
# ---------------------------------------------------------------------------

class SimpleLinearRegression:
    """Simple Bayesian linear regression for univariate models."""

    def __init__(
        self,
        n_predictors: int = 1,
        prior_scale: float = 1.0,
        noise_scale: float = 1.0,
        dtype=jnp.float32,
        n_obs: int = 100,
    ):
        self.n_predictors = n_predictors
        self.prior_scale = prior_scale
        self.noise_scale = noise_scale
        self.dtype = dtype
        self.n_obs = n_obs
        p0 = max(1, n_predictors // 2)
        self.tau0 = p0 / max(1, n_predictors - p0) * noise_scale / np.sqrt(n_obs)
        self.var_list = ['beta_raw', 'log_tau', 'log_local_scales', 'log_c',
                         'intercept', 'log_sigma']

    def create_prior(self):
        return tfd.JointDistributionNamed({
            'beta_raw': tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(self.n_predictors, dtype=self.dtype),
                    scale=jnp.ones(self.n_predictors, dtype=self.dtype)
                ), reinterpreted_batch_ndims=1
            ),
            'log_tau': tfd.Normal(loc=jnp.log(jnp.array(self.tau0, dtype=self.dtype)), scale=1.0),
            'log_local_scales': tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(self.n_predictors, dtype=self.dtype),
                    scale=jnp.ones(self.n_predictors, dtype=self.dtype)
                ), reinterpreted_batch_ndims=1
            ),
            'log_c': tfd.Normal(loc=jnp.log(jnp.array(2.0, dtype=self.dtype)), scale=0.5),
            'intercept': tfd.Normal(loc=jnp.zeros([], dtype=self.dtype), scale=self.prior_scale),
            'log_sigma': tfd.Normal(
                loc=jnp.log(jnp.array(self.noise_scale, dtype=self.dtype)), scale=1.0
            )
        })

    def _get_beta(self, params):
        beta_raw = params['beta_raw']
        log_tau = params['log_tau']
        log_local_scales = params['log_local_scales']
        log_c = params['log_c']
        if beta_raw.ndim == 1:
            return _horseshoe_reconstruct_beta(beta_raw, log_tau, log_local_scales, log_c)
        else:
            return jax.vmap(
                lambda br, lt, ll, lc: _horseshoe_reconstruct_beta(br, lt, ll, lc)
            )(beta_raw, log_tau, log_local_scales, log_c)

    def log_likelihood(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        X = jnp.asarray(data['X'], dtype=self.dtype)
        y = jnp.asarray(data['y'], dtype=self.dtype)
        beta = self._get_beta(params)
        intercept = params['intercept']
        log_sigma = params['log_sigma']

        if beta.ndim == 1:
            mu = jnp.dot(X, beta) + intercept
        else:
            mu = jnp.einsum('np,sp->sn', X, beta) + intercept[:, None]

        if log_sigma.ndim == 0:
            log_sigma = log_sigma[None]
        sigma = jnp.exp(log_sigma)

        residuals = y - mu if mu.ndim == 1 else y[None, :] - mu
        log_lik = -0.5 * jnp.log(2 * jnp.pi) - log_sigma[:, None] - 0.5 * (residuals / sigma[:, None])**2
        return log_lik

    def unormalized_log_prob(self, data: Dict[str, Any], scale_factor: float = 1.0, **params) -> jnp.ndarray:
        log_lik = self.log_likelihood(data, params)
        if 'weights' in data:
            w = jnp.asarray(data['weights'], dtype=self.dtype)
            total_log_lik = jnp.sum(w[None, :] * log_lik, axis=-1) * scale_factor if log_lik.ndim == 2 else jnp.sum(w * log_lik) * scale_factor
        else:
            total_log_lik = jnp.sum(log_lik, axis=-1) * scale_factor
        lp_horseshoe = _horseshoe_log_prior(
            params['log_tau'], params['log_local_scales'],
            params['log_c'], params['beta_raw'], self.tau0)
        lp_intercept = tfd.Normal(0.0, self.prior_scale).log_prob(params['intercept'])
        lp_log_sigma = tfd.Normal(jnp.log(jnp.array(self.noise_scale, dtype=self.dtype)), 1.0).log_prob(params['log_sigma'])
        return total_log_lik + lp_horseshoe + lp_intercept + lp_log_sigma


class SimpleLogisticRegression:
    """Simple Bayesian logistic regression for binary outcomes."""

    def __init__(
        self,
        n_predictors: int = 1,
        prior_scale: float = 1.0,
        dtype=jnp.float32,
        n_obs: int = 100,
    ):
        self.n_predictors = n_predictors
        self.prior_scale = prior_scale
        self.dtype = dtype
        self.n_obs = n_obs
        sigma_pseudo = np.pi / np.sqrt(3)
        p0 = max(1, n_predictors // 2)
        self.tau0 = p0 / max(1, n_predictors - p0) * sigma_pseudo / np.sqrt(n_obs)
        self.var_list = ['beta_raw', 'log_tau', 'log_local_scales', 'log_c', 'intercept']

    def create_prior(self):
        return tfd.JointDistributionNamed({
            'beta_raw': tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(self.n_predictors, dtype=self.dtype),
                    scale=jnp.ones(self.n_predictors, dtype=self.dtype)
                ), reinterpreted_batch_ndims=1
            ),
            'log_tau': tfd.Normal(loc=jnp.log(jnp.array(self.tau0, dtype=self.dtype)), scale=1.0),
            'log_local_scales': tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(self.n_predictors, dtype=self.dtype),
                    scale=jnp.ones(self.n_predictors, dtype=self.dtype)
                ), reinterpreted_batch_ndims=1
            ),
            'log_c': tfd.Normal(loc=jnp.log(jnp.array(2.0, dtype=self.dtype)), scale=0.5),
            'intercept': tfd.Normal(loc=jnp.zeros([], dtype=self.dtype), scale=self.prior_scale)
        })

    def _get_beta(self, params):
        beta_raw = params['beta_raw']
        log_tau = params['log_tau']
        log_local_scales = params['log_local_scales']
        log_c = params['log_c']
        if beta_raw.ndim == 1:
            return _horseshoe_reconstruct_beta(beta_raw, log_tau, log_local_scales, log_c)
        else:
            return jax.vmap(
                lambda br, lt, ll, lc: _horseshoe_reconstruct_beta(br, lt, ll, lc)
            )(beta_raw, log_tau, log_local_scales, log_c)

    def log_likelihood(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        X = jnp.asarray(data['X'], dtype=self.dtype)
        y = jnp.asarray(data['y'], dtype=self.dtype)
        beta = self._get_beta(params)
        intercept = params['intercept']

        if beta.ndim == 1:
            logits = jnp.dot(X, beta) + intercept
        else:
            if intercept.ndim == 0:
                intercept = intercept[None]
            logits = jnp.einsum('np,sp->sn', X, beta) + intercept[:, None]

        if logits.ndim == 1:
            log_lik = y * jax.nn.log_sigmoid(logits) + (1 - y) * jax.nn.log_sigmoid(-logits)
        else:
            log_lik = y[None, :] * jax.nn.log_sigmoid(logits) + (1 - y[None, :]) * jax.nn.log_sigmoid(-logits)
        return log_lik

    def unormalized_log_prob(self, data: Dict[str, Any], scale_factor: float = 1.0, **params) -> jnp.ndarray:
        log_lik = self.log_likelihood(data, params)
        if 'weights' in data:
            w = jnp.asarray(data['weights'], dtype=self.dtype)
            total_log_lik = jnp.sum(w[None, :] * log_lik, axis=-1) * scale_factor if log_lik.ndim == 2 else jnp.sum(w * log_lik) * scale_factor
        else:
            total_log_lik = jnp.sum(log_lik, axis=-1) * scale_factor
        lp_horseshoe = _horseshoe_log_prior(
            params['log_tau'], params['log_local_scales'],
            params['log_c'], params['beta_raw'], self.tau0)
        lp_intercept = tfd.Normal(0.0, self.prior_scale).log_prob(params['intercept'])
        return total_log_lik + lp_horseshoe + lp_intercept


class SimpleOrdinalLogisticRegression:
    """Simple Bayesian ordinal logistic regression."""

    def __init__(
        self,
        n_classes: int,
        n_predictors: int = 1,
        prior_scale: float = 1.0,
        dtype=jnp.float32,
        n_obs: int = 100,
    ):
        self.n_classes = n_classes
        self.n_cutpoints = n_classes - 1
        self.n_predictors = n_predictors
        self.prior_scale = prior_scale
        self.dtype = dtype
        self.n_obs = n_obs
        sigma_pseudo = np.pi / np.sqrt(3)
        p0 = max(1, n_predictors // 2)
        self.tau0 = p0 / max(1, n_predictors - p0) * sigma_pseudo / np.sqrt(n_obs)
        self.var_list = ['beta_raw', 'log_tau', 'log_local_scales', 'log_c', 'cutpoints_raw']

    def create_prior(self):
        return tfd.JointDistributionNamed({
            'beta_raw': tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(self.n_predictors, dtype=self.dtype),
                    scale=jnp.ones(self.n_predictors, dtype=self.dtype)
                ), reinterpreted_batch_ndims=1
            ),
            'log_tau': tfd.Normal(loc=jnp.log(jnp.array(self.tau0, dtype=self.dtype)), scale=1.0),
            'log_local_scales': tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(self.n_predictors, dtype=self.dtype),
                    scale=jnp.ones(self.n_predictors, dtype=self.dtype)
                ), reinterpreted_batch_ndims=1
            ),
            'log_c': tfd.Normal(loc=jnp.log(jnp.array(2.0, dtype=self.dtype)), scale=0.5),
            'cutpoints_raw': tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(self.n_cutpoints, dtype=self.dtype),
                    scale=5.0 * jnp.ones(self.n_cutpoints, dtype=self.dtype)
                ), reinterpreted_batch_ndims=1
            )
        })

    def _transform_cutpoints(self, cutpoints_raw):
        tfb = tfp.bijectors
        bij = tfb.Ascending()
        return bij.forward(cutpoints_raw)

    def _get_beta(self, params):
        beta_raw = params['beta_raw']
        log_tau = params['log_tau']
        log_local_scales = params['log_local_scales']
        log_c = params['log_c']
        if beta_raw.ndim == 1:
            return _horseshoe_reconstruct_beta(beta_raw, log_tau, log_local_scales, log_c)
        else:
            return jax.vmap(
                lambda br, lt, ll, lc: _horseshoe_reconstruct_beta(br, lt, ll, lc)
            )(beta_raw, log_tau, log_local_scales, log_c)

    def log_likelihood(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        X = jnp.asarray(data['X'], dtype=self.dtype)
        y = jnp.asarray(data['y'], dtype=self.dtype)
        beta = self._get_beta(params)
        cutpoints_raw = params['cutpoints_raw']

        if cutpoints_raw.ndim == 1:
            cutpoints = self._transform_cutpoints(cutpoints_raw)
        else:
            cutpoints = jax.vmap(self._transform_cutpoints)(cutpoints_raw)

        if beta.ndim == 1:
            eta = jnp.dot(X, beta)
        else:
            eta = jnp.einsum('np,sp->sn', X, beta)

        if cutpoints.ndim == 0:
            cutpoints = cutpoints[None]

        if cutpoints.ndim == 1 and beta.ndim == 1:
            dist = tfd.OrderedLogistic(cutpoints=cutpoints, loc=eta)
            return dist.log_prob(y)

        if cutpoints.ndim == 1:
            cutpoints_expanded = cutpoints[None, None, :]
        elif cutpoints.ndim == 2:
            cutpoints_expanded = cutpoints[:, None, :]
        else:
            cutpoints_expanded = cutpoints

        dist = tfd.OrderedLogistic(cutpoints=cutpoints_expanded, loc=eta)
        log_prob = dist.log_prob(y[None, :])
        return log_prob

    def unormalized_log_prob(self, data: Dict[str, Any], scale_factor: float = 1.0, **params) -> jnp.ndarray:
        log_lik = self.log_likelihood(data, params)
        if 'weights' in data:
            w = jnp.asarray(data['weights'], dtype=self.dtype)
            total_log_lik = jnp.sum(w[None, :] * log_lik, axis=-1) * scale_factor if log_lik.ndim == 2 else jnp.sum(w * log_lik) * scale_factor
        else:
            total_log_lik = jnp.sum(log_lik, axis=-1) * scale_factor
        lp_horseshoe = _horseshoe_log_prior(
            params['log_tau'], params['log_local_scales'],
            params['log_c'], params['beta_raw'], self.tau0)
        lp_cutpoints = tfd.Independent(
            tfd.Normal(
                loc=jnp.zeros(self.n_cutpoints, dtype=self.dtype),
                scale=5.0 * jnp.ones(self.n_cutpoints, dtype=self.dtype)
            ), reinterpreted_batch_ndims=1
        ).log_prob(params['cutpoints_raw'])
        return total_log_lik + lp_horseshoe + lp_cutpoints


class SimpleModelLikelihood(AutoDiffLikelihoodMixin, LikelihoodFunction):
    """Generic LikelihoodFunction wrapper for simple univariate models."""

    def __init__(self, model):
        self.model = model

    def log_likelihood(self, data, params):
        return self.model.log_likelihood(data, params)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def ordinal_one_hot_encode(data_matrix, max_val):
    """Encodes integer data matrix into ordinal one-hot (thermometer) format.

    Args:
        data_matrix: (N, M) matrix of integers 0..max_val
        max_val: Maximum possible value (determines vector length per feature)

    Returns:
        Encoded matrix of shape (N, M * max_val)
    """
    N, M = data_matrix.shape
    encoded = np.zeros((N, M * max_val), dtype=np.int8)
    for v in range(1, max_val + 1):
        mask = (data_matrix >= v)
        for col_idx in range(M):
            target_col = col_idx * max_val + (v - 1)
            encoded[:, target_col] = mask[:, col_idx].astype(int)
    return encoded


def infer_variable_type(values: np.ndarray) -> str:
    """Infer variable type from values."""
    unique_values = np.unique(values[~np.isnan(values)])
    if len(unique_values) == 2 and set(unique_values).issubset({0, 1, 0.0, 1.0}):
        return 'binary'
    if len(unique_values) >= 2 and len(unique_values) < 20:
        is_integer = np.all(np.mod(unique_values, 1) == 0)
        if is_integer:
            return 'ordinal'
    return 'continuous'


def check_for_nan(params: Optional[Dict[str, jnp.ndarray]], elbo: float) -> bool:
    """Check if parameters or ELBO contain NaN values."""
    if params is None:
        return False
    if np.isnan(elbo) or np.isinf(elbo):
        return True
    for key, value in params.items():
        if np.any(np.isnan(value)) or np.any(np.isinf(value)):
            return True
    return False


def run_pathfinder(
    model,
    data: Dict[str, Any],
    num_samples: int = 200,
    maxiter: int = 100,
    scale_factor: float = 1.0,
    seed: int = 42,
) -> Tuple[Optional[Dict[str, jnp.ndarray]], float, bool, Optional[callable]]:
    """Run Pathfinder variational inference.

    Returns:
        Tuple of (samples_dict, elbo, converged, surrogate_log_prob_fn)
    """
    key = jax.random.PRNGKey(seed)
    prior = model.create_prior()
    prior_sample = prior.sample(seed=key)
    template = prior_sample
    flat_template, unflatten_fn = jax.flatten_util.ravel_pytree(template)
    param_dim = flat_template.shape[0]

    def logprob_fn_flat(params_flat):
        params_dict = unflatten_fn(params_flat)
        return jnp.squeeze(model.unormalized_log_prob(data=data, scale_factor=scale_factor, **params_dict))

    initial_position = jax.random.normal(jax.random.PRNGKey(seed + 1), (param_dim,)) * 0.1

    try:
        state, info = pathfinder.approximate(
            rng_key=jax.random.PRNGKey(seed + 2),
            logdensity_fn=logprob_fn_flat,
            initial_position=initial_position,
            num_samples=num_samples,
            maxiter=maxiter,
            ftol=1e-6,
            gtol=1e-9,
        )

        elbo = float(state.elbo)
        converged = True

        sample_key = jax.random.PRNGKey(seed + 3)
        samples_result = pathfinder.sample(sample_key, state, num_samples=num_samples)
        samples_flat = samples_result[0] if isinstance(samples_result, tuple) else samples_result

        samples_dict = {var: [] for var in model.var_list}
        for i in range(num_samples):
            sample = unflatten_fn(samples_flat[i])
            for var in model.var_list:
                samples_dict[var].append(sample[var])

        for var in model.var_list:
            samples_dict[var] = jnp.stack(samples_dict[var], axis=0)

        sample_mean = {var: jnp.mean(samples_dict[var], axis=0) for var in model.var_list}
        sample_std = {var: jnp.maximum(jnp.std(samples_dict[var], axis=0), 1e-6) for var in model.var_list}

        def surrogate_log_prob_fn(params):
            lp = jnp.zeros(jax.tree_util.tree_leaves(params)[0].shape[0])
            for var in model.var_list:
                p = params[var]
                dist = tfd.Independent(
                    tfd.Normal(loc=sample_mean[var], scale=sample_std[var]),
                    reinterpreted_batch_ndims=max(sample_mean[var].ndim, 1)
                ) if sample_mean[var].ndim >= 1 else tfd.Normal(
                    loc=sample_mean[var], scale=sample_std[var]
                )
                lp = lp + dist.log_prob(p)
            return lp

        return samples_dict, elbo, converged, surrogate_log_prob_fn

    except Exception as e:
        _warn_fallback("Pathfinder inference failed, returning unconverged", e)
        import traceback
        traceback.print_exc()
        return None, float('-inf'), False, None


def run_advi(
    model,
    data: Dict[str, Any],
    num_samples: int = 200,
    maxiter: int = 100,
    scale_factor: float = 1.0,
    seed: int = 42,
) -> Tuple[Optional[Dict[str, jnp.ndarray]], float, bool, Optional[callable]]:
    """Run minibatch ADVI inference.

    Returns:
        Tuple of (samples_dict, elbo, converged, surrogate_log_prob_fn)
    """
    from bayesianquilts.vi.minibatch import minibatch_mc_variational_loss
    import optax

    key = jax.random.PRNGKey(seed)
    prior = model.create_prior()
    prior_sample = prior.sample(seed=key)

    def create_surrogate(params):
        surrogate_dists = {}
        for var_name in model.var_list:
            param_val = prior_sample[var_name]
            shape = param_val.shape if isinstance(param_val, jnp.ndarray) else ()
            loc = params[f'{var_name}_loc']
            log_scale = params[f'{var_name}_log_scale']
            if shape:
                surrogate_dists[var_name] = tfd.Independent(
                    tfd.Normal(loc=loc, scale=jnp.exp(log_scale)),
                    reinterpreted_batch_ndims=len(shape)
                )
            else:
                surrogate_dists[var_name] = tfd.Normal(loc=loc, scale=jnp.exp(log_scale))
        return tfd.JointDistributionNamed(surrogate_dists)

    surrogate_params = {}
    for var_name in model.var_list:
        param_val = prior_sample[var_name]
        shape = param_val.shape if isinstance(param_val, jnp.ndarray) else ()
        surrogate_params[f'{var_name}_loc'] = jnp.zeros(shape, dtype=model.dtype)
        surrogate_params[f'{var_name}_log_scale'] = jnp.zeros(shape, dtype=model.dtype) - 1.0

    def target_log_prob_fn(data, **params):
        model_params = {var: params[var] for var in model.var_list if var in params}
        return model.unormalized_log_prob(data=data, scale_factor=scale_factor, **model_params)

    optimizer = optax.adam(learning_rate=5e-3)
    opt_state = optimizer.init(surrogate_params)

    @jax.jit
    def update_step(params, opt_state, seed):
        def loss_fn(p):
            surrogate = create_surrogate(p)
            return minibatch_mc_variational_loss(
                target_log_prob_fn=target_log_prob_fn,
                surrogate_posterior=surrogate,
                dataset_size=1,
                batch_size=1,
                data=data,
                sample_size=10,
                seed=seed
            )
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state_new, loss

    best_loss = float('inf')
    patience_counter = 0
    max_patience = 20

    for step in range(maxiter):
        step_key = jax.random.fold_in(key, step)
        surrogate_params, opt_state, loss = update_step(surrogate_params, opt_state, step_key)
        if loss < best_loss:
            best_loss = float(loss)
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= max_patience:
            break

    try:
        final_surrogate = create_surrogate(surrogate_params)
        _, sample_key = jax.random.split(key)
        samples = final_surrogate.sample(num_samples, seed=sample_key)

        samples_dict = {}
        for var in model.var_list:
            samples_dict[var] = samples[var]

        captured_params = {k: jnp.array(v) for k, v in surrogate_params.items()}

        def surrogate_log_prob_fn(params):
            surr = create_surrogate(captured_params)
            return surr.log_prob(params)

        return samples_dict, -best_loss, True, surrogate_log_prob_fn

    except Exception as e:
        _warn_fallback("ADVI inference failed, returning unconverged", e)
        return None, float('-inf'), False, None


def run_inference_with_fallback(
    model,
    data_dict: Dict[str, Any],
    scale_factor: float,
    seed: int,
    current_dtype: jnp.dtype,
    inference_method: str = 'pathfinder',
    num_samples: int = 200,
    maxiter: int = 100,
    verbose: bool = True,
) -> Tuple[Optional[Dict[str, jnp.ndarray]], float, bool, jnp.dtype, Optional[callable]]:
    """Run inference with automatic fallback to float64 if NaN detected.

    Returns:
        Tuple of (params, elbo, converged, dtype_used, surrogate_log_prob_fn)
    """
    runner = run_advi if inference_method == 'advi' else run_pathfinder

    params, elbo, converged, surrogate_fn = runner(
        model, data_dict, num_samples=num_samples, maxiter=maxiter,
        scale_factor=scale_factor, seed=seed
    )

    if converged and check_for_nan(params, elbo):
        if current_dtype == jnp.float32:
            if verbose:
                print(f"    NaN detected with float32, retrying with float64...")

            old_dtype = model.dtype
            model.dtype = jnp.float64

            data_dict_f64 = {
                'X': data_dict['X'].astype(np.float64),
                'y': data_dict['y'].astype(np.float64)
            }

            params, elbo, converged, surrogate_fn = runner(
                model, data_dict_f64, num_samples=num_samples, maxiter=maxiter,
                scale_factor=scale_factor, seed=seed
            )

            model.dtype = old_dtype
            return params, elbo, converged, jnp.float64, surrogate_fn
        else:
            return None, float('-inf'), False, current_dtype, None

    return params, elbo, converged, current_dtype, surrogate_fn


def compute_loo_elpd(
    model,
    data: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
) -> Tuple[float, float, float, float]:
    """Compute LOO-ELPD using PSIS.

    If data contains a 'weights' key, the per-observation LOO log-likelihoods
    are multiplied by the weights before summing to obtain the weighted ELPD.

    Returns:
        Tuple of (elpd_loo, elpd_loo_se, khat_max, khat_mean)
    """
    log_lik = model.log_likelihood(data, params)
    log_lik_np = np.array(log_lik)
    loo, loos, ks = nppsis.psisloo(log_lik_np)

    if 'weights' in data:
        w = np.asarray(data['weights'], dtype=np.float64)
        loos = loos * w
        loo = float(np.sum(loos))

    n = len(loos)
    elpd_se = np.sqrt(n * np.var(loos))

    return float(loo), float(elpd_se), float(np.max(ks)), float(np.mean(ks)), loos.astype(np.float32)

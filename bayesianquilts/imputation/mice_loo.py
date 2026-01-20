"""
MICE LOO-CV Framework for Bayesian Multiple Imputation

This module implements a Bayesian framework for evaluating variable relationships
using Leave-One-Out Cross-Validation with Pathfinder variational inference.

For P variables, this framework fits:
- P zero-predictor (intercept-only) models
- Up to P*(P-1) one-predictor univariate models

Each univariate model predicts variable i using variable j, where the data
subset only includes observations where both variables are observed.

The framework stores:
- n_obs: Number of observations in the data subset
- elpd_loo_per_obs: LOO ELPD divided by n_obs (normalized ELPD)
- elpd_loo_per_obs_se: Standard error of the normalized ELPD
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import jax.flatten_util
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
import tensorflow_probability.substrates.jax as tfp
import blackjax.vi.pathfinder as pathfinder

from bayesianquilts.imputation.mice import MICELogistic, ordinal_one_hot_encode
from bayesianquilts.metrics import nppsis

tfd = tfp.distributions


@dataclass
class UnivariateModelResult:
    """Results from fitting a univariate model."""
    n_obs: int
    elpd_loo: float
    elpd_loo_per_obs: float
    elpd_loo_per_obs_se: float  # Standard error of ELPD per observation
    khat_max: float
    khat_mean: float
    predictor_idx: Optional[int]  # None for zero-predictor model
    target_idx: int
    converged: bool
    params: Optional[Dict[str, jnp.ndarray]] = None
    # Standardization parameters for prediction
    predictor_mean: Optional[float] = None
    predictor_std: Optional[float] = None
    # Point estimates for memory-efficient prediction
    beta_mean: Optional[Union[float, np.ndarray]] = None
    intercept_mean: Optional[float] = None
    cutpoints_mean: Optional[np.ndarray] = None


class SimpleLinearRegression:
    """Simple Bayesian linear regression for univariate models."""

    def __init__(
        self,
        n_predictors: int = 1,
        prior_scale: float = 1.0,
        noise_scale: float = 1.0,
        dtype=jnp.float32
    ):
        self.n_predictors = n_predictors
        self.prior_scale = prior_scale
        self.noise_scale = noise_scale
        self.dtype = dtype
        self.var_list = ['beta', 'intercept', 'log_sigma']

    def create_prior(self):
        """Create prior distribution."""
        return tfd.JointDistributionNamed({
            'beta': tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(self.n_predictors, dtype=self.dtype),
                    scale=self.prior_scale * jnp.ones(self.n_predictors, dtype=self.dtype)
                ),
                reinterpreted_batch_ndims=1
            ),
            'intercept': tfd.Normal(
                loc=jnp.zeros([], dtype=self.dtype),
                scale=self.prior_scale
            ),
            'log_sigma': tfd.Normal(
                loc=jnp.log(self.noise_scale),
                scale=1.0
            )
        })

    def log_likelihood(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        """Compute log-likelihood for each data point."""
        X = jnp.asarray(data['X'], dtype=self.dtype)
        y = jnp.asarray(data['y'], dtype=self.dtype)

        beta = params['beta']
        intercept = params['intercept']
        log_sigma = params['log_sigma']
        sigma = jnp.exp(log_sigma)

        # Handle batch dimensions
        if beta.ndim == 1:
            mu = jnp.dot(X, beta) + intercept
        else:
            # beta shape: (n_samples, n_predictors)
            mu = jnp.einsum('np,sp->sn', X, beta) + intercept[:, None]

        if log_sigma.ndim == 0:
            log_sigma = log_sigma[None]
        sigma = jnp.exp(log_sigma)
            
        # Log likelihood
        residuals = y - mu if mu.ndim == 1 else y[None, :] - mu
        
        # log_sigma is (S,) or (1,). log_sigma[:, None] is (S, 1) or (1, 1).
        log_lik = -0.5 * jnp.log(2 * jnp.pi) - log_sigma[:, None] - 0.5 * (residuals / sigma[:, None])**2

        return log_lik

    def unormalized_log_prob(self, data: Dict[str, Any], scale_factor: float = 1.0, **params) -> jnp.ndarray:
        """Compute unnormalized log probability (log joint)."""
        log_lik = self.log_likelihood(data, params)
        total_log_lik = jnp.sum(log_lik, axis=-1) * scale_factor

        prior = self.create_prior()
        log_prior = prior.log_prob(params)

        return total_log_lik + log_prior


class SimpleLogisticRegression:
    """Simple Bayesian logistic regression for binary outcomes."""

    def __init__(
        self,
        n_predictors: int = 1,
        prior_scale: float = 1.0,
        dtype=jnp.float32
    ):
        self.n_predictors = n_predictors
        self.prior_scale = prior_scale
        self.dtype = dtype
        self.var_list = ['beta', 'intercept']

    def create_prior(self):
        """Create prior distribution."""
        return tfd.JointDistributionNamed({
            'beta': tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(self.n_predictors, dtype=self.dtype),
                    scale=self.prior_scale * jnp.ones(self.n_predictors, dtype=self.dtype)
                ),
                reinterpreted_batch_ndims=1
            ),
            'intercept': tfd.Normal(
                loc=jnp.zeros([], dtype=self.dtype),
                scale=self.prior_scale
            )
        })

    def log_likelihood(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        """Compute log-likelihood for each data point."""
        X = jnp.asarray(data['X'], dtype=self.dtype)
        y = jnp.asarray(data['y'], dtype=self.dtype)

        beta = params['beta']
        intercept = params['intercept']

        # Handle batch dimensions
        if beta.ndim == 1:
            logits = jnp.dot(X, beta) + intercept
        else:
            # batch
            if intercept.ndim == 0:
                 intercept = intercept[None]
            logits = jnp.einsum('np,sp->sn', X, beta) + intercept[:, None]

        # Bernoulli log likelihood
        if logits.ndim == 1:
            log_lik = y * jax.nn.log_sigmoid(logits) + (1 - y) * jax.nn.log_sigmoid(-logits)
        else:
            log_lik = y[None, :] * jax.nn.log_sigmoid(logits) + (1 - y[None, :]) * jax.nn.log_sigmoid(-logits)

        return log_lik

    def unormalized_log_prob(self, data: Dict[str, Any], scale_factor: float = 1.0, **params) -> jnp.ndarray:
        """Compute unnormalized log probability (log joint)."""
        log_lik = self.log_likelihood(data, params)
        total_log_lik = jnp.sum(log_lik, axis=-1) * scale_factor

        prior = self.create_prior()
        log_prior = prior.log_prob(params)

        return total_log_lik + log_prior



class SimpleOrdinalLogisticRegression:
    """Simple Bayesian ordinal logistic regression."""

    def __init__(
        self,
        n_classes: int,
        n_predictors: int = 1,
        prior_scale: float = 1.0,
        dtype=jnp.float32
    ):
        self.n_classes = n_classes
        self.n_cutpoints = n_classes - 1
        self.n_predictors = n_predictors
        self.prior_scale = prior_scale
        self.dtype = dtype
        # Use unconstrained parameters for optimization stability
        self.var_list = ['beta', 'cutpoints_raw']

    def create_prior(self):
        """
        Create prior distribution.
        
        We use unconstrained 'cutpoints_raw' which will be transformed 
        to ordered 'cutpoints' via tfb.Ascending() (or Softplus-cumsum).
        """
        return tfd.JointDistributionNamed({
            'beta': tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(self.n_predictors, dtype=self.dtype),
                    scale=self.prior_scale * jnp.ones(self.n_predictors, dtype=self.dtype)
                ),
                reinterpreted_batch_ndims=1
            ),
            # Unconstrained prior for cutpoints parameters
            # We will transform these to ordered cutpoints in log_likelihood.
            # Good initialization of the prior mean helps:
            # If we want cutpoints roughly at [-2, -1, 0, 1], we can set prior locs 
            # such that forward(locs) ~= [-2, ...].
            # For simplicity, we use N(0, 5) but initialized loosely.
            # Ascending bijector: y[0] = x[0], y[i] = x[0] + sum_j=1^i exp(x[j]) ?
            # TFP Ascending is usually: [x0, x0 + exp(x1), x0 + exp(x1) + exp(x2)...]
            # So x0 is location, x1... are logs of gaps.
            'cutpoints_raw': tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(self.n_cutpoints, dtype=self.dtype),
                    scale=5.0 * jnp.ones(self.n_cutpoints, dtype=self.dtype)
                ),
                reinterpreted_batch_ndims=1
            )
        })

    def _transform_cutpoints(self, cutpoints_raw):
        """Transform unconstrained raw cutpoints to ordered cutpoints."""
        # Bijector for unconstrained -> ordered
        # tfb.Ascending() is effectively: [x0, x0 + softplus(x1), ...] or similar.
        # Let's verify TFP behavior or implement manually for safety.
        # Manual implementation of "ordered" transformation:
        # c[0] = raw[0]
        # c[i] = c[i-1] + softplus(raw[i]) for i > 0
        # This ensures strict ordering.
        
        # JAX/TFP implementation:
        # We can use tfp.bijectors.Ascending() if available, but manual is safer without verifying version.
        # Actually simplest: cumsum of softplus?
        # That forces c[0] > 0 if raw[0] is unconstrained? No.
        # Standard: c[0] = raw[0]. gaps = softplus(raw[1:]).
        # c = concat([raw[0], gaps]).cumsum()
        
        # Let's use tfb.Ascending() if we can import it, otherwise manual.
        # Assuming tfp.bijectors is not aliased as tfb in the file yet.
        # We imported tensorflow_probability.substrates.jax as tfp
        tfb = tfp.bijectors
        
        # tfb.Ascending() typically maps R^k -> {y in R^k : y0 < y1 < ... < yk}
        # It handles the Jacobian too if we used it in TransformedDistribution.
        # But here we just want the value transform.
        bij = tfb.Ascending()
        return bij.forward(cutpoints_raw)

    def log_likelihood(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        """Compute log-likelihood for each data point."""
        X = jnp.asarray(data['X'], dtype=self.dtype)
        y = jnp.asarray(data['y'], dtype=self.dtype) # y should be integer 0..K-1
        
        beta = params['beta']
        cutpoints_raw = params['cutpoints_raw']
        
        # Transform cutpoints
        # Handle batching
        if cutpoints_raw.ndim == 1:
            cutpoints = self._transform_cutpoints(cutpoints_raw)
        else:
            # vmap the transform over batch dimension
            cutpoints = jax.vmap(self._transform_cutpoints)(cutpoints_raw)
        
        # Linear predictor
        # beta: (n_samples, n_predictors) or (n_predictors,)
        if beta.ndim == 1:
            eta = jnp.dot(X, beta) # (N,)
        else:
            eta = jnp.einsum('np,sp->sn', X, beta) # (n_samples, N)
        
        # Align dimensions
        # Ensure cutpoints is at least 1D
        if cutpoints.ndim == 0:
            cutpoints = cutpoints[None]

        if cutpoints.ndim == 1 and beta.ndim == 1:
            # No batching
            dist = tfd.OrderedLogistic(cutpoints=cutpoints, loc=eta)
            return dist.log_prob(y)
        
        # Batching
        if cutpoints.ndim == 1:
            # cutpoints is (K-1,), eta is (S, N)
            # We need to broadcast cutpoints to (S, 1, K-1) or similar?
            # tfd.OrderedLogistic: loc (...,) cutpoints (..., K-1)
            # If loc is (S, N), we want cutpoints to be (S, 1, K-1) so it broadcasts to (S, N, K-1) 
            # effectively using the same cutpoints for all N.
            cutpoints_expanded = cutpoints[None, None, :] 
            # But wait, if cutpoints is just (K-1,), TFP might broadcast automatically?
            # loc (S, N), cutpoints (K-1). Broadcast -> (S, N, K-1) ?
            # Let's try explicit broadcast to match batch dims (S).
            # cutpoints_expanded = cutpoints[None, None, :]
        elif cutpoints.ndim == 2: # (S, K-1)
             cutpoints_expanded = cutpoints[:, None, :] # (S, 1, K-1)
        else:
             cutpoints_expanded = cutpoints # Should likely not happen or be handled
        
        dist = tfd.OrderedLogistic(cutpoints=cutpoints_expanded, loc=eta)
        log_prob = dist.log_prob(y[None, :]) # (S, N)
        return log_prob

    def unormalized_log_prob(self, data: Dict[str, Any], scale_factor: float = 1.0, **params) -> jnp.ndarray:
        """Compute unnormalized log probability (log joint).
        
        Args:
            data: Dictionary with 'X' and 'y'
            scale_factor: Factor to scale log-likelihood (for batch inference)
            **params: Model parameters
        """
        # Note: We define prior on cutpoints_raw (gaussian).
        # We do NOT include the Jacobian of the transform in the prior term 
        # because the prior is explicitly on the raw parameter space.
        # This effectively induces a prior on cutpoints that is pushforward of Normal.
        # This is fine and standard for VI.
        
        log_lik = self.log_likelihood(data, params)
        total_log_lik = jnp.sum(log_lik, axis=-1) * scale_factor

        prior = self.create_prior()
        log_prior = prior.log_prob(params)

        return total_log_lik + log_prior

class MICEBayesianLOO(MICELogistic):
    """
    MICE Bayesian LOO-CV Framework using Pathfinder variational inference.

    For P variables, this class fits:
    - P zero-predictor (intercept-only) models
    - Up to P*(P-1) one-predictor univariate models

    Each univariate model predicts variable i using variable j, where the data
    subset only includes observations where both variables are observed.

    Attributes:
        variable_names: List of variable names
        variable_types: Dict mapping variable name to type ('continuous' or 'binary')
        zero_predictor_results: Dict mapping variable index to UnivariateModelResult
        univariate_results: Dict mapping (target_idx, predictor_idx) to UnivariateModelResult
    """

    def __init__(
        self,
        n_imputations: int = 5,
        max_iter: int = 5,
        random_state: int = 42,
        prior_scale: float = 1.0,
        noise_scale: float = 1.0,
        pathfinder_num_samples: int = 200,
        pathfinder_maxiter: int = 100,
        min_obs: int = 5,
        batch_size: Optional[int] = None,
        inference_method: str = 'pathfinder',
        verbose: bool = True
    ):
        """
        Initialize MICEBayesianLOO.

        Args:
            n_imputations: Number of imputations (passed to parent)
            max_iter: Maximum MICE iterations (passed to parent)
            random_state: Random seed
            prior_scale: Prior scale for regression coefficients
            noise_scale: Prior scale for noise (continuous variables)
            pathfinder_num_samples: Number of samples for Pathfinder
            pathfinder_maxiter: Maximum iterations for Pathfinder
            min_obs: Minimum observations required to fit a model
            batch_size: Maximum observations per model fit (None = use all data).
                       For large datasets, use 512-1024 for memory efficiency.
            inference_method: 'pathfinder' or 'advi'. Pathfinder is faster but ADVI
                             may be more robust for difficult posteriors.
            verbose: Print progress information
        """
        super().__init__(n_imputations, max_iter, random_state, n_predictors=1)
        self.prior_scale = prior_scale
        self.noise_scale = noise_scale
        self.pathfinder_num_samples = pathfinder_num_samples
        self.pathfinder_maxiter = pathfinder_maxiter
        self.min_obs = min_obs
        self.batch_size = batch_size
        self.inference_method = inference_method
        self.verbose = verbose
        self.dtype = jnp.float32

        self.variable_names: List[str] = []
        self.variable_types: Dict[int, str] = {}
        self.zero_predictor_results: Dict[int, UnivariateModelResult] = {}
        self.univariate_results: Dict[Tuple[int, int], UnivariateModelResult] = {}
        self.prediction_graph: Dict[str, List[str]] = {}
        self.n_obs_total: int = 0  # Overall dataset size

    def _infer_variable_type(self, values: np.ndarray) -> str:
        """Infer variable type from values."""
        unique_values = np.unique(values[~np.isnan(values)])
        if len(unique_values) == 2 and set(unique_values).issubset({0, 1, 0.0, 1.0}):
            return 'binary'
        if len(unique_values) >= 2 and len(unique_values) < 20:
            # Check if values are effectively integers
            is_integer = np.all(np.mod(unique_values, 1) == 0)
            if is_integer:
                return 'ordinal'
            else:
                pass
        return 'continuous'

    def _get_observed_mask(self, data: np.ndarray, var_idx: int) -> np.ndarray:
        """Get mask for observed values of a variable."""
        return ~np.isnan(data[:, var_idx])

    def _get_overlapping_mask(self, data: np.ndarray, idx1: int, idx2: int) -> np.ndarray:
        """Get mask for observations where both variables are observed."""
        mask1 = self._get_observed_mask(data, idx1)
        mask2 = self._get_observed_mask(data, idx2)
        return mask1 & mask2

    def _check_for_nan(self, params: Optional[Dict[str, jnp.ndarray]], elbo: float) -> bool:
        """Check if parameters or ELBO contain NaN values."""
        if params is None:
            return False
        
        # Check ELBO
        if np.isnan(elbo) or np.isinf(elbo):
            return True
        
        # Check all parameters
        for key, value in params.items():
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                return True
        
        return False

    def _run_inference_with_fallback(
        self,
        model,
        data_dict: Dict[str, Any],
        scale_factor: float,
        seed: int,
        current_dtype: jnp.dtype
    ) -> Tuple[Optional[Dict[str, jnp.ndarray]], float, bool, jnp.dtype]:
        """
        Run inference with automatic fallback to float64 if NaN detected.
        
        Returns:
            Tuple of (params, elbo, converged, dtype_used)
        """
        # Try with current dtype
        if self.inference_method == 'advi':
            params, elbo, converged = self._run_advi(model, data_dict, scale_factor=scale_factor, seed=seed)
        else:
            params, elbo, converged = self._run_pathfinder(model, data_dict, scale_factor=scale_factor, seed=seed)
        
        # Check for NaN
        if converged and self._check_for_nan(params, elbo):
            if current_dtype == jnp.float32:
                if self.verbose:
                    print(f"    NaN detected with float32, retrying with float64...")
                
                # Recreate model with float64
                old_dtype = model.dtype
                model.dtype = jnp.float64
                
                # Convert data to float64
                data_dict_f64 = {
                    'X': data_dict['X'].astype(np.float64),
                    'y': data_dict['y'].astype(np.float64)
                }
                
                # Retry inference
                if self.inference_method == 'advi':
                    params, elbo, converged = self._run_advi(model, data_dict_f64, scale_factor=scale_factor, seed=seed)
                else:
                    params, elbo, converged = self._run_pathfinder(model, data_dict_f64, scale_factor=scale_factor, seed=seed)
                
                # Restore original dtype
                model.dtype = old_dtype
                
                return params, elbo, converged, jnp.float64
            else:
                # Already using float64 and still getting NaN
                return None, float('-inf'), False, current_dtype
        
        return params, elbo, converged, current_dtype

    def _run_pathfinder(
        self,
        model,
        data: Dict[str, Any],
        scale_factor: float = 1.0,
        seed: int = 42
    ) -> Tuple[Optional[Dict[str, jnp.ndarray]], float, bool]:
        """
        Run Pathfinder variational inference.

        Args:
            model: Model instance with unormalized_log_prob method
            data: Data dictionary
            scale_factor: Factor to scale log-likelihood (for batch inference)
            seed: Random seed

        Returns:
            Tuple of (samples_dict, elbo, converged)
        """
        # Setup parameter flattening
        key = jax.random.PRNGKey(seed)
        prior = model.create_prior()
        prior_sample = prior.sample(seed=key)
        template = prior_sample
        flat_template, unflatten_fn = jax.flatten_util.ravel_pytree(template)
        param_dim = flat_template.shape[0]

        # Define log probability for Pathfinder
        def logprob_fn_flat(params_flat):
            params_dict = unflatten_fn(params_flat)
            return jnp.squeeze(model.unormalized_log_prob(data=data, scale_factor=scale_factor, **params_dict))

        # Initial position
        initial_position = jax.random.normal(jax.random.PRNGKey(seed + 1), (param_dim,)) * 0.1

        try:
            # Run Pathfinder
            state, info = pathfinder.approximate(
                rng_key=jax.random.PRNGKey(seed + 2),
                logdensity_fn=logprob_fn_flat,
                initial_position=initial_position,
                num_samples=self.pathfinder_num_samples,
                maxiter=self.pathfinder_maxiter,
                ftol=1e-6,
                gtol=1e-9,
            )

            elbo = float(state.elbo)
            converged = True

            # Sample from approximate posterior
            sample_key = jax.random.PRNGKey(seed + 3)
            samples_result = pathfinder.sample(sample_key, state, num_samples=self.pathfinder_num_samples)
            samples_flat = samples_result[0] if isinstance(samples_result, tuple) else samples_result

            # Unflatten samples
            samples_dict = {var: [] for var in model.var_list}
            for i in range(self.pathfinder_num_samples):
                sample = unflatten_fn(samples_flat[i])
                for var in model.var_list:
                    samples_dict[var].append(sample[var])

            # Stack samples
            for var in model.var_list:
                samples_dict[var] = jnp.stack(samples_dict[var], axis=0)

            return samples_dict, elbo, converged

        except Exception as e:
            # Always print errors for debugging purposes
            print(f"    Pathfinder failed: {e}")
            import traceback
            traceback.print_exc()
            return None, float('-inf'), False

    def _run_advi(
        self,
        model,
        data: Dict[str, Any],
        scale_factor: float = 1.0,
        seed: int = 42
    ) -> Tuple[Optional[Dict[str, jnp.ndarray]], float, bool]:
        """
        Run minibatch ADVI inference.

        Args:
            model: Model instance with unormalized_log_prob method
            data: Data dictionary
            scale_factor: Factor to scale log-likelihood (for batch inference)
            seed: Random seed

        Returns:
            Tuple of (samples_dict, elbo, converged)
        """
        from bayesianquilts.vi.minibatch import minibatch_mc_variational_loss
        import optax
        
        # Setup parameter flattening
        key = jax.random.PRNGKey(seed)
        prior = model.create_prior()
        prior_sample = prior.sample(seed=key)
        
        # Create mean-field surrogate
        # Use same structure as prior but with trainable loc and scale
        def create_surrogate(params):
            """Create mean-field normal surrogate."""
            surrogate_dists = {}
            for var_name in model.var_list:
                # Get shape from prior sample
                param_val = prior_sample[var_name]
                if isinstance(param_val, jnp.ndarray):
                    shape = param_val.shape
                else:
                    shape = ()
                
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
        
        # Initialize surrogate parameters
        surrogate_params = {}
        for var_name in model.var_list:
            param_val = prior_sample[var_name]
            if isinstance(param_val, jnp.ndarray):
                shape = param_val.shape
            else:
                shape = ()
            
            surrogate_params[f'{var_name}_loc'] = jnp.zeros(shape, dtype=self.dtype)
            surrogate_params[f'{var_name}_log_scale'] = jnp.zeros(shape, dtype=self.dtype) - 1.0  # Start with small scale
        
        # Define target log prob
        def target_log_prob_fn(data, **params):
            # Extract actual model params from surrogate params
            model_params = {var: params[var] for var in model.var_list if var in params}
            return model.unormalized_log_prob(data=data, scale_factor=scale_factor, **model_params)
        
        # Optimization loop
        optimizer = optax.adam(learning_rate=5e-3)
        opt_state = optimizer.init(surrogate_params)
        
        @jax.jit
        def update_step(params, opt_state, seed):
            def loss_fn(p):
                surrogate = create_surrogate(p)
                return minibatch_mc_variational_loss(
                    target_log_prob_fn=target_log_prob_fn,
                    surrogate_posterior=surrogate,
                    dataset_size=1,  # Already scaled by scale_factor
                    batch_size=1,
                    data=data,
                    sample_size=10,
                    seed=seed
                )
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state, loss

        best_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        for step in range(self.pathfinder_maxiter):
            surrogate_params, opt_state, loss = update_step(surrogate_params, opt_state, seed + step)
            
            # Check convergence
            if loss < best_loss:
                best_loss = float(loss)
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                break
        
        # Sample from final surrogate
        try:
            final_surrogate = create_surrogate(surrogate_params)
            samples = final_surrogate.sample(self.pathfinder_num_samples, seed=jax.random.PRNGKey(seed + 1000))
            
            # Convert to dict format
            samples_dict = {}
            for var in model.var_list:
                samples_dict[var] = samples[var]
            
            return samples_dict, -best_loss, True
            
        except Exception as e:
            print(f"    ADVI sampling failed: {e}")
            return None, float('-inf'), False
        """
        Run Pathfinder variational inference.

        Returns:
            Tuple of (samples_dict, elbo, converged)
        """
        # Setup parameter flattening
        key = jax.random.PRNGKey(seed)
        prior = model.create_prior()
        prior_sample = prior.sample(seed=key)
        template = prior_sample
        flat_template, unflatten_fn = jax.flatten_util.ravel_pytree(template)
        param_dim = flat_template.shape[0]

        # Define log probability for Pathfinder
        def logprob_fn_flat(params_flat):
            params_dict = unflatten_fn(params_flat)
            return jnp.squeeze(model.unormalized_log_prob(data=data, **params_dict))

        # Initial position
        initial_position = jax.random.normal(jax.random.PRNGKey(seed + 1), (param_dim,)) * 0.1

        try:
            # Run Pathfinder
            state, info = pathfinder.approximate(
                rng_key=jax.random.PRNGKey(seed + 2),
                logdensity_fn=logprob_fn_flat,
                initial_position=initial_position,
                num_samples=self.pathfinder_num_samples,
                maxiter=self.pathfinder_maxiter,
                ftol=1e-6,
                gtol=1e-9,
            )

            elbo = float(state.elbo)
            converged = True

            # Sample from approximate posterior
            sample_key = jax.random.PRNGKey(seed + 3)
            samples_result = pathfinder.sample(sample_key, state, num_samples=self.pathfinder_num_samples)
            samples_flat = samples_result[0] if isinstance(samples_result, tuple) else samples_result

            # Unflatten samples
            samples_dict = {var: [] for var in model.var_list}
            for i in range(self.pathfinder_num_samples):
                sample = unflatten_fn(samples_flat[i])
                for var in model.var_list:
                    samples_dict[var].append(sample[var])

            # Stack samples
            for var in model.var_list:
                samples_dict[var] = jnp.stack(samples_dict[var], axis=0)

            return samples_dict, elbo, converged

        except Exception as e:
            # Always print errors for debugging purposes
            print(f"    Pathfinder failed: {e}")
            import traceback
            traceback.print_exc()
            return None, float('-inf'), False

    def _compute_loo_elpd(
        self,
        model,
        data: Dict[str, Any],
        params: Dict[str, jnp.ndarray]
    ) -> Tuple[float, float, float, float]:
        """
        Compute LOO-ELPD using PSIS.

        Returns:
            Tuple of (elpd_loo, elpd_loo_se, khat_max, khat_mean)
        """
        # Compute log-likelihood for each sample and data point
        log_lik = model.log_likelihood(data, params)  # (n_samples, n_data)

        # Convert to numpy for PSIS
        log_lik_np = np.array(log_lik)

        # Run PSIS-LOO
        loo, loos, ks = nppsis.psisloo(log_lik_np)

        # Compute standard error of LOO ELPD
        # SE = sqrt(n * var(loos)) where loos are pointwise contributions
        n = len(loos)
        elpd_se = np.sqrt(n * np.var(loos))

        return float(loo), float(elpd_se), float(np.max(ks)), float(np.mean(ks))


    def _fit_zero_predictor(
        self,
        data: np.ndarray,
        target_idx: int,
        seed: int = 42
    ) -> UnivariateModelResult:
        """Fit a zero-predictor (intercept-only) model."""
        # Get observed values
        mask = self._get_observed_mask(data, target_idx)
        y = data[mask, target_idx]
        n_obs = len(y)

        if n_obs < self.min_obs:
            return UnivariateModelResult(
                n_obs=n_obs,
                elpd_loo=float('-inf'),
                elpd_loo_per_obs=float('-inf'),
                elpd_loo_per_obs_se=float('inf'),
                khat_max=float('inf'),
                khat_mean=float('inf'),
                predictor_idx=None,
                target_idx=target_idx,
                converged=False
            )

        # Determine variable type
        var_type = self.variable_types.get(target_idx)
        if var_type is None:
            var_type = self._infer_variable_type(y)
            self.variable_types[target_idx] = var_type

        # Batch subsampling for large datasets
        scale_factor = 1.0
        if self.batch_size is not None and n_obs > self.batch_size:
            # Randomly subsample
            rng = np.random.RandomState(seed)
            subsample_idx = rng.choice(n_obs, size=self.batch_size, replace=False)
            y_batch = y[subsample_idx]
            scale_factor = n_obs / self.batch_size
        else:
            y_batch = y

        # For zero-predictor model, use X of zeros
        X = np.zeros((len(y_batch), 1), dtype=np.float32)
        data_dict = {'X': X, 'y': y_batch.astype(np.float32)}

        if var_type == 'binary':
            model = SimpleLogisticRegression(
                n_predictors=1,
                prior_scale=self.prior_scale,
                dtype=self.dtype
            )
        elif var_type == 'ordinal':
            # Use global ordinal values if available (for consistent n_classes)
            if getattr(self, 'global_ordinal_values', None) is not None:
                # Use global set
                unique_vals = self.global_ordinal_values
                n_classes = self.n_global_classes
            else:
                # Fallback to local unique values
                unique_vals = np.unique(y_batch)
                n_classes = len(unique_vals)
            
            # Map y_batch to 0..K-1 based on the CHOSEN unique values (global or local)
            val_map = {val: i for i, val in enumerate(sorted(unique_vals))}
            y_mapped = np.array([val_map[val] for val in y_batch], dtype=np.float32)
            
            data_dict['y'] = y_mapped 
            
            model = SimpleOrdinalLogisticRegression(
                n_classes=n_classes,
                n_predictors=1,
                prior_scale=self.prior_scale,
                dtype=self.dtype
            )
        else:
            model = SimpleLinearRegression(
                n_predictors=1,
                prior_scale=self.prior_scale,
                noise_scale=self.noise_scale,
                dtype=self.dtype
            )

        # Run inference (Pathfinder or ADVI)
        params, elbo, converged, _ = self._run_inference_with_fallback(
            model, data_dict, scale_factor=scale_factor, seed=seed, current_dtype=self.dtype
        )

        if not converged or params is None:
            return UnivariateModelResult(
                n_obs=n_obs,
                elpd_loo=float('-inf'),
                elpd_loo_per_obs=float('-inf'),
                elpd_loo_per_obs_se=float('inf'),
                khat_max=float('inf'),
                khat_mean=float('inf'),
                predictor_idx=None,
                target_idx=target_idx,
                converged=False
            )

        # Compute LOO-ELPD
        elpd_loo, elpd_se, khat_max, khat_mean = self._compute_loo_elpd(model, data_dict, params)

        if params is not None:
            # Compute point estimates before discarding params
            # Standardize on numpy arrays to avoid JAX device memory hold
            beta_mean = np.array(np.mean(params.get('beta', 0.0), axis=0)) if 'beta' in params else None
            intercept_mean = float(np.mean(params['intercept'])) if 'intercept' in params else None
            
            # For ordinal, cutpoints
            cutpoints_mean = None
            if 'cutpoints_raw' in params:
                 # Transform raw cutpoints to ordered ones and then average
                 # Vmap the transform over samples
                 transformed_cutpoints = jax.vmap(model._transform_cutpoints)(params['cutpoints_raw'])
                 cutpoints_mean = np.array(np.mean(transformed_cutpoints, axis=0))
        else:
            beta_mean = None
            intercept_mean = None
            cutpoints_mean = None

        return UnivariateModelResult(
            n_obs=n_obs,
            elpd_loo=elpd_loo,
            elpd_loo_per_obs=elpd_loo / n_obs if n_obs > 0 else float('-inf'),
            elpd_loo_per_obs_se=elpd_se / n_obs if n_obs > 0 else float('inf'),
            khat_max=khat_max,
            khat_mean=khat_mean,
            predictor_idx=None,
            target_idx=target_idx,
            converged=converged,
            params=None, # Drop large posterior samples to save memory
            beta_mean=beta_mean,
            intercept_mean=intercept_mean,
            cutpoints_mean=cutpoints_mean
        )

    def _fit_univariate(
        self,
        data: np.ndarray,
        target_idx: int,
        predictor_idx: int,
        seed: int = 42
    ) -> UnivariateModelResult:
        """Fit a one-predictor univariate model."""
        # Get overlapping observations
        mask = self._get_overlapping_mask(data, target_idx, predictor_idx)
        n_obs = int(np.sum(mask))

        if n_obs < self.min_obs:
            return UnivariateModelResult(
                n_obs=n_obs,
                elpd_loo=float('-inf'),
                elpd_loo_per_obs=float('-inf'),
                elpd_loo_per_obs_se=float('inf'),
                khat_max=float('inf'),
                khat_mean=float('inf'),
                predictor_idx=predictor_idx,
                target_idx=target_idx,
                converged=False
            )

        # Get data subset
        X_raw = data[mask, predictor_idx:predictor_idx+1].astype(np.float32)
        y = data[mask, target_idx].astype(np.float32)

        # Determine variable types for both target and predictor
        target_var_type = self.variable_types.get(target_idx)
        if target_var_type is None:
            target_var_type = self._infer_variable_type(y)
            self.variable_types[target_idx] = target_var_type

        predictor_var_type = self.variable_types.get(predictor_idx)
        if predictor_var_type is None:
            predictor_var_type = self._infer_variable_type(data[mask, predictor_idx])
            self.variable_types[predictor_idx] = predictor_var_type

        # Batch subsampling for large datasets
        scale_factor = 1.0
        if self.batch_size is not None and n_obs > self.batch_size:
            # Randomly subsample
            rng = np.random.RandomState(seed)
            subsample_idx = rng.choice(n_obs, size=self.batch_size, replace=False)
            X_raw_batch = X_raw[subsample_idx]
            y_batch = y[subsample_idx]
            scale_factor = n_obs / self.batch_size
        else:
            X_raw_batch = X_raw
            y_batch = y
            
        # Prepare X based on predictor type
        if predictor_var_type == 'ordinal':
            if getattr(self, 'global_ordinal_values', None) is not None:
                # Use global max for consistency across all models
                g_max = int(np.max(self.global_ordinal_values))
                max_val = g_max
            else:
                # Fallback to local max
                x_vals = X_raw_batch.flatten()
                max_val = int(np.max(x_vals))
            
            # Use ordinal one-hot encoding
            X = ordinal_one_hot_encode(X_raw_batch.astype(int), max_val)
            X = X.astype(np.float32)
            
            # Don't standardize one-hot encoded variables
            X_mean = 0.0
            X_std = 1.0
            
            n_predictors = X.shape[1]
            
        else:
            # Continuous/Standard handling
            X = X_raw_batch
            X_mean = float(np.mean(X))
            X_std = float(np.std(X))
            if X_std > 1e-6:
                X = (X - X_mean) / X_std
            else:
                X_std = 1.0  # Avoid division by zero
            n_predictors = 1

        data_dict = {'X': X, 'y': y_batch}

        # Create model
        if target_var_type == 'binary':
            model = SimpleLogisticRegression(
                n_predictors=n_predictors,
                prior_scale=self.prior_scale,
                dtype=self.dtype
            )
        elif target_var_type == 'ordinal':
             # Use global ordinal values if available (for consistent n_classes)
            if getattr(self, 'global_ordinal_values', None) is not None:
                unique_vals = self.global_ordinal_values
                n_classes = self.n_global_classes
            else:
                unique_vals = np.unique(y_batch)
                n_classes = len(unique_vals)
            
            val_map = {val: i for i, val in enumerate(sorted(unique_vals))}
            y_mapped = np.array([val_map[val] for val in y_batch], dtype=np.float32)
            data_dict['y'] = y_mapped
            
            model = SimpleOrdinalLogisticRegression(
                n_classes=n_classes,
                n_predictors=n_predictors,
                prior_scale=self.prior_scale,
                dtype=self.dtype
            )
        else:
            model = SimpleLinearRegression(
                n_predictors=n_predictors,
                prior_scale=self.prior_scale,
                noise_scale=self.noise_scale,
                dtype=self.dtype
            )

        # Run inference (Pathfinder or ADVI)
        params, elbo, converged, _ = self._run_inference_with_fallback(
            model, data_dict, scale_factor=scale_factor, seed=seed, current_dtype=self.dtype
        )

        if not converged or params is None:
            return UnivariateModelResult(
                n_obs=n_obs,
                elpd_loo=float('-inf'),
                elpd_loo_per_obs=float('-inf'),
                elpd_loo_per_obs_se=float('inf'),
                khat_max=float('inf'),
                khat_mean=float('inf'),
                predictor_idx=predictor_idx,
                target_idx=target_idx,
                converged=False,
                predictor_mean=X_mean,
                predictor_std=X_std
            )
            
        # Compute LOO-ELPD
        elpd_loo, elpd_se, khat_max, khat_mean = self._compute_loo_elpd(model, data_dict, params)

        if params is not None:
             # Compute point estimates before discarding params
             beta_mean = np.array(np.mean(params.get('beta', 0.0), axis=0)) if 'beta' in params else None
             intercept_mean = float(np.mean(params['intercept'])) if 'intercept' in params else None
            
             # For ordinal, cutpoints
             cutpoints_mean = None
             if 'cutpoints_raw' in params:
                  # Transform raw cutpoints to ordered ones and then average
                  transformed_cutpoints = jax.vmap(model._transform_cutpoints)(params['cutpoints_raw'])
                  cutpoints_mean = np.array(np.mean(transformed_cutpoints, axis=0))
        else:
             beta_mean = None
             intercept_mean = None
             cutpoints_mean = None

        return UnivariateModelResult(
            n_obs=n_obs,
            elpd_loo=elpd_loo,
            elpd_loo_per_obs=elpd_loo / n_obs if n_obs > 0 else float('-inf'),
            elpd_loo_per_obs_se=elpd_se / n_obs if n_obs > 0 else float('inf'),
            khat_max=khat_max,
            khat_mean=khat_mean,
            predictor_idx=predictor_idx,
            target_idx=target_idx,
            converged=converged,
            params=None, # Drop large posterior samples
            predictor_mean=X_mean,
            predictor_std=X_std,
            beta_mean=beta_mean,
            intercept_mean=intercept_mean,
            cutpoints_mean=cutpoints_mean
        )
        """
        Run Pathfinder variational inference.

        Returns:
            Tuple of (samples_dict, elbo, converged)
        """
        # Setup parameter flattening
        key = jax.random.PRNGKey(seed)
        prior = model.create_prior()
        prior_sample = prior.sample(seed=key)
        template = prior_sample
        flat_template, unflatten_fn = jax.flatten_util.ravel_pytree(template)
        param_dim = flat_template.shape[0]

        # Define log probability for Pathfinder
        def logprob_fn_flat(params_flat):
            params_dict = unflatten_fn(params_flat)
            return jnp.squeeze(model.unormalized_log_prob(data=data, **params_dict))

        # Initial position
        initial_position = jax.random.normal(jax.random.PRNGKey(seed + 1), (param_dim,)) * 0.1

        try:
            # Run Pathfinder
            state, info = pathfinder.approximate(
                rng_key=jax.random.PRNGKey(seed + 2),
                logdensity_fn=logprob_fn_flat,
                initial_position=initial_position,
                num_samples=self.pathfinder_num_samples,
                maxiter=self.pathfinder_maxiter,
                ftol=1e-6,
                gtol=1e-9,
            )

            elbo = float(state.elbo)
            converged = True

            # Sample from approximate posterior
            sample_key = jax.random.PRNGKey(seed + 3)
            samples_result = pathfinder.sample(sample_key, state, num_samples=self.pathfinder_num_samples)
            samples_flat = samples_result[0] if isinstance(samples_result, tuple) else samples_result

            # Unflatten samples
            samples_dict = {var: [] for var in model.var_list}
            for i in range(self.pathfinder_num_samples):
                sample = unflatten_fn(samples_flat[i])
                for var in model.var_list:
                    samples_dict[var].append(sample[var])

            # Stack samples
            for var in model.var_list:
                samples_dict[var] = jnp.stack(samples_dict[var], axis=0)

            return samples_dict, elbo, converged

        except Exception as e:
            if self.verbose:
                print(f"    Pathfinder failed: {e}")
            return None, float('-inf'), False

    def _compute_loo_elpd(
        self,
        model,
        data: Dict[str, Any],
        params: Dict[str, jnp.ndarray]
    ) -> Tuple[float, float, float, float]:
        """
        Compute LOO-ELPD using PSIS.

        Returns:
            Tuple of (elpd_loo, elpd_loo_se, khat_max, khat_mean)
        """
        # Compute log-likelihood for each sample and data point
        log_lik = model.log_likelihood(data, params)  # (n_samples, n_data)

        # Convert to numpy for PSIS
        log_lik_np = np.array(log_lik)

        # Run PSIS-LOO
        loo, loos, ks = nppsis.psisloo(log_lik_np)

        # Compute standard error of LOO ELPD
        # SE = sqrt(n * var(loos)) where loos are pointwise contributions
        n = len(loos)
        elpd_se = np.sqrt(n * np.var(loos))

        return float(loo), float(elpd_se), float(np.max(ks)), float(np.mean(ks))


    def fit_loo_models(
        self,
        X_df: pd.DataFrame,
        fit_zero_predictors: bool = True,
        seed: int = 42,
        save_dir: Optional[Union[str, Path]] = None,
        n_jobs: int = -1,
        n_top_features: int = 50
    ) -> 'MICEBayesianLOO':
        """
        Fit all univariate models for LOO-CV evaluation.

        Args:
            X_df: DataFrame with potentially missing values (NaN)
            fit_zero_predictors: Whether to fit zero-predictor (intercept-only) models
            seed: Random seed
            save_dir: Directory to save incremental results. If None, keeps all in memory.
            n_jobs: Number of parallel jobs. -1 uses all cores.
            n_top_features: Number of top correlated features to consider as predictors.

        Returns:
            self
        """
        try:
            from tqdm import tqdm
        except ImportError:
            # Fallback if tqdm not installed
            def tqdm(iterable, **kwargs): return iterable
        
        try:
            from joblib import Parallel, delayed
        except ImportError:
            print("Warning: joblib not installed. Falling back to sequential execution.")
            n_jobs = 1
            def Parallel(n_jobs=1, **kwargs):
                return lambda x: list(x)
            def delayed(func): return func

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                print(f"Saving incremental results to {save_dir}")

        data = X_df.values
        self.variable_names = list(X_df.columns)
        n_variables = data.shape[1]
        self.n_obs_total = data.shape[0]

        # 1. Infer Variable Types & Global Ordinal Values
        # The user requested to assume possible values are the same for all variables.
        # We'll collect all unique values from columns identified as ordinal.
        
        all_ordinal_values = set()
        
        for i in range(n_variables):
            # Check existing or infer
            var_type = self.variable_types.get(i)
            if var_type is None:
                # Basic inference on observed data
                y_obs = data[~np.isnan(data[:, i]), i]
                var_type = self._infer_variable_type(y_obs)
                self.variable_types[i] = var_type
            
            if var_type == 'ordinal':
                # Collect values
                y_obs = data[~np.isnan(data[:, i]), i]
                all_ordinal_values.update(np.unique(y_obs).tolist())

        # Sort global values (assuming they are comparable, likely integers)
        if all_ordinal_values:
            self.global_ordinal_values = np.array(sorted(list(all_ordinal_values)))
            self.n_global_classes = len(self.global_ordinal_values)
        else:
            self.global_ordinal_values = None
            self.n_global_classes = 0

        # Compute Spearman correlation matrix for feature selection
        if self.verbose:
            print("Computing feature correlations...")
        corr_matrix = X_df.corr(method='spearman').abs().values
        # Fill self-correlation with -1 to exclude from top selection (though we skip i==j explicitly anyway)
        np.fill_diagonal(corr_matrix, -1.0)
        
        if self.verbose:
            print(f"Fitting MICE Bayesian LOO-CV models with Pathfinder")
            print(f"  Variables: {n_variables}")
            print(f"  Observations: {self.n_obs_total}")
            print(f"  Min obs per model: {self.min_obs}")
            print(f"  Parallel jobs: {n_jobs}")
            print(f"  Top features per target: {n_top_features}")
            if self.n_global_classes > 0:
                print(f"  Global Ordinal Values: {self.global_ordinal_values} (n={self.n_global_classes})")

        # Fit zero-predictor models
        if fit_zero_predictors:
            if self.verbose:
                print("\nFitting zero-predictor models...")

            if self.verbose:
                print(f"  Scheduling {n_variables} zero-predictor jobs...")

            def fit_zero(i):
                return self._fit_zero_predictor(data, i, seed=seed + i)

            # Use return_as="generator" to allow tqdm to track completion
            results_gen = Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(fit_zero)(i) for i in range(n_variables)
            )
            
            # Wrap generator with tqdm if verbose is off
            if not self.verbose:
                results_gen = tqdm(results_gen, total=n_variables, desc="Zero-Predictor Models")

            # Collect results locally to avoid modifying self while pickling tasks
            local_results = []
            for i, result in zip(range(n_variables), results_gen):
                local_results.append((i, result))
                if self.verbose:
                     if result.converged:
                        print(f"  Var {i} ({self.variable_names[i]}): n_obs={result.n_obs}, elpd/n={result.elpd_loo_per_obs:.4f}")
                     else:
                        print(f"  Var {i} ({self.variable_names[i]}): FAILED/SKIPPED")

            # Update state after parallel execution finishes
            for i, result in local_results:
                self.zero_predictor_results[i] = result

        # Fit one-predictor models
        if self.verbose:
            print("\nFitting one-predictor models...")

        if not self.verbose:
            pbar = tqdm(range(n_variables), desc="Target Variables (One-Predictor)")
            iterator = pbar
        else:
            pbar = None
            iterator = range(n_variables)

        for i in iterator:
            target_name = self.variable_names[i]
            if pbar is not None:
                # Truncate if too long
                disp_name = target_name[:20] + "..." if len(target_name) > 20 else target_name
                pbar.set_description(f"Target: {disp_name}")
            
            # Identify candidate predictors based on correlation
            # Get correlations for target i
            target_corrs = corr_matrix[i, :]
            # Get indices of top k correlated features
            # argsort returns indices that sort the array. We take the last n_top_features.
            # But we must be careful with NaNs in correlation matrix (if data is all NaN or constant)
            # np.argsort handles NaNs by putting them at the end if kind='quicksort' (default).
            # But we want highest values.
            # Let's handle NaNs explicitly just in case.
            valid_corr_mask = np.isfinite(target_corrs)
            valid_corr_indices = np.where(valid_corr_mask)[0]
            
            if len(valid_corr_indices) == 0:
                top_features = []
            else:
                # Sort indices by correlation value descending
                sorted_indices = valid_corr_indices[np.argsort(target_corrs[valid_corr_indices])[::-1]]
                # Filter out self-correlation (should be -1 or 1, but we want to exclude i)
                sorted_indices = [idx for idx in sorted_indices if idx != i]
                # Take top N
                top_features = sorted_indices[:n_top_features]
            
            top_features_set = set(top_features)

            # Identify valid predictors (filtering step)
            valid_predictors = []
            
            # Only iterate over top features to check overlap condition
            # We must verify min_obs for these candidates
            for j in top_features:
                # Check overlap efficiently before scheduling job
                mask = self._get_overlapping_mask(data, i, j)
                if np.sum(mask) >= self.min_obs:
                    valid_predictors.append(j)
            
            if not valid_predictors:
                if self.verbose:
                    print(f"Skipping {target_name}: No valid top-{n_top_features} predictors with >= {self.min_obs} obs.")
                continue

            # Parallelize fitting of valid predictors for this target
            if self.verbose:
                print(f"  Processing {target_name} ({len(valid_predictors)} valid predictors)")

            def fit_single(j):
                return self._fit_univariate(
                    data, i, j,
                    seed=seed + n_variables + (i * n_variables + j) # Unique seed
                )

            # Use return_as="generator" for inner loop progress
            results_gen = Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(fit_single)(j) for j in valid_predictors
            )
            
            # Optional inner progress bar
            if not self.verbose:
                results_gen = tqdm(results_gen, total=len(valid_predictors), desc=f"Predictors for {target_name[:20]}...", leave=False)

            # Collect results for this target locally
            target_results = {}
            if fit_zero_predictors and i in self.zero_predictor_results:
                target_results['zero_predictor'] = self._result_to_dict(self.zero_predictor_results[i])
            
            univariate_list = []
            local_updates = []
            
            # Consume generator
            for j, result in zip(valid_predictors, results_gen):
                # Stash for update later
                local_updates.append((j, result))
                
                # Add to local list for saving
                uni_res_dict = self._result_to_dict(result)
                uni_res_dict['predictor_name'] = self.variable_names[j]
                uni_res_dict['predictor_idx'] = j
                univariate_list.append(uni_res_dict)
            
            # Update self state safely
            for j, result in local_updates:
                self.univariate_results[(i, j)] = result
            
            # Save results for this target variable if requested
            if save_dir is not None:
                target_results['univariate_models'] = univariate_list
                target_results['target_name'] = target_name
                target_results['target_idx'] = i
                
                # Sanitize filename
                safe_name = "".join(c if c.isalnum() or c in ('-','_') else '_' for c in target_name)
                # handle potential lengthy names
                if len(safe_name) > 100:
                     safe_name = safe_name[:100] + f"_hash{hash(target_name)}"

                file_path = save_dir / f"model_target_{i}_{safe_name}.yaml"
                self._save_yaml(file_path, target_results)

        # Populate prediction graph
        self.prediction_graph = {}
        for (target_idx, predictor_idx), result in self.univariate_results.items():
            if result.converged:
                target_name = self.variable_names[target_idx]
                predictor_name = self.variable_names[predictor_idx]
                if target_name not in self.prediction_graph:
                    self.prediction_graph[target_name] = []
                self.prediction_graph[target_name].append(predictor_name)

        return self

    def _save_yaml(self, file_path: Path, data: Dict[str, Any]):
        """Helper to save results to YAML."""
        import yaml
        
        # Custom representer for numpy scalars
        def repr_float(dumper, data):
            return dumper.represent_float(float(data))
        def repr_int(dumper, data):
            return dumper.represent_int(int(data))
            
        yaml.add_representer(np.float32, repr_float)
        yaml.add_representer(np.float64, repr_float)
        yaml.add_representer(np.int32, repr_int)
        yaml.add_representer(np.int64, repr_int)

        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                
                # Optional: clear from memory if trying to save RAM?
                # But we might need them for self.univariate_results access later?
                # The user just wants to save coefficients.
                # If N=500, self.univariate_results will grow large.
                # If we save to disk, we technically don't need to keep them in memory if we don't return them.
                # But the method returns 'self'.
                # For safety, I'll keep them in memory unless memory is tight.
                # Assuming 12GB RAM, 600MB is fine.

        if self.verbose:
             # ... (same summary stats)
             pass

        if self.verbose:
            n_converged_zero = sum(1 for r in self.zero_predictor_results.values() if r.converged)
            n_converged_uni = sum(1 for r in self.univariate_results.values() if r.converged)
            print(f"\nCompleted:")
            print(f"  Zero-predictor: {n_converged_zero}/{len(self.zero_predictor_results)} converged")
            print(f"  Univariate: {n_converged_uni}/{len(self.univariate_results)} converged")

        return self

    def get_elpd_matrix(self) -> np.ndarray:
        """
        Get matrix of ELPD per observation values.

        Returns:
            Array of shape (n_variables, n_variables) where entry [i, j] is
            the ELPD per observation for predicting variable i from variable j.
            Diagonal entries are from zero-predictor models.
        """
        n_variables = len(self.variable_names)
        matrix = np.full((n_variables, n_variables), np.nan)

        # Fill diagonal with zero-predictor results
        for i, result in self.zero_predictor_results.items():
            if result.converged:
                matrix[i, i] = result.elpd_loo_per_obs

        # Fill off-diagonal with univariate results
        for (target_idx, predictor_idx), result in self.univariate_results.items():
            if result.converged:
                matrix[target_idx, predictor_idx] = result.elpd_loo_per_obs

        return matrix

    def get_elpd_se_matrix(self) -> np.ndarray:
        """
        Get matrix of ELPD per observation standard errors.

        Returns:
            Array of shape (n_variables, n_variables) where entry [i, j] is
            the standard error of ELPD per observation for predicting variable i from variable j.
            Diagonal entries are from zero-predictor models.
        """
        n_variables = len(self.variable_names)
        matrix = np.full((n_variables, n_variables), np.nan)

        # Fill diagonal with zero-predictor results
        for i, result in self.zero_predictor_results.items():
            if result.converged:
                matrix[i, i] = result.elpd_loo_per_obs_se

        # Fill off-diagonal with univariate results
        for (target_idx, predictor_idx), result in self.univariate_results.items():
            if result.converged:
                matrix[target_idx, predictor_idx] = result.elpd_loo_per_obs_se

        return matrix

    def get_n_obs_matrix(self) -> np.ndarray:
        """
        Get matrix of observation counts.

        Returns:
            Array of shape (n_variables, n_variables) where entry [i, j] is
            the number of observations used for predicting variable i from variable j.
        """
        n_variables = len(self.variable_names)
        matrix = np.zeros((n_variables, n_variables), dtype=int)

        # Fill diagonal with zero-predictor results
        for i, result in self.zero_predictor_results.items():
            matrix[i, i] = result.n_obs

        # Fill off-diagonal with univariate results
        for (target_idx, predictor_idx), result in self.univariate_results.items():
            matrix[target_idx, predictor_idx] = result.n_obs

        return matrix

    def get_improvement_matrix(self) -> np.ndarray:
        """
        Get matrix of ELPD improvements over zero-predictor.

        Returns:
            Array of shape (n_variables, n_variables) where entry [i, j] is
            elpd[i, j] - elpd[i, i] (improvement from adding predictor j).
        """
        elpd_matrix = self.get_elpd_matrix()
        baseline = np.diag(elpd_matrix).copy()
        improvement = elpd_matrix - baseline[:, None]
        return improvement

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        elpd_matrix = self.get_elpd_matrix()
        n_obs_matrix = self.get_n_obs_matrix()
        improvement_matrix = self.get_improvement_matrix()
        se_matrix = self.get_elpd_se_matrix()

        n_variables = len(self.variable_names)

        # Find best predictors for each variable
        best_predictors = {}
        for i in range(n_variables):
            target = self.variable_names[i]
            row = improvement_matrix[i, :].copy()
            row[i] = -np.inf  # Exclude self
            if np.any(np.isfinite(row)):
                j = np.nanargmax(row)
                best_predictors[target] = {
                    'predictor': self.variable_names[j],
                    'improvement': row[j],
                    'elpd_per_obs': elpd_matrix[i, j],
                    'elpd_per_obs_se': se_matrix[i, j],
                    'n_obs': n_obs_matrix[i, j]
                }

        return {
            'n_variables': n_variables,
            'variable_names': self.variable_names,
            'variable_types': self.variable_types,
            'n_obs_total': self.n_obs_total,
            'elpd_matrix': elpd_matrix,
            'elpd_se_matrix': se_matrix,
            'n_obs_matrix': n_obs_matrix,
            'improvement_matrix': improvement_matrix,
            'best_predictors': best_predictors,
            'n_converged_zero': sum(1 for r in self.zero_predictor_results.values() if r.converged),
            'n_converged_univariate': sum(1 for r in self.univariate_results.values() if r.converged),
        }

    def _predict_single_univariate(
        self,
        uni_result: UnivariateModelResult,
        predictor_value: float,
        target_var_type: str
    ) -> float:
        """
        Helper to predict a single value using a univariate model.
        """
        # Standardize the predictor value
        X_mean = uni_result.predictor_mean if uni_result.predictor_mean is not None else 0.0
        X_std = uni_result.predictor_std if uni_result.predictor_std is not None else 1.0
        x_standardized = (predictor_value - X_mean) / X_std

        if uni_result.beta_mean is not None:
             # Use stored point estimates
             beta = uni_result.beta_mean
             intercept = uni_result.intercept_mean if uni_result.intercept_mean is not None else 0.0
             cutpoints = uni_result.cutpoints_mean
        elif uni_result.params is not None:
             # Fallback to computing from params if available (legacy support)
             beta = np.mean(uni_result.params.get('beta', 0.0), axis=0)
             intercept = np.mean(uni_result.params.get('intercept', 0.0))
             cutpoints = None
             if 'cutpoints_raw' in uni_result.params:
                  # Need the model to transform
                  # Since model isn't here, this is tricky. 
                  # But we shouldn't hit this often now that we store cutpoints_mean.
                  pass
        else:
             raise ValueError("No parameters available for prediction")

        if isinstance(beta, (np.ndarray, list)):
             beta_val = beta[0]
        else:
             beta_val = beta
             
        eta = float(x_standardized * beta_val + intercept)

        if target_var_type == 'binary':
            pred = 1.0 / (1.0 + np.exp(-eta))
        elif target_var_type == 'ordinal' and cutpoints is not None:
            # Ordinal prediction: Expected value
            # Probabilities for each category
            # P(Y <= k) = sigmoid(c_k - eta)
            # P(Y = k) = P(Y <= k) - P(Y <= k-1)
            def sigmoid(x):
                return 1.0 / (1.0 + np.exp(-x))
            
            p_le = sigmoid(cutpoints - eta)
            # Add boundary conditions
            p_le = np.concatenate([[0.0], p_le, [1.0]])
            p = np.diff(p_le)
            
            # Expected value: sum(k * p_k)
            categories = np.arange(len(p))
            pred = np.sum(categories * p)
        else:
            # Continuous or ordinal without cutpoints (fallback)
            pred = eta
            
        return float(pred)

    def _find_prediction_path(self, target: str, source: str) -> Optional[List[str]]:
        """
        Find a prediction path from target to source using BFS.
        
        Returns:
            List of variable names [target, intermediate_1, ..., source]
            representing the path of dependencies (target depends on int_1, which depends on source).
            Returns None if no path exists.
        """
        if target == source:
            return [target]
        
        queue = [(target, [target])]
        visited = {target}
        
        while queue:
            current_node, path = queue.pop(0)
            
            # Get predictors for current node
            predictors = self.prediction_graph.get(current_node, [])
            
            for predictor in predictors:
                if predictor == source:
                    return path + [source]
                
                if predictor not in visited:
                    visited.add(predictor)
                    queue.append((predictor, path + [predictor]))
                    
        return None

    def predict_chained(
        self,
        target: str,
        source: str,
        value: float
    ) -> Optional[float]:
        """
        Predict target variable from a distant source variable using a chain of models.

        Args:
            target: Name of the target variable to predict.
            source: Name of the source variable used for prediction.
            value: Value of the source variable.

        Returns:
            Predicted value of the target variable, or None if no path exists.

        Raises:
            ValueError: If finding a path fails or models are missing.
        """
        if target not in self.variable_names:
            raise ValueError(f"Target '{target}' not known.")
        if source not in self.variable_names:
            raise ValueError(f"Source '{source}' not known.")

        # Find path: [target, v1, v2, ..., source]
        # target depends on v1, v1 depends on v2, ..., depends on source
        path = self._find_prediction_path(target, source)
        
        if path is None:
            return None

        if self.verbose:
            print(f"Prediction path: {' <- '.join(path)}")

        # Traverse path from source to target (reverse order of dependency list)
        # path is [target, ..., source]
        # We start with source value, predict v_last, then v_last-1, ..., then target.
        
        current_value = value
        
        # Iterate backwards from source (last element) to target (first element)
        # We need pairs: (path[i-1], path[i]) where path[i] predicts path[i-1]
        # range(len(path) - 1, 0, -1) -> indices of predictors
        for i in range(len(path) - 1, 0, -1):
            predictor_name = path[i]
            target_name = path[i-1]
            
            predictor_idx = self.variable_names.index(predictor_name)
            target_idx = self.variable_names.index(target_name)
            
            key = (target_idx, predictor_idx)
            
            if key not in self.univariate_results:
                 raise ValueError(f"Model for {target_name} <- {predictor_name} missing despite being in graph.")
            
            uni_result = self.univariate_results[key]
            target_type = self.variable_types.get(target_idx, 'continuous')
            
            # Predict
            current_value = self._predict_single_univariate(uni_result, current_value, target_type)
            
        return current_value

    def estimate_chain_elpd(self, target: str, source: str) -> Optional[float]:
        """
        Estimate the LOO ELPD for a chained prediction from source to target.

        Uses Gaussian variance propagation:
        1. Convert LOO ELPD of each link to an effective noise variance.
           elpd_mean = elpd / n_obs
           sigma^2_eff = exp(-2 * elpd_mean) / (2 * pi * e)
        2. Propagate variance through the chain.
           Sigma^2_out = beta^2 * Sigma^2_in + sigma^2_eff
        3. Convert final variance back to ELPD.
           elpd_chain = n_total * (-0.5 * log(2 * pi * e * Sigma^2_final))

        Args:
            target: Name of the target variable.
            source: Name of the source variable.

        Returns:
            Estimated ELPD for the chain, or None if no path exists.
        """
        path = self._find_prediction_path(target, source)
        if path is None:
            return None

        # Initial variance of the source variable (observed, so 0 uncertainty relative to itself)
        current_variance = 0.0
        
        # Traverse path from source to target (reverse order)
        # path is [target, ..., source]
        for i in range(len(path) - 1, 0, -1):
            predictor_name = path[i]
            target_name = path[i-1]
            
            predictor_idx = self.variable_names.index(predictor_name)
            target_idx = self.variable_names.index(target_name)
            
            key = (target_idx, predictor_idx)
            uni_result = self.univariate_results[key]
            
            # 1. Effective Noise Variance from ELPD
            elpd_mean = uni_result.elpd_loo / uni_result.n_obs
            # sigma^2_eff = 1/(2*pi*e) * exp(-2*elpd_mean)
            sigma2_eff = np.exp(-2 * elpd_mean) / (2 * np.pi * np.e)
            
            # 2. Get coefficient (beta)
            if uni_result.beta_mean is not None:
                beta = uni_result.beta_mean
            elif uni_result.params is not None and 'beta' in uni_result.params:
                beta = np.mean(uni_result.params['beta'])
            else:
                beta = 0.0
            
            if isinstance(beta, (np.ndarray, list)) and len(beta) > 0:
                beta = beta[0]
            
            # 3. Propagate Variance
            # Sigma_out = beta^2 * Sigma_in + sigma_noise
            current_variance = (float(beta) ** 2) * current_variance + sigma2_eff
            
        # 4. Convert final variance back to ELPD
        # elpd_chain = N * (-0.5 * log(2*pi*e * current_variance))
        if current_variance <= 0:
            return -np.inf # Should not happen unless ELPD was inf
            
        elpd_per_obs = -0.5 * np.log(2 * np.pi * np.e * current_variance)
        estimated_elpd = elpd_per_obs * self.n_obs_total
        
        return float(estimated_elpd)

    def predict(
        self,
        items: Dict[str, float],
        target: str,
        return_details: bool = False,
        uncertainty_penalty: float = 1.0
    ) -> Union[float, Dict[str, Any]]:
        """
        Predict a target variable using stacked univariate models.

        Uses available predictors from the items dict to stack predictions from
        the zero-predictor model and univariate models. Stacking weights are
        computed as exp(elpd_loo - uncertainty_penalty * elpd_se) to hedge
        against uncertainty in the ELPD estimates.

        Args:
            items: Dict mapping variable names to their observed values.
                   Variables not in this dict are considered missing.
            target: Name of the target variable to predict.
            return_details: If True, return a dict with prediction, weights,
                           elpd_loo values, and n_obs_total for combining with
                           external models.
            uncertainty_penalty: Penalty factor for ELPD uncertainty. Higher values
                           give more weight to models with lower SE. Default 1.0
                           corresponds to using a ~68% lower confidence bound.
                           Use 0.0 to ignore uncertainty.

        Returns:
            If return_details is False: The stacked prediction (float).
            If return_details is True: Dict with keys:
                - 'prediction': The stacked prediction
                - 'weights': Dict mapping model name to weight
                - 'elpd_loo': Dict mapping model name to elpd_loo value
                - 'elpd_loo_se': Dict mapping model name to elpd_loo standard error
                - 'predictions': Dict mapping model name to individual prediction
                - 'n_obs_total': Overall dataset size

        Raises:
            ValueError: If target is not in variable_names or no models available.
        """
        if target not in self.variable_names:
            raise ValueError(f"Target '{target}' not in variable_names: {self.variable_names}")

        target_idx = self.variable_names.index(target)
        var_type = self.variable_types.get(target_idx, 'continuous')

        # Collect available models and their predictions
        # List of (name, elpd_loo, elpd_loo_se, prediction)
        models_info = []

        # 1. Zero-predictor model (always available if converged)
        if target_idx in self.zero_predictor_results:
            zero_result = self.zero_predictor_results[target_idx]
            if zero_result.converged:
                # Compute prediction from zero-predictor model
                # Use stored intercept_mean if available
                if zero_result.intercept_mean is not None:
                     intercept = zero_result.intercept_mean
                elif zero_result.params is not None:
                     intercept = np.mean(zero_result.params['intercept'])
                else:
                     # Should not happen if converged
                     intercept = 0.0 

                if var_type == 'binary':
                    # Logistic: sigmoid(intercept)
                    pred = 1.0 / (1.0 + np.exp(-intercept))
                elif var_type == 'ordinal' and zero_result.cutpoints_mean is not None:
                    # Ordinal: Expected value with eta=0 (or intercept)
                    def sigmoid(x):
                        return 1.0 / (1.0 + np.exp(-x))
                    
                    p_le = sigmoid(zero_result.cutpoints_mean - intercept)
                    p_le = np.concatenate([[0.0], p_le, [1.0]])
                    p = np.diff(p_le)
                    categories = np.arange(len(p))
                    pred = np.sum(categories * p)
                else:
                    # Linear: intercept
                    pred = intercept

                # SE for the total ELPD (not per obs)
                elpd_se = zero_result.elpd_loo_per_obs_se * zero_result.n_obs
                models_info.append(('intercept', zero_result.elpd_loo, elpd_se, float(pred)))

        # 2. Univariate models for available predictors
        for predictor_name, predictor_value in items.items():
            if predictor_name not in self.variable_names:
                continue
            if predictor_name == target:
                continue

            predictor_idx = self.variable_names.index(predictor_name)
            key = (target_idx, predictor_idx)

            if key not in self.univariate_results:
                continue

            uni_result = self.univariate_results[key]
            if not uni_result.converged:
                continue

            pred = self._predict_single_univariate(uni_result, predictor_value, var_type)

            # SE for the total ELPD (not per obs)
            elpd_se = uni_result.elpd_loo_per_obs_se * uni_result.n_obs
            models_info.append((predictor_name, uni_result.elpd_loo, elpd_se, float(pred)))

        if len(models_info) == 0:
            raise ValueError(f"No converged models available for target '{target}'")

        # Compute stacking weights: w_k = exp(elpd_loo_k - uncertainty_penalty * se_k)
        # This hedges against uncertainty by penalizing models with high SE
        elpd_values = np.array([m[1] for m in models_info])
        se_values = np.array([m[2] for m in models_info])

        # Compute uncertainty-adjusted ELPD (lower confidence bound)
        # Replace inf SE with large value for calculation
        se_safe = np.where(np.isfinite(se_values), se_values, 1e6)
        adjusted_elpd = elpd_values - uncertainty_penalty * se_safe

        # Handle -inf values
        finite_mask = np.isfinite(adjusted_elpd)
        if not np.any(finite_mask):
            # All models have -inf adjusted ELPD, use uniform weights
            weights = np.ones(len(models_info)) / len(models_info)
        else:
            # Subtract max for numerical stability
            max_adj_elpd = np.max(adjusted_elpd[finite_mask])
            log_weights = np.where(finite_mask, adjusted_elpd - max_adj_elpd, -np.inf)
            weights = np.exp(log_weights)
            weights = weights / np.sum(weights)

        # Compute stacked prediction
        predictions = np.array([m[3] for m in models_info])
        stacked_pred = float(np.sum(weights * predictions))

        if return_details:
            return {
                'prediction': stacked_pred,
                'weights': {m[0]: float(w) for m, w in zip(models_info, weights)},
                'elpd_loo': {m[0]: float(m[1]) for m in models_info},
                'elpd_loo_se': {m[0]: float(m[2]) for m in models_info},
                'predictions': {m[0]: float(m[3]) for m in models_info},
                'n_obs_total': self.n_obs_total
            }
        else:
            return stacked_pred

    def _result_to_dict(self, result: UnivariateModelResult) -> Dict[str, Any]:
        """Convert UnivariateModelResult to a serializable dict."""
        d = asdict(result)
        # Convert params (JAX arrays) to nested lists
        if d['params'] is not None:
            d['params'] = {
                k: np.array(v).tolist() for k, v in d['params'].items()
            }
        return d

    def _dict_to_result(self, d: Dict[str, Any]) -> UnivariateModelResult:
        """Convert dict back to UnivariateModelResult."""
        # Convert params back to JAX arrays
        if d['params'] is not None:
            d['params'] = {
                k: jnp.array(v, dtype=self.dtype) for k, v in d['params'].items()
            }
        return UnivariateModelResult(**d)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the fitted model to a YAML file.

        Args:
            path: Path to save the YAML file.
        """
        path = Path(path)

        # Build serializable state dict
        # Use list format for univariate_results to handle variable names with underscores
        state = {
            'version': '1.1',
            'config': {
                'prior_scale': self.prior_scale,
                'noise_scale': self.noise_scale,
                'pathfinder_num_samples': self.pathfinder_num_samples,
                'pathfinder_maxiter': self.pathfinder_maxiter,
                'min_obs': self.min_obs,
                'n_imputations': self.n_imputations,
                'max_iter': self.max_iter,
                'random_state': self.random_state,
            },
            'data': {
                'variable_names': self.variable_names,
                'variable_types': {int(k): v for k, v in self.variable_types.items()},
                'n_obs_total': self.n_obs_total,
            },
            'zero_predictor_results': {
                int(k): self._result_to_dict(v)
                for k, v in self.zero_predictor_results.items()
            },
            'univariate_results': [
                {
                    'target_idx': int(k[0]),
                    'predictor_idx': int(k[1]),
                    'result': self._result_to_dict(v)
                }
                for k, v in self.univariate_results.items()
            ],

            'prediction_graph': self.prediction_graph,
        }

        with open(path, 'w') as f:
            # Custom representer for numpy scalars (needed here too)
            import yaml
            def repr_float(dumper, data):
                return dumper.represent_float(float(data))
            def repr_int(dumper, data):
                return dumper.represent_int(int(data))
                
            yaml.add_representer(np.float32, repr_float)
            yaml.add_representer(np.float64, repr_float)
            yaml.add_representer(np.int32, repr_int)
            yaml.add_representer(np.int64, repr_int)
            
            yaml.dump(state, f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'MICEBayesianLOO':
        """
        Load a fitted model from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Loaded MICEBayesianLOO instance.
        """
        path = Path(path)

        with open(path, 'r') as f:
            state = yaml.safe_load(f)

        # Check version
        version = state.get('version', '1.0')
        if version not in ('1.0', '1.1'):
            raise ValueError(f"Unsupported model version: {version}")

        # Create instance with config
        config = state['config']
        instance = cls(
            n_imputations=config.get('n_imputations', 5),
            max_iter=config.get('max_iter', 5),
            random_state=config.get('random_state', 42),
            prior_scale=config.get('prior_scale', 1.0),
            noise_scale=config.get('noise_scale', 1.0),
            pathfinder_num_samples=config.get('pathfinder_num_samples', 200),
            pathfinder_maxiter=config.get('pathfinder_maxiter', 100),
            min_obs=config.get('min_obs', 5),
            verbose=False,
        )

        # Restore data attributes
        data = state['data']
        instance.variable_names = data['variable_names']
        instance.variable_types = {int(k): v for k, v in data['variable_types'].items()}
        instance.n_obs_total = data['n_obs_total']

        # Restore zero predictor results
        for k, v in state['zero_predictor_results'].items():
            instance.zero_predictor_results[int(k)] = instance._dict_to_result(v)

        # Restore univariate results (handle both v1.0 and v1.1 formats)
        univariate_data = state['univariate_results']
        if isinstance(univariate_data, list):
            # v1.1 format: list of dicts with target_idx, predictor_idx, result
            for item in univariate_data:
                key = (int(item['target_idx']), int(item['predictor_idx']))
                instance.univariate_results[key] = instance._dict_to_result(item['result'])
        else:
            # v1.0 format: dict with "idx_idx" keys (legacy, may break with underscores)
            for k, v in univariate_data.items():
                parts = k.split('_')
                key = (int(parts[0]), int(parts[1]))
                instance.univariate_results[key] = instance._dict_to_result(v)
        
        
        # Restore prediction graph
        instance.prediction_graph = state.get('prediction_graph', {})

        return instance

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the fitted model to a dictionary (for programmatic serialization).

        Returns:
            Dictionary representation of the model.
        """
        return {
            'version': '1.1',
            'config': {
                'prior_scale': self.prior_scale,
                'noise_scale': self.noise_scale,
                'pathfinder_num_samples': self.pathfinder_num_samples,
                'pathfinder_maxiter': self.pathfinder_maxiter,
                'min_obs': self.min_obs,
                'n_imputations': self.n_imputations,
                'max_iter': self.max_iter,
                'random_state': self.random_state,
            },
            'data': {
                'variable_names': self.variable_names,
                'variable_types': {int(k): v for k, v in self.variable_types.items()},
                'n_obs_total': self.n_obs_total,
            },
            'zero_predictor_results': {
                int(k): self._result_to_dict(v)
                for k, v in self.zero_predictor_results.items()
            },
            'univariate_results': [
                {
                    'target_idx': int(k[0]),
                    'predictor_idx': int(k[1]),
                    'result': self._result_to_dict(v)
                }
                for k, v in self.univariate_results.items()
            ],

            'prediction_graph': self.prediction_graph,
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> 'MICEBayesianLOO':
        """
        Create a fitted model from a dictionary.

        Args:
            state: Dictionary representation of the model.

        Returns:
            Loaded MICEBayesianLOO instance.
        """
        # Check version
        version = state.get('version', '1.0')
        if version not in ('1.0', '1.1'):
            raise ValueError(f"Unsupported model version: {version}")

        # Create instance with config
        config = state['config']
        instance = cls(
            n_imputations=config.get('n_imputations', 5),
            max_iter=config.get('max_iter', 5),
            random_state=config.get('random_state', 42),
            prior_scale=config.get('prior_scale', 1.0),
            noise_scale=config.get('noise_scale', 1.0),
            pathfinder_num_samples=config.get('pathfinder_num_samples', 200),
            pathfinder_maxiter=config.get('pathfinder_maxiter', 100),
            min_obs=config.get('min_obs', 5),
            verbose=False,
        )

        # Restore data attributes
        data = state['data']
        instance.variable_names = data['variable_names']
        instance.variable_types = {int(k): v for k, v in data['variable_types'].items()}
        instance.n_obs_total = data['n_obs_total']

        # Restore zero predictor results
        for k, v in state['zero_predictor_results'].items():
            instance.zero_predictor_results[int(k)] = instance._dict_to_result(v)

        # Restore univariate results (handle both v1.0 and v1.1 formats)
        univariate_data = state['univariate_results']
        if isinstance(univariate_data, list):
            # v1.1 format: list of dicts with target_idx, predictor_idx, result
            for item in univariate_data:
                key = (int(item['target_idx']), int(item['predictor_idx']))
                instance.univariate_results[key] = instance._dict_to_result(item['result'])
        else:
            # v1.0 format: dict with "idx_idx" keys (legacy, may break with underscores)
            for k, v in univariate_data.items():
                parts = k.split('_')
                key = (int(parts[0]), int(parts[1]))
                instance.univariate_results[key] = instance._dict_to_result(v)
        
        # Restore prediction graph
        instance.prediction_graph = state.get('prediction_graph', {})

        return instance

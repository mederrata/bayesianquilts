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

        if sigma.ndim == 0:
            sigma = sigma[None]

        # Log likelihood
        residuals = y - mu if mu.ndim == 1 else y[None, :] - mu
        log_lik = -0.5 * jnp.log(2 * jnp.pi) - log_sigma[:, None] - 0.5 * (residuals / sigma[:, None])**2

        return log_lik

    def unormalized_log_prob(self, data: Dict[str, Any], **params) -> jnp.ndarray:
        """Compute unnormalized log probability (log joint)."""
        log_lik = self.log_likelihood(data, params)
        total_log_lik = jnp.sum(log_lik, axis=-1)

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
            logits = jnp.einsum('np,sp->sn', X, beta) + intercept[:, None]

        # Bernoulli log likelihood
        if logits.ndim == 1:
            log_lik = y * jax.nn.log_sigmoid(logits) + (1 - y) * jax.nn.log_sigmoid(-logits)
        else:
            log_lik = y[None, :] * jax.nn.log_sigmoid(logits) + (1 - y[None, :]) * jax.nn.log_sigmoid(-logits)

        return log_lik

    def unormalized_log_prob(self, data: Dict[str, Any], **params) -> jnp.ndarray:
        """Compute unnormalized log probability (log joint)."""
        log_lik = self.log_likelihood(data, params)
        total_log_lik = jnp.sum(log_lik, axis=-1)

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
        self.var_list = ['beta', 'cutpoints']

    def create_prior(self):
        """Create prior distribution."""
        # Beta: Normal(0, scale)
        # Cutpoints: Ordered transform of Normal(0, scale) or similar.
        # Simpler: Ordered(Normal(0, 5))
        
        return tfd.JointDistributionNamed({
            'beta': tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(self.n_predictors, dtype=self.dtype),
                    scale=self.prior_scale * jnp.ones(self.n_predictors, dtype=self.dtype)
                ),
                reinterpreted_batch_ndims=1
            ),
            # Cutpoints modeled as ordered vector
            # Using TransformedDistribution with Ordered bijector roughly
            # Or just use an increasingly sorted prior. 
            # A common way for VI/Pathfinder is to parameterize cutpoints as:
            # c1, c2-c1, c3-c2 ... where differences are log-normal or positive.
            # But TFP OrderedLogistic expects sorted cutpoints.
            # For simplicity in AD/VI, we can use a parameter 'raw_cutpoints' and sort them via Bijector?
            # Or just put a loose Normal prior and let the Likelihood constrain order? No, that fails.
            # Let's use TFP's Ordered bijector if possible.
            # Actually, `tfd.OrderedLogistic` expects cutpoints.
            # For the prior on cutpoints, a simple approach is:
            # c_k ~ Normal(0, 5) constrained to be ordered.
            # We can use `tfd.TransformedDistribution` with `tfb.Ordered()`.
            # However, `tfb.Ordered` might not be in the JAX substrate or `blackjax` pathfinder might struggle.
            # Alternative: Parameterize as `first_cutpoint` + cumsum(exp(gaps)).
            # Let's try explicit implementation using cumulative probabilities for likelihood + simple priors if easy.
            # Start with `tfd.Normal` for cutpoints but use `tfb.Ordered()` if available.
            
            # Assuming tfp.bijectors as tfb
            'cutpoints': tfd.TransformedDistribution(
                distribution=tfd.Normal(
                    loc=jnp.linspace(-2, 2, self.n_cutpoints, dtype=self.dtype), # Initialize spread out
                    scale=jnp.ones(self.n_cutpoints, dtype=self.dtype)
                ),
                bijector=tfp.bijectors.Ordered(),
                name='cutpoints'
            )
        })

    def log_likelihood(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        """Compute log-likelihood for each data point."""
        X = jnp.asarray(data['X'], dtype=self.dtype)
        y = jnp.asarray(data['y'], dtype=self.dtype) # y should be integer 0..K-1
        
        beta = params['beta']
        cutpoints = params['cutpoints']
        
        # Linear predictor
        # Handle batch dimensions
        # beta: (n_samples, n_predictors) or (n_predictors,)
        if beta.ndim == 1:
            eta = jnp.dot(X, beta) # (N,)
        else:
            eta = jnp.einsum('np,sp->sn', X, beta) # (n_samples, N)
        
        # TFP OrderedLogistic
        # cutpoints shape: (n_samples, n_cutpoints) or (n_cutpoints,)
        # loc shape: same as eta
        
        # If batching:
        # dist = tfd.OrderedLogistic(cutpoints=cutpoints, loc=eta)
        # But eta has (n_samples, N). cutpoints (n_samples, K-1).
        # We need broadcast.
        
        # Let's align dimensions.
        # y: (N,)
        
        if cutpoints.ndim == 1 and beta.ndim == 1:
            # No batching
            dist = tfd.OrderedLogistic(cutpoints=cutpoints, loc=eta)
            return dist.log_prob(y)
        
        # Batching
        if cutpoints.ndim == 1:
            cutpoints = cutpoints[None, :] # (1, K-1)
        
        # eta is (S, N)
        # cutpoints (S, K-1)
        # We want log_prob of y (N,) for each S.
        # tfd.OrderedLogistic expects broadcastable shapes.
        # If we pass loc=(S, N) and cutpoints=(S, K-1), it expects output (S, N).
        
        # We need to make sure cutpoints broadcast against N? 
        # OrderedLogistic: loc (...,) cutpoints (..., K-1).
        # It treats the last dim of cutpoints as categories.
        # The batch shape is ...
        # If loc is (S, N), cutpoints should be (S, 1, K-1)? Or (S, K-1) is ambiguous?
        # Actually OrderedLogistic `loc` should match batch shape.
        # If we want a distribution per observation per sample:
        # batch shape (S, N).
        # cutpoints should be (S, 1, K-1) to broadcast to (S, N, K-1)?
        # Let's try expanding cutpoints to (S, 1, K-1) and loc to (S, N).
        
        # Expand cutpoints to (S, 1, K-1)
        cutpoints_expanded = cutpoints[:, None, :] # (S, 1, K-1)
        
        dist = tfd.OrderedLogistic(cutpoints=cutpoints_expanded, loc=eta)
        
        # y is (N,)
        # we want to broadcast y to (S, N) implicitly?
        log_prob = dist.log_prob(y[None, :]) # (S, N)
        
        return log_prob

    def unormalized_log_prob(self, data: Dict[str, Any], **params) -> jnp.ndarray:
        """Compute unnormalized log probability (log joint)."""
        log_lik = self.log_likelihood(data, params)
        total_log_lik = jnp.sum(log_lik, axis=-1)

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
            verbose: Print progress information
        """
        super().__init__(n_imputations, max_iter, random_state, n_predictors=1)
        self.prior_scale = prior_scale
        self.noise_scale = noise_scale
        self.pathfinder_num_samples = pathfinder_num_samples
        self.pathfinder_maxiter = pathfinder_maxiter
        self.min_obs = min_obs
        self.verbose = verbose
        self.dtype = jnp.float32

        self.variable_names: List[str] = []
        self.variable_types: Dict[int, str] = {}
        self.zero_predictor_results: Dict[int, UnivariateModelResult] = {}
        self.univariate_results: Dict[Tuple[int, int], UnivariateModelResult] = {}
        self.n_obs_total: int = 0  # Overall dataset size

    def _infer_variable_type(self, values: np.ndarray) -> str:
        """Infer variable type from values."""
        unique_values = np.unique(values[~np.isnan(values)])
        if len(unique_values) == 2 and set(unique_values).issubset({0, 1, 0.0, 1.0}):
            return 'binary'
        # Heuristic for ordinal: integers, few unique values
        if len(unique_values) > 2 and len(unique_values) < 20:
            # Check if values are effectively integers
            if np.all(np.mod(unique_values, 1) == 0):
                return 'ordinal'
        return 'continuous'

    def _get_observed_mask(self, data: np.ndarray, var_idx: int) -> np.ndarray:
        """Get mask for observed values of a variable."""
        return ~np.isnan(data[:, var_idx])

    def _get_overlapping_mask(self, data: np.ndarray, idx1: int, idx2: int) -> np.ndarray:
        """Get mask for observations where both variables are observed."""
        mask1 = self._get_observed_mask(data, idx1)
        mask2 = self._get_observed_mask(data, idx2)
        return mask1 & mask2

    def _run_pathfinder(
        self,
        model,
        data: Dict[str, Any],
        seed: int = 42
    ) -> Tuple[Optional[Dict[str, jnp.ndarray]], float, bool]:
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

        # For zero-predictor model, use X of zeros
        X = np.zeros((n_obs, 1), dtype=np.float32)
        data_dict = {'X': X, 'y': y.astype(np.float32)}

        if var_type == 'binary':
            model = SimpleLogisticRegression(
                n_predictors=1,
                prior_scale=self.prior_scale,
                dtype=self.dtype
            )
        elif var_type == 'ordinal':
            # Determine number of classes
            # Assumes y values are appropriate for use as indices or can be mapped?
            # SimpleOrdinalLogisticRegression uses y directly for log_prob if y is integer 0..K-1.
            # We should probably ensure y is 0-indexed integers.
            # But here we just need K.
            # If y contains [0, 2], classes=3? Or just map to 0, 1?
            # Ordinal regression assumes ordered categories. Mapping 0,2 to 0,1 preserves order.
            # Let's map unique sorted values to 0..K-1.
            unique_vals = np.unique(y)
            n_classes = len(unique_vals)
            
            # Map y to 0..K-1
            # We need to do this mapping for training and consistently for prediction?
            # For LOO we just need likelihoods.
            val_map = {val: i for i, val in enumerate(sorted(unique_vals))}
            y_mapped = np.array([val_map[val] for val in y], dtype=np.float32) # Wait, TFP expects float-like integer? Or int?
            # TFP OrderedLogistic log_prob(val) expects val to be compatible with cutpoints dtype (float) but represent rank.
            # Actually TFP docs say `value` should be broadcastable with cutpoints. 
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

        # Run Pathfinder
        params, elbo, converged = self._run_pathfinder(model, data_dict, seed=seed)

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
            params=params
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
            
        # Prepare X based on predictor type
        if predictor_var_type == 'ordinal':
            # Use global max for consistency if possible, or local max
            # For robustness, let's use the max of the observed predictor values in this subset
            # (using global might create all-zero columns which is fine for Bayes but maybe wasteful)
            # Actually, using local max is safer to avoid empty columns issues in Pathfinder optimization
            # unless we accept prior dominance.
            # But the user might want comparable coefficients.
            # Let's use local max_val for now to ensure numerical stability.
            x_vals = X_raw.flatten()
            max_val = int(np.max(x_vals))
            
            # Use ordinal one-hot encoding
            # ordinal_one_hot_encode expects (N, M) matrix of integers
            # We pass X_raw cast to int.
            # Note: ordinal_one_hot_encode imports from .mice
            X = ordinal_one_hot_encode(X_raw.astype(int), max_val)
            X = X.astype(np.float32)
            
            # Don't standardize one-hot encoded variables
            X_mean = 0.0
            X_std = 1.0
            
            n_predictors = X.shape[1]
            
        else:
            # Continuous/Standard handling
            X = X_raw
            X_mean = float(np.mean(X))
            X_std = float(np.std(X))
            if X_std > 1e-6:
                X = (X - X_mean) / X_std
            else:
                X_std = 1.0  # Avoid division by zero
            n_predictors = 1

        data_dict = {'X': X, 'y': y}

        # Create model
        if target_var_type == 'binary':
            model = SimpleLogisticRegression(
                n_predictors=n_predictors,
                prior_scale=self.prior_scale,
                dtype=self.dtype
            )
        elif target_var_type == 'ordinal':
             # Map y to 0..K-1 for ordinal regression
            unique_vals = np.unique(y)
            n_classes = len(unique_vals)
            val_map = {val: i for i, val in enumerate(sorted(unique_vals))}
            y_mapped = np.array([val_map[val] for val in y], dtype=np.float32)
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

        # Run Pathfinder
        params, elbo, converged = self._run_pathfinder(model, data_dict, seed=seed)

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
            params=params,
            predictor_mean=X_mean,
            predictor_std=X_std
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

        # For zero-predictor model, use X of zeros (intercept-only)
        X = np.zeros((n_obs, 1), dtype=np.float32)
        data_dict = {'X': X, 'y': y.astype(np.float32)}

        if var_type == 'binary':
            model = SimpleLogisticRegression(
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

        # Run Pathfinder
        params, elbo, converged = self._run_pathfinder(model, data_dict, seed=seed)

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
            params=params
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
        X = data[mask, predictor_idx:predictor_idx+1].astype(np.float32)
        y = data[mask, target_idx].astype(np.float32)

        # Standardize predictor and save stats for prediction
        X_mean = float(np.mean(X))
        X_std = float(np.std(X))
        if X_std > 1e-6:
            X = (X - X_mean) / X_std
        else:
            X_std = 1.0  # Avoid division by zero

        data_dict = {'X': X, 'y': y}

        # Determine variable type
        var_type = self.variable_types.get(target_idx)
        if var_type is None:
            var_type = self._infer_variable_type(y)
            self.variable_types[target_idx] = var_type

        # Create model
        if var_type == 'binary':
            model = SimpleLogisticRegression(
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

        # Run Pathfinder
        params, elbo, converged = self._run_pathfinder(model, data_dict, seed=seed)

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
            params=params,
            predictor_mean=X_mean,
            predictor_std=X_std
        )

    def fit_loo_models(
        self,
        X_df: pd.DataFrame,
        fit_zero_predictors: bool = True,
        seed: int = 42,
        save_dir: Optional[Union[str, Path]] = None
    ) -> 'MICEBayesianLOO':
        """
        Fit all univariate models for LOO-CV evaluation.

        Args:
            X_df: DataFrame with potentially missing values (NaN)
            fit_zero_predictors: Whether to fit zero-predictor (intercept-only) models
            seed: Random seed
            save_dir: Directory to save incremental results. If None, keeps all in memory.

        Returns:
            self
        """
        try:
            from tqdm import tqdm
        except ImportError:
            # Fallback if tqdm not installed
            def tqdm(iterable, **kwargs): return iterable

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                print(f"Saving incremental results to {save_dir}")

        data = X_df.values
        self.variable_names = list(X_df.columns)
        n_variables = data.shape[1]
        self.n_obs_total = data.shape[0]

        if self.verbose:
            print(f"Fitting MICE Bayesian LOO-CV models with Pathfinder")
            print(f"  Variables: {n_variables}")
            print(f"  Observations: {self.n_obs_total}")
            print(f"  Min obs per model: {self.min_obs}")

        # Fit zero-predictor models
        if fit_zero_predictors:
            if self.verbose:
                print("\nFitting zero-predictor models...")

            iterator = range(n_variables)
            if not self.verbose:
                iterator = tqdm(iterator, desc="Zero-Predictor Models")

            for i in iterator:
                var_name = self.variable_names[i]
                if self.verbose:
                    print(f"  [{i+1}/{n_variables}] {var_name}")

                result = self._fit_zero_predictor(data, i, seed=seed + i)
                self.zero_predictor_results[i] = result

                if self.verbose:
                    if result.converged:
                        print(f"    n_obs={result.n_obs}, elpd/n={result.elpd_loo_per_obs:.4f} "
                              f"(SE={result.elpd_loo_per_obs_se:.4f}), khat_max={result.khat_max:.3f}")
                    else:
                        print(f"    n_obs={result.n_obs}, FAILED/SKIPPED")

        # Fit one-predictor models
        if self.verbose:
            print("\nFitting one-predictor models...")

        model_count = 0
        total_models = n_variables * (n_variables - 1)
        
        # We need to restructure the loop to save per target variable if save_dir is used
        # Iterating by target variable i
        
        iterator = range(n_variables)
        if not self.verbose:
            iterator = tqdm(iterator, desc="Target Variables (One-Predictor)")

        for i in iterator:
            target_name = self.variable_names[i]
            
            # Dictionary to collect results for this target
            target_results = {}
            if fit_zero_predictors:
                target_results['zero_predictor'] = self._result_to_dict(self.zero_predictor_results[i])
            
            univariate_list = []

            for j in range(n_variables):
                if i == j:
                    continue

                predictor_name = self.variable_names[j]
                model_count += 1

                if self.verbose:
                    print(f"  [{model_count}/{total_models}] {target_name} ~ {predictor_name}")

                result = self._fit_univariate(
                    data, i, j,
                    seed=seed + n_variables + model_count
                )

                self.univariate_results[(i, j)] = result
                
                # Add to local list for saving
                uni_res_dict = self._result_to_dict(result)
                uni_res_dict['predictor_name'] = predictor_name
                uni_res_dict['predictor_idx'] = j
                univariate_list.append(uni_res_dict)

                if self.verbose:
                    if result.converged:
                        print(f"    n_obs={result.n_obs}, elpd/n={result.elpd_loo_per_obs:.4f} "
                              f"(SE={result.elpd_loo_per_obs_se:.4f}), khat_max={result.khat_max:.3f}")
                    else:
                        print(f"    n_obs={result.n_obs}, FAILED/SKIPPED")
            
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
                with open(file_path, 'w') as f:
                    yaml.dump(target_results, f, default_flow_style=False, allow_unicode=True)
                
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
            if zero_result.converged and zero_result.params is not None:
                # Compute prediction from zero-predictor model
                # For zero-predictor, X is zeros, so prediction is just intercept
                params = zero_result.params
                intercept = np.mean(params['intercept'])

                if var_type == 'binary':
                    # Logistic: sigmoid(intercept)
                    pred = 1.0 / (1.0 + np.exp(-intercept))
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
            if not uni_result.converged or uni_result.params is None:
                continue

            # Standardize the predictor value
            X_mean = uni_result.predictor_mean if uni_result.predictor_mean is not None else 0.0
            X_std = uni_result.predictor_std if uni_result.predictor_std is not None else 1.0
            x_standardized = (predictor_value - X_mean) / X_std

            # Compute prediction
            params = uni_result.params
            beta = np.mean(params['beta'], axis=0)  # Average over posterior samples
            intercept = np.mean(params['intercept'])

            linear_pred = float(x_standardized * beta[0] + intercept)

            if var_type == 'binary':
                pred = 1.0 / (1.0 + np.exp(-linear_pred))
            else:
                pred = linear_pred

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
        }

        with open(path, 'w') as f:
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

        return instance

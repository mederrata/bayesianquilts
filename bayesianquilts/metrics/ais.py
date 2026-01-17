#!/usr/bin/env python3
"""
Generalized Adaptive Importance Sampling (AIS) Framework

This module provides a flexible framework for adaptive importance sampling
with leave-one-out cross-validation. It generalizes the approach from the
logistic regression specific implementation to work with arbitrary likelihood
functions.

The framework implements several transformation strategies:
- Likelihood descent: T_ll
- KL divergence based: T_kl
- Variance based: T_var
- Identity: T_I

Each transformation can be applied to perform importance sampling for
leave-one-out cross-validation predictions.
"""

import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Tuple, Optional, List
from bayesianquilts.metrics import nppsis


class LikelihoodFunction(ABC):
    """Abstract base class for likelihood functions used in AIS."""

    @abstractmethod
    def log_likelihood(self, data: Any, params: Dict[str, Any]) -> jnp.ndarray:
        """Compute log-likelihood for each data point.

        Args:
            data: Input data
            params: Model parameters

        Returns:
            Array of shape (n_samples, n_data) with log-likelihood values
        """
        pass

    @abstractmethod
    def log_likelihood_gradient(self, data: Any, params: Dict[str, Any]) -> jnp.ndarray:
        """Compute gradient of log-likelihood w.r.t. parameters.

        Args:
            data: Input data
            params: Model parameters

        Returns:
            Gradient array of shape (n_samples, n_data, n_params)
        """
        pass

    @abstractmethod
    def log_likelihood_hessian_diag(self, data: Any, params: Dict[str, Any]) -> jnp.ndarray:
        """Compute diagonal of Hessian of log-likelihood w.r.t. parameters.

        Args:
            data: Input data
            params: Model parameters

        Returns:
            Hessian diagonal of shape (n_samples, n_data, n_params)
        """
        pass

    @abstractmethod
    def extract_parameters(self, params: Dict[str, Any]) -> jnp.ndarray:
        """Extract and flatten parameters into a single array.

        Args:
            params: Parameter dictionary

        Returns:
            Flattened parameter array of shape (n_samples, n_params)
        """
        pass

    @abstractmethod
    def reconstruct_parameters(self, flat_params: jnp.ndarray, template: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct parameter dictionary from flattened array.

        Args:
            flat_params: Flattened parameters of shape (..., n_params)
            template: Template parameter dictionary for structure

        Returns:
            Reconstructed parameter dictionary
        """
        pass


class AutoDiffLikelihoodMixin(LikelihoodFunction):
    """Mixin to provide automatic differentiation for LikelihoodFunction.
    
    This mixin implements log_likelihood_gradient and log_likelihood_hessian_diag
    using JAX automatic differentiation.
    """
    
    def log_likelihood_gradient(self, data: Any, params: Dict[str, Any]) -> jnp.ndarray:
        """Compute gradient of log-likelihood w.r.t. parameters using autodiff."""
        # We need a function that maps flattened params -> scalar log likelihood per data point
        
        # 1. Extract params
        flat_params = self.extract_parameters(params) # (S, K)
        n_samples, n_features = flat_params.shape
        
        # 2. Reconstruct parameters template
        # We need to know the structure to reconstruct inside the transform
        
        def single_point_ll(theta_flat, x_i, y_i):
            # theta_flat: (K,)
            # x_i: (F,) or whatever single data point structure
            # y_i: scalar or single label
            
            # Reconstruct params for single sample
            # We need to reshape theta_flat to be (1, K) for reconstruct_parameters
            # because reconstruct_parameters expects batch dim if params have batch dim
            # But here we just want to reconstruct a single set of params.
            
            # Let's assume reconstruct_parameters handles unbatched input or we add a batch dim
            params_reconstructed = self.reconstruct_parameters(theta_flat[jnp.newaxis, :], params)
            
            # Remove the batch dim we added, because we want single sample params
            params_single = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), params_reconstructed)
            
            # Data wrapper for single point
            data_single = {'X': x_i[jnp.newaxis, ...], 'y': y_i[jnp.newaxis, ...]}
            
            # Helper to wrap reconstruct_parameters properly is tricky because 
            # log_likelihood method expects batched params usually.
            
            # Alternative approach:
            # Vmap over samples (S) and data points (N)
            pass 

        # Let's use a cleaner approach:
        # Define a function f(theta_flat) -> log_likelihood(theta_flat, data)
        # But log_likelihood returns (N,). We need gradients per data point.
        # So we need Jacobian of F(theta) w.r.t theta, where output is N-dim.
        # Jac will be (N, K).
        
        X = data['X']
        y = data['y']
        n_data = X.shape[0]
        
        def batch_log_likelihood(theta_s):
            # theta_s: (K,)
            
            # Reconstruct params
            # We assume reconstruct_parameters preserves leading dims
            p = self.reconstruct_parameters(theta_s, params) 
            
            # Add singleton batch dimension to simulate batch size of 1
            # Params will go from (F,) -> (1, F)
            p_batched = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], p)
            
            ll = self.log_likelihood(data, p_batched)
            # If ll result has shape (1, N) squeeze it
            return jnp.squeeze(ll)

        # Grad of sum is sum of grads. We want per-point gradients.
        # Jacobian of mapping theta -> [ll_1, ll_2, ... ll_N]
        # Jacobian shape: (N, K)
        
        # We need to vmap this Jacobian computation over samples S
        
        jac_fn = jax.jacrev(batch_log_likelihood)
        
        # Vmap over samples
        # flat_params: (S, K)
        # Output: (S, N, K)
        gradients = jax.vmap(jac_fn)(flat_params)
        
        return gradients

    def log_likelihood_hessian_diag(self, data: Any, params: Dict[str, Any]) -> jnp.ndarray:
        """Compute diagonal of Hessian of log-likelihood w.r.t. parameters using autodiff."""
        
        flat_params = self.extract_parameters(params) # (S, K)
        
        def batch_log_likelihood(theta_s):
            p = self.reconstruct_parameters(theta_s, params)
            ll = self.log_likelihood(data, p)
            return jnp.squeeze(ll) # (N,)

        # Hessian is (N, K, K). We need diagonal (N, K).
        # Computing full Hessian is expensive (K*K).
        # We can computer diagonal directly if we treat it elementwise? No.
        
        # We want [d^2 L_i / d theta_j^2] for each i, j.
        # This is the diagonal of the Hessian of L_i w.r.t theta.
        
        def hessian_diag_fn(theta_s):
            # Returns (N, K)
            # define scalar function L_i(theta) -> Hessian_i(theta)
            
            # To define efficient diagonal hessian:
            # h(theta) = diagonal(Hessian(scalar_func))
            pass
            
            # Just use full Hessian and take diag for now for simplicity/correctness, 
            # unless K is large. K includes ALL features.
            # For Bouldering data, K might be small enough?
            # Or use forward-over-reverse.
            
            # Map over data points?
            # Let's compute Jacobian of Gradient.
            
            def grad_fn(t): 
                return jax.jacrev(batch_log_likelihood)(t) # (N, K)
                
            # We want diagonal of Jacobian of Gradient.
            # grad_fn returns (N, K). 
            # We want d(grad_fn_ij)/d(theta_j).
            
            # Let's iterate over N?
            # Or use a scan?
            pass
        
        # Fallback to full Hessian per point if needed, but slow.
        # Better:
        
        def per_point_hessian_diag(theta_s):
            # theta_s: (K,)
            
            # We effectively want to vmap the hessian diag over data points N?
            # But the likelihood function vectorizes over N internally.
            
            # Let's construct a function that returns the vector of LLs (N,)
            def val_fn(theta_s):
                p = self.reconstruct_parameters(theta_s, params)
                p_batched = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], p)
                ll = self.log_likelihood(data, p_batched)
                return jnp.squeeze(ll)

            
            # We want diag(H_i) for each i.
            # H_i is Hessian of i-th output w.r.t input.
            
            hessian_fn = jax.hessian(val_fn) # Returns (N, K, K)
            
            H = hessian_fn(theta_s)
            return jax.vmap(jnp.diag)(H) # (N, K)

        # Vmap over samples S
        hess_diags = jax.vmap(per_point_hessian_diag)(flat_params)
        
        return hess_diags

class AdaptiveImportanceSampler:
    """Generalized Adaptive Importance Sampling for Leave-One-Out Cross-Validation.

    This class implements adaptive importance sampling transformations that can work
    with any likelihood function. It provides several transformation strategies
    for generating importance weights.
    """

    def __init__(self,
                 likelihood_fn: LikelihoodFunction,
                 prior_log_prob_fn: Optional[Callable] = None,
                 surrogate_log_prob_fn: Optional[Callable] = None):
        """Initialize the adaptive importance sampler.

        Args:
            likelihood_fn: Likelihood function implementing LikelihoodFunction interface
            prior_log_prob_fn: Function computing log prior probability
            surrogate_log_prob_fn: Function computing log surrogate probability (for variational)
        """
        self.likelihood_fn = likelihood_fn
        self.prior_log_prob_fn = prior_log_prob_fn
        self.surrogate_log_prob_fn = surrogate_log_prob_fn

    def entropy(self, probs: jnp.ndarray) -> jnp.ndarray:
        """Compute entropy of probability distributions."""
        return -jnp.sum(probs * jnp.log(probs + 1e-10), axis=0)

    def adaptive_is_loo(self,
                        data: Dict[str, Any],
                        params: Dict[str, Any],
                        hbar: float = 1.0,
                        rhos: List[float] = None,
                        variational: bool = True,
                        transformations: List[str] = ['identity', 'll', 'kl', 'var']) -> Dict[str, Any]:
        """Perform adaptive IS for LOO-CV.
        
        Args:
            data: Dictionary of data (X, y)
            params: Dictionary of parameters (shape: S x ...)
            hbar: Step size for transformations (legacy single value)
            rhos: List of step sizes to sweep over. If provided, overrides hbar.
            variational: Whether parameters come from a VI approximation
            transformations: List of transformation names to try
            
        Returns:
            Dictionary of results for each transformation (+ 'best' if sweeping)
        """
        
        if rhos is None:
            rhos = [hbar]
            
        theta = self.likelihood_fn.extract_parameters(params)
        theta_std = jnp.std(theta, axis=0) # (K,)
        # Ensure std is not zero
        theta_std = jnp.where(theta_std < 1e-6, 1.0, theta_std)

        log_ell = self.likelihood_fn.log_likelihood(data, params) # (S, N)
        if variational and self.surrogate_log_prob_fn is not None:
            log_pi = self.surrogate_log_prob_fn(params) # (S,)
            # Reshape for broadcasting
            log_pi = log_pi[:, jnp.newaxis] # (S, 1)
        else:
            # If MCMC samples, log_pi is effectively constant/uniform if we assume samples 
            # are from the posterior. Or we can compute log_prob.
            # However, standard AIS usually starts with weights=1 for MCMC samples.
            # But the logic below expects log_pi. 
            # Let's set it to log_ell + log_prior (posterior)
            # Or just zeros if we treat MCMC samples as having proposal = target
            # For now, let's calculate the log joint prob as log_pi
            # This assumes a 'model' attribute exists, which is not defined in __init__.
            # For faithfulness to the instruction, I'm including it, but it might require
            # self.model to be passed in __init__ or set elsewhere.
            # If self.model is not available, this will cause an AttributeError.
            # A safer fallback for non-variational might be:
            # log_prior = self.prior_log_prob_fn(params) if self.prior_log_prob_fn else jnp.zeros(theta.shape[0])
            # log_pi = jnp.sum(log_ell, axis=1) + log_prior
            # log_pi = log_pi[:, jnp.newaxis]
            # For now, following the instruction's provided code:
            if hasattr(self, 'model') and hasattr(self.model, 'unormalized_log_prob'):
                log_pi = self.model.unormalized_log_prob(data, **params)
                log_pi = log_pi[:, jnp.newaxis]
            else:
                # Fallback if self.model is not available, using existing prior_log_prob_fn
                log_prior = self.prior_log_prob_fn(params) if self.prior_log_prob_fn else jnp.zeros(theta.shape[0])
                log_pi = jnp.sum(log_ell, axis=1) + log_prior
                log_pi = log_pi[:, jnp.newaxis]


        # Pre-compute gradients if needed (only once)
        log_ell_prime = self.likelihood_fn.log_likelihood_gradient(data, params) # (S, N, K)
        
        # Hessian diagonal for var-based transformation
        log_ell_doubleprime = None
        if 'var' in transformations:
            log_ell_doubleprime = self.likelihood_fn.log_likelihood_hessian_diag(data, params) # (S, N, K)
            
        grad_log_pi = None
        # grad_log_pi isn't strictly needed for current transformations but good for extensibility
        
        results = {}
        
        # Storage for best-khat logic
        # best_khat: (N,) initialized to inf
        # best_results: dictionaries per N? That's messy.
        # Structure: results['best'] = { 'khat': (N,), 'p_loo_eta': (N,), ... }
        
        n_data = log_ell.shape[1]
        best_khat = jnp.inf * jnp.ones(n_data)
        best_metrics = {
             'p_loo_eta': jnp.zeros(n_data),
             'p_loo_psis': jnp.zeros(n_data),
             'khat': jnp.inf * jnp.ones(n_data)
        }
        
        # Helper to map legacy names to methods
        method_map = {
            'identity': self._transform_identity,
            'll': self._transform_likelihood_descent,
            'kl': self._transform_kl_divergence,
            'var': self._transform_variance_based,
            'mm1': self._transform_mm1,
            'mm2': self._transform_mm2,
            'pmm1': self._transform_pmm1,
            'pmm2': self._transform_pmm2
        }

        for method_name in transformations:
            if method_name not in method_map:
                print(f"Warning: Unknown transformation {method_name}")
                continue
                
            transform_fn = method_map[method_name]
            
            # Determine which rhos to loop over for this method
            # Identity, MM1, MM2 don't use hbar/rho
            current_rhos = rhos
            if method_name in ['identity', 'mm1', 'mm2']:
                current_rhos = [1.0] # Dummy loop
                
            for rho in current_rhos:
                key = f"{method_name}"
                if len(rhos) > 1 and method_name not in ['identity', 'mm1', 'mm2']:
                     key = f"{method_name}_rho{rho}"

                # Call transformation
                # Note: kwargs are passed. PMM1/PMM2 use 'hbar'. LL/KL/Var use 'hbar'.
                res = transform_fn(
                    data=data,
                    params=params,
                    theta=theta,
                    log_ell=log_ell,
                    log_ell_prime=log_ell_prime,
                    log_ell_doubleprime=log_ell_doubleprime,
                    theta_std=theta_std,
                    hbar=rho,
                    variational=variational,
                    log_pi=log_pi,
                    grad_log_pi=grad_log_pi,
                    log_ell_original=log_ell
                )
                
                # Check for NaNs in khat
                khat_safe = jnp.where(jnp.isnan(res['khat']), jnp.inf, res['khat'])
                res['khat'] = khat_safe
                
                results[key] = res
                
                # Update best metrics
                # Find indices where current method is better
                improved_idx = khat_safe < best_khat
                best_khat = jnp.where(improved_idx, khat_safe, best_khat)
                
                best_metrics['khat'] = best_khat
                best_metrics['p_loo_eta'] = jnp.where(improved_idx, res['p_loo_eta'], best_metrics['p_loo_eta'])
                best_metrics['p_loo_psis'] = jnp.where(improved_idx, res['p_loo_psis'], best_metrics['p_loo_psis'])
                
        # Add 'best' entry if we swept
        if len(results) > 0:
            results['best'] = best_metrics
            
        return results

    def _compute_importance_weights(self,
                                  data: Any,
                                  params_original: Dict[str, Any],
                                  params_transformed: Dict[str, Any],
                                  log_jacobian: jnp.ndarray,
                                  variational: bool,
                                  log_pi_original: jnp.ndarray,
                                  log_ell_original: jnp.ndarray = None) -> Tuple[jnp.ndarray, ...]:
        """Compute importance weights and related quantities.
        
        For LOO importance sampling, we compute:
            w[s, i] ~ p_{-i}(theta_prime[s,i]) / p_{-i}(theta[s])
        
        Where p_{-i} is the LOO posterior (posterior without data point i).
        
        The LOO posterior satisfies: log p_{-i}(θ) = log p(θ) - log p(y_i | θ)
        
        Args:
            data: Input data
            params_original: Original (untransformed) parameters
            params_transformed: Transformed parameters (shape includes data dimension)
            log_jacobian: Log Jacobian of the transformation
            variational: Whether using variational approximation
            log_pi_original: Log posterior at original params, shape (n_samples,)
            log_ell_original: Original log likelihoods, shape (n_samples, n_data)
        """

        # Compute likelihood for transformed parameters
        log_ell_new = self.likelihood_fn.log_likelihood(data, params_transformed)

        if variational and self.surrogate_log_prob_fn is not None:
            # Trust variational approximation
            log_pi_trans = self.surrogate_log_prob_fn(params_transformed)
            delta_log_pi = (log_pi_trans -
                          log_pi_original[:, jnp.newaxis])
            delta_log_pi = delta_log_pi - jnp.max(delta_log_pi, axis=0, keepdims=True)
        else:
            # Compute LOO posterior difference properly
            # 
            # For LOO at data point i:
            #   log p_{-i}(θ') = log p(θ') - log p(y_i | θ')
            #   log p_{-i}(θ)  = log p(θ)  - log p(y_i | θ)
            # 
            # So: delta_log_pi[s, i] = [log p(θ'[s,i]) - log ell_new[s,i]] - [log p(θ[s]) - log ell[s,i]]
            #                        = log p(θ'[s,i]) - log p(θ[s]) - log ell_new[s,i] + log ell[s,i]
            #
            # Since we're doing leave-one-out, we assume log p(θ') ≈ log p(θ) for small transformations
            # (the transformation is designed to be a small perturbation along the gradient)
            # 
            # Simplified: delta_log_pi ≈ log_ell_original - log_ell_new
            
            if log_ell_original is not None:
                # Use the LOO formula: difference in log likelihoods at the left-out point
                delta_log_pi = log_ell_original - log_ell_new
            else:
                # Fallback: assume no posterior change (will give poor results)
                delta_log_pi = jnp.zeros_like(log_ell_new)

        # Compute importance weights
        # w ∝ exp(delta_log_pi + log_jacobian)
        # For LOO prediction of y_i, we don't want log_ell_new in the weights since
        # that's what we're trying to estimate
        log_eta_weights = delta_log_pi + log_jacobian
        log_eta_weights = log_eta_weights - jnp.max(log_eta_weights, axis=0, keepdims=True)

        # Apply PSIS
        psis_weights, khat = nppsis.psislw(log_eta_weights)

        # Normalize weights
        eta_weights = jnp.exp(log_eta_weights)
        eta_weights = eta_weights / jnp.sum(eta_weights, axis=0, keepdims=True)

        psis_weights = jnp.exp(psis_weights) 
        psis_weights = psis_weights / jnp.sum(psis_weights, axis=0, keepdims=True)

        return eta_weights, psis_weights, khat, log_ell_new


    def _transform_likelihood_descent(self,
                                    data: Any,
                                    params: Dict[str, Any],
                                    theta: jnp.ndarray,
                                    log_ell: jnp.ndarray,
                                    log_ell_prime: jnp.ndarray,
                                    log_ell_doubleprime: jnp.ndarray,
                                    theta_std: jnp.ndarray,
                                    hbar: float,
                                    variational: bool,
                                    log_pi: jnp.ndarray,
                                    grad_log_pi: jnp.ndarray,
                                    log_ell_original: jnp.ndarray = None,
                                    **kwargs) -> Dict[str, Any]:
        """Likelihood descent transformation T_ll."""

        # Compute direction: negative log-likelihood gradient
        Q = -log_ell_prime  # Shape: (n_samples, n_data, n_params)

        # Standardize the direction
        # Standardize the direction
        pad_dims = Q.ndim - 2 - theta_std.ndim
        theta_std_expanded = theta_std[jnp.newaxis, jnp.newaxis, ...]
        
        Q_standardized = Q / theta_std_expanded
        
        reduction_axes = tuple(range(2, Q_standardized.ndim))
        Q_norm = jnp.max(jnp.abs(Q_standardized), axis=reduction_axes, keepdims=True)
        Q_norm = jnp.max(Q_norm, axis=0, keepdims=True)  # (1, n_data, 1, 1...)

        # Compute step size
        h = hbar / Q_norm

        # Apply transformation
        theta_expanded = theta[:, jnp.newaxis, :]  # (n_samples, 1, n_params)
        hQ = h * Q  # (n_samples, n_data, n_params)
        theta_new = theta_expanded + hQ  # (n_samples, n_data, n_params)

        # Compute Jacobian approximation
        # log|J| ~ log|1 + h * Tr(H)|
        # Q = -grad(log_ell)
        # T(theta) = theta - h * grad(log_ell)
        # J = I - h * Hessian(log_ell)
        # Since Q is negative gradient, we have:
        # theta_new = theta + h * Q = theta - h * grad(log_ell)
        # So J = I - h * H
        # trace(J) approx is 1 - h * trace(H) ? No, determinant is product of eigenvalues.
        # det(I + A) ~ 1 + trace(A) for small A.
        # det(I - hH) ~ 1 - h * trace(H)
        
        # We need trace(H).
        # We can get diagonal of H.
        
        hess_diag = self.likelihood_fn.log_likelihood_hessian_diag(data, params) # (S, N, K)
        trace_H = jnp.sum(hess_diag, axis=-1) # (S, N)
        
        # The update rule in code is: theta_new = theta + h * Q
        # Q = -log_ell_prime (negative gradient)
        # So theta_new = theta - h * grad(log_ell)
        # Jacobian matrix J_matrix = d(theta_new)/d(theta) = I - h * Hessian(log_ell)
        
        # det(J_matrix) approx 1 - h * trace(H)
        # log det(J_matrix) approx log(abs(1 - h * trace(H)))
        
        log_jacobian = jnp.log(jnp.abs(1.0 - h * trace_H)) # (S, N) with broadcasting if h is (1, N, 1)
        
        if log_jacobian.ndim > 2:
             log_jacobian = jnp.squeeze(log_jacobian)

        # Reconstruct parameters
        params_new = {}
        for i in range(log_ell.shape[1]):
            params_new_i = self.likelihood_fn.reconstruct_parameters(theta_new[:, i, :], params)
            if i == 0:
                for key in params_new_i.keys():
                    params_new[key] = params_new_i[key][:, jnp.newaxis, ...]
            else:
                for key in params_new_i.keys():
                    params_new[key] = jnp.concatenate([
                        params_new[key],
                        params_new_i[key][:, jnp.newaxis, ...]
                    ], axis=1)

        # Compute importance weights
        eta_weights, psis_weights, khat, log_ell_new = self._compute_importance_weights(
            data, params, params_new, log_jacobian, variational, log_pi, log_ell_original=log_ell
        )

        # Compute predictions and diagnostics
        predictions = self.likelihood_fn.log_likelihood(data, params_new)

        weight_entropy = self.entropy(eta_weights)
        psis_entropy = self.entropy(psis_weights)

        p_loo_eta = jnp.sum(jnp.exp(predictions) * eta_weights, axis=0)
        p_loo_psis = jnp.sum(jnp.exp(predictions) * psis_weights, axis=0)

        ll_loo_eta = jnp.sum(eta_weights * jnp.exp(log_ell_new), axis=0)
        ll_loo_psis = jnp.sum(psis_weights * jnp.exp(log_ell_new), axis=0)

        return {
            'eta_weights': eta_weights,
            'psis_weights': psis_weights,
            'p_loo_eta': p_loo_eta,
            'p_loo_psis': p_loo_psis,
            'll_loo_eta': ll_loo_eta,
            'll_loo_psis': ll_loo_psis,
            'weight_entropy': weight_entropy,
            'psis_entropy': psis_entropy,
            'khat': khat,
            'predictions': predictions
        }

    def _transform_kl_divergence(self,
                               data: Any,
                               params: Dict[str, Any],
                               theta: jnp.ndarray,
                               log_ell: jnp.ndarray,
                               log_ell_prime: jnp.ndarray,
                               log_ell_doubleprime: jnp.ndarray,
                               theta_std: jnp.ndarray,
                               hbar: float,
                               variational: bool,
                               log_pi: jnp.ndarray,
                               grad_log_pi: jnp.ndarray,
                               log_ell_original: jnp.ndarray = None,
                               **kwargs) -> Dict[str, Any]:
        """KL divergence based transformation T_kl."""
        
        if variational and self.surrogate_log_prob_fn is not None:
             raise NotImplementedError("Variational AIS-KL not implemented yet")

        # This is a complex transformation
        log_pi_normalized = log_pi - jnp.max(log_pi, axis=0, keepdims=True)
        # log_pi (S,) or (S, 1) -> weights (S, 1)
        weights = jnp.exp(log_pi_normalized).reshape(log_pi.shape[0], 1)
        
        # Expand weights to match log_ell_prime rank (S, N, P...)
        # We want weights to be (S, 1, 1...)
        while weights.ndim < log_ell_prime.ndim:
            weights = weights[..., jnp.newaxis]
            
        Q = -log_ell_prime * weights
        
        # Standardize using theta_std
        # theta_std (P...) -> (1, 1, P...)
        pad_dims = Q.ndim - 2 - theta_std.ndim
        # If theta_std matches P dimensions exactly, pad_dims=0. 
        # But we need (1, 1) prefix.
        theta_std_expanded = theta_std[jnp.newaxis, jnp.newaxis, ...]
        
        Q_standardized = Q / theta_std_expanded
        
        # Reduce over all parameter dimensions (axes 2+) to get scalar h per (S, N)
        reduction_axes = tuple(range(2, Q_standardized.ndim))
        Q_norm = jnp.max(jnp.abs(Q_standardized), axis=reduction_axes, keepdims=True)
        # Reduce over samples (axis 0) ?? Logic says 'axis=0' in original code.
        # Original: Q_norm = jnp.max(Q_norm, axis=0, keepdims=True)
        # We'll keep that generic normalization across samples.
        Q_norm = jnp.max(Q_norm, axis=0, keepdims=True)

        h = hbar / Q_norm
        theta_new = theta[:, jnp.newaxis, :] + h * Q

        log_jacobian = jnp.zeros((theta.shape[0], log_ell.shape[1]))

        # Reconstruct parameters (similar to likelihood descent)
        params_new = {}
        for i in range(log_ell.shape[1]):
            params_new_i = self.likelihood_fn.reconstruct_parameters(theta_new[:, i, :], params)
            if i == 0:
                for key in params_new_i.keys():
                    params_new[key] = params_new_i[key][:, jnp.newaxis, ...]
            else:
                for key in params_new_i.keys():
                    params_new[key] = jnp.concatenate([
                        params_new[key],
                        params_new_i[key][:, jnp.newaxis, ...]
                    ], axis=1)
        


        # Compute importance weights and return results
        eta_weights, psis_weights, khat, log_ell_new = self._compute_importance_weights(
            data, params, params_new, log_jacobian, variational, log_pi, log_ell_original=log_ell
        )

        predictions = self.likelihood_fn.log_likelihood(data, params_new)
        weight_entropy = self.entropy(eta_weights)
        psis_entropy = self.entropy(psis_weights)

        p_loo_eta = jnp.sum(jnp.exp(predictions) * eta_weights, axis=0)
        p_loo_psis = jnp.sum(jnp.exp(predictions) * psis_weights, axis=0)
        ll_loo_eta = jnp.sum(eta_weights * jnp.exp(log_ell_new), axis=0)
        ll_loo_psis = jnp.sum(psis_weights * jnp.exp(log_ell_new), axis=0)

        return {
            'eta_weights': eta_weights,
            'psis_weights': psis_weights,
            'p_loo_eta': p_loo_eta,
            'p_loo_psis': p_loo_psis,
            'll_loo_eta': ll_loo_eta,
            'll_loo_psis': ll_loo_psis,
            'weight_entropy': weight_entropy,
            'psis_entropy': psis_entropy,
            'khat': khat,
            'predictions': predictions
        }

    def _transform_variance_based(self,
                                data: Any,
                                params: Dict[str, Any],
                                theta: jnp.ndarray,
                                log_ell: jnp.ndarray,
                                log_ell_prime: jnp.ndarray,
                                log_ell_doubleprime: jnp.ndarray,
                                theta_std: jnp.ndarray,
                                hbar: float,
                                variational: bool,
                                log_pi: jnp.ndarray,
                                grad_log_pi: jnp.ndarray,
                                log_ell_original: jnp.ndarray = None,
                                **kwargs) -> Dict[str, Any]:
        """Variance-based transformation T_var."""

        # Use Hessian information for the transformation direction
        # This is simplified - full implementation would use second-order information
        log_pi_normalized = log_pi - jnp.max(log_pi, axis=0, keepdims=True)
        weights = jnp.exp(log_pi_normalized).reshape(log_pi.shape[0], 1)
        
        while weights.ndim < log_ell_prime.ndim:
            weights = weights[..., jnp.newaxis]

        # Direction based on curvature information
        Q = -log_ell_prime * weights * jnp.abs(log_ell_doubleprime)

        Q_standardized = Q / theta_std[jnp.newaxis, jnp.newaxis, ...]
        
        reduction_axes = tuple(range(2, Q_standardized.ndim))
        Q_norm = jnp.max(jnp.abs(Q_standardized), axis=reduction_axes, keepdims=True)
        Q_norm = jnp.max(Q_norm, axis=0, keepdims=True)

        h = hbar / Q_norm
        theta_new = theta[:, jnp.newaxis, :] + h * Q

        log_jacobian = jnp.zeros((theta.shape[0], log_ell.shape[1]))

        # Reconstruct parameters
        params_new = {}
        for i in range(log_ell.shape[1]):
            params_new_i = self.likelihood_fn.reconstruct_parameters(theta_new[:, i, :], params)
            if i == 0:
                for key in params_new_i.keys():
                    params_new[key] = params_new_i[key][:, jnp.newaxis, ...]
            else:
                for key in params_new_i.keys():
                    params_new[key] = jnp.concatenate([
                        params_new[key],
                        params_new_i[key][:, jnp.newaxis, ...]
                    ], axis=1)

        eta_weights, psis_weights, khat, log_ell_new = self._compute_importance_weights(
            data, params, params_new, log_jacobian, variational, log_pi, log_ell_original=log_ell
        )

        predictions = self.likelihood_fn.log_likelihood(data, params_new)
        weight_entropy = self.entropy(eta_weights)
        psis_entropy = self.entropy(psis_weights)

        p_loo_eta = jnp.sum(jnp.exp(predictions) * eta_weights, axis=0)
        p_loo_psis = jnp.sum(jnp.exp(predictions) * psis_weights, axis=0)
        ll_loo_eta = jnp.sum(eta_weights * jnp.exp(log_ell_new), axis=0)
        ll_loo_psis = jnp.sum(psis_weights * jnp.exp(log_ell_new), axis=0)

        return {
            'eta_weights': eta_weights,
            'psis_weights': psis_weights,
            'p_loo_eta': p_loo_eta,
            'p_loo_psis': p_loo_psis,
            'll_loo_eta': ll_loo_eta,
            'll_loo_psis': ll_loo_psis,
            'weight_entropy': weight_entropy,
            'psis_entropy': psis_entropy,
            'khat': khat,
            'predictions': predictions
        }

    def _transform_identity(self,
                           data: Any,
                           params: Dict[str, Any],
                           theta: jnp.ndarray,
                           log_ell: jnp.ndarray,
                           log_ell_prime: jnp.ndarray,
                           log_ell_doubleprime: jnp.ndarray,
                           theta_std: jnp.ndarray,
                           hbar: float,
                           variational: bool,
                           log_pi: jnp.ndarray,
                           grad_log_pi: jnp.ndarray,
                           log_ell_original: jnp.ndarray = None,
                           **kwargs) -> Dict[str, Any]:
        """Identity transformation T_I (no transformation)."""

        # No transformation - just compute standard importance weights
        n_samples, n_data = log_ell.shape

        # Create "transformed" parameters that are identical to original
        params_new = {}
        for key, value in params.items():
            # Add data dimension: (n_samples,) -> (n_samples, n_data, ...)
            if value.ndim == 1:
                params_new[key] = jnp.tile(value[:, jnp.newaxis], (1, n_data))
            else:
                # For multi-dimensional parameters, tile appropriately
                tile_pattern = [1] * value.ndim
                tile_pattern.insert(1, n_data)
                params_new[key] = jnp.tile(value[:, jnp.newaxis, ...], tile_pattern)

        log_jacobian = jnp.zeros((n_samples, n_data))

        eta_weights, psis_weights, khat, log_ell_new = self._compute_importance_weights(
            data, params, params_new, log_jacobian, variational, log_pi, log_ell_original=log_ell
        )

        predictions = log_ell  # No transformation, so predictions are original likelihoods
        weight_entropy = self.entropy(eta_weights)
        psis_entropy = self.entropy(psis_weights)

        p_loo_eta = jnp.sum(jnp.exp(predictions) * eta_weights, axis=0)
        p_loo_psis = jnp.sum(jnp.exp(predictions) * psis_weights, axis=0)
        ll_loo_eta = jnp.sum(eta_weights * jnp.exp(log_ell_new), axis=0)
        ll_loo_psis = jnp.sum(psis_weights * jnp.exp(log_ell_new), axis=0)

        return {
            'eta_weights': eta_weights,
            'psis_weights': psis_weights,
            'p_loo_eta': p_loo_eta,
            'p_loo_psis': p_loo_psis,
            'll_loo_eta': ll_loo_eta,
            'll_loo_psis': ll_loo_psis,
            'weight_entropy': weight_entropy,
            'psis_entropy': psis_entropy,
            'khat': khat,
            'predictions': predictions
        }


    def _compute_moments(self, params: Dict[str, Any], weights: jnp.ndarray) -> Dict[str, Any]:
        """Compute weighted and unweighted moments for parameters.
        
        Args:
            params: Dictionary of parameters (shape: S x ...)
            weights: Importance weights (shape: S x N)
            
        Returns:
            Dictionary containing means and variances
        """
        moments = {}
        # Normalize weights along sample dimension (axis 0)
        # weights is S x N
        # We want weights sum to 1 for each n
        w_sum = jnp.sum(weights, axis=0, keepdims=True)
        w_norm = weights / (w_sum + 1e-10)
        
        for name, value in params.items():
            # value shape: S x ... (e.g., S x K or S)
            # weights shape: S x N
            
            # Weighted mean
            # Need to broadcast weights to match value dimensions
            # If value is S x K, we want result N x K
            # If value is S, we want result N
            
            if value.ndim == 1: # Shape S
                # Broadcast value to S x N
                v_expanded = value[:, jnp.newaxis] # S x 1
                
                # Weighted mean: sum(w * v, axis=0) -> N
                mean_w = jnp.sum(w_norm * v_expanded, axis=0)
                mean = jnp.mean(value, axis=0) # Scalar
                
                # Expand mean for variance calc: N
                mean_expanded_w = mean_w
                mean_expanded = mean      

                # Weighted variance
                var_w = jnp.sum(w_norm * (v_expanded - mean_expanded_w)**2, axis=0)
                var = jnp.mean((value - mean)**2, axis=0)
                
            else: # Shape S x K or S x P1 x P2...
                # Broadcast weights to matches value dimensions
                # value: (S, P...) -> v_expanded (S, 1, P...)
                v_expanded = value[:, jnp.newaxis, ...]
                
                # w_norm: (S, N) -> w_expanded (S, N, 1...) to match v_expanded rank
                w_expanded = w_norm
                while w_expanded.ndim < v_expanded.ndim:
                    w_expanded = w_expanded[..., jnp.newaxis]
                
                # Weighted mean: sum(w * v, axis=0) -> N x P...
                mean_w = jnp.sum(w_expanded * v_expanded, axis=0)
                mean = jnp.mean(value, axis=0) # P...
                
                # Weighted variance: sum(w * (v - mean)^2, axis=0) -> N x P...
                # Note: mean_w is (N, P...), v_expanded is (S, 1, P...)
                # Broadcasting (S, 1, P...) with (N, P...) -> (S, N, P...)
                var_w = jnp.sum(w_expanded * (v_expanded - mean_w)**2, axis=0)
                
                # Unweighted variance
                # mean is (P...). value is (S, P...). Broadcasts fine.
                var = jnp.mean((value - mean)**2, axis=0)
            
            moments[name] = {
                'mean': mean,
                'mean_w': mean_w,
                'var': var,
                'var_w': var_w
            }
            
        return moments

    def _transform_mm1(self, params: Dict[str, Any], data: Dict[str, Any],
                       weight_fn=None, log_ell_original=None, variational=False, log_pi=None, **kwargs) -> Dict[str, Any]:
        """Moment Matching transformation 1 (Shift by weighted mean diff)."""
        if log_ell_original is None:
             raise ValueError("log_ell_original is required for MM1 transformation")
        # Fix: Compute weights correctly using the helper (which computes log_ell)
        # We need log_ell_original if not provided
        log_w = -log_ell_original
        log_w = jax.lax.stop_gradient(log_w)
        weights = jnp.exp(log_w)
        
        moments = self._compute_moments(params, weights)
        
        new_params = {}
        for name, value in params.items():
            m = moments[name]
            # beta_adj = beta + 1 * (-beta_hat + beta_hat_w)
            # Shapes: value (S x ...), mean (Scalar or K), mean_w (N or N x K)
            
            if value.ndim == 1:
                # Value: S
                # Mean: Scalar
                # Mean_w: N
                diff = -m['mean'] + m['mean_w'] # N
                new_params[name] = value[:, jnp.newaxis] + diff[jnp.newaxis, :] # S x N
            else:
                # Value: S x K
                # Mean: K
                # Mean_w: N x K
                diff = -m['mean'][jnp.newaxis, :] + m['mean_w'] # N x K
                new_params[name] = value[:, jnp.newaxis, :] + diff[jnp.newaxis, :, :] # S x N x K
                
        # MM1 returns 0 log Jacobian adjustment (see validaty in notebook) or implies volume preservation?
        # The notebook returns tf.zeros_like(ell). It's a pure shift so jacobian is 0.
        log_jacobian = jnp.zeros_like(log_w)
        
        # Compute final importance weights
        eta_weights, psis_weights, khat, log_ell_new = self._compute_importance_weights(
            data, params, new_params, log_jacobian, variational, log_pi_original=log_pi, log_ell_original=log_ell_original
        )
        
        # Compute predictions and diagnostics
        predictions = self.likelihood_fn.log_likelihood(data, new_params)
        
        weight_entropy = self.entropy(eta_weights)
        psis_entropy = self.entropy(psis_weights)
        
        p_loo_eta = jnp.sum(jnp.exp(predictions) * eta_weights, axis=0)
        p_loo_psis = jnp.sum(jnp.exp(predictions) * psis_weights, axis=0)
        
        ll_loo_eta = jnp.sum(eta_weights * jnp.exp(log_ell_new), axis=0)
        ll_loo_psis = jnp.sum(psis_weights * jnp.exp(log_ell_new), axis=0)
        
        return {
            'eta_weights': eta_weights,
            'psis_weights': psis_weights,
            'p_loo_eta': p_loo_eta,
            'p_loo_psis': p_loo_psis,
            'll_loo_eta': ll_loo_eta,
            'll_loo_psis': ll_loo_psis,
            'weight_entropy': weight_entropy,
            'psis_entropy': psis_entropy,
            'khat': khat
        }

    def _transform_mm2(self, params: Dict[str, Any], data: Dict[str, Any],
                       weight_fn=None, log_ell_original=None, variational=False, log_pi=None, **kwargs) -> Dict[str, Any]:
        """Moment Matching transformation 2 (Shift + Scale)."""
        if log_ell_original is None:
              raise ValueError("log_ell_original is required for MM2 transformation")
        
        log_w = -log_ell_original
        log_w = jax.lax.stop_gradient(log_w)
        weights = jnp.exp(log_w)
        
        moments = self._compute_moments(params, weights)
        
        new_params = {}
        # log_det_jac needs to match shape of log_w (S, N)
        # Calculating per-parameter contribution and summing
        log_det_jac = jnp.zeros_like(log_w)
        
        for name, value in params.items():
            m = moments[name]
            # beta_adj = beta + 1 * ( (sqrt(v_w/v) - 1)*beta - sqrt(v_w/v)*beta_hat + beta_hat_w )
            # This simplifies to: beta_adj = sqrt(v_w/v)*(beta - beta_hat) + beta_hat_w
            
            if value.ndim == 1:
                ratio = jnp.sqrt(m['var_w'] / (m['var'] + 1e-10)) # N
                # Term 1: ratio * (value - mean)
                term1 = ratio[jnp.newaxis, :] * (value[:, jnp.newaxis] - m['mean']) # S x N
                new_value = term1 + m['mean_w'][jnp.newaxis, :] # S x N
                new_params[name] = new_value
                
                # Log Jacobian: sum log(ratio)
                # Here dimension is 1, so just log(ratio)
                log_det_jac += jnp.log(ratio)[jnp.newaxis, :]
            else:
                # Ratio: N x K
                ratio = jnp.sqrt(m['var_w'] / (m['var'][jnp.newaxis, :] + 1e-10)) 
                
                # Term 1: ratio * (value - mean)
                # value: S x K -> S x 1 x K
                # mean: K -> 1 x 1 x K
                # term1: S x N x K
                term1 = ratio[jnp.newaxis, :, :] * (value[:, jnp.newaxis, :] - m['mean'][jnp.newaxis, jnp.newaxis, :])
                new_value = term1 + m['mean_w'][jnp.newaxis, :, :]
                new_params[name] = new_value
                
                # Log Jacobian: sum over parameter dimensions
                reduction_axes = tuple(range(1, ratio.ndim))
                log_det_jac += jnp.sum(jnp.log(ratio), axis=reduction_axes)[jnp.newaxis, :]
                
        log_jacobian = log_det_jac
        
        # Compute final importance weights
        eta_weights, psis_weights, khat, log_ell_new = self._compute_importance_weights(
            data=data, params_original=params, params_transformed=new_params, log_jacobian=log_jacobian, variational=variational, log_pi_original=log_pi, log_ell_original=log_ell_original
        )
        
        # Compute predictions and diagnostics
        predictions = self.likelihood_fn.log_likelihood(data, new_params)
        
        weight_entropy = self.entropy(eta_weights)
        psis_entropy = self.entropy(psis_weights)
        
        p_loo_eta = jnp.sum(jnp.exp(predictions) * eta_weights, axis=0)
        p_loo_psis = jnp.sum(jnp.exp(predictions) * psis_weights, axis=0)
        
        ll_loo_eta = jnp.sum(eta_weights * jnp.exp(log_ell_new), axis=0)
        ll_loo_psis = jnp.sum(psis_weights * jnp.exp(log_ell_new), axis=0)
        
        return {
            'eta_weights': eta_weights,
            'psis_weights': psis_weights,
            'p_loo_eta': p_loo_eta,
            'p_loo_psis': p_loo_psis,
            'll_loo_eta': ll_loo_eta,
            'll_loo_psis': ll_loo_psis,
            'weight_entropy': weight_entropy,
            'psis_entropy': psis_entropy,
            'khat': khat
        }

    def _transform_pmm1(self, params: Dict[str, Any], data: Dict[str, Any],
                        weight_fn=None, log_ell_original=None, hbar=1.0, variational=False, log_pi=None, **kwargs) -> Dict[str, Any]:
        """Partial Moment Matching 1 (MM1 with step size hbar)."""
        # Exactly like MM1 but scaled by hbar
        # beta_adj = beta + hbar * (-beta_hat + beta_hat_w)
        
        if log_ell_original is None:
              raise ValueError("log_ell_original is required for PMM1 transformation")
        
        log_w = -log_ell_original
        log_w = jax.lax.stop_gradient(log_w)
        weights = jnp.exp(log_w)
        
        moments = self._compute_moments(params, weights)
        
        new_params = {}
        for name, value in params.items():
            m = moments[name]
            
            if value.ndim == 1:
                diff = -m['mean'] + m['mean_w'] # N
                new_params[name] = value[:, jnp.newaxis] + hbar * diff[jnp.newaxis, :]
            else:
                diff = -m['mean'][jnp.newaxis, :] + m['mean_w'] # N x K
                new_params[name] = value[:, jnp.newaxis, :] + hbar * diff[jnp.newaxis, :, :]
                
        # Jacobian is still 0 because it's a shift
        log_jacobian = jnp.zeros_like(log_w)
        
        # Compute final importance weights
        eta_weights, psis_weights, khat, log_ell_new = self._compute_importance_weights(
            data=data, params_original=params, params_transformed=new_params, log_jacobian=log_jacobian, variational=variational, log_pi_original=log_pi, log_ell_original=log_ell_original
        )
        
        # Compute predictions and diagnostics
        predictions = self.likelihood_fn.log_likelihood(data, new_params)
        
        weight_entropy = self.entropy(eta_weights)
        psis_entropy = self.entropy(psis_weights)
        
        p_loo_eta = jnp.sum(jnp.exp(predictions) * eta_weights, axis=0)
        p_loo_psis = jnp.sum(jnp.exp(predictions) * psis_weights, axis=0)
        
        ll_loo_eta = jnp.sum(eta_weights * jnp.exp(log_ell_new), axis=0)
        ll_loo_psis = jnp.sum(psis_weights * jnp.exp(log_ell_new), axis=0)
        
        return {
            'eta_weights': eta_weights,
            'psis_weights': psis_weights,
            'p_loo_eta': p_loo_eta,
            'p_loo_psis': p_loo_psis,
            'll_loo_eta': ll_loo_eta,
            'll_loo_psis': ll_loo_psis,
            'weight_entropy': weight_entropy,
            'psis_entropy': psis_entropy,
            'khat': khat
        }

    def _transform_pmm2(self, params: Dict[str, Any], data: Dict[str, Any],
                        weight_fn=None, log_ell_original=None, hbar=1.0, variational=False, log_pi=None, **kwargs) -> Dict[str, Any]:
        """Partial Moment Matching 2 (MM2 with step size hbar)."""
        # beta_adj = beta + hbar * ( (sqrt(v_w/v) - 1)*beta - sqrt(v_w/v)*beta_hat + beta_hat_w )
        # beta_adj = beta + hbar * ( (ratio - 1)*beta - ratio*mean + mean_w )
        # beta_adj = beta * (1 + hbar*(ratio - 1)) + hbar*(mean_w - ratio*mean)
        
        if log_ell_original is None:
             raise ValueError("log_ell_original is required for PMM2 transformation")
        
        log_w = -log_ell_original
        log_w = jax.lax.stop_gradient(log_w)
        weights = jnp.exp(log_w)
        
        moments = self._compute_moments(params, weights)
        
        new_params = {}
        log_det_jac = jnp.zeros_like(log_w)
        
        for name, value in params.items():
            m = moments[name]
            
            if value.ndim == 1:
                ratio = jnp.sqrt(m['var_w'] / (m['var'] + 1e-10)) # N
                
                # Scaling factor: 1 + hbar * (ratio - 1)
                scale = 1.0 + hbar * (ratio - 1.0) # N
                
                # Shift: hbar * (mean_w - ratio * mean)
                shift = hbar * (m['mean_w'] - ratio * m['mean']) # N
                
                new_params[name] = value[:, jnp.newaxis] * scale[jnp.newaxis, :] + shift[jnp.newaxis, :]
                
                # Jacobian: log(scale)
                log_det_jac += jnp.log(jnp.abs(scale))[jnp.newaxis, :]
                
            else:
                # Ratio: N x K
                ratio = jnp.sqrt(m['var_w'] / (m['var'][jnp.newaxis, :] + 1e-10)) 
                
                scale = 1.0 + hbar * (ratio - 1.0) # N x K
                shift = hbar * (m['mean_w'] - ratio * m['mean'][jnp.newaxis, :]) # N x K
                
                new_params[name] = value[:, jnp.newaxis, :] * scale[jnp.newaxis, :, :] + shift[jnp.newaxis, :, :]
                
                # Jacobian: sum(log(scale)) over parameter dimensions
                reduction_axes = tuple(range(1, scale.ndim))
                log_det_jac += jnp.sum(jnp.log(jnp.abs(scale)), axis=reduction_axes)[jnp.newaxis, :]
                
        log_jacobian = log_det_jac
        
        # Compute final importance weights
        eta_weights, psis_weights, khat, log_ell_new = self._compute_importance_weights(
            data=data, params_original=params, params_transformed=new_params, log_jacobian=log_jacobian, variational=variational, log_pi_original=log_pi, log_ell_original=log_ell_original
        )
        
        # Compute predictions and diagnostics
        predictions = self.likelihood_fn.log_likelihood(data, new_params)
        
        weight_entropy = self.entropy(eta_weights)
        psis_entropy = self.entropy(psis_weights)
        
        p_loo_eta = jnp.sum(jnp.exp(predictions) * eta_weights, axis=0)
        p_loo_psis = jnp.sum(jnp.exp(predictions) * psis_weights, axis=0)
        
        ll_loo_eta = jnp.sum(eta_weights * jnp.exp(log_ell_new), axis=0)
        ll_loo_psis = jnp.sum(psis_weights * jnp.exp(log_ell_new), axis=0)
        
        return {
            'eta_weights': eta_weights,
            'psis_weights': psis_weights,
            'p_loo_eta': p_loo_eta,
            'p_loo_psis': p_loo_psis,
            'll_loo_eta': ll_loo_eta,
            'll_loo_psis': ll_loo_psis,
            'weight_entropy': weight_entropy,
            'psis_entropy': psis_entropy,
            'khat': khat
        }

# Legacy classes for backward compatibility
class AdaptiveIsSampler(ABC):
    def __init__(self):
        return

class Bijection(ABC):
    @abstractmethod
    def call(self, data, **params):
        return

    @abstractmethod
    def inverse(self):
        return

    @abstractmethod
    def forward_grad(self):
        return

    def __init__(self):
        return

class AutoDiffBijection(Bijection):
    def __init__(self, model, hbar=1.0):
        self.model = model
        self.hbar = hbar
        return

    def call(self, data, params):
        return self.model.adaptive_is_loo(data, params, self.hbar)

class SmallStepTransformation(Bijection):
    def call(self):
        return



# Example implementations for common likelihood functions

class LogisticRegressionLikelihood(LikelihoodFunction):
    """Likelihood function for logistic regression."""

    def __init__(self, dtype=jnp.float32):
        self.dtype = dtype

    def log_likelihood(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        """Compute log-likelihood for logistic regression."""
        X = jnp.asarray(data["X"], dtype=self.dtype)  # (n_data, n_features)
        y = jnp.asarray(data["y"], dtype=self.dtype)  # (n_data,)

        beta = params["beta"]  # (n_samples, n_features)
        intercept = params["intercept"]  # (n_samples,)

        # Linear predictor: X @ beta.T + intercept
        if beta.ndim == 3:
            mu = jnp.einsum('df,sdf->sd', X, beta) + intercept
        else:
            mu = jnp.einsum('df,sf->sd', X, beta) + intercept[:, jnp.newaxis]

        # Sigmoid and log-likelihood
        sigma = jax.nn.sigmoid(mu)
        log_lik = y[jnp.newaxis, :] * jnp.log(sigma + 1e-10) + (1 - y[jnp.newaxis, :]) * jnp.log(1 - sigma + 1e-10)

        return log_lik

    def log_likelihood_gradient(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        """Compute gradient of log-likelihood w.r.t. parameters."""
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]

        mu = jnp.einsum('df,sf->sd', X, beta) + intercept[:, jnp.newaxis]
        sigma = jax.nn.sigmoid(mu)

        # Gradient w.r.t. linear predictor
        grad_mu = y[jnp.newaxis, :] - sigma  # (n_samples, n_data)

        # Gradient w.r.t. beta: X.T @ grad_mu
        grad_beta = jnp.einsum('df,sd->sdf', X, grad_mu)  # (n_samples, n_data, n_features)

        # Gradient w.r.t. intercept
        grad_intercept = grad_mu[..., jnp.newaxis]  # (n_samples, n_data, 1)

        # Concatenate gradients
        gradients = jnp.concatenate([grad_beta, grad_intercept], axis=-1)  # (n_samples, n_data, n_features + 1)

        return gradients

    def log_likelihood_hessian_diag(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        """Compute diagonal of Hessian of log-likelihood."""
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]

        mu = jnp.einsum('df,sf->sd', X, beta) + intercept[:, jnp.newaxis]
        sigma = jax.nn.sigmoid(mu)

        # Hessian diagonal w.r.t. linear predictor
        hess_diag_mu = -sigma * (1 - sigma)  # (n_samples, n_data)

        # Hessian diagonal w.r.t. beta: X^2 * hess_diag_mu
        hess_diag_beta = jnp.einsum('df,sd->sdf', X**2, hess_diag_mu)  # (n_samples, n_data, n_features)

        # Hessian diagonal w.r.t. intercept
        hess_diag_intercept = hess_diag_mu[..., jnp.newaxis]  # (n_samples, n_data, 1)

        # Concatenate
        hess_diag = jnp.concatenate([hess_diag_beta, hess_diag_intercept], axis=-1)

        return hess_diag

    def extract_parameters(self, params: Dict[str, Any]) -> jnp.ndarray:
        """Extract parameters into flattened array."""
        beta = params["beta"]  # (n_samples, n_features)
        intercept = params["intercept"]  # (n_samples,)

        # Concatenate
        theta = jnp.concatenate([beta, intercept[:, jnp.newaxis]], axis=-1)
        return theta

    def reconstruct_parameters(self, flat_params: jnp.ndarray, template: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct parameters from flattened array."""
        n_features = template["beta"].shape[-1]

        beta = flat_params[..., :n_features]
        intercept = flat_params[..., n_features]

        return {"beta": beta, "intercept": intercept}





class LinearRegressionLikelihood(LikelihoodFunction):
    """Likelihood function for linear regression with Gaussian errors."""

    def __init__(self, dtype=jnp.float32):
        self.dtype = dtype

    def log_likelihood(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        """Compute log-likelihood for linear regression."""
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]
        log_sigma = params.get("log_sigma", jnp.zeros((beta.shape[0],)))  # Log noise std

        # Predictions
        if beta.ndim == 3:
            mu = jnp.einsum('df,sdf->sd', X, beta) + intercept
        else:
            mu = jnp.einsum('df,sf->sd', X, beta) + intercept[:, jnp.newaxis]
        
        sigma = jnp.exp(log_sigma)
        if log_sigma.ndim == 1:
             sigma = sigma[:, jnp.newaxis]

        # Gaussian log-likelihood
        residuals = y[jnp.newaxis, :] - mu
        log_lik = -0.5 * jnp.log(2 * jnp.pi) - log_sigma[:, jnp.newaxis] - 0.5 * (residuals / sigma)**2

        return log_lik

    def log_likelihood_gradient(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        """Compute gradient of log-likelihood."""
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]
        log_sigma = params.get("log_sigma", jnp.zeros((beta.shape[0],)))

        mu = jnp.einsum('df,sf->sd', X, beta) + intercept[:, jnp.newaxis]
        sigma = jnp.exp(log_sigma)[:, jnp.newaxis]
        residuals = y[jnp.newaxis, :] - mu

        # Gradients
        grad_beta = jnp.einsum('df,sd->sdf', X, residuals / sigma**2)
        grad_intercept = (residuals / sigma**2)[..., jnp.newaxis]
        grad_log_sigma = (-1 + (residuals / sigma)**2)[..., jnp.newaxis]

        gradients = jnp.concatenate([grad_beta, grad_intercept, grad_log_sigma], axis=-1)
        return gradients

    def log_likelihood_hessian_diag(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        """Compute diagonal of Hessian."""
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]
        log_sigma = params.get("log_sigma", jnp.zeros((beta.shape[0],)))

        mu = jnp.einsum('df,sf->sd', X, beta) + intercept[:, jnp.newaxis]
        sigma = jnp.exp(log_sigma)[:, jnp.newaxis]
        residuals = y[jnp.newaxis, :] - mu

        # Hessian diagonals
        hess_diag_beta = jnp.einsum('df,sd->sdf', -X**2, 1 / sigma**2)
        hess_diag_intercept = (-1 / sigma**2)[..., jnp.newaxis]
        hess_diag_log_sigma = (-2 * (residuals / sigma)**2)[..., jnp.newaxis]

        hess_diag = jnp.concatenate([hess_diag_beta, hess_diag_intercept, hess_diag_log_sigma], axis=-1)
        return hess_diag

    def extract_parameters(self, params: Dict[str, Any]) -> jnp.ndarray:
        """Extract parameters into flattened array."""
        beta = params["beta"]
        intercept = params["intercept"]
        log_sigma = params.get("log_sigma", jnp.zeros((beta.shape[0],)))

        theta = jnp.concatenate([beta, intercept[:, jnp.newaxis], log_sigma[:, jnp.newaxis]], axis=-1)
        return theta

    def reconstruct_parameters(self, flat_params: jnp.ndarray, template: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct parameters from flattened array."""
        n_features = template["beta"].shape[-1]

        beta = flat_params[..., :n_features]
        intercept = flat_params[..., n_features]
        log_sigma = flat_params[..., n_features + 1]

        result = {"beta": beta, "intercept": intercept}
        if "log_sigma" in template:
            result["log_sigma"] = log_sigma

        return result


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
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Tuple, Optional
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
                       data: Any,
                       params: Dict[str, Any],
                       hbar: float = 1.0,
                       variational: bool = True,
                       transformations: Optional[list] = None) -> Dict[str, Any]:
        """Perform adaptive importance sampling leave-one-out cross-validation.

        Args:
            data: Input data for evaluation
            params: Posterior parameter samples
            hbar: Step size parameter for transformations
            variational: Whether to use variational approximation
            transformations: List of transformation names to apply

        Returns:
            Dictionary containing results for each transformation
        """
        if transformations is None:
            transformations = ['ll', 'kl', 'var', 'identity']

        # Compute base quantities
        log_ell = self.likelihood_fn.log_likelihood(data, params)
        log_ell_prime = self.likelihood_fn.log_likelihood_gradient(data, params)
        log_ell_doubleprime = self.likelihood_fn.log_likelihood_hessian_diag(data, params)

        # Get initial PSIS diagnostics
        _, khat0 = nppsis.psislw(-log_ell)

        # Extract parameter arrays
        theta = self.likelihood_fn.extract_parameters(params)
        n_samples, n_params = theta.shape
        n_data = log_ell.shape[1]

        # Compute posterior gradient if needed
        if variational and self.surrogate_log_prob_fn is not None:
            # Use surrogate distribution gradients
            log_pi = self.surrogate_log_prob_fn(params)
            log_pi = log_pi - jnp.max(log_pi, axis=0)
            grad_log_pi = jax.grad(self.surrogate_log_prob_fn)(params)
            grad_log_pi = self.likelihood_fn.extract_parameters(grad_log_pi)
        else:
            # Use full posterior gradients
            if self.prior_log_prob_fn is not None:
                log_prior = self.prior_log_prob_fn(params)
                grad_log_prior = jax.grad(self.prior_log_prob_fn)(params)
                grad_log_prior = self.likelihood_fn.extract_parameters(grad_log_prior)
            else:
                log_prior = jnp.zeros(n_samples)
                grad_log_prior = jnp.zeros_like(theta)

            log_pi = jnp.sum(log_ell, axis=1) + log_prior
            grad_log_pi = jnp.sum(log_ell_prime, axis=1) + grad_log_prior

        # Compute parameter standard deviations for standardization
        theta_std = jnp.std(theta, axis=0, keepdims=True)
        theta_std = jnp.maximum(theta_std, 1e-6)  # Avoid division by zero

        results = {}

        for transform_name in transformations:
            if transform_name == 'll':
                result = self._transform_likelihood_descent(
                    data, params, theta, log_ell, log_ell_prime, log_ell_doubleprime,
                    theta_std, hbar, variational, log_pi, grad_log_pi
                )
            elif transform_name == 'kl':
                result = self._transform_kl_divergence(
                    data, params, theta, log_ell, log_ell_prime, log_ell_doubleprime,
                    theta_std, hbar, variational, log_pi, grad_log_pi
                )
            elif transform_name == 'var':
                result = self._transform_variance_based(
                    data, params, theta, log_ell, log_ell_prime, log_ell_doubleprime,
                    theta_std, hbar, variational, log_pi, grad_log_pi
                )
            elif transform_name == 'identity':
                result = self._transform_identity(
                    data, params, theta, log_ell, log_ell_prime, log_ell_doubleprime,
                    theta_std, hbar, variational, log_pi, grad_log_pi
                )
            else:
                raise ValueError(f"Unknown transformation: {transform_name}")

            results[transform_name] = result

        return results

    def _compute_importance_weights(self,
                                  data: Any,
                                  params_original: Dict[str, Any],
                                  params_transformed: Dict[str, Any],
                                  log_jacobian: jnp.ndarray,
                                  variational: bool,
                                  log_pi_original: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        """Compute importance weights and related quantities."""

        # Compute likelihood for transformed parameters
        log_ell_new = self.likelihood_fn.log_likelihood(data, params_transformed)

        if variational and self.surrogate_log_prob_fn is not None:
            # Trust variational approximation
            delta_log_pi = (self.surrogate_log_prob_fn(params_transformed) -
                          log_pi_original[:, jnp.newaxis])
            delta_log_pi = delta_log_pi - jnp.max(delta_log_pi, axis=0, keepdims=True)
        else:
            # Compute full posterior difference
            if self.prior_log_prob_fn is not None:
                log_prior_new = self.prior_log_prob_fn(params_transformed)
                log_prior_old = self.prior_log_prob_fn(params_original)
                delta_log_prior = log_prior_new - log_prior_old[:, jnp.newaxis]
            else:
                delta_log_prior = 0.0

            log_pi_new = jnp.sum(log_ell_new, axis=-1)
            delta_log_pi = log_pi_new - log_pi_original[:, jnp.newaxis] + delta_log_prior

        # Compute importance weights
        log_eta_weights = delta_log_pi - log_ell_new + log_jacobian
        log_eta_weights = log_eta_weights - jnp.max(log_eta_weights, axis=0)

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
                                    grad_log_pi: jnp.ndarray) -> Dict[str, Any]:
        """Likelihood descent transformation T_ll."""

        # Compute direction: negative log-likelihood gradient
        Q = -log_ell_prime  # Shape: (n_samples, n_data, n_params)

        # Standardize the direction
        Q_standardized = Q / theta_std[jnp.newaxis, jnp.newaxis, :]
        Q_norm = jnp.max(jnp.abs(Q_standardized), axis=-1)  # (n_samples, n_data)
        Q_norm = jnp.max(Q_norm, axis=0, keepdims=True)  # (1, n_data)

        # Compute step size
        h = hbar / Q_norm[..., jnp.newaxis]  # (1, n_data, 1)

        # Apply transformation
        theta_new = theta[:, jnp.newaxis, :] + h * Q  # (n_samples, n_data, n_params)

        # Compute Jacobian (for simple additive transformation, this is often identity in log space)
        # For more complex transformations, this would need to be computed properly
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

        # Compute importance weights
        eta_weights, psis_weights, khat, log_ell_new = self._compute_importance_weights(
            data, params, params_new, log_jacobian, variational, log_pi
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
                               grad_log_pi: jnp.ndarray) -> Dict[str, Any]:
        """KL divergence based transformation T_kl."""

        # This is a more complex transformation that uses the posterior weights
        log_pi_normalized = log_pi - jnp.max(log_pi, axis=0, keepdims=True)

        # The direction involves weighted likelihood gradients
        # This is a simplified version - the full implementation would be more complex
        weights = jnp.exp(log_pi_normalized)[:, jnp.newaxis]
        Q = -log_ell_prime * weights[..., jnp.newaxis]

        # Standardize and compute step size (similar to likelihood descent)
        Q_standardized = Q / theta_std[jnp.newaxis, jnp.newaxis, :]
        Q_norm = jnp.max(jnp.abs(Q_standardized), axis=-1)
        Q_norm = jnp.max(Q_norm, axis=0, keepdims=True)

        h = hbar / Q_norm[..., jnp.newaxis]
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
            data, params, params_new, log_jacobian, variational, log_pi
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
                                grad_log_pi: jnp.ndarray) -> Dict[str, Any]:
        """Variance-based transformation T_var."""

        # Use Hessian information for the transformation direction
        # This is simplified - full implementation would use second-order information
        log_pi_normalized = log_pi - jnp.max(log_pi, axis=0, keepdims=True)
        weights = jnp.exp(log_pi_normalized)[:, jnp.newaxis]

        # Direction based on curvature information
        Q = -log_ell_prime * weights[..., jnp.newaxis] * jnp.abs(log_ell_doubleprime)

        Q_standardized = Q / theta_std[jnp.newaxis, jnp.newaxis, :]
        Q_norm = jnp.max(jnp.abs(Q_standardized), axis=-1)
        Q_norm = jnp.max(Q_norm, axis=0, keepdims=True)

        h = hbar / Q_norm[..., jnp.newaxis]
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
            data, params, params_new, log_jacobian, variational, log_pi
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
                           grad_log_pi: jnp.ndarray) -> Dict[str, Any]:
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
            data, params, params_new, log_jacobian, variational, log_pi
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
        mu = jnp.einsum('df,sf->sd', X, beta) + intercept[:, jnp.newaxis]  # (n_samples, n_data)

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


class PoissonRegressionLikelihood(LikelihoodFunction):
    """Likelihood function for Poisson regression."""

    def __init__(self, dtype=jnp.float32):
        self.dtype = dtype

    def log_likelihood(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        """Compute log-likelihood for Poisson regression."""
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)  # counts

        beta = params["beta"]
        intercept = params["intercept"]

        # Log rate
        log_rate = jnp.einsum('df,sf->sd', X, beta) + intercept[:, jnp.newaxis]
        rate = jnp.exp(log_rate)

        # Poisson log-likelihood: y * log(rate) - rate - log(y!)
        # We omit log(y!) as it doesn't depend on parameters
        log_lik = y[jnp.newaxis, :] * log_rate - rate

        return log_lik

    def log_likelihood_gradient(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        """Compute gradient of log-likelihood."""
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]

        log_rate = jnp.einsum('df,sf->sd', X, beta) + intercept[:, jnp.newaxis]
        rate = jnp.exp(log_rate)

        # Gradient w.r.t. log rate
        grad_log_rate = y[jnp.newaxis, :] - rate

        # Gradient w.r.t. beta
        grad_beta = jnp.einsum('df,sd->sdf', X, grad_log_rate)

        # Gradient w.r.t. intercept
        grad_intercept = grad_log_rate[..., jnp.newaxis]

        gradients = jnp.concatenate([grad_beta, grad_intercept], axis=-1)
        return gradients

    def log_likelihood_hessian_diag(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        """Compute diagonal of Hessian."""
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]

        log_rate = jnp.einsum('df,sf->sd', X, beta) + intercept[:, jnp.newaxis]
        rate = jnp.exp(log_rate)

        # Hessian diagonal w.r.t. log rate
        hess_diag_log_rate = -rate

        # Hessian diagonal w.r.t. beta
        hess_diag_beta = jnp.einsum('df,sd->sdf', X**2, hess_diag_log_rate)

        # Hessian diagonal w.r.t. intercept
        hess_diag_intercept = hess_diag_log_rate[..., jnp.newaxis]

        hess_diag = jnp.concatenate([hess_diag_beta, hess_diag_intercept], axis=-1)
        return hess_diag

    def extract_parameters(self, params: Dict[str, Any]) -> jnp.ndarray:
        """Extract parameters into flattened array."""
        beta = params["beta"]
        intercept = params["intercept"]
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
        mu = jnp.einsum('df,sf->sd', X, beta) + intercept[:, jnp.newaxis]
        sigma = jnp.exp(log_sigma)[:, jnp.newaxis]

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
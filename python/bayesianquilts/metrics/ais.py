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
from functools import partial
from bayesianquilts.metrics import psis


# JIT-compiled utility functions for performance
@partial(jax.jit, static_argnums=())
def _normalize_weights(log_weights: jnp.ndarray) -> jnp.ndarray:
    """Normalize log weights to probabilities (JIT-compiled)."""
    log_weights = log_weights - jnp.max(log_weights, axis=0, keepdims=True)
    weights = jnp.exp(log_weights)
    return weights / jnp.sum(weights, axis=0, keepdims=True)


@jax.jit
def _compute_entropy(weights: jnp.ndarray) -> jnp.ndarray:
    """Compute entropy of weights (JIT-compiled)."""
    return -jnp.sum(weights * jnp.log(weights + 1e-12), axis=0)


@jax.jit
def _weighted_sum(weights: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """Compute weighted sum along axis 0 (JIT-compiled)."""
    return jnp.sum(weights * values, axis=0)


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
    def log_likelihood_gradient(self, data: Any, params: Dict[str, Any]) -> Any:
        """Compute gradient of log-likelihood w.r.t. parameters.

        Args:
            data: Input data
            params: Model parameters (Dict/PyTree)

        Returns:
            Gradient matching the structure of params, with extra (N,) dimensions.
            (e.g., if param is (S, ...), grad is (S, N, ...))
        """
        pass


    @abstractmethod
    def log_likelihood_hessian_diag(
        self, data: Any, params: Dict[str, Any]
    ) -> Any:
        """Compute diagonal of Hessian of log-likelihood w.r.t. parameters.

        Args:
            data: Input data
            params: Model parameters

        Returns:
            Hessian diagonal matching params structure + (N,) dim.
        """
        pass


class Transformation(ABC):
    """Abstract base class for AIS transformations."""

    def __init__(self, likelihood_fn: LikelihoodFunction):
        self.likelihood_fn = likelihood_fn

    @abstractmethod
    def __call__(
        self,
        max_iter: int,
        params: Dict[str, Any],
        theta: jnp.ndarray,
        data: Any,
        log_ell: jnp.ndarray,  # (S, N)
        **kwargs,
    ) -> Dict[str, Any]:
        """Apply transformation.

        Returns dictionary containing:
            - theta_new: Transformed parameters (S, N, K)
            - log_jacobian: Log Jacobian determinant (S, N)
            - params_new: Transformed parameters as dict
            - ... other metrics
        """
        pass

    @staticmethod
    def compute_moments(params: Dict[str, Any], weights: jnp.ndarray) -> Dict[str, Any]:
        """Compute weighted and unweighted moments for parameters.

        Args:
            params: Dictionary of parameters where each value has shape (S, ...).
                    S is the sample dimension. Additional dimensions are parameter-specific.
            weights: Importance weights with shape (S, N) where N is the data dimension.

        Returns:
            Dictionary mapping parameter names to dicts with keys:
                - mean: Unweighted mean over S, shape (...)
                - mean_w: Weighted mean, shape (N, ...)
                - var: Unweighted variance over S, shape (...)
                - var_w: Weighted variance, shape (N, ...)
        """
        moments = {}

        # Normalize weights along sample dimension (axis 0) - single computation
        w_norm = _normalize_weights(jnp.log(weights + 1e-10))  # (S, N)

        for name, value in params.items():
            # value shape: (S, ...) where ... can be empty, (K,), (K, L), etc.
            # weights shape: (S, N)

            # Unweighted statistics (reduce over S)
            mean = jnp.mean(value, axis=0)  # (...)
            var = jnp.var(value, axis=0)  # (...) - use jnp.var for efficiency

            # For weighted statistics, we need to broadcast:
            # - value from (S, ...) to (S, 1, ...) so N dimension can be inserted
            # - weights from (S, N) to (S, N, 1, 1, ...) to match value's trailing dims

            # Insert N dimension into value at position 1
            v_expanded = jnp.expand_dims(value, axis=1)  # (S, 1, ...)

            # Expand weights to match v_expanded's rank
            w_expanded = w_norm
            for _ in range(v_expanded.ndim - w_expanded.ndim):
                w_expanded = jnp.expand_dims(w_expanded, axis=-1)  # (S, N, 1, 1, ...)

            # Weighted mean: sum over S -> (N, ...)
            mean_w = jnp.sum(w_expanded * v_expanded, axis=0)

            # Weighted variance using online formula to avoid recomputing mean_w
            # var_w = E[x^2] - E[x]^2
            mean_sq_w = jnp.sum(w_expanded * v_expanded ** 2, axis=0)
            var_w = mean_sq_w - mean_w ** 2
            # Clip negative values due to numerical precision
            var_w = jnp.maximum(var_w, 0.0)

            moments[name] = {"mean": mean, "mean_w": mean_w, "var": var, "var_w": var_w}

        return moments

    def compute_importance_weights_helper(
        self,
        likelihood_fn: LikelihoodFunction,
        data: Any,
        params_original: Dict[str, Any],
        params_transformed: Dict[str, Any],
        log_jacobian: jnp.ndarray,
        variational: bool,
        log_pi_original: jnp.ndarray,
        log_ell_original: jnp.ndarray = None,
        surrogate_log_prob_fn=None,
    ) -> Tuple[jnp.ndarray, ...]:
        """Compute importance weights for transformed parameters.

        Returns:
            Tuple of (eta_weights, psis_weights, khat, log_ell_new).
        """
        log_ell_new = likelihood_fn.log_likelihood(data, params_transformed)

        if variational and surrogate_log_prob_fn is not None:
            log_pi_trans = surrogate_log_prob_fn(params_transformed)

            # Broadcast if log_pi_trans is (S,)
            if log_pi_trans.ndim == 1:
                log_pi_trans = log_pi_trans[:, jnp.newaxis]

            delta_log_pi = log_pi_trans - log_pi_original[:, jnp.newaxis]
        else:
            if log_ell_original is not None:
                delta_log_pi = log_ell_original - log_ell_new
            else:
                delta_log_pi = jnp.zeros_like(log_ell_new)

        # Combine all log terms first, then apply max-subtract for numerical stability
        log_eta_weights = delta_log_pi + log_jacobian
        log_eta_weights = log_eta_weights.astype(jnp.float64)

        # Use JIT-compiled normalization
        eta_weights = _normalize_weights(log_eta_weights)

        # PSIS smoothing
        psis_log_weights, khat = psis.psislw(
            log_eta_weights - jnp.max(log_eta_weights, axis=0, keepdims=True)
        )
        psis_weights = _normalize_weights(psis_log_weights)

        return eta_weights, psis_weights, khat, log_ell_new

    def entropy(self, weights):
        """Compute entropy of weights."""
        return _compute_entropy(weights)


class SmallStepTransformation(Transformation):
    """Base class for transformations of the form T(theta) = theta + h * Q(theta)."""

    @abstractmethod
    def compute_Q(
        self,
        theta: Any,
        data: Any,
        params: Dict[str, Any],
        current_log_ell: jnp.ndarray,
        **kwargs,
    ) -> Any:
        """Compute the vector field Q(theta). Returns PyTree matching theta."""
        pass

    def compute_divergence_Q(
        self,
        theta: Any,
        data: Any,
        params: Dict[str, Any],
        current_log_ell: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        """Compute divergence of Q using generic autodiff (Trace of Jacobian).

        The divergence div(Q) is used to compute the log Jacobian determinant:
            log|J| ≈ log(1 + h * div(Q))

        **Zero Divergence Fallback**: Returns zero divergence when:
        - PyTree is empty
        - Leaves have fewer than 2 dimensions
        - Leaves have inconsistent batch shapes
        - Any leaf has more than 3 dimensions (complex structure)

        This approximation is reasonable because:
        1. The Jacobian correction term is often small relative to other terms
        2. Many LOO implementations (e.g., standard PSIS-LOO) ignore this term
        3. Computing exact divergence for complex PyTrees is O(K) expensive
        4. Zero divergence corresponds to assuming volume-preserving transformation

        For simple 3D PyTrees (all leaves shape (S, N, K)), exact divergence
        is computed via JVP-based trace estimation.

        Args:
            theta: Current parameter PyTree
            data: Input data
            params: Original model parameters
            current_log_ell: Current log-likelihood values (S, N)

        Returns:
            Divergence array of shape (S, N), or zeros if fallback is used.
        """
        leaves, treedef = jax.tree_util.tree_flatten(theta)

        if not leaves:
            return jnp.zeros(())

        # Check if all leaves have consistent shapes for divergence computation
        # We need all leaves to have shape (S, N, K) where the last dim is the parameter dim
        # If any leaf has more than 3 dimensions, we can't compute divergence simply
        first_leaf = leaves[0]

        # Determine the batch shape (S, N) from the first two dimensions
        if first_leaf.ndim < 2:
            # Scalar or 1D - return zero
            return jnp.zeros(first_leaf.shape)

        batch_shape = first_leaf.shape[:2]  # (S, N)

        # Check if all leaves have consistent batch shapes and are 3D (S, N, K)
        can_compute_divergence = True
        for leaf in leaves:
            if leaf.ndim != 3:
                can_compute_divergence = False
                break
            if leaf.shape[:2] != batch_shape:
                can_compute_divergence = False
                break

        if not can_compute_divergence:
            # Return zero divergence for complex PyTree structures
            return jnp.zeros(batch_shape, dtype=first_leaf.dtype)

        # Simple case: all leaves are (S, N, K_i) - compute divergence
        # Note: We don't pass pre-computed gradients (log_ell_prime) to allow JVP to work.
        # Filter out pre-computed values that would make Q constant w.r.t. theta.
        kwargs_for_jvp = {
            k: v for k, v in kwargs.items()
            if k not in ('log_ell_prime', 'log_ell_doubleprime', 'grad_log_f')
        }

        def func(t):
            return self.compute_Q(t, data, params, current_log_ell, **kwargs_for_jvp)

        # Compute total number of parameters
        total_K = sum(L.shape[-1] for L in leaves)

        # Build index mapping: flat_idx -> (leaf_idx, feature_idx)
        leaf_indices = []
        feat_indices = []
        for i, L in enumerate(leaves):
            K_i = L.shape[-1]
            leaf_indices.extend([i] * K_i)
            feat_indices.extend(range(K_i))

        leaf_indices = jnp.array(leaf_indices)
        feat_indices = jnp.array(feat_indices)

        # Vectorized JVP computation using vmap
        def single_jvp(flat_idx):
            leaf_idx = leaf_indices[flat_idx]
            feat_idx = feat_indices[flat_idx]

            # Create one-hot tangent vector
            tangents = []
            for i, L in enumerate(leaves):
                t = jnp.zeros_like(L)
                # Use where to conditionally set the tangent
                mask = jnp.arange(L.shape[-1]) == feat_idx
                t = jnp.where(
                    (i == leaf_idx) & mask[jnp.newaxis, jnp.newaxis, :],
                    1.0, 0.0
                )
                tangents.append(t)

            tangent_tree = jax.tree_util.tree_unflatten(treedef, tangents)
            _, tangent_out = jax.jvp(func, (theta,), (tangent_tree,))

            q_out_leaves = jax.tree_util.tree_leaves(tangent_out)

            # Sum contributions from this parameter
            div_contrib = jnp.zeros(batch_shape, dtype=first_leaf.dtype)
            for i, q_leaf in enumerate(q_out_leaves):
                if i == leaf_idx:
                    if q_leaf.ndim == 2:
                        div_contrib = q_leaf
                    else:
                        div_contrib = q_leaf[..., feat_idx]
            return div_contrib

        # Try to compute divergence; fall back to zero if JVP fails
        try:
            # Use lax.fori_loop for memory efficiency instead of vmap
            # (vmap would create total_K copies of all intermediates)
            def body_fn(i, acc):
                return acc + single_jvp(i)

            divergence = jax.lax.fori_loop(
                0, total_K, body_fn,
                jnp.zeros(batch_shape, dtype=first_leaf.dtype)
            )
        except (ValueError, TypeError) as e:
            # JVP failed - return zero divergence
            return jnp.zeros(batch_shape, dtype=first_leaf.dtype)

        return divergence

    def normalize_vector_field(
        self, Q: Any, theta_std: Any = None
    ) -> Tuple[Any, jnp.ndarray]:
        """Normalize the vector field Q (PyTree).

        Args:
            Q: Vector field as PyTree with leaves of shape (S, N, ...) or (S, N)
            theta_std: Optional standardization PyTree matching Q structure

        Returns:
            Tuple of (Q_standardized, Q_norm_max) where Q_norm_max has shape (1, N).
        """
        # Standardize component-wise (fused with max computation for efficiency)
        if theta_std is not None:
            Q = jax.tree_util.tree_map(lambda q, s: q / (s + 1e-6), Q, theta_std)

        # Compute max absolute value across all leaves in a single pass
        # Using reduce instead of stack to avoid creating intermediate arrays
        leaves = jax.tree_util.tree_leaves(Q)

        if not leaves:
            return Q, jnp.zeros((1, 1))

        def leaf_abs_max(leaf):
            if leaf.ndim > 2:
                # Reduce over trailing dimensions
                return jnp.max(jnp.abs(leaf), axis=tuple(range(2, leaf.ndim)))
            return jnp.abs(leaf)

        # Compute max across all leaves without stacking
        first_mag = leaf_abs_max(leaves[0])
        Q_norm = first_mag
        for leaf in leaves[1:]:
            Q_norm = jnp.maximum(Q_norm, leaf_abs_max(leaf))

        Q_norm_max = jnp.max(Q_norm, axis=0, keepdims=True)  # (1, N)

        return Q, Q_norm_max

    def __call__(
        self,
        max_iter: int,
        params: Dict[str, Any],
        theta: Any,
        data: Any,
        log_ell: jnp.ndarray,
        hbar: float = 1.0,
        theta_std: Any = None,
        variational: bool = False,
        log_pi: jnp.ndarray = None,
        log_ell_original: jnp.ndarray = None,
        surrogate_log_prob_fn=None,
        **kwargs,
    ):

        # 1. Compute Q
        Q = self.compute_Q(
            theta,
            data,
            params,
            log_ell,
            log_pi=log_pi,
            log_ell_original=log_ell_original,
            **kwargs,
        )

        # 2. Normalize
        Q_standardized, Q_norm_max = self.normalize_vector_field(Q, theta_std)

        # h = rho / norm(Q)
        # h: (1, N)
        h = hbar / (Q_norm_max + 1e-8)

        # 3. Step: theta_new = theta + h * Q
        # h is (1, N).
        # For scalar params q is (S, N). h * q works.
        # For vector params q is (S, N, K). h * q needs h expanded.
        
        def update_step(t, q):
            # t: (S, 1, K...) or (S, 1, 1)
            # q: (S, N, K...) or (S, N)
            
            # If t has shape (S, 1, 1) [from scalar param], q might satisfy q.ndim == 2 (S, N)
            # We want to add them to get (S, N, 1).
            # t is already (S, 1, 1).
            # If q is (S, N), we need to expand it to (S, N, 1).
            
            q_expanded = q
            if t.ndim > q.ndim:
                 # Check if t is (...) -> (S, 1, ...)
                 # and q is (S, N)
                 # We need to insert dimensions into q to match t's rank
                 # t: (S, 1, D1, D2...)
                 # q: (S, N)
                 # q -> (S, N, 1, 1...)
                 for _ in range(t.ndim - q.ndim):
                     q_expanded = q_expanded[..., jnp.newaxis]
            
            # Now q_expanded should have same rank as t, but with N in axis 1.
            # h is (1, N).
            
            # Expand h to match q_expanded rank
            h_exp = h
            for _ in range(q_expanded.ndim - h.ndim):
                h_exp = h_exp[..., jnp.newaxis]
                
            return t + h_exp * q_expanded

        theta_new = jax.tree_util.tree_map(update_step, theta, Q)

        # 4. Jacobian Approximation
        div_Q = self.compute_divergence_Q(
            theta,
            data,
            params,
            log_ell,
            log_pi=log_pi,
            log_ell_original=log_ell_original,
            **kwargs,
        )

        # log|J| ~ log(1 + h * div(Q))
        # h (1, N), div_Q (S, N)
        # h * div_Q -> (S, N)
        term = 1.0 + h * div_Q
        log_jacobian = jnp.log(jnp.abs(term))

        theta_new_params = theta_new # It IS the params structure now
        
        # We need to call compute_importance_weights_helper with new params
        # But wait, compute_importance_weights_helper expects (params_original, params_transformed)
        # where params_transformed IS theta_new (since theta is params pytree).
        
        eta_weights, psis_weights, khat, log_ell_new = (
            self.compute_importance_weights_helper(
                self.likelihood_fn,
                data,
                params,                   # Original params
                theta_new_params,         # Transformed params
                log_jacobian,
                variational,
                log_pi,
                log_ell_original,
                surrogate_log_prob_fn
            )
        )

        predictions = log_ell_new
        exp_predictions = jnp.exp(predictions)
        exp_log_ell_new = jnp.exp(log_ell_new)

        # Use JIT-compiled entropy and weighted sum
        weight_entropy = _compute_entropy(eta_weights)
        psis_entropy = _compute_entropy(psis_weights)

        p_loo_eta = _weighted_sum(eta_weights, exp_predictions)
        p_loo_psis = _weighted_sum(psis_weights, exp_predictions)

        ll_loo_eta = _weighted_sum(eta_weights, exp_log_ell_new)
        ll_loo_psis = _weighted_sum(psis_weights, exp_log_ell_new)

        return {
            "theta_new": theta_new,
            "log_jacobian": log_jacobian,
            "eta_weights": eta_weights,
            "psis_weights": psis_weights,
            "khat": khat,
            "predictions": predictions,
            "log_ell_new": log_ell_new,
            "weight_entropy": weight_entropy,
            "psis_entropy": psis_entropy,
            "p_loo_eta": p_loo_eta,
            "p_loo_psis": p_loo_psis,
            "ll_loo_eta": ll_loo_eta,
            "ll_loo_psis": ll_loo_psis,
        }


class LikelihoodDescent(SmallStepTransformation):
    """Likelihood Descent transformation (Gradient Ascent on Likelihood)."""

    def compute_Q(
        self,
        theta: jnp.ndarray,
        data: Any,
        params: Dict[str, Any],
        current_log_ell: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:

        # Q = grad(log_ell)
        if "log_ell_prime" in kwargs and kwargs["log_ell_prime"] is not None:
            grad_ll = kwargs["log_ell_prime"]
        else:
            grad_ll = self.likelihood_fn.log_likelihood_gradient(data, params)

        return -grad_ll


class KLDivergence(SmallStepTransformation):
    """KL Divergence transformation."""

    def compute_Q(
        self,
        theta: jnp.ndarray,
        data: Any,
        params: Dict[str, Any],
        current_log_ell: jnp.ndarray,
        log_pi: jnp.ndarray = None,
        log_ell_original: jnp.ndarray = None,
        **kwargs,
    ) -> jnp.ndarray:

        if log_pi is None:
            raise ValueError("log_pi required for KLDivergence")

        # Q_i = pi(theta|D) * grad(1/ell_i)
        # pi(theta|D) = exp(log_pi)
        # grad(1/ell) = -1/ell^2 * grad(ell) = -1/ell * grad(log_ell)
        # So Q_i = - exp(log_pi) / ell_i * grad(log_ell_i)
        #        = - exp(log_pi - log_ell_i) * grad(log_ell_i)

        # log_pi (S) or (S, 1)
        # current_log_ell (S, N)
        # grad_ll (S, N, K)

        # log_pi_normalized = log_pi - jnp.max(log_pi, axis=0, keepdims=True)
        # We need pi(theta|D) normalized properly?
        # The formula usually assumes pi is the posterior.
        # Yes, let's normalize weights across S.

        log_pi = log_pi - jnp.max(log_pi, axis=0, keepdims=True)

        if "log_ell_prime" in kwargs and kwargs["log_ell_prime"] is not None:
            grad_ll = kwargs["log_ell_prime"]
        else:
            grad_ll = self.likelihood_fn.log_likelihood_gradient(data, params)

        # Expand terms for broadcasting
        # log_pi: (S,) -> (S, 1, 1)
        # current_log_ell: (S, N) -> (S, N, 1)
        # grad_ll: (S, N, K)

        log_pi_expanded = log_pi[:, jnp.newaxis, jnp.newaxis]
        while log_pi_expanded.ndim < grad_ll.ndim:
            log_pi_expanded = log_pi_expanded[..., jnp.newaxis]

        log_ell_expanded = current_log_ell[..., jnp.newaxis]
        
        # scaling factor: - exp(log_pi - log_ell)
        # (S, N, 1)
        scaling = -jnp.exp(log_pi_expanded - log_ell_expanded)
        
        # Apply scaling to gradient tree
        # grad_ll leaves are (S, N, ...)
        # scaling (S, N, 1) should broadcast against (S, N, K) or (S, N, 1)
        
        def apply_scale(g):
            # Safe broadcast check
            # if g is (S, N), we want (S, N) scaling.
            # scaling is (S, N, 1). 
            if g.ndim == 2:
                return scaling[..., 0] * g
            else:
                return scaling * g
                
        return jax.tree_util.tree_map(apply_scale, grad_ll)


class Variance(SmallStepTransformation):
    """Variance-based transformation with optional target function f.

    Formula: Q = pi * (f/ell)^2 * grad(log(f/ell))
               = pi * exp(2*log_f - 2*log_ell) * (grad_log_f - grad_log_ell)
    """

    def __init__(
        self, likelihood_fn: LikelihoodFunction, f_fn: Callable
    ):
        super().__init__(likelihood_fn)
        self.f_fn = f_fn

    def compute_Q(
        self,
        theta: Any,
        data: Any,
        params: Dict[str, Any],
        current_log_ell: jnp.ndarray,
        log_pi: jnp.ndarray = None,
        **kwargs,
    ) -> Any:
        """Compute variance-based vector field Q as PyTree.

        Args:
            theta: Current parameters as PyTree
            data: Input data
            params: Original model parameters
            current_log_ell: Current log-likelihood (S, N)
            log_pi: Log posterior/surrogate probability (S,)

        Returns:
            Q as PyTree matching theta structure
        """
        if log_pi is None:
            raise ValueError("log_pi required")

        # 1. Compute gradients of log_ell
        if "log_ell_prime" in kwargs and kwargs["log_ell_prime"] is not None:
            grad_log_ell = kwargs["log_ell_prime"]
        else:
            grad_log_ell = self.likelihood_fn.log_likelihood_gradient(data, params)

        # 2. Compute log_f and grad_log_f
        log_f = kwargs.get("log_f", None)
        grad_log_f = kwargs.get("grad_log_f", None)

        if log_f is None:
            log_f = jnp.log(self.f_fn(data, **params))

        if grad_log_f is None:
            # Compute gradient of log_f w.r.t. params using autodiff
            def log_f_sum(p):
                return jnp.sum(jnp.log(self.f_fn(data, **p)))

            grad_log_f = jax.grad(log_f_sum)(params)
            # grad_log_f is PyTree with leaves of shape (...) matching params
            # We need to expand it to (S, N, ...) for broadcasting

        # 3. Compute weights
        log_pi_expanded = log_pi[:, jnp.newaxis] if log_pi.ndim == 1 else log_pi
        log_weights = log_pi_expanded + 2.0 * log_f - 2.0 * current_log_ell
        weights = jnp.exp(log_weights)  # (S, N)

        # 4. Assemble Q = weights * (grad_log_f - grad_log_ell)
        def compute_q_leaf(g_f, g_ell):
            # g_ell is (S, N, ...) from likelihood gradient
            # g_f might be (...) from autodiff grad - need to broadcast
            if g_f.ndim < g_ell.ndim:
                # Expand g_f to match g_ell shape
                g_f_expanded = g_f
                for _ in range(g_ell.ndim - g_f.ndim):
                    g_f_expanded = jnp.expand_dims(g_f_expanded, axis=0)
                g_f_expanded = jnp.broadcast_to(g_f_expanded, g_ell.shape)
            else:
                g_f_expanded = g_f

            diff = g_f_expanded - g_ell

            # Expand weights for broadcasting: (S, N) -> (S, N, 1, 1, ...)
            w = weights
            for _ in range(diff.ndim - 2):
                w = w[..., jnp.newaxis]

            return w * diff

        Q = jax.tree_util.tree_map(compute_q_leaf, grad_log_f, grad_log_ell)
        return Q


class PMM1(SmallStepTransformation):
    """Partial Moment Matching 1 (Shift based)."""

    def compute_Q(
        self,
        theta: jnp.ndarray,
        data: Any,
        params: Dict[str, Any],
        current_log_ell: jnp.ndarray,
        log_ell_original: jnp.ndarray = None,
        **kwargs,
    ) -> jnp.ndarray:

        if log_ell_original is None:
            raise ValueError("log_ell_original required")

        log_w = -log_ell_original
        log_w = jax.lax.stop_gradient(log_w)
        weights = jnp.exp(log_w)

        moments = Transformation.compute_moments(params, weights)

        # Q = mean_w - mean (Broadcasted to theta shape)
        # We need to reconstruct Q vector from parameter dict diffs

        # Construct diff dict
        diff_params = {}
        for name, value in params.items():
            m = moments[name]
            if value.ndim == 1:
                d = -m["mean"] + m["mean_w"]  # N
                # Broadcast to (S, N)
                diff_params[name] = jnp.tile(d[jnp.newaxis, :], (value.shape[0], 1))
            else:
                d = -m["mean"][jnp.newaxis, :] + m["mean_w"]  # N x K
                # Broadcast to (S, N, K)
                diff_params[name] = jnp.tile(
                    d[jnp.newaxis, :, :], (value.shape[0], 1, 1)
                )

        # Return Q as PyTree
        Q = diff_params
        return Q


class PMM2(SmallStepTransformation):
    """Partial Moment Matching 2 (Scale + Shift)."""

    def compute_Q(
        self,
        theta: jnp.ndarray,
        data: Any,
        params: Dict[str, Any],
        current_log_ell: jnp.ndarray,
        log_ell_original: jnp.ndarray = None,
        **kwargs,
    ) -> jnp.ndarray:

        if log_ell_original is None:
            raise ValueError("log_ell_original required")

        log_w = -log_ell_original
        log_w = jax.lax.stop_gradient(log_w)
        weights = jnp.exp(log_w)

        moments = Transformation.compute_moments(params, weights)

        Q_dict = {}

        for name, value in params.items():
            m = moments[name]

            # Q = (ratio - 1)*val + (mean_w - ratio*mean)
            # ratio = sqrt(var_w / var)

            if value.ndim == 1:
                ratio = jnp.sqrt(m["var_w"] / (m["var"] + 1e-10))  # N
                term1 = (ratio - 1.0)[jnp.newaxis, :] * value[:, jnp.newaxis]
                term2 = (m["mean_w"] - ratio * m["mean"])[jnp.newaxis, :]
                Q_dict[name] = term1 + term2
            else:
                ratio = jnp.sqrt(
                    m["var_w"] / (m["var"][jnp.newaxis, :] + 1e-10)
                )  # N x K
                term1 = (ratio - 1.0)[jnp.newaxis, :, :] * value[:, jnp.newaxis, :]
                term2 = (
                    m["mean_w"][jnp.newaxis, :, :]
                    - ratio[jnp.newaxis, :, :] * m["mean"][jnp.newaxis, jnp.newaxis, :]
                )
                Q_dict[name] = term1 + term2

        Q = Q_dict
        return Q


class GlobalTransformation(Transformation):
    """Transformations that don't depend on small steps/gradients."""

    pass


class MM1(GlobalTransformation):
    def __call__(
        self,
        max_iter,
        params,
        theta,
        data,
        log_ell,
        log_pi=None,
        log_ell_original=None,
        **kwargs,
    ):
        """Apply MM1 (Moment Matching 1) transformation - shift only."""
        if log_ell_original is None:
            raise ValueError("log_ell_original is required for MM1")
        log_w = -log_ell_original
        weights = jnp.exp(log_w)
        moments = Transformation.compute_moments(params, weights)

        new_params = {}
        for name, value in params.items():
            m = moments[name]
            if value.ndim == 1:
                diff = -m["mean"] + m["mean_w"]
                new_params[name] = value[:, jnp.newaxis] + diff[jnp.newaxis, :]
            else:
                diff = -m["mean"][jnp.newaxis, :] + m["mean_w"]
                new_params[name] = value[:, jnp.newaxis, :] + diff[jnp.newaxis, :, :]

        # MM1 Jac is 0
        log_jac = jnp.zeros_like(log_w)
        # We need theta_new to return
        theta_new = new_params

        eta_weights, psis_weights, khat, log_ell_new = (
            self.compute_importance_weights_helper(
                self.likelihood_fn,
                data,
                params,
                new_params,
                log_jac,
                False,
                log_pi,
                log_ell_original=log_ell_original,
            )
        )

        predictions = self.likelihood_fn.log_likelihood(data, new_params)
        exp_predictions = jnp.exp(predictions)
        exp_log_ell_new = jnp.exp(log_ell_new)

        weight_entropy = _compute_entropy(eta_weights)
        psis_entropy = _compute_entropy(psis_weights)

        p_loo_eta = _weighted_sum(eta_weights, exp_predictions)
        p_loo_psis = _weighted_sum(psis_weights, exp_predictions)

        ll_loo_eta = _weighted_sum(eta_weights, exp_log_ell_new)
        ll_loo_psis = _weighted_sum(psis_weights, exp_log_ell_new)

        return {
            "theta_new": theta_new,
            "log_jacobian": log_jac,
            "eta_weights": eta_weights,
            "psis_weights": psis_weights,
            "khat": khat,
            "predictions": predictions,
            "log_ell_new": log_ell_new,
            "weight_entropy": weight_entropy,
            "psis_entropy": psis_entropy,
            "p_loo_eta": p_loo_eta,
            "p_loo_psis": p_loo_psis,
            "ll_loo_eta": ll_loo_eta,
            "ll_loo_psis": ll_loo_psis,
        }


class MM2(GlobalTransformation):
    """Moment Matching 2 transformation - shift and scale."""

    def __call__(
        self,
        max_iter,
        params,
        theta,
        data,
        log_ell,
        log_pi=None,
        log_ell_original=None,
        **kwargs,
    ):
        """Apply MM2 (Moment Matching 2) transformation - shift and scale."""
        if log_ell_original is None:
            raise ValueError("log_ell_original is required for MM2")
        log_w = -log_ell_original
        weights = jnp.exp(log_w)
        moments = Transformation.compute_moments(params, weights)

        new_params = {}
        log_det_jac = jnp.zeros_like(log_w)

        for name, value in params.items():
            m = moments[name]
            if value.ndim == 1:
                ratio = jnp.sqrt(m["var_w"] / (m["var"] + 1e-10))
                term1 = ratio[jnp.newaxis, :] * (value[:, jnp.newaxis] - m["mean"])
                new_params[name] = term1 + m["mean_w"][jnp.newaxis, :]
                log_det_jac += jnp.log(ratio)[jnp.newaxis, :]
            else:
                ratio = jnp.sqrt(m["var_w"] / (m["var"][jnp.newaxis, :] + 1e-10))
                term1 = ratio[jnp.newaxis, :, :] * (
                    value[:, jnp.newaxis, :] - m["mean"][jnp.newaxis, jnp.newaxis, :]
                )
                new_params[name] = term1 + m["mean_w"][jnp.newaxis, :, :]
                log_det_jac += jnp.sum(
                    jnp.log(ratio), axis=tuple(range(1, ratio.ndim))
                )[jnp.newaxis, :]

        theta_new = new_params

        eta_weights, psis_weights, khat, log_ell_new = (
            self.compute_importance_weights_helper(
                self.likelihood_fn,
                data,
                params,
                new_params,
                log_det_jac,
                False,
                log_pi,
                log_ell_original=log_ell_original,
            )
        )

        predictions = self.likelihood_fn.log_likelihood(data, new_params)
        exp_predictions = jnp.exp(predictions)
        exp_log_ell_new = jnp.exp(log_ell_new)

        weight_entropy = _compute_entropy(eta_weights)
        psis_entropy = _compute_entropy(psis_weights)

        p_loo_eta = _weighted_sum(eta_weights, exp_predictions)
        p_loo_psis = _weighted_sum(psis_weights, exp_predictions)

        ll_loo_eta = _weighted_sum(eta_weights, exp_log_ell_new)
        ll_loo_psis = _weighted_sum(psis_weights, exp_log_ell_new)

        return {
            "theta_new": theta_new,
            "log_jacobian": log_det_jac,
            "eta_weights": eta_weights,
            "psis_weights": psis_weights,
            "khat": khat,
            "predictions": predictions,
            "log_ell_new": log_ell_new,
            "weight_entropy": weight_entropy,
            "psis_entropy": psis_entropy,
            "p_loo_eta": p_loo_eta,
            "p_loo_psis": p_loo_psis,
            "ll_loo_eta": ll_loo_eta,
            "ll_loo_psis": ll_loo_psis,
        }


class AutoDiffLikelihoodMixin(LikelihoodFunction):
    """Mixin to provide automatic differentiation for LikelihoodFunction.

    This mixin implements log_likelihood_gradient and log_likelihood_hessian_diag
    using JAX automatic differentiation.
    """

    
    def log_likelihood_gradient(self, data: Any, params: Dict[str, Any]) -> Any:
        """Compute gradient of log-likelihood w.r.t. parameters using autodiff."""
        # params is a PyTree (Dict)
        
        # We need a function that maps params -> log_likelihood (N,)
        # and we want derivatives w.r.t params. 
        # Output should be PyTree of shapes (S, N, K_i) matching params structure.

        def batch_ll(p_tree):
             # p_tree (PyTree) where each leaf is (S, ...)
             # log_likelihood expects (S, ..) input and returns (S, N)
             ll = self.log_likelihood(data, p_tree)
             return ll

        # If params has shape (S, ...), and LL is (S, N).
        # We want grad per (S, N). 
        # jax.grad sums output. jax.jacobian returns huge matrix.
        
        # We can map over S, N? Or just S?
        # params leafs are (S, ...).
        # We can VMAP over S.
        
        def single_sample_ll(p_sample):
             # p_sample has leaves (...) (no S dim)
             # Expand trace to satisfy model requirements? 
             # Model usually handles broadcasting.
             # But let's assume model can handle single sample input -> (N,) output.
             # We might need to add singleton dimension if model STRICTLY expects (S,...)
             
             # Heuristic: Add singleton dim to leaves
             p_exp = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), p_sample)
             ll = self.log_likelihood(data, p_exp) # (1, N)
             return jnp.squeeze(ll, 0) # (N,)

        # Jacobian of (N,) -> params_structure
        # jacrev is better for N > K usually? Or N large? 
        # params size K is smallish?
        # N is data size (e.g. 1000).
        # jacobian will be tree of (N, K_i...)
        
        jac_fn = jax.jacrev(single_sample_ll)
        
        # Vmap over S
        # params is (S, ...)
        # We want input to vmap to be the tree of params slicing S dimension.
        # jax.vmap default in_axes=0 handles PyTrees correctly if all leaves have leading dim S.
        
        gradients = jax.vmap(jac_fn)(params)
        # gradients is PyTree where leaves are (S, N, K_i...)
        
        return gradients

    def log_likelihood_hessian_diag(
        self, data: Any, params: Dict[str, Any]
    ) -> Any:
        """Compute diagonal of Hessian of log-likelihood w.r.t. parameters using autodiff.

        Uses forward-over-reverse mode autodiff for O(K) complexity instead of O(K^2)
        from computing the full Hessian.
        """
        # params: PyTree (S, ...)
        # Output: PyTree (S, N, ...) matches params structure

        n_data = jax.tree_util.tree_leaves(data)[0].shape[0]

        def single_sample_hess_diag(p_sample):
            # p_sample: PyTree (...)

            def point_ll(p, i):
                p_exp = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), p)
                ll = self.log_likelihood(data, p_exp)  # (1, N)
                return ll[0, i]

            # Flatten parameters for efficient diagonal Hessian computation
            flat_p, unravel = jax.flatten_util.ravel_pytree(p_sample)
            K = flat_p.shape[0]

            def flat_point_ll(fp, i):
                p = unravel(fp)
                return point_ll(p, i)

            # Compute diagonal Hessian using forward-over-reverse mode
            # This is O(K) instead of O(K^2) for full Hessian
            def diag_hess_single_point(i):
                # Gradient function
                grad_fn = jax.grad(lambda fp: flat_point_ll(fp, i))

                # For diagonal, we compute d/d(theta_k) of grad_k
                # Using forward mode on reverse mode gradient
                def hvp_diag(fp):
                    # Compute gradient
                    g = grad_fn(fp)
                    # For diagonal, compute JVP with each basis vector
                    # But we can be smarter: use vmap over basis vectors
                    def single_diag(k):
                        tangent = jnp.zeros_like(fp).at[k].set(1.0)
                        _, jvp_val = jax.jvp(grad_fn, (fp,), (tangent,))
                        return jvp_val[k]

                    return jax.vmap(single_diag)(jnp.arange(K))

                return hvp_diag(flat_p)

            # vmap over data points N
            h_flat_N = jax.vmap(diag_hess_single_point)(jnp.arange(n_data))  # (N, K)

            # Unravel back to PyTree
            h_tree_N = jax.vmap(unravel)(h_flat_N)
            return h_tree_N

        # Output shape: PyTree (S, N, ...)
        hessians = jax.vmap(single_sample_hess_diag)(params)
        return hessians


class AdaptiveImportanceSampler:
    """Generalized Adaptive Importance Sampling for Leave-One-Out Cross-Validation.

    This class implements adaptive importance sampling transformations that can work
    with any likelihood function. It provides several transformation strategies
    for generating importance weights.
    """

    def __init__(
        self,
        likelihood_fn: LikelihoodFunction,
        prior_log_prob_fn: Optional[Callable] = None,
        surrogate_log_prob_fn: Optional[Callable] = None,
    ):
        """Initialize the adaptive importance sampler.

        Args:
            likelihood_fn: Likelihood function implementing LikelihoodFunction interface
            prior_log_prob_fn: Function computing log prior probability
            surrogate_log_prob_fn: Function computing log surrogate probability (for variational)

        Raises:
            TypeError: If likelihood_fn is not a LikelihoodFunction instance.
        """
        if not isinstance(likelihood_fn, LikelihoodFunction):
            raise TypeError(
                f"likelihood_fn must be a LikelihoodFunction instance, got {type(likelihood_fn)}"
            )
        self.likelihood_fn = likelihood_fn
        self.prior_log_prob_fn = prior_log_prob_fn
        self.surrogate_log_prob_fn = surrogate_log_prob_fn

    def entropy(self, probs: jnp.ndarray) -> jnp.ndarray:
        """Compute entropy of probability distributions."""
        return _compute_entropy(probs)

    # Default transformation order by computational complexity (least to most expensive)
    DEFAULT_TRANSFORMATION_ORDER = [
        'identity',   # No computation - just standard PSIS-LOO
        'mm1',        # Global moment matching (shift only)
        'mm2',        # Global moment matching (shift + scale)
        'pmm1',       # Partial moment matching (shift)
        'pmm2',       # Partial moment matching (shift + scale)
        'll',         # Likelihood descent (requires gradient)
        'kl',         # KL divergence (requires gradient + posterior weights)
        'var',        # Variance-based (requires gradient + Hessian + f_fn)
    ]

    def adaptive_is_loo(
        self,
        data: Any,
        params: Dict[str, Any],
        n_sweeps: int = 4,
        max_iter: int = 10,
        initial_step_size: float = 1.0,
        variational: bool = False,
        verbose: bool = False,
        f_fn: Optional[Callable] = None,
        rhos: jnp.ndarray = None,
        transformations: List[str] = None,
        try_all_transformations: bool = False,
        khat_threshold: float = 0.7,
    ):
        """Perform Adaptive Importance Sampling for Leave-One-Out CV.

        By default, transformations are tried in order of computational complexity
        (identity < mm1/mm2 < pmm1/pmm2 < ll < kl < var). Once a data point achieves
        khat < khat_threshold, it is considered "adapted" and subsequent transformations
        are skipped for that point (unless try_all_transformations=True).

        Args:
            data: Input data
            params: Initial model parameters
            n_sweeps: Number of grid search sweeps
            max_iter: Maximum iterations for transformations (used by iterative ones)
            initial_step_size: Initial step size for search
            variational: Whether using variational approximation
            verbose: Verbosity flag
            f_fn: Optional function returning f for Variance transformation.
                  Signature: f_fn(data, **params) -> (S, N)
            rhos: Optional manual grid of step sizes. If None, generated from n_sweeps.
            transformations: Optional list of transformation names to run.
                             Available: 'll', 'kl', 'var', 'pmm1', 'pmm2', 'mm1', 'mm2', 'identity'.
                             If None, uses DEFAULT_TRANSFORMATION_ORDER.
            try_all_transformations: If False (default), skip transformations for data
                             points that have already achieved khat < khat_threshold.
                             If True, try all transformations for all points.
            khat_threshold: Threshold for considering a point "adapted" (default 0.7).
                           Points with khat below this threshold won't be processed
                           by subsequent transformations unless try_all_transformations=True.

        Returns:
            Dictionary of best results

        Raises:
            ValueError: If data or params are empty or have invalid shapes.
        """
        # Input validation
        if not params:
            raise ValueError("params cannot be empty")

        # Validate params has array-like leaves
        param_leaves = jax.tree_util.tree_leaves(params)
        if not param_leaves:
            raise ValueError("params must contain at least one parameter array")

        first_param = param_leaves[0]
        if first_param.ndim < 1:
            raise ValueError("Parameter arrays must have at least 1 dimension (samples)")
        n_samples = first_param.shape[0]
        if n_samples == 0:
            raise ValueError("Parameter arrays must have at least 1 sample")

        # Initial computations
        log_ell = self.likelihood_fn.log_likelihood(data, params)  # (S, N)
        
        # Use params directly as theta (PyTree)
        theta = params 

        # Precompute potentially expensive derivatives once
        log_ell_prime = self.likelihood_fn.log_likelihood_gradient(
            data, params
        )

        # Hessian diagonal - generally used for Variance transform
        log_ell_doubleprime = self.likelihood_fn.log_likelihood_hessian_diag(
            data, params
        )
        
        # Determine log_pi for transformations
        if variational and self.surrogate_log_prob_fn is not None:
            log_pi = self.surrogate_log_prob_fn(params)
            
            # gradient of surrogate log prob w.r.t params
            # jax.grad returns structure matching params
            grad_log_pi = jax.grad(lambda p: jnp.sum(self.surrogate_log_prob_fn(p)))(
                params
            )
        else:
            if hasattr(self, "target_log_prob_fn") and self.target_log_prob_fn:
                log_pi = self.target_log_prob_fn(data, **params)
                grad_log_pi = None
            elif hasattr(self, "model") and hasattr(self.model, "unormalized_log_prob"):
                log_pi = self.model.unormalized_log_prob(data, **params)
                grad_log_pi = None
            else:
                # Fallback
                log_prior = (
                    self.prior_log_prob_fn(params)
                    if self.prior_log_prob_fn
                    else jnp.zeros(log_ell.shape[0])
                )
                log_pi = jnp.sum(log_ell, axis=1) + log_prior
                grad_log_pi = None

        # log_pi should be (S,) at this point

        # Standard deviation of parameters (for standardization)
        # theta is a PyTree (dict). We want std per leaf.
        # theta shape (S, ...). std over S (axis 0).
        
        def compute_std(x):
            s = jnp.std(x, axis=0) # (...,)
            return jnp.where(s < 1e-6, 1.0, s)

        theta_std = jax.tree_util.tree_map(compute_std, theta)

        # theta_expanded
        # We need to broadcast theta (S, ...) to (S, 1, ...) for "N" dimension
        # Transformations expect theta as (S, N, ...) sometimes?
        # SmallStepTransformation normalizes Q (S, N, K) with theta_std (K).
        # And updates theta (S, 1, K) + h(1, N, 1)*Q(S, N, K) -> theta_new (S, N, K)
        
        # So yes, we should expand theta to have the N dimension.
        # theta leaves are (S, K_i...) or (S,).
        # We want (S, 1, K_i...)
        
        def expand_dims(x):
            # x is (S, ...)
            # We want (S, 1, K...). 
            # If x is (S,) -> (S, 1, 1)
            # If x is (S, D) -> (S, 1, D)
            if x.ndim == 1:
                return x[:, jnp.newaxis, jnp.newaxis]
            else:
                return jnp.expand_dims(x, axis=1)
            
        theta_expanded = jax.tree_util.tree_map(expand_dims, theta)

        # Define search grid for rho (step size factor)
        if rhos is None:
            rhos = jnp.logspace(-2, 1, 7) * initial_step_size
        elif not isinstance(rhos, jnp.ndarray):
            rhos = jnp.array(rhos)

        # Instantiate transformations
        all_transforms = {
            "ll": LikelihoodDescent(self.likelihood_fn),
            "likelihood_descent": LikelihoodDescent(self.likelihood_fn),
            "kl": KLDivergence(self.likelihood_fn),
            "kl_divergence": KLDivergence(self.likelihood_fn),
            "pmm1": PMM1(self.likelihood_fn),
            "pmm2": PMM2(self.likelihood_fn),
            "mm1": MM1(self.likelihood_fn),
            "mm2": MM2(self.likelihood_fn),
        }

        if f_fn is not None:
             all_transforms["var"] = Variance(self.likelihood_fn, f_fn=f_fn)
             all_transforms["variance_based"] = Variance(self.likelihood_fn, f_fn=f_fn)

        # Determine transformation order
        if transformations is not None:
            # User-specified order
            transform_order = [t for t in transformations if t in all_transforms or t == 'identity']
        else:
            # Default order by computational complexity
            # Filter to only include transforms that are available
            transform_order = [
                t for t in self.DEFAULT_TRANSFORMATION_ORDER
                if t in all_transforms or t == 'identity'
            ]
            # Exclude 'var' from default if f_fn not provided
            if f_fn is None and 'var' in transform_order:
                transform_order.remove('var')

        # Build transforms dict preserving order
        transforms = {}
        for name in transform_order:
            if name != 'identity' and name in all_transforms:
                transforms[name] = all_transforms[name]
        # 'identity' is handled separately below

        # Initialize best results
        n_data = log_ell.shape[1]
        best_metrics = {
            "khat": jnp.full(n_data, jnp.inf),  # (N,)
            "p_loo_eta": jnp.zeros(n_data),
            "p_loo_psis": jnp.zeros(n_data),
            "ll_loo_eta": jnp.full(n_data, -jnp.inf),
            "ll_loo_psis": jnp.full(n_data, -jnp.inf),
        }

        # Track which points are considered "adapted" (khat < threshold)
        # Once adapted, skip further transformations for that point (unless try_all)
        adapted_mask = jnp.zeros(n_data, dtype=bool)

        results = {}

        # Handle 'identity' sweep explicitly if in transform_order
        if "identity" in transform_order:
            if verbose:
                print("Running identity...")
            # Compute identity importance weights (standard PSIS-LOO)
            # log_eta = -log_ell
            log_eta_weights = (-log_ell).astype(jnp.float64)

            # Use JIT-compiled normalization
            eta_weights = _normalize_weights(log_eta_weights)

            # PSIS smoothing
            psis_log_weights, khat = psis.psislw(
                log_eta_weights - jnp.max(log_eta_weights, axis=0, keepdims=True)
            )
            psis_weights = _normalize_weights(psis_log_weights)

            log_ell_new = log_ell
            log_jac_identity = jnp.zeros_like(log_ell)

            predictions = log_ell  # Identity has same predictions
            exp_predictions = jnp.exp(predictions)
            exp_log_ell_new = jnp.exp(log_ell_new)

            # Use JIT-compiled functions
            weight_entropy = _compute_entropy(eta_weights)
            psis_entropy = _compute_entropy(psis_weights)

            p_loo_eta = _weighted_sum(eta_weights, exp_predictions)
            p_loo_psis = _weighted_sum(psis_weights, exp_predictions)

            ll_loo_eta = _weighted_sum(eta_weights, exp_log_ell_new)
            ll_loo_psis = _weighted_sum(psis_weights, exp_log_ell_new)

            res_identity = {
                "theta_new": theta_expanded,
                "log_jacobian": log_jac_identity,
                "eta_weights": eta_weights,
                "psis_weights": psis_weights,
                "khat": khat,
                "predictions": predictions,
                "log_ell_new": log_ell_new,
                "weight_entropy": weight_entropy,
                "psis_entropy": psis_entropy,
                "p_loo_eta": p_loo_eta,
                "p_loo_psis": p_loo_psis,
                "ll_loo_eta": ll_loo_eta,
                "ll_loo_psis": ll_loo_psis,
            }
            results["identity"] = res_identity

            # Update best metrics
            idx = khat < best_metrics["khat"]
            best_metrics["khat"] = jnp.where(idx, khat, best_metrics["khat"])
            best_metrics["p_loo_eta"] = jnp.where(
                idx, p_loo_eta, best_metrics["p_loo_eta"]
            )
            best_metrics["p_loo_psis"] = jnp.where(
                idx, p_loo_psis, best_metrics["p_loo_psis"]
            )
            best_metrics["ll_loo_eta"] = jnp.where(
                idx, ll_loo_eta, best_metrics["ll_loo_eta"]
            )
            best_metrics["ll_loo_psis"] = jnp.where(
                idx, ll_loo_psis, best_metrics["ll_loo_psis"]
            )

            # Update adapted mask - points with khat below threshold are considered adapted
            if not try_all_transformations:
                newly_adapted = khat < khat_threshold
                adapted_mask = adapted_mask | newly_adapted
                if verbose:
                    n_adapted = jnp.sum(adapted_mask).item()
                    print(f"  {n_adapted}/{n_data} points adapted (khat < {khat_threshold})")

        # Common kwargs for all transforms
        common_kwargs = {
            "log_ell": log_ell,
            "log_ell_prime": log_ell_prime,
            "log_ell_doubleprime": log_ell_doubleprime,
            "log_pi": log_pi,
            "grad_log_pi": grad_log_pi,
            "log_ell_original": log_ell,
            "theta_std": theta_std,
            "variational": variational,
            "surrogate_log_prob_fn": self.surrogate_log_prob_fn,
        }

        # Run sweeps in order of computational complexity
        for name, transform in transforms.items():
            # Early exit if all points are adapted and we're not forcing all transformations
            if not try_all_transformations and jnp.all(adapted_mask):
                if verbose:
                    print(f"All {n_data} points adapted, skipping remaining transformations")
                break

            if verbose:
                if not try_all_transformations:
                    n_remaining = n_data - jnp.sum(adapted_mask).item()
                    print(f"Running {name}... ({n_remaining} points remaining)")
                else:
                    print(f"Running {name}...")

            # Determine rhos for this transform
            current_rhos = rhos
            if name in ["mm1", "mm2"]:
                current_rhos = [
                    1.0
                ]  # These don't use step size in the same way (or are global)

            # For each rho
            for rho in current_rhos:
                try:
                    res = transform(
                        max_iter=max_iter,
                        params=params,
                        theta=theta_expanded,
                        data=data,
                        hbar=rho,
                        **common_kwargs,
                    )

                    shorthand_map = {
                        "likelihood_descent": "ll",
                        "kl_divergence": "kl",
                        "variance_based": "var",
                        "pmm1": "pmm1",
                        "pmm2": "pmm2",
                        "mm1": "mm1",
                        "mm2": "mm2",
                    }
                    shorthand = shorthand_map.get(name, name)
                    key = f"{shorthand}"
                    if len(current_rhos) > 1:
                        key = f"{shorthand}_rho{rho:.2e}"

                    results[key] = res

                    # Update best metrics only for non-adapted points (or all if try_all)
                    khat_safe = jnp.where(jnp.isnan(res["khat"]), jnp.inf, res["khat"])

                    # Only consider improvement for points not already adapted
                    if try_all_transformations:
                        improvement = khat_safe < best_metrics["khat"]
                    else:
                        improvement = (khat_safe < best_metrics["khat"]) & (~adapted_mask)

                    best_metrics["khat"] = jnp.where(
                        improvement, khat_safe, best_metrics["khat"]
                    )
                    best_metrics["p_loo_eta"] = jnp.where(
                        improvement, res["p_loo_eta"], best_metrics["p_loo_eta"]
                    )
                    best_metrics["p_loo_psis"] = jnp.where(
                        improvement, res["p_loo_psis"], best_metrics["p_loo_psis"]
                    )
                    best_metrics["ll_loo_eta"] = jnp.where(
                        improvement, res["ll_loo_eta"], best_metrics["ll_loo_eta"]
                    )
                    best_metrics["ll_loo_psis"] = jnp.where(
                        improvement, res["ll_loo_psis"], best_metrics["ll_loo_psis"]
                    )

                    # Update adapted mask after this transformation
                    if not try_all_transformations:
                        newly_adapted = best_metrics["khat"] < khat_threshold
                        adapted_mask = adapted_mask | newly_adapted

                except Exception as e:
                    if verbose:
                        print(f"Failed {name} rho={rho}: {e}")
                    continue

        results["best"] = best_metrics
        return results


# Example implementations for common likelihood functions


class LogisticRegressionLikelihood(LikelihoodFunction):
    """Likelihood function for logistic regression."""

    def __init__(self, dtype=jnp.float64):
        self.dtype = dtype

    def log_likelihood(
        self, data: Dict[str, Any], params: Dict[str, Any]
    ) -> jnp.ndarray:
        """Compute log-likelihood for logistic regression."""
        X = jnp.asarray(data["X"], dtype=self.dtype)  # (n_data, n_features)
        y = jnp.asarray(data["y"], dtype=self.dtype)  # (n_data,)

        beta = params["beta"]  # (n_samples, n_features) or (S, N, F)
        intercept = params["intercept"]  # (n_samples,) or (S, N) or (S, N, 1)

        # Squeeze trailing singleton from intercept if present (from transformations)
        if intercept.ndim > 1 and intercept.shape[-1] == 1:
            intercept = jnp.squeeze(intercept, axis=-1)

        # Linear predictor: X @ beta.T + intercept
        if beta.ndim == 3:
            mu = jnp.einsum("df,sdf->sd", X, beta) + intercept
        else:
            mu = jnp.einsum("df,sf->sd", X, beta) + intercept[..., jnp.newaxis]

        # Sigmoid and log-likelihood
        sigma = jax.nn.sigmoid(mu)
        log_lik = y[jnp.newaxis, :] * jnp.log(sigma + 1e-10) + (
            1 - y[jnp.newaxis, :]
        ) * jnp.log(1 - sigma + 1e-10)

        return log_lik

    def log_likelihood_gradient(
        self, data: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute gradient of log-likelihood w.r.t. parameters.

        Returns:
            PyTree matching params structure with shapes (S, N, ...).
        """
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]

        # Squeeze trailing singleton from intercept if present (from transformations)
        if intercept.ndim > 1 and intercept.shape[-1] == 1:
            intercept = jnp.squeeze(intercept, axis=-1)

        if beta.ndim == 3:
            mu = jnp.einsum("df,sdf->sd", X, beta) + intercept
        else:
            mu = jnp.einsum("df,sf->sd", X, beta) + intercept[..., jnp.newaxis]
        sigma = jax.nn.sigmoid(mu)

        # Gradient w.r.t. linear predictor
        grad_mu = y[jnp.newaxis, :] - sigma  # (S, N)

        # Gradient w.r.t. beta: (S, N, n_features)
        grad_beta = jnp.einsum("df,sd->sdf", X, grad_mu)

        # Gradient w.r.t. intercept: (S, N)
        grad_intercept = grad_mu

        return {"beta": grad_beta, "intercept": grad_intercept}

    def log_likelihood_hessian_diag(
        self, data: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute diagonal of Hessian of log-likelihood.

        Returns:
            PyTree matching params structure with shapes (S, N, ...).
        """
        X = jnp.asarray(data["X"], dtype=self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]

        # Squeeze trailing singleton from intercept if present (from transformations)
        if intercept.ndim > 1 and intercept.shape[-1] == 1:
            intercept = jnp.squeeze(intercept, axis=-1)

        if beta.ndim == 3:
            mu = jnp.einsum("df,sdf->sd", X, beta) + intercept
        else:
            mu = jnp.einsum("df,sf->sd", X, beta) + intercept[..., jnp.newaxis]
        sigma = jax.nn.sigmoid(mu)

        # Hessian diagonal w.r.t. linear predictor
        hess_diag_mu = -sigma * (1 - sigma)  # (S, N)

        # Hessian diagonal w.r.t. beta: (S, N, n_features)
        hess_diag_beta = jnp.einsum("df,sd->sdf", X**2, hess_diag_mu)

        # Hessian diagonal w.r.t. intercept: (S, N)
        hess_diag_intercept = hess_diag_mu

        return {"beta": hess_diag_beta, "intercept": hess_diag_intercept}


class LinearRegressionLikelihood(LikelihoodFunction):
    """Likelihood function for linear regression with Gaussian errors."""

    def __init__(self, dtype=jnp.float32):
        self.dtype = dtype

    def log_likelihood(
        self, data: Dict[str, Any], params: Dict[str, Any]
    ) -> jnp.ndarray:
        """Compute log-likelihood for linear regression."""
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]
        log_sigma = params.get(
            "log_sigma", jnp.zeros((beta.shape[0],))
        )  # Log noise std

        # Predictions
        if beta.ndim == 3:
            mu = jnp.einsum("df,sdf->sd", X, beta) + intercept
        else:
            mu = jnp.einsum("df,sf->sd", X, beta) + intercept[..., jnp.newaxis]

        sigma = jnp.exp(log_sigma)
        if log_sigma.ndim == 1:
            sigma = sigma[:, jnp.newaxis]

        # Gaussian log-likelihood
        residuals = y[jnp.newaxis, :] - mu
        log_lik = (
            -0.5 * jnp.log(2 * jnp.pi)
            - log_sigma[..., jnp.newaxis]
            - 0.5 * (residuals / sigma) ** 2
        )

        return log_lik

    def log_likelihood_gradient(
        self, data: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute gradient of log-likelihood.

        Returns:
            PyTree matching params structure with shapes (S, N, ...).
        """
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]
        log_sigma = params.get("log_sigma", jnp.zeros((beta.shape[0],)))

        if beta.ndim == 3:
            mu_lin = jnp.einsum("df,sdf->sd", X, beta)
        else:
            mu_lin = jnp.einsum("df,sf->sd", X, beta)

        # Consistent broadcasting for intercept and sigma
        intercept_bc = intercept[:, jnp.newaxis] if intercept.ndim == 1 else intercept
        log_sigma_bc = log_sigma[:, jnp.newaxis] if log_sigma.ndim == 1 else log_sigma

        mu = mu_lin + jnp.broadcast_to(intercept_bc, mu_lin.shape)
        sigma = jnp.exp(jnp.broadcast_to(log_sigma_bc, mu_lin.shape))
        residuals = y[jnp.newaxis, :] - mu

        # Gradients as PyTree
        grad_beta = jnp.einsum("df,sd->sdf", X, residuals / sigma**2)
        grad_intercept = residuals / sigma**2  # (S, N)
        grad_log_sigma = -1 + (residuals / sigma) ** 2  # (S, N)

        result = {"beta": grad_beta, "intercept": grad_intercept}
        if "log_sigma" in params:
            result["log_sigma"] = grad_log_sigma
        return result

    def log_likelihood_hessian_diag(
        self, data: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute diagonal of Hessian.

        Returns:
            PyTree matching params structure with shapes (S, N, ...).
        """
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]
        log_sigma = params.get("log_sigma", jnp.zeros((beta.shape[0],)))

        if beta.ndim == 3:
            mu_lin = jnp.einsum("df,sdf->sd", X, beta)
        else:
            mu_lin = jnp.einsum("df,sf->sd", X, beta)

        # Consistent broadcasting for intercept and sigma
        intercept_bc = intercept[:, jnp.newaxis] if intercept.ndim == 1 else intercept
        log_sigma_bc = log_sigma[:, jnp.newaxis] if log_sigma.ndim == 1 else log_sigma

        mu = mu_lin + jnp.broadcast_to(intercept_bc, mu_lin.shape)
        sigma = jnp.exp(jnp.broadcast_to(log_sigma_bc, mu_lin.shape))
        residuals = y[jnp.newaxis, :] - mu
        sigma_sq = sigma**2

        # Hessian diagonals as PyTree
        hess_diag_beta = jnp.einsum("df,sd->sdf", -(X**2), 1.0 / sigma_sq)
        hess_diag_intercept = -1.0 / sigma_sq  # (S, N)
        hess_diag_log_sigma = -2.0 * (residuals / sigma) ** 2  # (S, N)

        result = {"beta": hess_diag_beta, "intercept": hess_diag_intercept}
        if "log_sigma" in params:
            result["log_sigma"] = hess_diag_log_sigma
        return result

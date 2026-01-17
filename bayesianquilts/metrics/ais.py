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
    def log_likelihood_hessian_diag(
        self, data: Any, params: Dict[str, Any]
    ) -> jnp.ndarray:
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

            if value.ndim == 1:  # Shape S
                # Broadcast value to S x N
                v_expanded = value[:, jnp.newaxis]  # S x 1

                # Weighted mean: sum(w * v, axis=0) -> N
                mean_w = jnp.sum(w_norm * v_expanded, axis=0)
                mean = jnp.mean(value, axis=0)  # Scalar

                # Expand mean for variance calc: N
                mean_expanded_w = mean_w

                # Weighted variance
                var_w = jnp.sum(w_norm * (v_expanded - mean_expanded_w) ** 2, axis=0)
                # Unweighted variance
                var = jnp.mean((value - mean) ** 2, axis=0)

            else:  # Shape S x K or S x P1...
                # Broadcast weights to matches value dimensions
                # value: (S, P...) -> v_expanded (S, 1, P...)
                v_expanded = value[:, jnp.newaxis, ...]

                # w_norm: (S, N) -> w_expanded (S, N, 1...) to match v_expanded rank
                w_expanded = w_norm
                while w_expanded.ndim < v_expanded.ndim:
                    w_expanded = w_expanded[..., jnp.newaxis]

                # Weighted mean: sum(w * v, axis=0) -> N x P...
                mean_w = jnp.sum(w_expanded * v_expanded, axis=0)
                mean = jnp.mean(value, axis=0)  # P...

                # Weighted variance: sum(w * (v - mean)^2, axis=0) -> N x P...
                # Note: mean_w is (N, P...), v_expanded is (S, 1, P...)
                # Broadcasting (S, 1, P...) with (N, P...) -> (S, N, P...)
                # Need to be careful. v_expanded is (S, 1, P...), mean_w is (N, P...)
                # (S, 1, P) - (N, P) -> (S, N, P) check
                var_w = jnp.sum(w_expanded * (v_expanded - mean_w) ** 2, axis=0)

                # Unweighted variance
                var = jnp.mean((value - mean) ** 2, axis=0)

            moments[name] = {"mean": mean, "mean_w": mean_w, "var": var, "var_w": var_w}

        return moments

    def _compute_importance_weights(
        self,
        data: Any,
        params_original: Dict[str, Any],
        params_transformed: Dict[str, Any],
        log_jacobian: jnp.ndarray,
        variational: bool,
        log_pi_original: jnp.ndarray,
        log_ell_original: jnp.ndarray = None,
    ) -> Tuple[jnp.ndarray, ...]:
        """Compute importance weights."""
        # Placeholder mainly, but helper method below does the work.
        pass

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

        # Copied logic from _compute_importance_weights
        log_ell_new = likelihood_fn.log_likelihood(data, params_transformed)

        if variational and surrogate_log_prob_fn is not None:
            log_pi_trans = surrogate_log_prob_fn(params_transformed)
            delta_log_pi = log_pi_trans - log_pi_original[:, jnp.newaxis]
            delta_log_pi = delta_log_pi - jnp.max(delta_log_pi, axis=0, keepdims=True)
        else:
            if log_ell_original is not None:
                delta_log_pi = log_ell_original - log_ell_new
            else:
                delta_log_pi = jnp.zeros_like(log_ell_new)

        log_eta_weights = delta_log_pi + log_jacobian
        log_eta_weights = log_eta_weights - jnp.max(
            log_eta_weights, axis=0, keepdims=True
        )

        psis_weights, khat = nppsis.psislw(log_eta_weights)

        eta_weights = jnp.exp(log_eta_weights)
        eta_weights = eta_weights / jnp.sum(eta_weights, axis=0, keepdims=True)

        psis_weights = jnp.exp(psis_weights)
        psis_weights = psis_weights / jnp.sum(psis_weights, axis=0, keepdims=True)

        return eta_weights, psis_weights, khat, log_ell_new

    def entropy(self, weights):
        return -jnp.sum(weights * jnp.log(weights + 1e-12), axis=0)


class SmallStepTransformation(Transformation):
    """Base class for transformations of the form T(theta) = theta + h * Q(theta)."""

    @abstractmethod
    def compute_Q(
        self,
        theta: jnp.ndarray,
        data: Any,
        params: Dict[str, Any],
        current_log_ell: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        """Compute the vector field Q(theta)."""
        pass

    def compute_divergence_Q(
        self,
        theta: jnp.ndarray,
        data: Any,
        params: Dict[str, Any],
        current_log_ell: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        """Compute divergence of Q using generic autodiff (Trace of Jacobian)."""

        # div(Q) = sum_k dQ_k/dtheta_k
        # We use jax.jvp looping over k basis vectors.
        # This allows computing divergence for any compute_Q without
        # needing analytical derivation, and respects batching (S, N).

        K = theta.shape[-1]
        divergence = jnp.zeros(theta.shape[:-1], dtype=theta.dtype)  # (S, N)

        def func(t):
            return self.compute_Q(t, data, params, current_log_ell, **kwargs)

        # Identity matrix for basis vectors
        eye = jnp.eye(K, dtype=theta.dtype)

        for k in range(K):
            # Basis vector e_k broadcast to (S, N, K)
            # eye[k] is (K,). We need (S, N, K)
            v_k = jnp.zeros_like(theta)
            v_k = v_k + eye[k]  # Broadcasting adds (K,) to (S, N, K) last dim matches

            # JVP: directional derivative of Q along e_k
            # primals_out is Q(theta), tangents_out is (nabla Q) . e_k = dQ/dtheta_k (vector)
            _, tangent_out = jax.jvp(func, (theta,), (v_k,))

            # We want dQ_k / dtheta_k which is k-th component of tangent_out
            divergence = divergence + tangent_out[..., k]

        return divergence

    def normalize_vector_field(
        self, Q: jnp.ndarray, theta_std: jnp.ndarray = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Normalize the vector field Q.

        Args:
            Q: Vector field (S, N, K)
            theta_std: Parameter standard deviations (optional)

        Returns:
            Q_standardized: Standardized Q
            Q_norm_max: Max norm scalar (or per-sample)
        """
        if theta_std is not None:
            # theta_std (K) -> expand to match Q
            theta_std_expanded = theta_std
            # Q is usually (S, N, K). theta_std usually (K) or (1, K)
            # We want (1, 1, K)
            while theta_std_expanded.ndim < Q.ndim:
                theta_std_expanded = theta_std_expanded[jnp.newaxis, ...]

            Q_standardized = Q / (theta_std_expanded + 1e-6)
        else:
            Q_standardized = Q

        # Norm over K dims (typically last dim)
        # Use range(2, Q.ndim) for (S, N, K1, ...)
        # Actually Q is usually (S, N, K) rank 3. range(2, 3) -> (2,)

        reduction_axes = tuple(range(2, Q.ndim))
        if len(reduction_axes) == 0:
            # Fallback if Q is lower rank? Shouldn't happen given logic.
            reduction_axes = (-1,)

        Q_norm = jnp.max(
            jnp.abs(Q_standardized), axis=reduction_axes, keepdims=True
        )  # (S, N, 1)
        # Max over samples S only. Resulting shape (1, N, 1)
        Q_norm_max = jnp.max(Q_norm, axis=0, keepdims=True)

        # Force shape to (1, N) for broadcasting with (S, N)
        if Q_norm_max.ndim >= 2:
            Q_norm_max = Q_norm_max.reshape(1, -1)

        return Q_standardized, Q_norm_max

    def __call__(
        self,
        max_iter: int,
        params: Dict[str, Any],
        theta: jnp.ndarray,
        data: Any,
        log_ell: jnp.ndarray,
        hbar: float = 1.0,
        theta_std: jnp.ndarray = None,
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

        # 2. Normalize and determine step size h
        Q_standardized, Q_norm_max = self.normalize_vector_field(Q, theta_std)

        # h = rho / norm(Q), where rho is passed as hbar
        # h should be (1, N) or broadcastable to (S, N)
        h = hbar / (Q_norm_max + 1e-8)

        # 3. Step
        # h is (1, N). Q is (S, N, K).
        # Need h as (1, N, 1) to multiply Q
        h_expanded = h[..., jnp.newaxis]
        theta_new = theta + h_expanded * Q

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

        # log|J| ~ log|1 + h * div(Q)|
        # h is (1, N). div_Q is (S, N).
        # Should broadcast fine: (1, N) * (S, N) -> (S, N)
        log_jac = jnp.log(jnp.abs(1.0 + h * div_Q))

        if log_jac.ndim > 2:
            log_jac = jnp.squeeze(log_jac)

        # Reconstruct params to compute weights
        params_new = {}
        # theta_new is (S, N, K)
        # We need to reconstruct.
        # This part is model specific... extracting K back to dict structure.
        # For now assuming we can loop over N efficiently or vmap?

        # Reconstruct logic copied/adapted:
        # NOTE: This reconstruction loop is slow in Python.
        # But we must do it to get params_new for log_likelihood call.

        # Optimization: if we can, avoid full reconstruction or do it vectorized.
        # But `reconstruct_parameters` is abstract.
        # Let's assume we do it per N for now as in legacy.

        # Pre-calc shapes
        S, N = log_ell.shape

        for i in range(N):
            p_i = self.likelihood_fn.reconstruct_parameters(theta_new[:, i, :], params)
            if i == 0:
                for k, v in p_i.items():
                    # v is (S, ...) -> (S, 1, ...)
                    params_new[k] = v[:, jnp.newaxis, ...]
            else:
                for k, v in p_i.items():
                    params_new[k] = jnp.concatenate(
                        [params_new[k], v[:, jnp.newaxis, ...]], axis=1
                    )

        # Compute weights
        eta_weights, psis_weights, khat, log_ell_new = (
            self.compute_importance_weights_helper(
                self.likelihood_fn,
                data,
                params,
                params_new,
                log_jac,
                variational,
                log_pi,
                log_ell_original,
                surrogate_log_prob_fn,
            )
        )

        predictions = self.likelihood_fn.log_likelihood(data, params_new)

        weight_entropy = self.entropy(eta_weights)
        psis_entropy = self.entropy(psis_weights)

        p_loo_eta = jnp.sum(jnp.exp(predictions) * eta_weights, axis=0)
        p_loo_psis = jnp.sum(jnp.exp(predictions) * psis_weights, axis=0)

        ll_loo_eta = jnp.sum(eta_weights * jnp.exp(log_ell_new), axis=0)
        ll_loo_psis = jnp.sum(psis_weights * jnp.exp(log_ell_new), axis=0)

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

        return grad_ll


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
        scaling = -jnp.exp(log_pi_expanded - log_ell_expanded)

        return scaling * grad_ll


class Variance(SmallStepTransformation):
    """Variance-based transformation with optional target function f.

    Formula: Q = pi * (f/ell)^2 * grad(log(f/ell))
               = pi * exp(2*log_f - 2*log_ell) * (grad_log_f - grad_log_ell)
    """

    def __init__(
        self, likelihood_fn: LikelihoodFunction, log_f_fn: Optional[Callable] = None
    ):
        super().__init__(likelihood_fn)
        self.log_f_fn = log_f_fn

    def compute_Q(
        self,
        theta: jnp.ndarray,
        data: Any,
        params: Dict[str, Any],
        current_log_ell: jnp.ndarray,
        log_pi: jnp.ndarray = None,
        **kwargs,
    ) -> jnp.ndarray:

        if log_pi is None:
            raise ValueError("log_pi required")

        # 1. Compute gradients of log_ell (grad_log_ell)
        if "log_ell_prime" in kwargs and kwargs["log_ell_prime"] is not None:
            grad_log_ell = kwargs["log_ell_prime"]
        else:
            grad_log_ell = self.likelihood_fn.log_likelihood_gradient(data, params)

        # 2. Compute log_f and grad_log_f
        if self.log_f_fn is not None:
            # Helper for single sample
            def single_sample_log_f(t):
                # Shape adjustment for reconstruct which expects (S,...)
                p = self.likelihood_fn.reconstruct_parameters(t[jnp.newaxis, :], params)
                val = self.log_f_fn(data, **p)
                return jnp.squeeze(val, axis=0)  # (N,)

            log_f = kwargs.get("log_f", None)
            grad_log_f = kwargs.get("grad_log_f", None)

            if log_f is None:
                log_f = self.log_f_fn(data, **params)

            if grad_log_f is None:
                # Compute Jacobian: (S, N, K)
                jac_fn = jax.jacrev(single_sample_log_f)
                grad_log_f = jax.vmap(jac_fn)(theta)

            if grad_log_f.ndim == 4 and grad_log_f.shape[2] == 1:
                grad_log_f = jnp.squeeze(grad_log_f, axis=2)

        else:
            log_f = jnp.zeros_like(current_log_ell)
            grad_log_f = jnp.zeros_like(grad_log_ell)

        # 3. Assemble Q
        # Q = pi * exp(2*log_f - 2*log_ell) * (grad_log_f - grad_log_ell)

        log_pi_expanded = log_pi
        if log_pi_expanded.ndim == 1:
            log_pi_expanded = log_pi_expanded[:, jnp.newaxis]  # (S, 1)

        # log_pi (S, 1) to (S, N) broadcast
        log_pi_broadcasted = log_pi_expanded
        if log_pi_broadcasted.ndim == 2 and log_pi_broadcasted.shape[1] == 1:
            log_pi_broadcasted = jnp.tile(
                log_pi_broadcasted, (1, current_log_ell.shape[1])
            )

        # log_weights = log_pi + 2*log_f - 2*log_ell
        log_weights = log_pi_broadcasted + 2.0 * log_f - 2.0 * current_log_ell

        weights = jnp.exp(log_weights)  # (S, N)

        # Expand weights for gradient mult: (S, N, 1)
        weights_expanded = weights[..., jnp.newaxis]

        # diff_grad: (S, N, K)
        diff_grad = grad_log_f - grad_log_ell

        Q = weights_expanded * diff_grad

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

        # Flattened logic:
        # We have params dict.
        # We calculate diff for each param.
        # Then flatten to get Q vector.

        # This requires matching theta's flattened structure.
        # Ais uses `theta` as flattened (S, N, K).
        # extract_parameters does that.

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

        # Flatten diff_params to get Q
        Q = self.likelihood_fn.extract_parameters(diff_params)
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

        # We assume theta is flattened order of params iteration
        # extract_parameters usually iterates keys() or similar.
        # We need to be careful with ordering.
        # Ideally we use extract_parameters on a constructed dict.

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

        Q = self.likelihood_fn.extract_parameters(Q_dict)
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
        # Legacy _transform_mm1 logic
        # ...
        # For refactor, we can just invoke _transform_mm1 if we port it?
        # Or reimplement.
        # Reimplementing briefly for cleanliness.

        if log_ell_original is None:
            raise ValueError("required")
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
        theta_new = self.likelihood_fn.extract_parameters(new_params)

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

        weight_entropy = self.entropy(eta_weights)
        psis_entropy = self.entropy(psis_weights)

        p_loo_eta = jnp.sum(jnp.exp(predictions) * eta_weights, axis=0)
        p_loo_psis = jnp.sum(jnp.exp(predictions) * psis_weights, axis=0)

        ll_loo_eta = jnp.sum(eta_weights * jnp.exp(log_ell_new), axis=0)
        ll_loo_psis = jnp.sum(psis_weights * jnp.exp(log_ell_new), axis=0)

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
        if log_ell_original is None:
            raise ValueError("required")
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

        theta_new = self.likelihood_fn.extract_parameters(new_params)

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

        weight_entropy = self.entropy(eta_weights)
        psis_entropy = self.entropy(psis_weights)

        p_loo_eta = jnp.sum(jnp.exp(predictions) * eta_weights, axis=0)
        p_loo_psis = jnp.sum(jnp.exp(predictions) * psis_weights, axis=0)

        ll_loo_eta = jnp.sum(eta_weights * jnp.exp(log_ell_new), axis=0)
        ll_loo_psis = jnp.sum(psis_weights * jnp.exp(log_ell_new), axis=0)

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

    def log_likelihood_gradient(self, data: Any, params: Dict[str, Any]) -> jnp.ndarray:
        """Compute gradient of log-likelihood w.r.t. parameters using autodiff."""
        flat_params = self.extract_parameters(params)  # (S, K)

        # We need a function that maps (K,) -> (N,)
        def batch_ll(theta_s):
            # theta_s: (K,)
            # Reconstruct params to match the structure the model expects
            # Most models expect a sample dimension, so we add a singleton one
            p = self.reconstruct_parameters(theta_s[jnp.newaxis, ...], params)
            ll = self.log_likelihood(data, p)
            # Remove the added sample dimension from results (1, N) -> (N,)
            return jnp.squeeze(ll, axis=0)

        # Jacobian of mapping theta -> [ll_1, ll_2, ... ll_N]
        # Jacobian shape for one sample: (N, K)
        jac_fn = jax.jacrev(batch_ll)

        # Vmap over samples S
        # Output: (S, N, K)
        gradients = jax.vmap(jac_fn)(flat_params)

        return gradients

    def log_likelihood_hessian_diag(
        self, data: Any, params: Dict[str, Any]
    ) -> jnp.ndarray:
        """Compute diagonal of Hessian of log-likelihood w.r.t. parameters using autodiff."""

        flat_params = self.extract_parameters(params)  # (S, K)

        def batch_ll(theta_s):
            p = self.reconstruct_parameters(theta_s[jnp.newaxis, ...], params)
            ll = self.log_likelihood(data, p)
            return jnp.squeeze(ll, axis=0)  # (N,)

        # We want the diagonal of the Hessian for each data point i: d^2 L_i / d theta_j^2
        # jax.hessian(batch_ll) would give (N, K, K).
        # We can use jax.vmap over data points and compute diagonal Hessian for each.

        def point_ll(theta_s, i):
            p = self.reconstruct_parameters(theta_s[jnp.newaxis, ...], params)
            ll = self.log_likelihood(data, p)
            return ll[0, i]  # Scalar

        def diag_hess_fn(theta_s):
            # Returns (N, K)
            # Use vmap to compute diagonal Hessian for each data point i
            n_data = jax.tree_util.tree_leaves(data)[0].shape[0]

            def point_diag_hess(i):
                # Diagonal of Hessian of point_ll w.r.t theta_s
                # For small K, jax.hessian is okay. For large K, this might be slow.
                # But AIS is generally used when we can afford some computation.
                h = jax.hessian(lambda t: point_ll(t, i))(theta_s)
                return jnp.diag(h)

            return jax.vmap(point_diag_hess)(jnp.arange(n_data))

        # Vmap over samples S
        # Output: (S, N, K)
        hessian_diag = jax.vmap(diag_hess_fn)(flat_params)

        return hessian_diag


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
        """
        self.likelihood_fn = likelihood_fn
        self.prior_log_prob_fn = prior_log_prob_fn
        self.surrogate_log_prob_fn = surrogate_log_prob_fn

    def entropy(self, probs: jnp.ndarray) -> jnp.ndarray:
        """Compute entropy of probability distributions."""
        return -jnp.sum(probs * jnp.log(probs + 1e-10), axis=0)

    def adaptive_is_loo(
        self,
        data: Any,
        params: Dict[str, Any],
        n_sweeps: int = 4,
        max_iter: int = 10,
        initial_step_size: float = 1.0,
        variational: bool = False,
        verbose: bool = False,
        log_f_fn: Optional[Callable] = None,
        rhos: jnp.ndarray = None,
        transformations: List[str] = None,
    ):
        """Perform Adaptive Importance Sampling for Leave-One-Out CV.

        Args:
            data: Input data
            params: Initial model parameters
            n_sweeps: Number of grid search sweeps
            max_iter: Maximum iterations for transformations (used by iterative ones)
            initial_step_size: Initial step size for search
            variational: Whether using variational approximation
            verbose: Verbosity flag
            log_f_fn: Optional function returning log(f) for Variance transformation.
                      Signature: log_f_fn(data, **params) -> (S, N)
            rhos: Optional manual grid of step sizes. If None, generated from n_sweeps.
            transformations: Optional list of transformation names to run.
                             Available: 'll', 'kl', 'var', 'pmm1', 'pmm2', 'mm1', 'mm2', 'identity'.

        Returns:
            Dictionary of best results
        """

        # Initial computations
        log_ell = self.likelihood_fn.log_likelihood(data, params)  # (S, N)
        theta = self.likelihood_fn.extract_parameters(params)  # (S, K)

        # Precompute potentially expensive derivatives once
        log_ell_prime = self.likelihood_fn.log_likelihood_gradient(
            data, params
        )  # (S, N, K)
        log_ell_doubleprime = None
        # Only compute Hessian diagonal if needed (e.g. for Var transform)
        # But we do it once for efficiency if safe
        log_ell_doubleprime = self.likelihood_fn.log_likelihood_hessian_diag(
            data, params
        )  # (S, N, K)

        # Determine log_pi for transformations
        if variational and self.surrogate_log_prob_fn is not None:
            log_pi = self.surrogate_log_prob_fn(params)
            grad_log_pi = jax.grad(lambda p: jnp.sum(self.surrogate_log_prob_fn(p)))(
                params
            )
            grad_log_pi = self.likelihood_fn.extract_parameters(grad_log_pi)
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
                    else jnp.zeros(theta.shape[0])
                )
                log_pi = jnp.sum(log_ell, axis=1) + log_prior
                grad_log_pi = None

        # Ensure log_pi shape
        if log_pi.ndim == 1:
            log_pi = log_pi  # (S,) usually, but transforms expect broadcasting.
            # Some transforms expect (S, 1) or similar.
            # KLDivergence expects log_pi.
            # Let's ensure it's (S,) or (S,1) consistently.
            pass

        # Standard deviation of parameters (for standardization)
        theta_std = jnp.std(theta, axis=0)  # (K,)
        theta_std = jnp.where(theta_std < 1e-6, 1.0, theta_std)

        theta_expanded = theta[
            :, jnp.newaxis, :
        ]  # (S, 1, K) broadcastable to (S, N, K)

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
            "var": Variance(self.likelihood_fn, log_f_fn=log_f_fn),
            "variance_based": Variance(self.likelihood_fn, log_f_fn=log_f_fn),
            "pmm1": PMM1(self.likelihood_fn),
            "pmm2": PMM2(self.likelihood_fn),
            "mm1": MM1(self.likelihood_fn),
            "mm2": MM2(self.likelihood_fn),
        }

        # Filter transformations if requested
        if transformations is not None:
            transforms = {
                name: all_transforms[name]
                for name in transformations
                if name in all_transforms
            }
            # Handle 'identity' as a special case if requested (no-op transform)
        else:
            transforms = all_transforms
            # Remove aliases from default run to avoid duplication
            transforms.pop("ll", None)

        # Initialize best results
        best_metrics = {
            "khat": jnp.full(log_ell.shape[1], jnp.inf),  # (N,)
            "p_loo_eta": jnp.zeros(log_ell.shape[1]),
            "p_loo_psis": jnp.zeros(log_ell.shape[1]),
        }

        results = {}

        # Handle 'identity' sweep explicitly if requested or if transformations is None
        if transformations is None or "identity" in transformations:
            if verbose:
                print("Running identity...")
            # Compute identity importance weights (standard PSIS-LOO)
            # log_jacobian is 0
            log_jac_identity = jnp.zeros_like(log_ell)
            eta_weights, psis_weights, khat, log_ell_new = (
                Transformation.compute_importance_weights_helper(
                    None,
                    self.likelihood_fn,
                    data,
                    params,
                    params,
                    log_jac_identity,
                    variational,
                    log_pi,
                    log_ell,
                    self.surrogate_log_prob_fn,
                )
            )

            predictions = log_ell  # Identity has same predictions

            # Use self.entropy if available or compute directly
            weight_entropy = self.entropy(eta_weights)
            psis_entropy = self.entropy(psis_weights)

            p_loo_eta = jnp.sum(jnp.exp(predictions) * eta_weights, axis=0)
            p_loo_psis = jnp.sum(jnp.exp(predictions) * psis_weights, axis=0)

            ll_loo_eta = jnp.sum(eta_weights * jnp.exp(log_ell_new), axis=0)
            ll_loo_psis = jnp.sum(psis_weights * jnp.exp(log_ell_new), axis=0)

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
            if "ll_loo_eta" not in best_metrics:
                best_metrics["ll_loo_eta"] = jnp.zeros_like(p_loo_eta)
                best_metrics["ll_loo_psis"] = jnp.zeros_like(p_loo_psis)
            best_metrics["ll_loo_eta"] = jnp.where(
                idx, ll_loo_eta, best_metrics["ll_loo_eta"]
            )
            best_metrics["ll_loo_psis"] = jnp.where(
                idx, ll_loo_psis, best_metrics["ll_loo_psis"]
            )

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

        theta_expanded = theta[
            :, jnp.newaxis, :
        ]  # (S, 1, K) broadcastable to (S, N, K)

        # Run sweeps
        for name, transform in transforms.items():
            if verbose:
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

                    # Update best
                    khat_safe = jnp.where(jnp.isnan(res["khat"]), jnp.inf, res["khat"])

                    improvement = khat_safe < best_metrics["khat"]
                    best_metrics["khat"] = jnp.where(
                        improvement, khat_safe, best_metrics["khat"]
                    )
                    best_metrics["p_loo_eta"] = jnp.where(
                        improvement, res["p_loo_eta"], best_metrics["p_loo_eta"]
                    )
                    best_metrics["p_loo_psis"] = jnp.where(
                        improvement, res["p_loo_psis"], best_metrics["p_loo_psis"]
                    )

                except Exception as e:
                    if verbose:
                        print(f"Failed {name} rho={rho}: {e}")
                    # import traceback
                    # traceback.print_exc()
                    continue

        results["best"] = best_metrics
        return results

    # Legacy methods have been removed in favor of Transformation subclasses.


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

    def log_likelihood(
        self, data: Dict[str, Any], params: Dict[str, Any]
    ) -> jnp.ndarray:
        """Compute log-likelihood for logistic regression."""
        X = jnp.asarray(data["X"], dtype=self.dtype)  # (n_data, n_features)
        y = jnp.asarray(data["y"], dtype=self.dtype)  # (n_data,)

        beta = params["beta"]  # (n_samples, n_features)
        intercept = params["intercept"]  # (n_samples,)

        # Linear predictor: X @ beta.T + intercept
        if beta.ndim == 3:
            mu = jnp.einsum("df,sdf->sd", X, beta) + intercept
        else:
            mu = jnp.einsum("df,sf->sd", X, beta) + intercept[:, jnp.newaxis]

        # Sigmoid and log-likelihood
        sigma = jax.nn.sigmoid(mu)
        log_lik = y[jnp.newaxis, :] * jnp.log(sigma + 1e-10) + (
            1 - y[jnp.newaxis, :]
        ) * jnp.log(1 - sigma + 1e-10)

        return log_lik

    def log_likelihood_gradient(
        self, data: Dict[str, Any], params: Dict[str, Any]
    ) -> jnp.ndarray:
        """Compute gradient of log-likelihood w.r.t. parameters."""
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]

        mu = jnp.einsum("df,sf->sd", X, beta) + intercept[:, jnp.newaxis]
        sigma = jax.nn.sigmoid(mu)

        # Gradient w.r.t. linear predictor
        grad_mu = y[jnp.newaxis, :] - sigma  # (n_samples, n_data)

        # Gradient w.r.t. beta: X.T @ grad_mu
        grad_beta = jnp.einsum(
            "df,sd->sdf", X, grad_mu
        )  # (n_samples, n_data, n_features)

        # Gradient w.r.t. intercept
        grad_intercept = grad_mu[..., jnp.newaxis]  # (n_samples, n_data, 1)

        # Concatenate gradients
        gradients = jnp.concatenate(
            [grad_beta, grad_intercept], axis=-1
        )  # (n_samples, n_data, n_features + 1)

        return gradients

    def log_likelihood_hessian_diag(
        self, data: Dict[str, Any], params: Dict[str, Any]
    ) -> jnp.ndarray:
        """Compute diagonal of Hessian of log-likelihood."""
        X = jnp.asarray(data["X"], dtype=self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]

        mu = jnp.einsum("df,sf->sd", X, beta) + intercept[:, jnp.newaxis]
        sigma = jax.nn.sigmoid(mu)

        # Hessian diagonal w.r.t. linear predictor
        hess_diag_mu = -sigma * (1 - sigma)  # (n_samples, n_data)

        # Hessian diagonal w.r.t. beta: X^2 * hess_diag_mu
        hess_diag_beta = jnp.einsum(
            "df,sd->sdf", X**2, hess_diag_mu
        )  # (n_samples, n_data, n_features)

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

    def reconstruct_parameters(
        self, flat_params: jnp.ndarray, template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Reconstruct parameters from flattened array."""
        n_features = template["beta"].shape[-1]

        beta = flat_params[..., :n_features]
        intercept = flat_params[..., n_features]

        return {"beta": beta, "intercept": intercept}


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
            mu = jnp.einsum("df,sf->sd", X, beta) + intercept[:, jnp.newaxis]

        sigma = jnp.exp(log_sigma)
        if log_sigma.ndim == 1:
            sigma = sigma[:, jnp.newaxis]

        # Gaussian log-likelihood
        residuals = y[jnp.newaxis, :] - mu
        log_lik = (
            -0.5 * jnp.log(2 * jnp.pi)
            - log_sigma[:, jnp.newaxis]
            - 0.5 * (residuals / sigma) ** 2
        )

        return log_lik

    def log_likelihood_gradient(
        self, data: Dict[str, Any], params: Dict[str, Any]
    ) -> jnp.ndarray:
        """Compute gradient of log-likelihood."""
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]
        log_sigma = params.get("log_sigma", jnp.zeros((beta.shape[0],)))

        mu = jnp.einsum("df,sf->sd", X, beta) + intercept[:, jnp.newaxis]
        sigma = jnp.exp(log_sigma)[:, jnp.newaxis]
        residuals = y[jnp.newaxis, :] - mu

        # Gradients
        grad_beta = jnp.einsum("df,sd->sdf", X, residuals / sigma**2)
        grad_intercept = (residuals / sigma**2)[..., jnp.newaxis]
        grad_log_sigma = (-1 + (residuals / sigma) ** 2)[..., jnp.newaxis]

        gradients = jnp.concatenate(
            [grad_beta, grad_intercept, grad_log_sigma], axis=-1
        )
        return gradients

    def log_likelihood_hessian_diag(
        self, data: Dict[str, Any], params: Dict[str, Any]
    ) -> jnp.ndarray:
        """Compute diagonal of Hessian."""
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]
        log_sigma = params.get("log_sigma", jnp.zeros((beta.shape[0],)))

        mu = jnp.einsum("df,sf->sd", X, beta) + intercept[:, jnp.newaxis]
        sigma = jnp.exp(log_sigma)[:, jnp.newaxis]
        residuals = y[jnp.newaxis, :] - mu

        # Hessian diagonals
        hess_diag_beta = jnp.einsum("df,sd->sdf", -(X**2), 1 / sigma**2)
        hess_diag_intercept = (-1 / sigma**2)[..., jnp.newaxis]
        hess_diag_log_sigma = (-2 * (residuals / sigma) ** 2)[..., jnp.newaxis]

        hess_diag = jnp.concatenate(
            [hess_diag_beta, hess_diag_intercept, hess_diag_log_sigma], axis=-1
        )
        return hess_diag

    def extract_parameters(self, params: Dict[str, Any]) -> jnp.ndarray:
        """Extract parameters into flattened array."""
        beta = params["beta"]
        intercept = params["intercept"]
        log_sigma = params.get("log_sigma", jnp.zeros((beta.shape[0],)))

        theta = jnp.concatenate(
            [beta, intercept[:, jnp.newaxis], log_sigma[:, jnp.newaxis]], axis=-1
        )
        return theta

    def reconstruct_parameters(
        self, flat_params: jnp.ndarray, template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Reconstruct parameters from flattened array."""
        n_features = template["beta"].shape[-1]

        beta = flat_params[..., :n_features]
        intercept = flat_params[..., n_features]
        log_sigma = flat_params[..., n_features + 1]

        result = {"beta": beta, "intercept": intercept}
        if "log_sigma" in template:
            result["log_sigma"] = log_sigma

        return result

#!/usr/bin/env python3
"""
GRAMIS: Gradient-based Adaptive Multiple Importance Sampling

Implements the GRAMIS algorithm from Elvira et al. 2023 (arXiv:2210.10785v3)
as a baseline comparison for LOO-CV importance sampling.

The algorithm adapts N Gaussian proposals over T iterations using:
- Gradient ascent on proposal means with backtracking line search
- Hessian-based covariance adaptation
- Repulsive forces between proposals to maintain diversity
- Deterministic Mixture MIS (DM-MIS) importance weights

For LOO-CV, the target for each observation i is the LOO posterior:
    pi_{-i}(theta) propto pi_0(theta) * prod_{j != i} ell(y_j | theta)
"""

import jax
import jax.numpy as jnp
import jax.flatten_util
from typing import Dict, Any, Optional, Callable
from functools import partial

from bayesianquilts.metrics.ais import LikelihoodFunction
from bayesianquilts.metrics import psis


def _multivariate_normal_logpdf(x, mean, cov):
    """Log-pdf of multivariate normal distribution.

    Args:
        x: points, shape (..., D)
        mean: mean, shape (D,)
        cov: covariance, shape (D, D)

    Returns:
        Log-pdf values, shape (...)
    """
    d = mean.shape[0]
    diff = x - mean
    # Use Cholesky for numerical stability
    L = jnp.linalg.cholesky(cov)
    # Solve L z = diff^T => z = L^{-1} diff^T
    z = jax.scipy.linalg.solve_triangular(L, diff.T, lower=True)  # (D, ...)
    maha = jnp.sum(z ** 2, axis=0)  # (...)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det + maha)


class GRAMISSampler:
    """GRAMIS baseline sampler for LOO-CV importance sampling.

    Implements the GRAMIS algorithm (Elvira et al. 2023) adapted for
    leave-one-out cross-validation. Uses gradient-based adaptation of
    Gaussian mixture proposals with repulsive forces and Hessian-based
    covariance updates.
    """

    def __init__(
        self,
        likelihood_fn: LikelihoodFunction,
        prior_log_prob_fn: Optional[Callable] = None,
    ):
        """Initialize the GRAMIS sampler.

        Args:
            likelihood_fn: Likelihood function implementing LikelihoodFunction
                interface. Must provide log_likelihood(data, params) -> (S, N).
            prior_log_prob_fn: Function computing log prior probability.
                Signature: prior_log_prob_fn(params) -> scalar or (S,).
                If None, a flat (improper) prior is assumed.

        Raises:
            TypeError: If likelihood_fn is not a LikelihoodFunction instance.
        """
        if not isinstance(likelihood_fn, LikelihoodFunction):
            raise TypeError(
                f"likelihood_fn must be a LikelihoodFunction instance, "
                f"got {type(likelihood_fn)}"
            )
        self.likelihood_fn = likelihood_fn
        self.prior_log_prob_fn = prior_log_prob_fn

    def _flatten_params(self, params):
        """Flatten a params PyTree to a 2D array (S, D).

        Args:
            params: Dict of arrays, each with shape (S, ...).

        Returns:
            flat: array of shape (S, D) where D is total flattened param dim.
            unravel_fn: function that maps (D,) -> params PyTree (without S dim).
        """
        # Take first sample to get the unravel function
        first_sample = jax.tree_util.tree_map(lambda x: x[0], params)
        _, unravel_fn = jax.flatten_util.ravel_pytree(first_sample)

        # Flatten all samples
        flat = jax.vmap(lambda p: jax.flatten_util.ravel_pytree(p)[0])(params)
        return flat, unravel_fn

    def _make_params_from_flat(self, flat_vec, unravel_fn):
        """Convert a flat vector (D,) back to a params PyTree with S=1 dim.

        Args:
            flat_vec: array of shape (D,)
            unravel_fn: function mapping (D,) -> params PyTree

        Returns:
            params PyTree where each leaf has shape (1, ...)
        """
        p = unravel_fn(flat_vec)
        return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), p)

    def _loo_log_target(self, flat_vec, unravel_fn, data, full_ll, obs_idx):
        """Compute log LOO posterior target for leaving out observation obs_idx.

        log pi_{-i}(theta) = log pi_0(theta) + sum_{j != i} log ell(y_j | theta)

        Args:
            flat_vec: flattened parameter vector, shape (D,)
            unravel_fn: function to reconstruct params PyTree
            data: input data
            full_ll: NOT used (recomputed for the new params)
            obs_idx: index of observation to leave out

        Returns:
            Scalar log target density value.
        """
        params = self._make_params_from_flat(flat_vec, unravel_fn)
        ll = self.likelihood_fn.log_likelihood(data, params)  # (1, N)
        ll = ll[0]  # (N,)

        # Sum all log-likelihoods except observation obs_idx
        n_data = ll.shape[0]
        mask = jnp.ones(n_data, dtype=jnp.float64)
        mask = mask.at[obs_idx].set(0.0)
        loo_ll = jnp.sum(ll * mask)

        # Add prior
        if self.prior_log_prob_fn is not None:
            log_prior = self.prior_log_prob_fn(params)
            log_prior = jnp.squeeze(log_prior)
        else:
            log_prior = 0.0

        return loo_ll + log_prior

    def _full_log_target(self, flat_vec, unravel_fn, data):
        """Compute log full posterior target (all observations).

        Args:
            flat_vec: flattened parameter vector, shape (D,)
            unravel_fn: function to reconstruct params PyTree
            data: input data

        Returns:
            Scalar log target density value.
        """
        params = self._make_params_from_flat(flat_vec, unravel_fn)
        ll = self.likelihood_fn.log_likelihood(data, params)  # (1, N)
        total_ll = jnp.sum(ll)

        if self.prior_log_prob_fn is not None:
            log_prior = self.prior_log_prob_fn(params)
            log_prior = jnp.squeeze(log_prior)
        else:
            log_prior = 0.0

        return total_ll + log_prior

    def _backtracking_step_size(
        self,
        log_target_fn,
        mu,
        grad_log_target,
        cov,
        max_iter=10,
    ):
        """Backtracking line search for step size (Eq. 12 of Elvira et al.).

        Halves theta starting from 1 until pi(mu + theta * Sigma * grad) >= pi(mu).

        Args:
            log_target_fn: callable, log target density as function of flat vec
            mu: current mean, shape (D,)
            grad_log_target: gradient of log target at mu, shape (D,)
            cov: covariance matrix, shape (D, D)
            max_iter: maximum halving iterations

        Returns:
            Scalar step size theta.
        """
        direction = cov @ grad_log_target  # (D,)
        current_val = log_target_fn(mu)

        def cond_fn(state):
            theta, i, found = state
            return jnp.logical_and(i < max_iter, jnp.logical_not(found))

        def body_fn(state):
            theta, i, found = state
            candidate = mu + theta * direction
            candidate_val = log_target_fn(candidate)
            improved = candidate_val >= current_val
            return (
                jnp.where(improved, theta, theta * 0.5),
                i + 1,
                improved,
            )

        theta_init = jnp.float64(1.0)
        theta, _, found = jax.lax.while_loop(
            cond_fn, body_fn, (theta_init, 0, False)
        )
        # If no improvement found after max_iter, use a very small step
        return jnp.where(found, theta, jnp.float64(1e-4))

    def _compute_repulsion(self, means, n_idx, G_t, dim):
        """Compute repulsive force on proposal n from all other proposals (Eq. 10).

        r_{n,j} = G_t * (m_n * m_j / ||d_{n,j}||^{d_x}) * d_{n,j}

        For simplicity, we set m_n = m_j = 1 (uniform proposal masses).

        Args:
            means: array of shape (N_proposals, D) with current proposal means
            n_idx: index of the proposal to compute repulsion for
            G_t: repulsion strength at current iteration
            dim: parameter dimensionality D

        Returns:
            Repulsion vector, shape (D,).
        """
        mu_n = means[n_idx]
        n_proposals = means.shape[0]

        def repulsion_from_j(j):
            d_nj = mu_n - means[j]
            dist = jnp.sqrt(jnp.sum(d_nj ** 2) + 1e-12)
            # r_{n,j} = G_t * (1 / ||d||^{d_x}) * d
            r = G_t * d_nj / (dist ** dim + 1e-12)
            # Zero out self-interaction
            return jnp.where(j == n_idx, jnp.zeros_like(r), r)

        # Sum repulsion from all other proposals
        repulsions = jax.vmap(repulsion_from_j)(jnp.arange(n_proposals))
        return jnp.sum(repulsions, axis=0)

    def _adapt_covariance(self, log_target_fn, mu, current_cov, dim):
        """Adapt covariance using negative inverse Hessian (Eq. 13).

        Sigma_n^(t) = (-nabla^2 log pi(mu_n^(t)))^{-1}
        if positive definite, else keep current_cov.

        Args:
            log_target_fn: callable, log target as function of flat vec
            mu: current mean, shape (D,)
            current_cov: current covariance, shape (D, D)
            dim: parameter dimensionality

        Returns:
            Updated covariance matrix, shape (D, D).
        """
        # Compute Hessian via autodiff
        hessian = jax.hessian(log_target_fn)(mu)  # (D, D)
        neg_hessian = -hessian

        # Check positive definiteness via Cholesky
        # If Cholesky fails (not PD), keep current covariance
        try_chol = jnp.linalg.cholesky(neg_hessian)
        is_pd = jnp.all(jnp.isfinite(try_chol))

        # Invert if PD
        new_cov = jnp.linalg.inv(neg_hessian)
        # Also check the inverse is finite and PD
        new_cov_chol = jnp.linalg.cholesky(new_cov)
        new_cov_ok = jnp.all(jnp.isfinite(new_cov_chol))

        use_new = jnp.logical_and(is_pd, new_cov_ok)
        return jnp.where(use_new, new_cov, current_cov)

    def gramis_loo(
        self,
        data: Any,
        params: Dict[str, Any],
        n_proposals: int = 20,
        n_samples_per_proposal: int = 10,
        n_iterations: int = 10,
        repulsion_G0: float = 0.05,
        repulsion_decay: Optional[float] = None,
        backtrack_max_iter: int = 10,
        khat_threshold: float = 0.7,
        verbose: bool = False,
        rng_key: Optional[jnp.ndarray] = None,
        adapt_covariance: bool = False,
    ) -> dict:
        """Run GRAMIS for LOO-CV.

        For each observation i, adapts Gaussian mixture proposals to target
        the LOO posterior pi_{-i}, draws weighted samples, and computes
        LOO log-likelihood estimates with PSIS diagnostics.

        Args:
            data: Input data (passed to likelihood_fn).
            params: MCMC posterior samples as dict {name: array(S, ...)}.
                S is the number of posterior samples.
            n_proposals: N, number of Gaussian proposals in the mixture.
            n_samples_per_proposal: K, samples drawn from each proposal.
            n_iterations: T, number of GRAMIS adaptation iterations.
            repulsion_G0: Initial repulsion strength G_0.
            repulsion_decay: Decay rate beta for G_t = exp(-beta*t).
                If None, automatically set so G_T = 0.01 * G_0.
            backtrack_max_iter: Maximum iterations for backtracking line search.
            khat_threshold: PSIS khat threshold for diagnostics.
            verbose: If True, print progress information.
            rng_key: JAX PRNG key. If None, uses PRNGKey(0).
            adapt_covariance: If True, adapt covariance using Hessian (Eq. 13).
                This requires computing the full Hessian per proposal per
                iteration and can be very expensive for high-dimensional params.
                Default False uses empirical covariance from posterior samples.

        Returns:
            Dictionary with:
                - 'khat': array of shape (N_data,) with PSIS khat diagnostics
                - 'p_loo_psis': LOO predictive probabilities (N_data,)
                - 'll_loo_psis': LOO log-likelihoods (N_data,)
                - 'n_iterations': actual number of iterations used
                - 'proposal_means': final proposal means (N_proposals, D)
                - 'weights': raw log importance weights
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        # Flatten params to (S, D)
        flat_params, unravel_fn = self._flatten_params(params)
        flat_params = flat_params.astype(jnp.float64)
        S, D = flat_params.shape

        if verbose:
            print(f"GRAMIS: S={S} posterior samples, D={D} parameters")

        # Compute full posterior log-likelihoods for reference
        ll_full = self.likelihood_fn.log_likelihood(data, params)  # (S, N)
        ll_full = ll_full.astype(jnp.float64)
        N_data = ll_full.shape[1]

        if verbose:
            print(f"GRAMIS: N_data={N_data} observations")

        # --- Initialize proposals ---
        # Select n_proposals samples from the posterior as initial means
        n_proposals = min(n_proposals, S)
        rng_key, subkey = jax.random.split(rng_key)
        proposal_indices = jax.random.choice(
            subkey, S, shape=(n_proposals,), replace=False
        )
        means = flat_params[proposal_indices]  # (N_proposals, D)

        # Initial covariance: empirical covariance of posterior samples
        emp_cov = jnp.cov(flat_params.T)  # (D, D)
        if D == 1:
            emp_cov = emp_cov.reshape(1, 1)
        # Add small ridge for numerical stability
        emp_cov = emp_cov + 1e-6 * jnp.eye(D, dtype=jnp.float64)
        # Each proposal starts with the same covariance
        covs = jnp.broadcast_to(emp_cov, (n_proposals, D, D)).copy()

        # Repulsion decay
        if repulsion_decay is None:
            # Set so G_T = 0.01 * G_0, i.e. exp(-beta*T) = 0.01
            repulsion_decay = jnp.log(100.0) / max(n_iterations, 1)

        # --- Build log target functions ---
        # Full posterior target (for mean adaptation)
        def full_log_target(flat_vec):
            return self._full_log_target(flat_vec, unravel_fn, data)

        grad_full_log_target = jax.grad(full_log_target)

        # LOO target for observation i
        def loo_log_target_i(flat_vec, obs_idx):
            return self._loo_log_target(
                flat_vec, unravel_fn, data, None, obs_idx
            )

        # --- GRAMIS iterations ---
        # We adapt proposals using the FULL posterior first (not per-observation),
        # then use the adapted proposals for all LOO evaluations.
        # This is the practical approach since per-obs adaptation is O(N_data) times
        # more expensive.

        for t in range(n_iterations):
            G_t = repulsion_G0 * jnp.exp(-repulsion_decay * t)

            if verbose:
                print(f"  Iteration {t+1}/{n_iterations}, G_t={G_t:.6f}")

            new_means = jnp.zeros_like(means)

            for n in range(n_proposals):
                mu_n = means[n]

                # Gradient of log target at current mean
                grad_n = grad_full_log_target(mu_n)

                # Handle NaN gradients
                grad_n = jnp.where(jnp.isfinite(grad_n), grad_n, 0.0)

                # Backtracking line search for step size
                theta_n = self._backtracking_step_size(
                    full_log_target, mu_n, grad_n, covs[n],
                    max_iter=backtrack_max_iter,
                )

                # Repulsion force
                repulsion = self._compute_repulsion(means, n, G_t, D)

                # Update mean (Eq. 9)
                mu_new = mu_n + theta_n * (covs[n] @ grad_n) + repulsion
                mu_new = jnp.where(jnp.isfinite(mu_new), mu_new, mu_n)
                new_means = new_means.at[n].set(mu_new)

            means = new_means

            # Covariance adaptation (Eq. 13) - optional, expensive
            if adapt_covariance:
                for n in range(n_proposals):
                    covs = covs.at[n].set(
                        self._adapt_covariance(
                            full_log_target, means[n], covs[n], D
                        )
                    )

        # --- Sampling phase ---
        # Draw samples from each adapted proposal
        total_samples = n_proposals * n_samples_per_proposal
        all_samples = jnp.zeros((total_samples, D), dtype=jnp.float64)
        # Track which proposal each sample came from
        proposal_ids = jnp.zeros(total_samples, dtype=jnp.int32)

        for n in range(n_proposals):
            rng_key, subkey = jax.random.split(rng_key)
            samples_n = jax.random.multivariate_normal(
                subkey, means[n], covs[n],
                shape=(n_samples_per_proposal,),
                dtype=jnp.float64,
            )  # (K, D)
            start = n * n_samples_per_proposal
            end = start + n_samples_per_proposal
            all_samples = all_samples.at[start:end].set(samples_n)
            proposal_ids = proposal_ids.at[start:end].set(n)

        if verbose:
            print(f"GRAMIS: Drew {total_samples} total samples")

        # --- Compute DM-MIS mixture denominator ---
        # For each sample x, compute (1/N) * sum_j q_j(x)
        # where q_j is the j-th Gaussian proposal
        # log_q_mix[k] = log( (1/N) * sum_j exp(log q_j(x_k)) )
        log_q_components = jnp.zeros(
            (total_samples, n_proposals), dtype=jnp.float64
        )
        for n in range(n_proposals):
            log_q_n = _multivariate_normal_logpdf(
                all_samples, means[n], covs[n]
            )  # (total_samples,)
            log_q_components = log_q_components.at[:, n].set(log_q_n)

        # log( (1/N) * sum_j exp(log q_j(x)) )
        log_q_mix = (
            jax.scipy.special.logsumexp(log_q_components, axis=1)
            - jnp.log(jnp.float64(n_proposals))
        )  # (total_samples,)

        # --- Compute log-likelihoods for all samples ---
        # Build params PyTree for all samples at once
        # all_samples: (total_samples, D) -> need to unravel to params dict
        def unravel_batch(flat_batch):
            """Unravel a batch of flat vectors to params PyTree."""
            return jax.vmap(unravel_fn)(flat_batch)

        all_params = unravel_batch(all_samples)
        # all_params: PyTree where each leaf has shape (total_samples, ...)

        # Compute log-likelihoods: (total_samples, N_data)
        ll_samples = self.likelihood_fn.log_likelihood(data, all_params)
        ll_samples = ll_samples.astype(jnp.float64)  # (total_samples, N_data)

        # Compute log prior for all samples
        if self.prior_log_prob_fn is not None:
            log_prior_samples = self.prior_log_prob_fn(all_params)
            log_prior_samples = jnp.squeeze(log_prior_samples)
            if log_prior_samples.ndim == 0:
                log_prior_samples = jnp.broadcast_to(
                    log_prior_samples, (total_samples,)
                )
        else:
            log_prior_samples = jnp.zeros(total_samples, dtype=jnp.float64)

        # --- Compute LOO quantities ---
        # For each observation i:
        #   log pi_{-i}(x) = log_prior(x) + sum_{j != i} ll(x, j)
        #   w_k^{(i)} = pi_{-i}(x_k) / q_mix(x_k)

        # Total log-likelihood across all observations for each sample
        ll_total = jnp.sum(ll_samples, axis=1)  # (total_samples,)

        # LOO log-target for observation i:
        # log pi_{-i}(x_k) = log_prior(x_k) + ll_total(x_k) - ll(x_k, i)
        # log_numerator[k, i] = log_prior_samples[k] + ll_total[k] - ll_samples[k, i]
        log_numerator = (
            log_prior_samples[:, None]
            + ll_total[:, None]
            - ll_samples
        )  # (total_samples, N_data)

        # DM-MIS log weights: log w = log pi_{-i}(x) - log q_mix(x)
        log_weights_raw = log_numerator - log_q_mix[:, None]  # (total_samples, N_data)

        # --- PSIS smoothing and LOO estimates ---
        khat = jnp.zeros(N_data, dtype=jnp.float64)
        ll_loo_psis = jnp.zeros(N_data, dtype=jnp.float64)

        # PSIS expects log weights of shape (n_samples, n_data)
        # psislw operates column-by-column
        lw_smoothed, khat_vals = psis.psislw(log_weights_raw)

        # LOO log-likelihood estimate:
        # E_{pi_{-i}}[log ell(y_i | theta)] approx sum_k w_k^{(i)} * ll(x_k, i)
        # Using PSIS-smoothed normalized weights

        # Normalized PSIS weights (already log-normalized by psislw)
        # lw_smoothed are log normalized weights
        # LOO log predictive: log( sum_k exp(lw_smoothed_k + ll_k_i) )
        # But lw_smoothed from psislw are already normalized (sum to 1).
        # So: ll_loo_i = logsumexp(lw_smoothed[:, i] + ll_samples[:, i])

        ll_loo_psis = jax.scipy.special.logsumexp(
            lw_smoothed + ll_samples, axis=0
        )  # (N_data,)

        # LOO predictive probabilities (exponentiated)
        p_loo_psis = jnp.exp(ll_loo_psis)

        if verbose:
            n_good = jnp.sum(khat_vals < khat_threshold)
            n_ok = jnp.sum(
                jnp.logical_and(khat_vals >= khat_threshold, khat_vals < 1.0)
            )
            n_bad = jnp.sum(khat_vals >= 1.0)
            print(
                f"GRAMIS results: "
                f"khat < {khat_threshold}: {n_good}/{N_data}, "
                f"{khat_threshold} <= khat < 1.0: {n_ok}/{N_data}, "
                f"khat >= 1.0: {n_bad}/{N_data}"
            )
            print(
                f"  Mean khat: {jnp.mean(khat_vals):.4f}, "
                f"Max khat: {jnp.max(khat_vals):.4f}"
            )
            print(f"  Mean LOO ll: {jnp.mean(ll_loo_psis):.4f}")

        return {
            'khat': khat_vals,
            'p_loo_psis': p_loo_psis,
            'll_loo_psis': ll_loo_psis,
            'n_iterations': n_iterations,
            'proposal_means': means,
            'proposal_covs': covs,
            'weights': log_weights_raw,
            'lw_smoothed': lw_smoothed,
        }

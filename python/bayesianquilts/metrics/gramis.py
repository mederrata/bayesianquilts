#!/usr/bin/env python3
"""
GRAMIS: Gradient-based Adaptive Multiple Importance Sampling

Implements the GRAMIS algorithm from Elvira et al. 2023 (arXiv:2210.10785v3)
as a baseline comparison for LOO-CV importance sampling.

The algorithm adapts N Gaussian proposals over T iterations using:
- Gradient ascent on proposal means with backtracking line search
- Optional Hessian-based covariance adaptation
- Repulsive forces between proposals to maintain diversity
- Deterministic Mixture MIS (DM-MIS) importance weights using ALL proposals
  across ALL iterations

For LOO-CV, the target for each observation i is the LOO posterior:
    pi_{-i}(theta) propto pi_0(theta) * prod_{j != i} ell(y_j | theta)
"""

import jax
import jax.numpy as jnp
import jax.flatten_util
from typing import Dict, Any, Optional, Callable

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
    L = jnp.linalg.cholesky(cov)
    z = jax.scipy.linalg.solve_triangular(L, diff.T, lower=True)  # (D, ...)
    maha = jnp.sum(z ** 2, axis=0)  # (...)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det + maha)


class GRAMISSampler:
    """GRAMIS baseline sampler for LOO-CV importance sampling.

    Implements the GRAMIS algorithm (Elvira et al. 2023) adapted for
    leave-one-out cross-validation. Uses gradient-based adaptation of
    Gaussian mixture proposals with repulsive forces.

    Following the paper, samples are drawn at every iteration and the
    DM-MIS denominator uses all proposals from all iterations.
    """

    def __init__(
        self,
        likelihood_fn: LikelihoodFunction,
        prior_log_prob_fn: Optional[Callable] = None,
    ):
        if not isinstance(likelihood_fn, LikelihoodFunction):
            raise TypeError(
                f"likelihood_fn must be a LikelihoodFunction instance, "
                f"got {type(likelihood_fn)}"
            )
        self.likelihood_fn = likelihood_fn
        self.prior_log_prob_fn = prior_log_prob_fn

    def _flatten_params(self, params):
        """Flatten a params PyTree to a 2D array (S, D)."""
        first_sample = jax.tree_util.tree_map(lambda x: x[0], params)
        _, unravel_fn = jax.flatten_util.ravel_pytree(first_sample)
        flat = jax.vmap(lambda p: jax.flatten_util.ravel_pytree(p)[0])(params)
        return flat, unravel_fn

    def _make_params_from_flat(self, flat_vec, unravel_fn):
        """Convert a flat vector (D,) back to a params PyTree with S=1 dim."""
        p = unravel_fn(flat_vec)
        return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), p)

    def _full_log_target(self, flat_vec, unravel_fn, data):
        """Compute log full posterior target (all observations)."""
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
        self, log_target_fn, mu, grad_log_target, cov, max_iter=10,
    ):
        """Backtracking line search for step size (Eq. 12).

        Halves theta starting from 1 until pi(mu + theta * Sigma * grad) >= pi(mu).
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
        return jnp.where(found, theta, jnp.float64(1e-4))

    def _compute_repulsion(self, means, n_idx, G_t, dim):
        """Compute repulsive force on proposal n from all other proposals (Eq. 10).

        r_{n,j} = G_t * d_{n,j} / ||d_{n,j}||^{d_x}
        """
        mu_n = means[n_idx]
        n_proposals = means.shape[0]

        def repulsion_from_j(j):
            d_nj = mu_n - means[j]
            dist = jnp.sqrt(jnp.sum(d_nj ** 2) + 1e-12)
            r = G_t * d_nj / (dist ** dim + 1e-12)
            return jnp.where(j == n_idx, jnp.zeros_like(r), r)

        repulsions = jax.vmap(repulsion_from_j)(jnp.arange(n_proposals))
        return jnp.sum(repulsions, axis=0)

    def _adapt_covariance(self, log_target_fn, mu, current_cov, dim):
        """Adapt covariance using negative inverse Hessian (Eq. 13)."""
        hessian = jax.hessian(log_target_fn)(mu)  # (D, D)
        neg_hessian = -hessian

        try_chol = jnp.linalg.cholesky(neg_hessian)
        is_pd = jnp.all(jnp.isfinite(try_chol))

        new_cov = jnp.linalg.inv(neg_hessian)
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

        Following Elvira et al. 2023, at each iteration:
        1. Adapt proposal means via natural gradient + repulsion
        2. Draw K samples from each proposal
        3. Store all proposal means/covs for DM-MIS denominator

        After T iterations, the DM-MIS denominator uses ALL N*T proposals.

        Args:
            data: Input data (passed to likelihood_fn).
            params: MCMC posterior samples as dict {name: array(S, ...)}.
            n_proposals: N, number of Gaussian proposals in the mixture.
            n_samples_per_proposal: K, samples drawn from each proposal per iteration.
            n_iterations: T, number of GRAMIS adaptation iterations.
            repulsion_G0: Initial repulsion strength G_0.
            repulsion_decay: Decay rate for G_t = G_0 * exp(-decay*t).
            backtrack_max_iter: Maximum iterations for backtracking line search.
            khat_threshold: PSIS khat threshold for diagnostics.
            verbose: If True, print progress information.
            rng_key: JAX PRNG key. If None, uses PRNGKey(0).
            adapt_covariance: If True, adapt covariance using Hessian (Eq. 13).

        Returns:
            Dictionary with:
                - 'khat': array of shape (N_data,) with PSIS khat diagnostics
                - 'p_loo_psis': LOO predictive probabilities (N_data,)
                - 'll_loo_psis': LOO log-likelihoods (N_data,)
                - 'n_iterations': actual number of iterations used
                - 'proposal_means': final proposal means (N_proposals, D)
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
        n_proposals = min(n_proposals, S)
        rng_key, subkey = jax.random.split(rng_key)
        proposal_indices = jax.random.choice(
            subkey, S, shape=(n_proposals,), replace=False
        )
        means = flat_params[proposal_indices]  # (N, D)

        # Initial covariance: empirical covariance of posterior samples
        emp_cov = jnp.cov(flat_params.T)  # (D, D)
        if D == 1:
            emp_cov = emp_cov.reshape(1, 1)
        emp_cov = emp_cov + 1e-6 * jnp.eye(D, dtype=jnp.float64)
        covs = jnp.broadcast_to(emp_cov, (n_proposals, D, D)).copy()

        # Repulsion decay
        if repulsion_decay is None:
            repulsion_decay = jnp.log(100.0) / max(n_iterations, 1)

        # Full posterior target for adaptation
        def full_log_target(flat_vec):
            return self._full_log_target(flat_vec, unravel_fn, data)

        grad_full_log_target = jax.grad(full_log_target)

        # --- GRAMIS iterations: adapt and sample at each iteration ---
        # Store all proposal means and covs across iterations for DM-MIS
        all_means_history = []  # list of (N, D) arrays
        all_covs_history = []   # list of (N, D, D) arrays
        all_samples_list = []   # list of (N*K, D) arrays

        for t in range(n_iterations):
            G_t = repulsion_G0 * jnp.exp(-repulsion_decay * t)

            if verbose:
                print(f"  Iteration {t+1}/{n_iterations}, G_t={G_t:.6f}")

            # Store current proposals before adaptation
            all_means_history.append(means.copy())
            all_covs_history.append(covs.copy())

            # Draw K samples from each current proposal
            iter_samples = []
            for n in range(n_proposals):
                rng_key, subkey = jax.random.split(rng_key)
                samples_n = jax.random.multivariate_normal(
                    subkey, means[n], covs[n],
                    shape=(n_samples_per_proposal,),
                    dtype=jnp.float64,
                )  # (K, D)
                iter_samples.append(samples_n)
            iter_samples = jnp.concatenate(iter_samples, axis=0)  # (N*K, D)
            all_samples_list.append(iter_samples)

            # Adapt proposal means
            new_means = jnp.zeros_like(means)
            for n in range(n_proposals):
                mu_n = means[n]
                grad_n = grad_full_log_target(mu_n)
                grad_n = jnp.where(jnp.isfinite(grad_n), grad_n, 0.0)

                theta_n = self._backtracking_step_size(
                    full_log_target, mu_n, grad_n, covs[n],
                    max_iter=backtrack_max_iter,
                )

                repulsion = self._compute_repulsion(means, n, G_t, D)

                # Update mean (Eq. 9)
                mu_new = mu_n + theta_n * (covs[n] @ grad_n) + repulsion
                mu_new = jnp.where(jnp.isfinite(mu_new), mu_new, mu_n)
                new_means = new_means.at[n].set(mu_new)

            means = new_means

            # Covariance adaptation (Eq. 13) - optional
            if adapt_covariance:
                for n in range(n_proposals):
                    covs = covs.at[n].set(
                        self._adapt_covariance(
                            full_log_target, means[n], covs[n], D
                        )
                    )

        # Concatenate all samples across iterations
        all_samples = jnp.concatenate(all_samples_list, axis=0)
        # (N * K * T, D)
        total_samples = all_samples.shape[0]

        if verbose:
            print(f"GRAMIS: {total_samples} total samples across {n_iterations} iterations")

        # --- Compute DM-MIS mixture denominator ---
        # Per the paper (Eq. "all" denominator), use ALL proposals from ALL iterations:
        # Phi(x) = (1 / (N*T)) * sum_{n=1}^N sum_{t=1}^T q(x | mu_n^(t), Sigma_n^(t))
        n_total_proposals = n_proposals * n_iterations
        log_q_components = jnp.zeros(
            (total_samples, n_total_proposals), dtype=jnp.float64
        )
        col = 0
        for t in range(n_iterations):
            for n in range(n_proposals):
                log_q_nt = _multivariate_normal_logpdf(
                    all_samples, all_means_history[t][n], all_covs_history[t][n]
                )  # (total_samples,)
                log_q_components = log_q_components.at[:, col].set(log_q_nt)
                col += 1

        # log( (1/(N*T)) * sum_{n,t} exp(log q_{n,t}(x)) )
        log_q_mix = (
            jax.scipy.special.logsumexp(log_q_components, axis=1)
            - jnp.log(jnp.float64(n_total_proposals))
        )  # (total_samples,)

        # --- Compute log-likelihoods for all samples ---
        def unravel_batch(flat_batch):
            return jax.vmap(unravel_fn)(flat_batch)

        all_params = unravel_batch(all_samples)

        # Compute log-likelihoods: (total_samples, N_data)
        ll_samples = self.likelihood_fn.log_likelihood(data, all_params)
        ll_samples = ll_samples.astype(jnp.float64)

        # Compute log prior for all samples
        if self.prior_log_prob_fn is not None:
            log_prior_samples = self.prior_log_prob_fn(all_params)
            log_prior_samples = jnp.atleast_1d(jnp.squeeze(log_prior_samples))
            if log_prior_samples.shape[0] != total_samples:
                log_prior_samples = jnp.broadcast_to(
                    log_prior_samples, (total_samples,)
                )
        else:
            log_prior_samples = jnp.zeros(total_samples, dtype=jnp.float64)

        # --- Compute LOO quantities ---
        # log pi_{-i}(x_k) = log_prior(x_k) + sum_{j != i} ll(x_k, j)
        ll_total = jnp.sum(ll_samples, axis=1)  # (total_samples,)
        log_numerator = (
            log_prior_samples[:, None]
            + ll_total[:, None]
            - ll_samples
        )  # (total_samples, N_data)

        # DM-MIS log weights
        log_weights_raw = log_numerator - log_q_mix[:, None]  # (total_samples, N_data)

        # --- PSIS smoothing and LOO estimates ---
        lw_smoothed, khat_vals = psis.psislw(log_weights_raw)

        # Ensure khat_vals is always 1-D (psislw returns scalar when N_data=1)
        khat_vals = jnp.atleast_1d(khat_vals)

        # Ensure lw_smoothed is always 2-D
        if lw_smoothed.ndim == 1:
            lw_smoothed = lw_smoothed[:, None]

        # LOO log-likelihood estimate
        ll_loo_psis = jax.scipy.special.logsumexp(
            lw_smoothed + ll_samples, axis=0
        )  # (N_data,)

        p_loo_psis = jnp.exp(ll_loo_psis)

        if verbose:
            n_good = jnp.sum(khat_vals < khat_threshold)
            n_bad = jnp.sum(khat_vals >= 1.0)
            print(
                f"GRAMIS results: "
                f"khat < {khat_threshold}: {n_good}/{N_data}, "
                f"khat >= 1.0: {n_bad}/{N_data}"
            )
            print(
                f"  Mean khat: {jnp.mean(khat_vals):.4f}, "
                f"Max khat: {jnp.max(khat_vals):.4f}"
            )

        return {
            'khat': khat_vals,
            'p_loo_psis': p_loo_psis,
            'll_loo_psis': ll_loo_psis,
            'n_iterations': n_iterations,
            'proposal_means': means,
        }

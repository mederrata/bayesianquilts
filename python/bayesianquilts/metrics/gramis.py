#!/usr/bin/env python3
"""
GRAMIS: Gradient-based Adaptive Multiple Importance Sampling for LOO-CV

Implements GRAMIS (Elvira et al. 2023, arXiv:2210.10785v3) as a baseline
importance sampler for LOO-CV. For each observation i, GRAMIS builds a
Gaussian mixture proposal adapted to the LOO posterior:
    pi_{-i}(theta) propto pi_0(theta) * prod_{j != i} ell(y_j | theta)

This implementation uses a **defensive mixture** approach that combines:
1. The original S posterior samples (reweighted by 1/ell_i as in standard
   PSIS-LOO), which provide good baseline coverage of the LOO posterior
2. Fresh samples from an adapted Gaussian mixture, which provide additional
   coverage in regions where the posterior-to-LOO shift is large

The DM-MIS weighting scheme properly accounts for both sample sources.

Algorithm:
1. Initialize proposals from posterior samples
2. Adapt proposal means via gradient ascent on the full posterior (shared)
3. For each LOO observation i:
   a. Compute standard PSIS weights for the posterior samples (1/ell_i)
   b. Shift Gaussian means toward pi_{-i} (perturbative correction)
   c. Adapt proposal covariance via diagonal Hessian
   d. Draw fresh samples from the adapted mixture
   e. Combine posterior + mixture samples with DM-MIS weights
   f. PSIS smooth the combined weights
"""

import jax
import jax.numpy as jnp
import jax.flatten_util
from typing import Dict, Any, Optional, Callable

from bayesianquilts.metrics.ais import LikelihoodFunction
from bayesianquilts.metrics import psis


def _diagonal_normal_logpdf(x, mean, log_scale):
    """Log-pdf of diagonal-covariance multivariate normal.

    Args:
        x: points, shape (..., D)
        mean: mean, shape (D,)
        log_scale: log standard deviations, shape (D,)

    Returns:
        Log-pdf values, shape (...)
    """
    d = mean.shape[0]
    diff = x - mean
    inv_var = jnp.exp(-2.0 * log_scale)
    maha = jnp.sum(diff ** 2 * inv_var, axis=-1)
    log_det = 2.0 * jnp.sum(log_scale)
    return -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det + maha)


class GRAMISSampler:
    """GRAMIS sampler adapted for LOO-CV importance sampling.

    Uses a defensive mixture of posterior samples + adapted Gaussian
    proposals. The posterior samples provide robust baseline weights
    (equivalent to PSIS-LOO), while the Gaussian proposals provide
    additional samples that may improve estimates for problematic
    observations.

    Uses diagonal covariance for numerical stability in high-D.
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

    def gramis_loo(
        self,
        data: Any,
        params: Dict[str, Any],
        full_data: Any = None,
        n_proposals: int = 20,
        n_samples_per_proposal: int = 10,
        n_iterations: int = 10,
        repulsion_G0: float = 0.05,
        repulsion_decay: Optional[float] = None,
        backtrack_max_iter: int = 10,
        khat_threshold: float = 0.7,
        verbose: bool = False,
        rng_key: Optional[jnp.ndarray] = None,
        adapt_covariance: bool = True,
        defensive_alpha: float = 0.5,
    ) -> dict:
        """Run GRAMIS for LOO-CV with defensive mixture.

        Combines posterior samples (standard PSIS weights) with fresh
        samples from an adapted Gaussian mixture. The defensive mixture
        ensures the estimator is at least as good as standard PSIS.

        Args:
            data: Observations to compute LOO for (batch of bad obs).
            params: Posterior samples as dict {name: array(S, ...)}.
            full_data: Full dataset for the full posterior target.
            n_proposals: Number of Gaussian proposals.
            n_samples_per_proposal: Samples drawn from each proposal.
            n_iterations: Adaptation iterations for shared pass.
            repulsion_G0: Initial repulsion strength.
            repulsion_decay: Decay rate for repulsion.
            backtrack_max_iter: Max backtracking iterations.
            khat_threshold: PSIS khat threshold.
            verbose: Print progress.
            rng_key: JAX PRNG key.
            adapt_covariance: If True, adapt proposal covariance using
                diagonal Hessian of the LOO target.
            defensive_alpha: Weight of posterior samples in defensive
                mixture (0 < alpha < 1). Higher = more weight on posterior
                samples (safer). Default 0.5.

        Returns:
            Dictionary with 'khat', 'p_loo_psis', 'll_loo_psis', etc.
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        if full_data is None:
            full_data = data

        # Flatten params to (S, D)
        flat_params, unravel_fn = self._flatten_params(params)
        flat_params = flat_params.astype(jnp.float64)
        S, D = flat_params.shape

        # Disable defensive mixture for high-D: a diagonal Gaussian
        # approximation to the posterior is meaningless when D >> ~50,
        # causing all weights to become uniform and khat artificially low.
        MAX_D_DEFENSIVE = 50
        use_defensive = defensive_alpha > 0 and D <= MAX_D_DEFENSIVE

        if verbose:
            print(f"GRAMIS: S={S} posterior samples, D={D} parameters")
            if not use_defensive and defensive_alpha > 0:
                print(f"GRAMIS: D={D} > {MAX_D_DEFENSIVE}, "
                      f"disabling defensive mixture (Gaussian approx unreliable)")

        # Compute per-observation log-likelihoods on the LOO batch
        ll_batch = self.likelihood_fn.log_likelihood(data, params)  # (S, N_batch)
        ll_batch = ll_batch.astype(jnp.float64)
        N_batch = ll_batch.shape[1]

        if verbose:
            print(f"GRAMIS: N_batch={N_batch} LOO obs, "
                  f"defensive={'on' if use_defensive else 'off'} "
                  f"(alpha={defensive_alpha:.2f})")

        # --- Initialize proposals ---
        n_proposals = min(n_proposals, S)
        rng_key, subkey = jax.random.split(rng_key)
        proposal_indices = jax.random.choice(
            subkey, S, shape=(n_proposals,), replace=False
        )
        means_init = flat_params[proposal_indices]  # (N_proposals, D)

        # Initial scale: per-dimension std of posterior
        emp_std = jnp.maximum(jnp.std(flat_params, axis=0), 1e-6)
        log_scales_init = jnp.log(emp_std)  # (D,)

        # Repulsion decay
        if repulsion_decay is None:
            repulsion_decay = jnp.log(100.0) / max(n_iterations, 1)

        # --- Define target functions ---
        def _log_lik_obs_i(flat_vec, i):
            """Log-likelihood of single observation i from the LOO batch."""
            p = self._make_params_from_flat(flat_vec, unravel_fn)
            ll = self.likelihood_fn.log_likelihood(data, p)  # (1, N_batch)
            return ll[0, i]

        def _log_lik_full(flat_vec):
            """Full log-likelihood (all observations)."""
            p = self._make_params_from_flat(flat_vec, unravel_fn)
            ll = self.likelihood_fn.log_likelihood(full_data, p)  # (1, N_full)
            return jnp.sum(ll)

        def _log_prior(flat_vec):
            if self.prior_log_prob_fn is not None:
                p = self._make_params_from_flat(flat_vec, unravel_fn)
                return jnp.squeeze(self.prior_log_prob_fn(p))
            return jnp.float64(0.0)

        def full_log_target(flat_vec):
            return _log_lik_full(flat_vec) + _log_prior(flat_vec)

        grad_full = jax.grad(full_log_target)
        grad_lik_obs = jax.grad(_log_lik_obs_i)

        # For covariance adaptation: compute Hessian diagonal of log ell_i
        if adapt_covariance:
            def _hessian_diag_obs_i(flat_vec, i):
                """Compute diagonal of Hessian of log ell_i via forward-over-reverse AD."""
                g_fn = jax.grad(lambda x: _log_lik_obs_i(x, i))

                def hvp_col(e_j):
                    _, col = jax.jvp(g_fn, (flat_vec,), (e_j,))
                    return col

                eye = jnp.eye(D, dtype=flat_vec.dtype)
                H_cols = jax.vmap(hvp_col)(eye)  # (D, D)
                return jnp.diag(H_cols)  # (D,)

            def _estimate_loo_variance(flat_vec, i, var_prior):
                """Estimate LOO posterior variance via perturbative Hessian update.

                Sigma_{-i}^{-1} = Sigma^{-1} + H_i  (H_i <= 0 for log-concave)
                Diagonal: var_{-i,j} = var_j / (1 + var_j * H_{i,jj})
                """
                hess_diag = _hessian_diag_obs_i(flat_vec, i)
                hess_diag = jnp.where(jnp.isfinite(hess_diag), hess_diag, 0.0)
                denom = 1.0 + var_prior * hess_diag
                safe_denom = jnp.where(denom > 0.1, denom, 0.1)
                return var_prior / safe_denom

        max_grad_norm = 10.0 * jnp.sqrt(jnp.float64(D))

        # ============================================================
        # Phase 1: Shared adaptation toward full posterior
        # ============================================================
        means = means_init.copy()
        all_log_scales = jnp.broadcast_to(
            log_scales_init, (n_proposals, D)
        ).copy()

        for t in range(n_iterations):
            G_t = repulsion_G0 * jnp.exp(-repulsion_decay * t)

            new_means = jnp.zeros_like(means)
            for n in range(n_proposals):
                mu_n = means[n]
                grad_n = grad_full(mu_n)
                grad_n = jnp.where(jnp.isfinite(grad_n), grad_n, 0.0)

                # Clip gradient
                gn = jnp.sqrt(jnp.sum(grad_n ** 2) + 1e-12)
                grad_n = jnp.where(gn > max_grad_norm,
                                   grad_n * max_grad_norm / gn, grad_n)

                var_n = jnp.exp(2.0 * all_log_scales[n])
                direction = var_n * grad_n

                # Backtracking line search
                current_val = full_log_target(mu_n)
                current_val = jnp.where(
                    jnp.isfinite(current_val), current_val, jnp.float64(-1e30))

                theta = jnp.float64(1.0)
                for _ in range(backtrack_max_iter):
                    cand = mu_n + theta * direction
                    cand_val = full_log_target(cand)
                    ok = jnp.logical_and(jnp.isfinite(cand_val),
                                         cand_val >= current_val)
                    theta = jnp.where(ok, theta, theta * 0.5)

                repulsion = self._compute_repulsion(means, n, G_t)
                mu_new = mu_n + theta * direction + repulsion
                mu_new = jnp.where(jnp.isfinite(mu_new), mu_new, mu_n)
                new_means = new_means.at[n].set(mu_new)

            means = new_means

        if verbose:
            print(f"GRAMIS: Shared adaptation done ({n_iterations} iters).")

        # ============================================================
        # Fit Gaussian to posterior samples for defensive mixture
        # ============================================================
        if use_defensive:
            post_mean = jnp.mean(flat_params, axis=0)
            post_var = jnp.var(flat_params, axis=0)
            post_var = jnp.maximum(post_var, 1e-12)
            post_log_scale = 0.5 * jnp.log(post_var)

        # ============================================================
        # Phase 2: Per-observation LOO (defensive or pure GRAMIS)
        # ============================================================
        khat_all = jnp.full(N_batch, jnp.float64(10.0))
        ll_loo_all = jnp.full(N_batch, jnp.float64(-jnp.inf))

        for i in range(N_batch):
            rng_key, subkey = jax.random.split(rng_key)

            # --- Shift Gaussian proposals toward pi_{-i} ---
            shifted_means = jnp.zeros_like(means)
            shifted_log_scales = jnp.zeros_like(all_log_scales)

            for n in range(n_proposals):
                grad_obs_i = grad_lik_obs(means[n], i)
                grad_obs_i = jnp.where(
                    jnp.isfinite(grad_obs_i), grad_obs_i, 0.0)
                gi_norm = jnp.sqrt(jnp.sum(grad_obs_i ** 2) + 1e-12)
                grad_obs_i = jnp.where(
                    gi_norm > max_grad_norm,
                    grad_obs_i * max_grad_norm / gi_norm,
                    grad_obs_i)

                var_n = jnp.exp(2.0 * all_log_scales[n])
                mu_shifted = means[n] - var_n * grad_obs_i
                mu_shifted = jnp.where(
                    jnp.isfinite(mu_shifted), mu_shifted, means[n])
                shifted_means = shifted_means.at[n].set(mu_shifted)

                if adapt_covariance:
                    loo_var = _estimate_loo_variance(mu_shifted, i, var_n)
                    loo_var = jnp.where(
                        jnp.isfinite(loo_var), loo_var, var_n * 2.0)
                    loo_var = jnp.clip(loo_var, var_n * 0.1, var_n * 10.0)
                    new_log_scale = 0.5 * jnp.log(loo_var)
                    shifted_log_scales = shifted_log_scales.at[n].set(
                        new_log_scale)
                else:
                    shifted_log_scales = shifted_log_scales.at[n].set(
                        all_log_scales[n])

            # --- Draw fresh samples from the shifted mixture ---
            n_mix = n_proposals * n_samples_per_proposal
            mix_samples = []
            for n in range(n_proposals):
                subkey, sk = jax.random.split(subkey)
                z = jax.random.normal(
                    sk, shape=(n_samples_per_proposal, D), dtype=jnp.float64)
                scale = jnp.exp(shifted_log_scales[n])
                samples_n = shifted_means[n] + scale * z
                mix_samples.append(samples_n)
            mix_samples = jnp.concatenate(mix_samples, axis=0)  # (n_mix, D)

            # --- Evaluate ell_i at mixture samples ---
            mix_params = jax.vmap(unravel_fn)(mix_samples)
            ll_mix_batch = self.likelihood_fn.log_likelihood(
                data, mix_params)  # (n_mix, N_batch)
            ll_mix_batch = ll_mix_batch.astype(jnp.float64)
            ll_mix_obs_i = ll_mix_batch[:, i]

            # --- Compute importance weights ---
            # log q_mix at mixture samples (needed for both modes)
            log_q_components = jnp.full(
                (n_mix, n_proposals), -jnp.inf, dtype=jnp.float64)
            for n in range(n_proposals):
                lq = _diagonal_normal_logpdf(
                    mix_samples, shifted_means[n], shifted_log_scales[n])
                lq = jnp.where(jnp.isfinite(lq), lq, jnp.float64(-1e30))
                log_q_components = log_q_components.at[:, n].set(lq)
            log_q_mix_at_mix = (
                jax.scipy.special.logsumexp(log_q_components, axis=1)
                - jnp.log(jnp.float64(n_proposals))
            )

            if use_defensive:
                # --- Defensive mixture: combine posterior + mixture samples ---
                log_alpha = jnp.log(defensive_alpha)
                log_1malpha = jnp.log(1.0 - defensive_alpha)

                # log p_post (Gaussian approx) at mixture samples
                log_ppost_at_mix = _diagonal_normal_logpdf(
                    mix_samples, post_mean, post_log_scale)
                log_qdm_at_mix = jnp.logaddexp(
                    log_alpha + log_ppost_at_mix,
                    log_1malpha + log_q_mix_at_mix)
                log_w_mix = -ll_mix_obs_i + log_ppost_at_mix - log_qdm_at_mix

                # Posterior sample weights
                log_q_at_post = jnp.full(
                    (S, n_proposals), -jnp.inf, dtype=jnp.float64)
                for n in range(n_proposals):
                    lq = _diagonal_normal_logpdf(
                        flat_params, shifted_means[n], shifted_log_scales[n])
                    lq = jnp.where(jnp.isfinite(lq), lq, jnp.float64(-1e30))
                    log_q_at_post = log_q_at_post.at[:, n].set(lq)
                log_q_mix_at_post = (
                    jax.scipy.special.logsumexp(log_q_at_post, axis=1)
                    - jnp.log(jnp.float64(n_proposals))
                )
                log_ppost_at_post = _diagonal_normal_logpdf(
                    flat_params, post_mean, post_log_scale)
                log_qdm_at_post = jnp.logaddexp(
                    log_alpha + log_ppost_at_post,
                    log_1malpha + log_q_mix_at_post)
                log_w_post = -ll_batch[:, i] + log_ppost_at_post - log_qdm_at_post

                # Combine
                log_w_all = jnp.concatenate([log_w_post, log_w_mix])
                ll_obs_i_all = jnp.concatenate([ll_batch[:, i], ll_mix_obs_i])
            else:
                # --- Pure GRAMIS: fresh samples only, weight = pi_{-i}/q_mix ---
                # Evaluate full log-likelihood at mixture samples
                ll_mix_full = self.likelihood_fn.log_likelihood(
                    full_data, mix_params)  # (n_mix, N_full)
                ll_mix_full = ll_mix_full.astype(jnp.float64)
                ll_mix_full = jnp.where(
                    jnp.isfinite(ll_mix_full), ll_mix_full, jnp.float64(-1e30))
                ll_full_at_mix = jnp.sum(ll_mix_full, axis=1)

                # Prior at mixture samples
                if self.prior_log_prob_fn is not None:
                    lp = self.prior_log_prob_fn(mix_params)
                    lp = jnp.atleast_1d(jnp.squeeze(lp))
                    if lp.shape[0] != n_mix:
                        lp = jnp.broadcast_to(lp, (n_mix,))
                    lp = jnp.where(jnp.isfinite(lp), lp, jnp.float64(-1e30))
                else:
                    lp = jnp.zeros(n_mix, dtype=jnp.float64)

                # LOO log-target: prior + full LL - ell_i
                log_loo_target = lp + ll_full_at_mix - ll_mix_obs_i
                log_w_all = log_loo_target - log_q_mix_at_mix
                ll_obs_i_all = ll_mix_obs_i

            lw_smooth, khat_i = psis.psislw(log_w_all[:, None])
            khat_i = jnp.atleast_1d(khat_i)
            if lw_smooth.ndim == 2:
                lw_smooth = lw_smooth[:, 0]

            khat_all = khat_all.at[i].set(khat_i[0])

            ll_loo_i = jax.scipy.special.logsumexp(lw_smooth + ll_obs_i_all)
            ll_loo_all = ll_loo_all.at[i].set(ll_loo_i)

        p_loo_psis = jnp.exp(ll_loo_all)

        if verbose:
            n_good = jnp.sum(khat_all < khat_threshold)
            n_bad = jnp.sum(khat_all >= 1.0)
            print(
                f"GRAMIS results: "
                f"khat < {khat_threshold}: {n_good}/{N_batch}, "
                f"khat >= 1.0: {n_bad}/{N_batch}"
            )
            print(
                f"  Mean khat: {jnp.mean(khat_all):.4f}, "
                f"Max khat: {jnp.max(khat_all):.4f}"
            )

        return {
            'khat': khat_all,
            'p_loo_psis': p_loo_psis,
            'll_loo_psis': ll_loo_all,
            'n_iterations': n_iterations,
            'proposal_means': means,
        }

    def _compute_repulsion(self, means, n_idx, G_t):
        """Compute repulsive force on proposal n from all others.

        Uses Euclidean exponent (dim=2) to avoid overflow in high-D.
        """
        mu_n = means[n_idx]
        n_proposals = means.shape[0]

        def repulsion_from_j(j):
            d_nj = mu_n - means[j]
            dist_sq = jnp.sum(d_nj ** 2) + 1e-12
            r = G_t * d_nj / dist_sq
            return jnp.where(j == n_idx, jnp.zeros_like(r), r)

        repulsions = jax.vmap(repulsion_from_j)(jnp.arange(n_proposals))
        total = jnp.sum(repulsions, axis=0)
        rep_norm = jnp.sqrt(jnp.sum(total ** 2) + 1e-12)
        max_rep = jnp.std(means, axis=0).mean() * 0.1
        max_rep = jnp.maximum(max_rep, 1e-6)
        total = jnp.where(rep_norm > max_rep, total * max_rep / rep_norm, total)
        return total

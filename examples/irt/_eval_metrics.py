"""Manuscript-style RMSE + PSIS-LOO ELPD from a marginal-MCMC NPZ.

This mirrors ``notebooks/irt/eval_mcmc.compute_metrics_from_npz`` so the
example pipelines (``fit_imputed_irt.py`` etc.) report the same numbers
that journal_article.tex's tables use.

Usage::

    from _eval_metrics import compute_metrics_from_npz, print_metrics_table
    m = compute_metrics_from_npz(npz_path, batch, item_keys, K, num_people)
    print_metrics_table({'baseline': m_baseline, 'imputed': m_imputed})
"""

from __future__ import annotations

import gc
from typing import Dict, Mapping, Optional, Sequence

import numpy as np


def compute_metrics_from_npz(
    npz_path,
    batch: Mapping[str, np.ndarray],
    item_keys: Sequence[str],
    K: int,
    num_people: int,
    *,
    eap_fallback: Optional[np.ndarray] = None,
    max_S: int = 200,
) -> Dict[str, float]:
    """Compute RMSE + PSIS-LOO ELPD from a marginal-MCMC NPZ.

    The NPZ schema matches ``run_marginal_mcmc.py``'s output:
    ``discriminations``, ``difficulties0``, ``ddifficulties`` arrays of
    shape ``(chains, samples, ...)`` plus ``eap`` of shape ``(N,)``.

    Returns a dict with ``rmse``, ``rmse_se``, ``elpd``, ``elpd_per_person``,
    ``elpd_se_per_person``, ``elpd_per_resp``, ``elpd_se_per_resp``,
    ``n_obs``, ``n_bad_khat``.
    """
    from bayesianquilts.metrics.nppsis import psisloo

    mcmc = np.load(str(npz_path))
    disc = mcmc['discriminations']
    diff0 = mcmc['difficulties0']
    ddiff = mcmc.get('ddifficulties', None)

    n_chains, n_samp = disc.shape[:2]
    S = n_chains * n_samp
    I = len(item_keys)

    disc_squeezed = disc.reshape(S, -1).astype(np.float64)
    if disc_squeezed.size == S * I:
        disc_flat = disc_squeezed.reshape(S, I)
    elif disc_squeezed.size == S:
        disc_flat = np.broadcast_to(disc_squeezed.reshape(S, 1),
                                    (S, I)).copy()
    else:
        disc_flat = disc.squeeze().reshape(S, I).astype(np.float64)
    diff0_flat = diff0.squeeze().reshape(S, I, -1).astype(np.float64)
    ddiff_flat = (ddiff.squeeze().reshape(S, I, -1).astype(np.float64)
                  if ddiff is not None and ddiff.size > 0 else None)

    eap = mcmc.get('eap', eap_fallback)
    if eap is None:
        raise ValueError("No EAP abilities in npz or fallback")
    eap = np.asarray(eap, dtype=np.float64).flatten()

    rng = np.random.default_rng(42)
    use_S = min(S, max_S)
    idx = rng.choice(S, use_S, replace=False) if use_S < S else np.arange(S)
    disc_use = disc_flat[idx]
    diff0_use = diff0_flat[idx]
    ddiff_use = ddiff_flat[idx] if ddiff_flat is not None else None
    if ddiff_use is not None:
        diffs_use = np.cumsum(
            np.concatenate([diff0_use, ddiff_use], axis=-1), axis=-1)
    else:
        diffs_use = diff0_use

    disc_mean = disc_flat.mean(0)
    diff0_mean = diff0_flat.mean(0)
    if ddiff_flat is not None:
        ddiff_mean = ddiff_flat.mean(0)
        diffs_mean = np.cumsum(
            np.concatenate([diff0_mean, ddiff_mean], axis=-1), axis=-1)
    else:
        diffs_mean = diff0_mean

    obs_matrix = np.full((num_people, I), -1.0, dtype=np.float64)
    for i, key in enumerate(item_keys):
        obs = np.asarray(batch[key], dtype=np.float64)
        obs_matrix[:, i] = np.where(
            np.isnan(obs) | (obs < 0) | (obs >= K), -1.0, obs)
    obs_int = obs_matrix.astype(int)
    mask = (obs_matrix >= 0) & (obs_matrix < K)
    n_obs = int(mask.sum())

    # RMSE via posterior mean
    logits_mean = (disc_mean[None, :, None]
                   * (eap[:, None, None] - diffs_mean[None, :, :]))
    cum_p_mean = 1.0 / (1.0 + np.exp(-logits_mean))
    p_mean = np.zeros((num_people, I, K))
    p_mean[:, :, 0] = 1.0 - cum_p_mean[:, :, 0]
    for k in range(1, K - 1):
        p_mean[:, :, k] = cum_p_mean[:, :, k - 1] - cum_p_mean[:, :, k]
    p_mean[:, :, K - 1] = cum_p_mean[:, :, K - 2]
    p_mean = np.maximum(p_mean, 1e-30)
    p_mean /= p_mean.sum(axis=-1, keepdims=True)
    expected = np.sum(p_mean * np.arange(K, dtype=np.float64)[None, None, :],
                      axis=-1)
    sq_errors = (obs_matrix[mask] - expected[mask]) ** 2
    rmse = float(np.sqrt(np.mean(sq_errors)))
    n_resp = len(sq_errors)
    rmse_se = (float(np.std(sq_errors, ddof=1)) / np.sqrt(n_resp) / (2 * rmse)
               if n_resp > 1 and rmse > 0 else float('nan'))

    # PSIS-LOO ELPD (vectorized, chunked)
    bytes_per_s = num_people * I * K * 8
    chunk_S = max(1, min(use_S, int(2e9 / bytes_per_s)))

    log_lik = np.zeros((use_S, num_people))
    for start in range(0, use_S, chunk_S):
        end = min(start + chunk_S, use_S)
        s_chunk = end - start
        disc_c = disc_use[start:end]
        diffs_c = diffs_use[start:end]

        theta = eap[None, :, None, None]
        diff = diffs_c[:, None, :, :]
        a = disc_c[:, None, :, None]
        logits = a * (theta - diff)
        cum_p = 1.0 / (1.0 + np.exp(-logits))

        p = np.zeros((s_chunk, num_people, I, K))
        p[:, :, :, 0] = 1.0 - cum_p[:, :, :, 0]
        for k in range(1, K - 1):
            p[:, :, :, k] = cum_p[:, :, :, k - 1] - cum_p[:, :, :, k]
        p[:, :, :, K - 1] = cum_p[:, :, :, K - 2]
        p = np.maximum(p, 1e-30)
        p /= p.sum(axis=-1, keepdims=True)

        obs_clamped = np.where(obs_int >= 0, obs_int, 0)
        s_idx = np.arange(s_chunk)[:, None, None]
        n_idx = np.arange(num_people)[None, :, None]
        i_idx = np.arange(I)[None, None, :]
        log_p_obs = np.log(p[s_idx, n_idx, i_idx, obs_clamped[None, :, :]])
        log_p_obs *= mask[None, :, :]
        log_lik[start:end] = log_p_obs.sum(axis=-1)
        del logits, cum_p, p, log_p_obs
        gc.collect()

    loo, loos, ks = psisloo(log_lik)
    n_bad_k = int(np.sum(ks > 0.7))
    return {
        'rmse': rmse,
        'rmse_se': float(rmse_se),
        'n_obs': n_obs,
        'elpd': float(loo),
        'elpd_per_person': float(loo / num_people),
        'elpd_se_per_person': float(np.std(loos) * np.sqrt(num_people)
                                    / num_people),
        'elpd_per_resp': float(loo / n_obs),
        'elpd_se_per_resp': float(np.std(loos) * np.sqrt(num_people) / n_obs),
        'n_bad_khat': n_bad_k,
    }


def print_metrics_table(metrics: Mapping[str, Mapping[str, float]]) -> None:
    """Manuscript-style summary table for one or more variants."""
    print(f"\n{'='*78}")
    print(f"  {'Variant':<12} {'RMSE':>16} {'ELPD/n':>22} "
          f"{'ELPD/resp':>22}")
    print(f"  {'-'*76}")
    for variant, m in metrics.items():
        rmse = f"{m['rmse']:.4f} ({m['rmse_se']:.4f})"
        elpd_n = (f"{m['elpd_per_person']:.4f} "
                  f"({m['elpd_se_per_person']:.4f})")
        elpd_r = (f"{m['elpd_per_resp']:.4f} "
                  f"({m['elpd_se_per_resp']:.4f})")
        bad_k = f" [k>0.7: {m['n_bad_khat']}]" if m.get('n_bad_khat') else ""
        print(f"  {variant:<12} {rmse:>16} {elpd_n:>22} {elpd_r:>22}{bad_k}")
    print(f"{'='*78}\n")

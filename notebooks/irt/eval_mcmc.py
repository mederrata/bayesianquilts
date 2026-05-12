#!/usr/bin/env python
"""Evaluate MCMC-fitted models: Pred RMSE and ELPD from MCMC samples.

Evaluates all 3 variants (baseline, pairwise, mixed) when available.
Uses MCMC posterior samples directly for PSIS-LOO ELPD computation.
"""

import os
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import argparse
import importlib
import inspect
import gc
import numpy as np
import jax.numpy as jnp
from pathlib import Path

from run_single_notebook import DATASET_CONFIGS, make_data_dict, calibrate_manually
from bayesianquilts.metrics.nppsis import psisloo


VARIANTS = ['baseline', 'pairwise', 'mixed']


def compute_pred_rmse(model, batch, item_keys, K):
    """Compute predictive RMSE from calibrated model expectations."""
    ce = model.calibrated_expectations
    probs = model.grm_model_prob_d(
        ce['abilities'], ce['discriminations'],
        ce['difficulties0'], ce.get('ddifficulties'))
    categories = jnp.arange(K, dtype=jnp.float64)
    expected = jnp.sum(probs * categories[None, :], axis=-1)

    se_sum = 0.0
    count = 0
    for i, key in enumerate(item_keys):
        obs = np.array(batch[key], dtype=np.float64)
        pred_i = np.array(expected[:, i])
        valid = ~np.isnan(obs) & (obs >= 0) & (obs < K)
        se_sum += np.sum((obs[valid] - pred_i[valid]) ** 2)
        count += int(np.sum(valid))
    return float(np.sqrt(se_sum / count))


def compute_metrics_from_npz(npz_path, batch, item_keys, K, num_people,
                              eap_fallback=None, max_S=200):
    """Compute RMSE + PSIS-LOO ELPD from MCMC samples (vectorized, chunked).

    Mirrors eval_promis_new.compute_metrics_from_npz: posterior mean params
    for RMSE; subsampled posterior for ELPD; thresholds reconstructed as
    cumsum(concat([d0, dd])) without any softplus -- MCMC samples are
    already in constrained (positive) space because the prior is HalfNormal
    and marginal_log_prob evaluates the constrained density directly.
    """
    mcmc = np.load(str(npz_path))
    disc = mcmc['discriminations']
    diff0 = mcmc['difficulties0']
    ddiff = mcmc.get('ddifficulties', None)

    n_chains, n_samp = disc.shape[:2]
    S = n_chains * n_samp
    I = len(item_keys)

    # Squeeze trailing/middle singleton axes so disc is (S, I), diff0/ddiff are (S, I, *).
    # Shared-discrimination GRM stores a single (1, D, 1, 1) per sample -> broadcast to I.
    disc_squeezed = disc.reshape(S, -1).astype(np.float64)
    if disc_squeezed.size == S * I:
        disc_flat = disc_squeezed.reshape(S, I)
    elif disc_squeezed.size == S:
        disc_flat = np.broadcast_to(disc_squeezed.reshape(S, 1), (S, I)).copy()
    else:
        disc_flat = disc.squeeze().reshape(S, I).astype(np.float64)
    diff0_flat = diff0.squeeze().reshape(S, I, -1).astype(np.float64)
    if ddiff is not None and ddiff.size > 0:
        ddiff_flat = ddiff.squeeze().reshape(S, I, -1).astype(np.float64)
    else:
        ddiff_flat = None

    eap = mcmc.get('eap', eap_fallback)
    if eap is None:
        raise ValueError("No EAP abilities in npz or fallback")
    eap = np.array(eap, dtype=np.float64).flatten()

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

    # Posterior mean for RMSE
    disc_mean = disc_flat.mean(0)
    diff0_mean = diff0_flat.mean(0)
    if ddiff_flat is not None:
        ddiff_mean = ddiff_flat.mean(0)
        diffs_mean = np.cumsum(
            np.concatenate([diff0_mean, ddiff_mean], axis=-1), axis=-1)
    else:
        diffs_mean = diff0_mean

    # Observation matrix (N, I), -1 for missing
    obs_matrix = np.full((num_people, I), -1.0, dtype=np.float64)
    for i, key in enumerate(item_keys):
        obs = np.asarray(batch[key], dtype=np.float64)
        obs_matrix[:, i] = np.where(
            np.isnan(obs) | (obs < 0) | (obs >= K), -1.0, obs)
    obs_int = obs_matrix.astype(int)
    mask = (obs_matrix >= 0) & (obs_matrix < K)
    n_obs = int(mask.sum())

    # RMSE via posterior mean params
    logits_mean = disc_mean[None, :, None] * (eap[:, None, None] - diffs_mean[None, :, :])
    cum_p_mean = 1.0 / (1.0 + np.exp(-logits_mean))
    p_mean = np.zeros((num_people, I, K))
    p_mean[:, :, 0] = 1.0 - cum_p_mean[:, :, 0]
    for k in range(1, K - 1):
        p_mean[:, :, k] = cum_p_mean[:, :, k - 1] - cum_p_mean[:, :, k]
    p_mean[:, :, K - 1] = cum_p_mean[:, :, K - 2]
    p_mean = np.maximum(p_mean, 1e-30)
    p_mean /= p_mean.sum(axis=-1, keepdims=True)
    expected = np.sum(p_mean * np.arange(K, dtype=np.float64)[None, None, :], axis=-1)
    sq_errors = (obs_matrix[mask] - expected[mask]) ** 2
    rmse = float(np.sqrt(np.mean(sq_errors)))
    n_resp = len(sq_errors)
    if n_resp > 1 and rmse > 0:
        se_mean_sq = float(np.std(sq_errors, ddof=1)) / np.sqrt(n_resp)
        rmse_se = se_mean_sq / (2 * rmse)
    else:
        rmse_se = float('nan')

    # LOO-ELPD vectorized, chunked
    bytes_per_s = num_people * I * K * 8
    chunk_S = max(1, min(use_S, int(2e9 / bytes_per_s)))
    print(f"    Vectorized: use_S={use_S}, N={num_people}, I={I}, K={K}, chunk_S={chunk_S}",
          flush=True)

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
        if end < use_S:
            print(f"      chunk {end}/{use_S}", flush=True)

    loo, loos, ks = psisloo(log_lik)
    n_bad_k = int(np.sum(ks > 0.7))
    return {
        'rmse': rmse,
        'rmse_se': float(rmse_se),
        'n_obs': n_obs,
        'elpd': float(loo),
        'elpd_per_person': float(loo / num_people),
        'elpd_se_per_person': float(np.std(loos) * np.sqrt(num_people) / num_people),
        'elpd_per_resp': float(loo / n_obs),
        'elpd_se_per_resp': float(np.std(loos) * np.sqrt(num_people) / n_obs),
        'n_bad_khat': n_bad_k,
    }


def eval_mcmc_dataset(dataset_name):
    from bayesianquilts.irt.grm import GRModel

    config = DATASET_CONFIGS[dataset_name]
    mod = importlib.import_module(config['module'])
    item_keys = mod.item_keys
    response_cardinality = mod.response_cardinality
    K = response_cardinality

    work_dir = Path(__file__).parent / dataset_name
    os.chdir(work_dir)

    get_data_kwargs = {'polars_out': True}
    if 'reorient' in inspect.signature(mod.get_data).parameters:
        get_data_kwargs['reorient'] = True
    df, num_people = mod.get_data(**get_data_kwargs)
    batch = make_data_dict(df)

    n_observed_responses = int(sum(
        np.sum((batch[k] >= 0) & (batch[k] < K) & ~np.isnan(batch[k]))
        for k in item_keys
    ))

    print(f"\n{'='*80}")
    print(f"  {dataset_name.upper()}: N={num_people}, items={len(item_keys)}, K={K}")
    print(f"{'='*80}")

    results = {}

    for variant in VARIANTS:
        npz_path = work_dir / 'mcmc_samples' / f'mcmc_{variant}.npz'
        if not npz_path.exists():
            print(f"  {variant}: no npz, skipping")
            continue

        print(f"\n  --- {variant} ---", flush=True)

        try:
            m = compute_metrics_from_npz(
                npz_path, batch, item_keys, K, num_people, max_S=200)
            print(f"    RMSE:      {m['rmse']:.4f} ({m['rmse_se']:.4f})", flush=True)
            print(f"    ELPD/n:    {m['elpd_per_person']:.4f} ({m['elpd_se_per_person']:.4f})", flush=True)
            print(f"    ELPD/resp: {m['elpd_per_resp']:.4f} ({m['elpd_se_per_resp']:.4f})", flush=True)
            print(f"    n_bad_k:   {m['n_bad_khat']}/{num_people}", flush=True)
            results[variant] = {'pred_rmse': m['rmse'], **m}
        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
            results[variant] = {'error': str(e)}

        gc.collect()

    # Summary
    print(f"\n  {'Model':<12} {'RMSE':>8} {'ELPD/n':>14} {'ELPD/resp':>14}")
    print(f"  {'-'*52}")
    for variant in VARIANTS:
        if variant not in results:
            continue
        r = results[variant]
        rmse = f"{r['pred_rmse']:.4f}"
        en = f"{r['elpd_per_person']:.4f} ({r['elpd_se_per_person']:.4f})" if 'elpd_per_person' in r else "--"
        er = f"{r['elpd_per_resp']:.4f} ({r['elpd_se_per_resp']:.4f})" if 'elpd_per_resp' in r else "--"
        print(f"  {variant:<12} {rmse:>8} {en:>14} {er:>14}")

    return {
        'dataset': dataset_name,
        'num_people': num_people,
        'n_observed_responses': n_observed_responses,
        'variants': results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    datasets = ['scs', 'gcbs', 'grit', 'rwa', 'tma'] if args.dataset == 'all' else [args.dataset]

    all_results = {}
    for ds in datasets:
        try:
            all_results[ds] = eval_mcmc_dataset(ds)
        except Exception as e:
            import traceback
            print(f"ERROR on {ds}: {e}")
            traceback.print_exc()

    if len(all_results) > 1:
        print(f"\n\n{'='*80}")
        print(f"  MCMC SUMMARY — ALL DATASETS")
        print(f"{'='*80}")
        print(f"{'Dataset':<8} {'Variant':<12} {'RMSE':>8} {'ELPD/n':>18} {'ELPD/resp':>18}")
        print(f"{'-'*68}")
        for ds, dr in all_results.items():
            for variant in VARIANTS:
                if variant not in dr['variants']:
                    continue
                r = dr['variants'][variant]
                rmse = f"{r['pred_rmse']:.4f}"
                en = f"{r['elpd_per_person']:.4f} ({r['elpd_se_per_person']:.4f})" if 'elpd_per_person' in r else "--"
                er = f"{r['elpd_per_resp']:.4f} ({r['elpd_se_per_resp']:.4f})" if 'elpd_per_resp' in r else "--"
                print(f"{ds.upper():<8} {variant:<12} {rmse:>8} {en:>18} {er:>18}")
            print(f"{'-'*68}")


if __name__ == '__main__':
    main()

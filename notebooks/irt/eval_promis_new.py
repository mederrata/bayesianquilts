#!/usr/bin/env python
"""Evaluate PROMIS datasets (W1 + SU banks + sleep) from npz files.

Vectorized computation of R-hat, RMSE (delta-method SE), and LOO-ELPD
per-person and per-response (PSIS).

Usage:
    python eval_promis_new.py --datasets promis_w1__anger promis_su__bank2
    python eval_promis_new.py --all-tankie
    python eval_promis_new.py --all-local
"""

import argparse
import gc
import importlib
import importlib.util
import inspect
import json
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')

import numpy as np
from pathlib import Path

IRT_DIR = Path(__file__).parent.resolve()
PYDIR = IRT_DIR.parent.parent / 'python'

sys.path.insert(0, str(IRT_DIR))
sys.path.insert(0, str(PYDIR))

from bayesianquilts.metrics.nppsis import psisloo


# ---- R-hat ------------------------------------------------------------------

def compute_rhat(samples):
    n_chains, n_samp = samples.shape[:2]
    if n_chains < 2:
        return np.array([float('nan')])
    flat = samples.reshape(n_chains, n_samp, -1)
    chain_means = np.mean(flat, axis=1)
    grand_mean = np.mean(chain_means, axis=0)
    B = n_samp / (n_chains - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)
    W = np.mean(np.var(flat, axis=1, ddof=1), axis=0)
    var_plus = (n_samp - 1) / n_samp * W + B / n_samp
    rhat = np.sqrt(var_plus / np.maximum(W, 1e-20))
    return rhat


# ---- GRM log-likelihood (vectorized) ----------------------------------------

def grm_loglik_vectorized(disc_samp, diffs_samp, obs_matrix, eap, K):
    """Compute log P(Y|theta, params) for each sample and person.

    Args:
        disc_samp:  (use_S, I)          discrimination params
        diffs_samp: (use_S, I, K-1)     threshold params
        obs_matrix: (N, I)              observed responses (-1 = missing)
        eap:        (N,)                person abilities (EAP)
        K:          int                 response categories

    Returns:
        log_lik:    (use_S, N)          sum log-lik over items per person
    """
    use_S, I = disc_samp.shape
    N = len(eap)

    # logits: (use_S, N, I, K-1)
    # theta[n] - diff[s,i,k]: broadcast (N,) and (use_S, I, K-1)
    theta = eap[None, :, None, None]          # (1, N, 1, 1)
    diff = diffs_samp[:, None, :, :]          # (use_S, 1, I, K-1)
    a = disc_samp[:, None, :, None]           # (use_S, 1, I, 1)
    logits = a * (theta - diff)               # (use_S, N, I, K-1)
    cum_p = 1.0 / (1.0 + np.exp(-logits))    # (use_S, N, I, K-1)

    # Category probs: (use_S, N, I, K)
    p = np.zeros((use_S, N, I, K))
    p[..., 0] = 1.0 - cum_p[..., 0]
    for k in range(1, K - 1):
        p[..., k] = cum_p[..., k - 1] - cum_p[..., k]
    p[..., K - 1] = cum_p[..., K - 2]
    p = np.maximum(p, 1e-30)
    p /= p.sum(axis=-1, keepdims=True)

    # Select category based on observed response
    # obs_matrix: (N, I) with values 0..K-1 or -1
    obs_valid = obs_matrix.copy()
    obs_valid[obs_valid < 0] = 0    # will be masked out below
    obs_valid = obs_valid.astype(int)

    # (use_S, N, I) log-probs of observed category
    # Use advanced indexing: p[s, n, i, y[n,i]]
    s_idx = np.arange(use_S)[:, None, None]   # (use_S, 1, 1)
    n_idx = np.arange(N)[None, :, None]        # (1, N, 1)
    i_idx = np.arange(I)[None, None, :]        # (1, 1, I)
    log_p_obs = np.log(p[s_idx, n_idx, i_idx, obs_valid[None, :, :]])
    # (use_S, N, I)

    # Mask out missing responses
    mask = (obs_matrix >= 0) & (obs_matrix < K)  # (N, I) bool
    log_p_obs *= mask[None, :, :]                 # zero missing
    log_lik = log_p_obs.sum(axis=-1)              # (use_S, N)

    return log_lik


# ---- Data loaders -----------------------------------------------------------

def _load_wave1_module():
    pyc_path = PYDIR / 'bayesianquilts/data/__pycache__/promis_wave1.cpython-313.pyc'
    if not pyc_path.exists():
        raise FileNotFoundError(f"wave1 pyc not found: {pyc_path}")
    key = 'bayesianquilts.data.promis_wave1'
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, str(pyc_path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def load_wave1_domain(domain, wave1_tab_path):
    mod = _load_wave1_module()
    df, num_people = mod.get_data(
        polars_out=True,
        cache_dir=str(Path(wave1_tab_path).parent),
        domain=domain,
    )
    item_keys = [c for c in df.columns if c != 'person']
    batch = {col: df[col].to_numpy().astype(np.float32) for col in item_keys}
    return batch, item_keys, num_people


def load_su_bank(work_dir, su_tab_dir):
    """Load a SU bank using item_keys from the GRM baseline config."""
    import yaml
    cfg_path = work_dir / 'grm_baseline' / 'config.yaml'
    with open(str(cfg_path)) as f:
        cfg = yaml.safe_load(f)
    item_keys = cfg['item_keys']
    K = cfg.get('response_cardinality', 5)
    num_people = cfg['num_people']

    import importlib as imp
    mod = imp.import_module('bayesianquilts.data.promis_substance_use')
    mod.item_keys.clear()
    df_full, _ = mod.get_data(polars_out=True, cache_dir=str(su_tab_dir), reorient=True)
    all_keys = mod.item_keys

    item_keys_valid = [k for k in item_keys if k in all_keys]
    if not item_keys_valid:
        raise ValueError(f"No bank items found in full substance-use dataset")
    batch = {col: df_full[col].to_numpy().astype(np.float32) for col in item_keys_valid}
    return batch, item_keys_valid, num_people, K


def load_module_dataset(module_name, work_dir=None):
    import importlib as imp
    mod = imp.import_module(module_name)
    kw = {'polars_out': True}
    if 'reorient' in inspect.signature(mod.get_data).parameters:
        kw['reorient'] = True
    if work_dir and 'cache_dir' in inspect.signature(mod.get_data).parameters:
        kw['cache_dir'] = str(work_dir)
    df, num_people = mod.get_data(**kw)
    item_keys = [c for c in df.columns if c != 'person']
    batch = {col: df[col].to_numpy().astype(np.float32) for col in item_keys}
    return batch, item_keys, num_people, mod.response_cardinality


# ---- Core metric computation ------------------------------------------------

def compute_metrics_from_npz(npz_path, batch, item_keys, K, num_people,
                              max_S=200):
    """Compute RMSE and LOO-ELPD from an npz MCMC sample file.

    Vectorized implementation -- processes all samples and people at once.
    Memory: use_S * N * I * K * 8 bytes.  For W1: 200*15k*56*5*8 = ~6.7GB.
    Use chunked approach to stay within ~4GB.
    """
    mcmc = np.load(str(npz_path))
    disc = mcmc['discriminations']   # (chains, samp, 1, 1, I, 1)
    diff0 = mcmc['difficulties0']    # (chains, samp, 1, 1, I, 1)
    ddiff = mcmc.get('ddifficulties', None)

    n_chains, n_samp = disc.shape[:2]
    S = n_chains * n_samp
    I = len(item_keys)

    rhat_d = compute_rhat(disc)
    rhat_b = compute_rhat(diff0)
    max_rhat = float(max(np.nanmax(rhat_d), np.nanmax(rhat_b)))

    disc_flat = disc.reshape(S, I).astype(np.float64)
    diff0_flat = diff0.reshape(S, I, -1).astype(np.float64)
    if ddiff is not None and ddiff.size > 0:
        ddiff_flat = ddiff.reshape(S, I, -1).astype(np.float64)
    else:
        ddiff_flat = None

    eap = np.array(mcmc.get('eap', np.zeros(num_people)), dtype=np.float64).flatten()

    # Sub-sample
    rng = np.random.default_rng(42)
    use_S = min(S, max_S)
    idx = rng.choice(S, use_S, replace=False)
    disc_use = disc_flat[idx]     # (use_S, I)
    diff0_use = diff0_flat[idx]   # (use_S, I, 1)
    ddiff_use = ddiff_flat[idx] if ddiff_flat is not None else None

    # Build diffs_use: (use_S, I, K-1)
    # Thresholds: cumsum(concat([d0, dd])) matches GRModel.predictive_distribution
    # exactly. dd is in constrained (positive) space because MCMC targets
    # HalfNormal(0.5) directly on ddifficulties without an unconstraining
    # bijector, so the saved samples are already positive.
    if ddiff_use is not None:
        diffs_use = np.cumsum(
            np.concatenate([diff0_use, ddiff_use], axis=-1), axis=-1
        )
    else:
        diffs_use = diff0_use  # (use_S, I, K-1) if K-1 == 1

    # Posterior mean params for RMSE
    disc_mean = disc_flat.mean(0)
    diff0_mean = diff0_flat.mean(0)
    if ddiff_flat is not None:
        ddiff_mean = ddiff_flat.mean(0)
        diffs_mean = np.cumsum(
            np.concatenate([diff0_mean, ddiff_mean], axis=-1), axis=-1
        )
    else:
        diffs_mean = diff0_mean

    # Build observation matrix (N, I)
    obs_matrix = np.full((num_people, I), -1.0, dtype=np.float64)
    for i, key in enumerate(item_keys):
        obs = batch[key].astype(np.float64)
        obs_matrix[:, i] = np.where(
            np.isnan(obs) | (obs < 0) | (obs >= K), -1.0, obs)
    obs_int = obs_matrix.astype(int)  # -1 for missing
    mask = (obs_matrix >= 0) & (obs_matrix < K)   # (N, I) bool

    n_obs = int(mask.sum())

    # ---- RMSE ---------------------------------------------------------------
    # Use posterior mean params
    logits_mean = disc_mean[None, :, None] * (eap[:, None, None] - diffs_mean[None, :, :])
    # (N, I, K-1)
    cum_p_mean = 1.0 / (1.0 + np.exp(-logits_mean))
    p_mean = np.zeros((num_people, I, K))
    p_mean[:, :, 0] = 1.0 - cum_p_mean[:, :, 0]
    for k in range(1, K - 1):
        p_mean[:, :, k] = cum_p_mean[:, :, k - 1] - cum_p_mean[:, :, k]
    p_mean[:, :, K - 1] = cum_p_mean[:, :, K - 2]
    p_mean = np.maximum(p_mean, 1e-30)
    p_mean /= p_mean.sum(axis=-1, keepdims=True)
    expected = np.sum(p_mean * np.arange(K, dtype=np.float64)[None, None, :], axis=-1)
    # (N, I)
    sq_errors_all = np.where(mask, (obs_matrix - expected) ** 2, np.nan)
    sq_errors = sq_errors_all[mask]  # flat valid entries

    rmse = float(np.sqrt(np.mean(sq_errors)))
    n_resp = len(sq_errors)
    if n_resp > 1 and rmse > 0:
        se_mean_sq = float(np.std(sq_errors, ddof=1)) / np.sqrt(n_resp)
        rmse_se = se_mean_sq / (2 * rmse)
    else:
        rmse_se = float('nan')

    # ---- LOO-ELPD (vectorized, chunked) -------------------------------------
    # Memory estimate per chunk: chunk_S * N * I * K * 8 bytes
    # Target ~2GB per chunk
    bytes_per_s = num_people * I * K * 8
    chunk_S = max(1, min(use_S, int(2e9 / bytes_per_s)))
    print(f"    Vectorized ELPD: use_S={use_S}, N={num_people}, I={I}, "
          f"K={K}, chunk_S={chunk_S}", flush=True)

    log_lik = np.zeros((use_S, num_people))

    for start in range(0, use_S, chunk_S):
        end = min(start + chunk_S, use_S)
        s_chunk = end - start
        disc_c = disc_use[start:end]    # (s_chunk, I)
        diffs_c = diffs_use[start:end]  # (s_chunk, I, K-1)

        # logits: (s_chunk, N, I, K-1)
        theta = eap[None, :, None, None]           # (1, N, 1, 1)
        diff = diffs_c[:, None, :, :]              # (s_chunk, 1, I, K-1)
        a = disc_c[:, None, :, None]               # (s_chunk, 1, I, 1)
        logits = a * (theta - diff)                # (s_chunk, N, I, K-1)
        cum_p = 1.0 / (1.0 + np.exp(-logits))

        p = np.zeros((s_chunk, num_people, I, K))
        p[:, :, :, 0] = 1.0 - cum_p[:, :, :, 0]
        for k in range(1, K - 1):
            p[:, :, :, k] = cum_p[:, :, :, k - 1] - cum_p[:, :, :, k]
        p[:, :, :, K - 1] = cum_p[:, :, :, K - 2]
        p = np.maximum(p, 1e-30)
        p /= p.sum(axis=-1, keepdims=True)

        # log prob of observed category
        obs_clamped = np.where(obs_int >= 0, obs_int, 0)  # (N, I)
        # gather: p[s, n, i, obs_clamped[n,i]]
        # obs_clamped broadcast: (1, N, I)
        s_idx = np.arange(s_chunk)[:, None, None]
        n_idx = np.arange(num_people)[None, :, None]
        i_idx = np.arange(I)[None, None, :]
        log_p_obs = np.log(p[s_idx, n_idx, i_idx, obs_clamped[None, :, :]])
        # (s_chunk, N, I)
        log_p_obs *= mask[None, :, :]
        log_lik[start:end] = log_p_obs.sum(axis=-1)  # (s_chunk, N)

        del logits, cum_p, p, log_p_obs
        gc.collect()

        if end < use_S:
            print(f"      chunk {end}/{use_S}", flush=True)

    loo, loos, ks = psisloo(log_lik)
    n_bad_k = int(np.sum(ks > 0.7))

    return {
        'max_rhat': max_rhat,
        'rmse': rmse,
        'rmse_se': float(rmse_se),
        'n_obs': n_obs,
        'N': num_people,
        'I': I,
        'K': K,
        'elpd': float(loo),
        'elpd_per_person': float(loo / num_people),
        'elpd_se_per_person': float(np.std(loos) * np.sqrt(num_people) / num_people),
        'elpd_per_resp': float(loo / n_obs),
        'elpd_se_per_resp': float(np.std(loos) * np.sqrt(num_people) / n_obs),
        'n_bad_khat': n_bad_k,
    }


# ---- Dataset registry -------------------------------------------------------

def get_dataset_config():
    wave1_tab = IRT_DIR / 'promis_wave1' / 'promis_wave1.tab'
    su_tab_dir = IRT_DIR / 'promis_substance_use'

    return {
        'promis_w1__anger': {
            'loader': 'wave1', 'domain': 'anger', 'K': 5,
            'wave1_tab': wave1_tab,
        },
        'promis_w1__anxiety': {
            'loader': 'wave1', 'domain': 'anxiety', 'K': 5,
            'wave1_tab': wave1_tab,
        },
        'promis_w1__depression': {
            'loader': 'wave1', 'domain': 'depression', 'K': 5,
            'wave1_tab': wave1_tab,
        },
        'promis_w1__fatigue_experience': {
            'loader': 'wave1', 'domain': 'fatigue_experience', 'K': 5,
            'wave1_tab': wave1_tab,
        },
        'promis_w1__fatigue_impact': {
            'loader': 'wave1', 'domain': 'fatigue_impact', 'K': 5,
            'wave1_tab': wave1_tab,
        },
        'promis_w1__physical_function_a': {
            'loader': 'wave1', 'domain': 'physical_function_a', 'K': 5,
            'wave1_tab': wave1_tab,
        },
        'promis_sleep': {
            'loader': 'module',
            'module': 'bayesianquilts.data.promis_sleep',
            'work_dir': IRT_DIR / 'promis_sleep',
            'sleep_data_dir': Path.home() / 'workspace/bayesianquilts',
        },
        'promis_su__bank2': {
            'loader': 'su_bank', 'K': 5, 'su_tab_dir': su_tab_dir,
        },
        'promis_su__bank3': {
            'loader': 'su_bank', 'K': 5, 'su_tab_dir': su_tab_dir,
        },
        'promis_su__bank4': {
            'loader': 'su_bank', 'K': 5, 'su_tab_dir': su_tab_dir,
        },
        'promis_su__bank5': {
            'loader': 'su_bank', 'K': 5, 'su_tab_dir': su_tab_dir,
        },
        'promis_su__bank6': {
            'loader': 'su_bank', 'K': 5, 'su_tab_dir': su_tab_dir,
        },
    }


def load_data_for_dataset(ds_name, config):
    loader = config['loader']
    if loader == 'wave1':
        batch, item_keys, num_people = load_wave1_domain(
            config['domain'], config['wave1_tab'])
        K = config.get('K', 5)
        return batch, item_keys, num_people, K
    elif loader == 'su_bank':
        work_dir = IRT_DIR / ds_name
        return load_su_bank(work_dir, config['su_tab_dir'])
    elif loader == 'module':
        kw = {'polars_out': True}
        import importlib as imp
        mod = imp.import_module(config['module'])
        if 'reorient' in inspect.signature(mod.get_data).parameters:
            kw['reorient'] = True
        # For sleep, use the special data dir if specified
        data_dir = config.get('sleep_data_dir') or config.get('work_dir')
        if data_dir and 'cache_dir' in inspect.signature(mod.get_data).parameters:
            kw['cache_dir'] = str(data_dir)
        df, num_people = mod.get_data(**kw)
        item_keys = [c for c in df.columns if c != 'person']
        batch = {col: df[col].to_numpy().astype(np.float32) for col in item_keys}
        return batch, item_keys, num_people, mod.response_cardinality
    else:
        raise ValueError(f"Unknown loader: {loader}")


def eval_dataset(ds_name, config):
    work_dir = IRT_DIR / ds_name

    print(f"\n{'='*60}")
    print(f"  {ds_name}", flush=True)
    print(f"{'='*60}", flush=True)

    batch, item_keys, num_people, K = load_data_for_dataset(ds_name, config)
    print(f"  N={num_people}, I={len(item_keys)}, K={K}")

    total_cells = num_people * len(item_keys)
    n_missing = sum(
        int(np.sum(np.isnan(batch[k]) | (batch[k] < 0) | (batch[k] >= K)))
        for k in item_keys
    )
    miss_frac = n_missing / total_cells
    print(f"  Missingness: {miss_frac*100:.1f}%")

    results = {
        'N': num_people,
        'I': len(item_keys),
        'K': K,
        'missing_frac': float(miss_frac),
        'variants': {},
    }

    for variant in ['baseline', 'pairwise', 'mixed']:
        npz_path = work_dir / 'mcmc_samples' / f'mcmc_{variant}.npz'
        if not npz_path.exists():
            print(f"  {variant}: no npz, skipping")
            continue

        print(f"\n  --- {variant} ---", flush=True)

        mcmc = np.load(str(npz_path))
        disc = mcmc['discriminations']
        diff0 = mcmc['difficulties0']
        rhat_d = compute_rhat(disc)
        rhat_b = compute_rhat(diff0)
        max_rhat = float(max(np.nanmax(rhat_d), np.nanmax(rhat_b)))
        print(f"    max R-hat: {max_rhat:.4f}", end='', flush=True)

        if max_rhat > 1.1:
            print(f" -- FAILED (> 1.1), excluding")
            results['variants'][variant] = {'max_rhat': max_rhat, 'failed': True}
            del mcmc
            gc.collect()
            continue
        print()
        del mcmc
        gc.collect()

        try:
            m = compute_metrics_from_npz(
                npz_path, batch, item_keys, K, num_people, max_S=200)
            print(f"    RMSE:      {m['rmse']:.4f} ({m['rmse_se']:.4f})")
            print(f"    ELPD/n:    {m['elpd_per_person']:.4f} ({m['elpd_se_per_person']:.4f})")
            print(f"    ELPD/resp: {m['elpd_per_resp']:.4f} ({m['elpd_se_per_resp']:.4f})")
            print(f"    n_bad_k:   {m['n_bad_khat']}/{num_people}")
            results['variants'][variant] = m
        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
            results['variants'][variant] = {'max_rhat': max_rhat, 'error': str(e)}

        gc.collect()

    return results


def main():
    configs = get_dataset_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None,
                        choices=list(configs.keys()))
    parser.add_argument('--all-tankie', action='store_true')
    parser.add_argument('--all-local', action='store_true')
    parser.add_argument('--output', default='promis_eval_results.json')
    args = parser.parse_args()

    tankie_ds = [
        'promis_w1__anger', 'promis_w1__anxiety', 'promis_w1__depression',
        'promis_w1__fatigue_experience', 'promis_w1__fatigue_impact',
        'promis_w1__physical_function_a',
        'promis_sleep',
        'promis_su__bank2', 'promis_su__bank3', 'promis_su__bank4',
        'promis_su__bank5', 'promis_su__bank6',
    ]

    if args.all_tankie:
        datasets_to_run = {k: configs[k] for k in tankie_ds if k in configs}
    elif args.datasets:
        datasets_to_run = {k: configs[k] for k in args.datasets if k in configs}
    else:
        datasets_to_run = configs

    all_results = {}
    for ds_name, config in datasets_to_run.items():
        try:
            all_results[ds_name] = eval_dataset(ds_name, config)
        except Exception as e:
            import traceback
            print(f"\nERROR on {ds_name}: {e}")
            traceback.print_exc()
            all_results[ds_name] = {'error': str(e)}

    out_path = IRT_DIR / args.output
    with open(str(out_path), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    print(f"\n{'='*90}")
    print(f"SUMMARY")
    print(f"{'='*90}")
    for ds, res in all_results.items():
        if 'error' in res:
            print(f"{ds}: ERROR -- {res['error'][:60]}")
            continue
        for v, vr in res.get('variants', {}).items():
            if vr.get('failed'):
                print(f"{ds:48s} {v:12s} FAILED R-hat={vr.get('max_rhat', 0):.4f}")
            elif 'elpd_per_person' in vr:
                print(f"{ds:48s} {v:12s} "
                      f"RMSE={vr['rmse']:.4f}({vr['rmse_se']:.4f}) "
                      f"ELPD/n={vr['elpd_per_person']:.4f}({vr['elpd_se_per_person']:.4f}) "
                      f"ELPD/resp={vr['elpd_per_resp']:.4f}({vr['elpd_se_per_resp']:.4f})")
            elif 'error' in vr:
                print(f"{ds:48s} {v:12s} ERROR: {vr['error'][:50]}")


if __name__ == '__main__':
    main()

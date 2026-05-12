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


def compute_elpd_from_npz(npz_path, batch, item_keys, K, num_people,
                          n_observed_responses, eap_fallback=None, max_S=200):
    """Compute PSIS-LOO ELPD from MCMC samples stored in .npz."""
    mcmc_data = np.load(str(npz_path))
    disc_samples = mcmc_data['discriminations']
    diff0_samples = mcmc_data['difficulties0']
    ddiff_samples = mcmc_data.get('ddifficulties', None)

    n_chains, n_samp = disc_samples.shape[:2]
    S = n_chains * n_samp
    I = len(item_keys)
    N = num_people

    disc_flat = disc_samples.reshape(S, -1)
    diff0_flat = diff0_samples.reshape(S, disc_flat.shape[1], -1)
    if ddiff_samples is not None and ddiff_samples.size > 0:
        ddiff_flat = ddiff_samples.reshape(S, disc_flat.shape[1], -1)
    else:
        ddiff_flat = None

    eap = mcmc_data.get('eap', eap_fallback)
    if eap is None:
        raise ValueError("No EAP abilities in npz or fallback")
    eap = np.array(eap).flatten()

    use_S = min(S, max_S)
    rng = np.random.default_rng(42)
    if use_S < S:
        idx = rng.choice(S, use_S, replace=False)
        disc_use = disc_flat[idx]
        diff0_use = diff0_flat[idx]
        ddiff_use = ddiff_flat[idx] if ddiff_flat is not None else None
    else:
        disc_use = disc_flat
        diff0_use = diff0_flat
        ddiff_use = ddiff_flat

    log_lik = np.zeros((use_S, N))

    for s in range(use_S):
        a_s = disc_use[s]
        d0_s = diff0_use[s]

        if ddiff_use is not None and ddiff_use.shape[-1] > 0:
            dd_s = ddiff_use[s]
            diffs = np.cumsum(
                np.concatenate([d0_s, dd_s], axis=-1), axis=-1)
        else:
            diffs = d0_s

        for n in range(N):
            theta_n = eap[n]
            ll_n = 0.0
            for i, key in enumerate(item_keys):
                y = batch[key][n]
                if np.isnan(y) or y < 0 or y >= K:
                    continue
                y_int = int(y)

                logits = a_s[i] * (theta_n - diffs[i])
                cum_probs = 1.0 / (1.0 + np.exp(-logits))

                probs_cat = np.zeros(K)
                for k_cat in range(K):
                    if k_cat == 0:
                        probs_cat[k_cat] = 1.0 - (cum_probs[0] if len(logits) > 0 else 0.0)
                    elif k_cat == K - 1:
                        probs_cat[k_cat] = cum_probs[k_cat-1] if (k_cat-1) < len(logits) else 0.0
                    else:
                        probs_cat[k_cat] = cum_probs[k_cat-1] - cum_probs[k_cat]

                probs_cat = np.maximum(probs_cat, 1e-30)
                probs_cat /= probs_cat.sum()
                ll_n += np.log(probs_cat[y_int])

            log_lik[s, n] = ll_n

        if (s + 1) % 50 == 0:
            print(f"      Sample {s+1}/{use_S}")

    loo, loos, ks = psisloo(log_lik)
    n_bad_k = int(np.sum(ks > 0.7))
    return {
        'elpd': float(loo),
        'elpd_per_person': float(loo / N),
        'elpd_se_per_person': float(np.std(loos) * np.sqrt(N) / N),
        'elpd_per_resp': float(loo / n_observed_responses),
        'elpd_se_per_resp': float(np.std(loos) * np.sqrt(N) / n_observed_responses),
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
        model_path = work_dir / f'grm_mcmc_{variant}'
        npz_path = work_dir / 'mcmc_samples' / f'mcmc_{variant}.npz'

        if not (model_path / 'params.h5').exists():
            print(f"  {variant}: no model, skipping")
            continue

        print(f"\n  --- {variant} ---")
        model = GRModel.load_from_disk(model_path)
        calibrate_manually(model, n_samples=32, seed=101)

        rmse = compute_pred_rmse(model, batch, item_keys, K)
        print(f"    Pred RMSE: {rmse:.4f}")

        elpd_result = None
        if npz_path.exists():
            eap_fallback = np.array(model.calibrated_expectations.get('abilities', None))
            if eap_fallback is not None:
                eap_fallback = eap_fallback.flatten()
            print(f"    Computing ELPD-LOO from MCMC samples...")
            try:
                elpd_result = compute_elpd_from_npz(
                    npz_path, batch, item_keys, K, num_people,
                    n_observed_responses, eap_fallback=eap_fallback)
                print(f"    ELPD/n: {elpd_result['elpd_per_person']:.4f} "
                      f"(SE: {elpd_result['elpd_se_per_person']:.4f}), "
                      f"k>0.7: {elpd_result['n_bad_khat']}/{num_people}")
            except Exception as e:
                print(f"    ELPD failed: {e}")

        results[variant] = {
            'pred_rmse': rmse,
            **(elpd_result if elpd_result else {}),
        }

        del model
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

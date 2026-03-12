#!/usr/bin/env python
"""Comprehensive evaluation: Pred RMSE, LOO-RMSE, ELPD, k-hat for all models."""

import os
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import argparse
import importlib
import inspect
import gc
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from run_single_notebook import make_data_dict, calibrate_manually, predictive_rmse


def eval_dataset(dataset_name):
    from run_single_notebook import DATASET_CONFIGS
    from bayesianquilts.irt.grm import GRModel

    config = DATASET_CONFIGS[dataset_name]
    mod = importlib.import_module(config['module'])
    item_keys = mod.item_keys
    response_cardinality = mod.response_cardinality

    work_dir = Path(__file__).parent / dataset_name
    os.chdir(work_dir)

    get_data_kwargs = {'polars_out': True}
    if 'reorient' in inspect.signature(mod.get_data).parameters:
        get_data_kwargs['reorient'] = True
    df, num_people = mod.get_data(**get_data_kwargs)
    batch = make_data_dict(df)

    n_observed_responses = sum(
        np.sum((batch[k] >= 0) & (batch[k] < response_cardinality) & ~np.isnan(batch[k]))
        for k in item_keys
    )

    # Build data factory for ELPD computation
    batch_size = 256
    subsample_n = num_people

    def data_factory():
        for start in range(0, subsample_n, batch_size):
            end = min(start + batch_size, subsample_n)
            sub = {k: batch[k][start:end] for k in batch}
            yield sub

    results = []

    for label, model_dir in [('Baseline', 'grm_baseline'),
                              ('MICE-only', 'grm_mice_only'),
                              ('Mixed', 'grm_imputed')]:
        model_path = work_dir / model_dir
        if not (model_path / 'params.h5').exists():
            results.append({
                'label': label, 'pred_rmse': None, 'loo_rmse': None,
                'elpd': None, 'elpd_se': None, 'n_obs': None,
                'khat_bad': None, 'khat_max': None, 'khat_mean': None,
            })
            continue

        mdl = GRModel.load_from_disk(model_path)
        calibrate_manually(mdl, n_samples=32, seed=101)

        # Pred RMSE
        try:
            rmse = predictive_rmse(mdl, batch, item_keys, response_cardinality)
        except Exception:
            rmse = None

        # LOO-RMSE
        try:
            loo_rmse = predictive_rmse(mdl, batch, item_keys, response_cardinality,
                                        loo=True, n_samples=100)
        except Exception:
            loo_rmse = None

        # ELPD-LOO with k-hat
        try:
            mdl._compute_elpd_loo(data_factory, n_samples=100, seed=101, use_ais=True)
            elpd = float(mdl.elpd_loo)
            elpd_se = float(mdl.elpd_loo_se)
            n_obs = int(mdl.elpd_loo_n_obs)
            ks = np.asarray(mdl.elpd_loo_khat)
            khat_bad = int(np.sum(ks > 0.7))
            finite_ks = ks[np.isfinite(ks)]
            khat_max = float(np.max(finite_ks)) if len(finite_ks) > 0 else float('inf')
            khat_mean = float(np.mean(finite_ks)) if len(finite_ks) > 0 else float('inf')
            khat_n_inf = int(np.sum(~np.isfinite(ks)))
        except Exception as e:
            print(f"  ELPD failed for {label}: {e}")
            elpd = elpd_se = n_obs = None
            khat_bad = khat_max = khat_mean = khat_n_inf = None

        results.append({
            'label': label, 'pred_rmse': rmse, 'loo_rmse': loo_rmse,
            'elpd': elpd, 'elpd_se': elpd_se, 'n_obs': n_obs,
            'n_resp': n_observed_responses,
            'khat_bad': khat_bad, 'khat_max': khat_max, 'khat_mean': khat_mean,
            'khat_n_inf': khat_n_inf,
        })

        del mdl
        gc.collect()

    # Print table
    print(f"\n{'='*140}")
    print(f"  {dataset_name.upper()} (K={response_cardinality}, N={num_people}, "
          f"items={len(item_keys)}, observed={n_observed_responses})")
    print(f"{'='*140}")
    print(f"{'Model':<12} {'Pred RMSE':>10} {'LOO-RMSE':>10} "
          f"{'ELPD/person':>16} {'ELPD/resp':>16} "
          f"{'k>0.7':>10} {'k_inf':>6} {'k_max*':>8} {'k_mean*':>8}")
    print(f"  (* = finite k-hat only)")
    print(f"{'-'*140}")

    for r in results:
        rmse_s = f"{r['pred_rmse']:.4f}" if r['pred_rmse'] is not None else "--"
        loo_s = f"{r['loo_rmse']:.4f}" if r['loo_rmse'] is not None else "--"

        if r['elpd'] is not None and not np.isnan(r['elpd']):
            n = r['n_obs']
            n_resp = r.get('n_resp', n_observed_responses)
            ep = f"{r['elpd']/n:.4f}±{r['elpd_se']/n:.4f}"
            er = f"{r['elpd']/n_resp:.4f}±{r['elpd_se']/n_resp:.4f}"
            kb = f"{r['khat_bad']}/{n}"
            ki = f"{r.get('khat_n_inf', 0)}"
            km = f"{r['khat_max']:.3f}" if np.isfinite(r['khat_max']) else "inf"
            ka = f"{r['khat_mean']:.3f}" if np.isfinite(r['khat_mean']) else "inf"
        else:
            ep = er = "--"
            kb = ki = km = ka = "--"

        print(f"{r['label']:<12} {rmse_s:>10} {loo_s:>10} "
              f"{ep:>16} {er:>16} "
              f"{kb:>10} {ki:>6} {km:>8} {ka:>8}")

    print(f"{'='*140}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    all_results = {}
    datasets = ['grit', 'tma', 'rwa', 'npi', 'wpi'] if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        try:
            all_results[ds] = eval_dataset(ds)
        except Exception as e:
            import traceback
            print(f"ERROR on {ds}: {e}")
            traceback.print_exc()

    if len(all_results) > 1:
        print(f"\n\n{'#'*150}")
        print(f"  COMBINED RESULTS")
        print(f"{'#'*150}")
        print(f"{'Dataset':<8} {'Model':<12} {'Pred RMSE':>10} {'LOO-RMSE':>10} "
              f"{'ELPD/person':>16} {'ELPD/resp':>16} "
              f"{'k>0.7':>10} {'k_inf':>6} {'k_max*':>8} {'k_mean*':>8}")
        print(f"{'-'*150}")
        for ds, results in all_results.items():
            for r in results:
                rmse_s = f"{r['pred_rmse']:.4f}" if r['pred_rmse'] is not None else "--"
                loo_s = f"{r['loo_rmse']:.4f}" if r['loo_rmse'] is not None else "--"
                if r['elpd'] is not None and not np.isnan(r['elpd']):
                    n = r['n_obs']
                    n_resp = r.get('n_resp', n)
                    ep = f"{r['elpd']/n:.4f}±{r['elpd_se']/n:.4f}"
                    er = f"{r['elpd']/n_resp:.4f}±{r['elpd_se']/n_resp:.4f}"
                    kb = f"{r['khat_bad']}/{n}"
                    ki = f"{r.get('khat_n_inf', 0)}"
                    km = f"{r['khat_max']:.3f}" if np.isfinite(r['khat_max']) else "inf"
                    ka = f"{r['khat_mean']:.3f}" if np.isfinite(r['khat_mean']) else "inf"
                else:
                    ep = er = "--"
                    kb = ki = km = ka = "--"
                print(f"{ds.upper():<8} {r['label']:<12} {rmse_s:>10} {loo_s:>10} "
                      f"{ep:>16} {er:>16} "
                      f"{kb:>10} {ki:>6} {km:>8} {ka:>8}")
            print(f"{'-'*150}")
        print(f"{'#'*150}")


if __name__ == '__main__':
    main()

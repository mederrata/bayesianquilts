#!/usr/bin/env python
"""Evaluate LOO-RMSE and Pred RMSE from saved models (no refitting)."""

import os
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import argparse
import importlib
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

    import inspect
    get_data_kwargs = {'polars_out': True}
    if 'reorient' in inspect.signature(mod.get_data).parameters:
        get_data_kwargs['reorient'] = True
    df, num_people = mod.get_data(**get_data_kwargs)
    batch = make_data_dict(df)

    n_observed_responses = sum(
        np.sum((batch[k] >= 0) & (batch[k] < response_cardinality) & ~np.isnan(batch[k]))
        for k in item_keys
    )

    print(f"\n{'='*110}")
    print(f"SUMMARY: {dataset_name.upper()} (K={response_cardinality}, N={num_people}, items={len(item_keys)})")
    print(f"{'='*110}")
    print(f"{'Model':<15} {'Pred RMSE':>10} {'LOO-RMSE':>10} {'ELPD/person':>20} {'ELPD/response':>20}")
    print(f"{'-'*110}")

    for label, model_dir in [('Baseline', 'grm_baseline'),
                              ('MICE-only', 'grm_mice_only'),
                              ('Mixed', 'grm_imputed')]:
        model_path = work_dir / model_dir
        if not (model_path / 'params.h5').exists():
            print(f"{label:<15} {'--':>10} {'--':>10} {'--':>20} {'--':>20}")
            continue

        mdl = GRModel.load_from_disk(model_path)
        calibrate_manually(mdl, n_samples=32, seed=101)

        # ELPD - values may be stored as Flax Field objects or plain scalars
        def _scalar(v, default=None):
            if v is None:
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        elpd = _scalar(getattr(mdl, 'elpd_loo', None))
        se = _scalar(getattr(mdl, 'elpd_loo_se', None))
        n = _scalar(getattr(mdl, 'elpd_loo_n_obs', None)) or num_people
        if elpd is not None and not np.isnan(elpd):
            pp = f"{elpd/n:.4f} ± {se/n:.4f}"
            pr = f"{elpd/n_observed_responses:.4f} ± {se/n_observed_responses:.4f}"
        else:
            pp = "nan"
            pr = "nan"

        # Pred RMSE
        try:
            rmse = predictive_rmse(mdl, batch, item_keys, response_cardinality)
            rmse_str = f"{rmse:.4f}"
        except Exception as e:
            rmse_str = f"err"

        # LOO-RMSE
        try:
            loo_rmse = predictive_rmse(mdl, batch, item_keys, response_cardinality, loo=True, n_samples=100)
            loo_rmse_str = f"{loo_rmse:.4f}"
        except Exception as e:
            import traceback
            print(f"  LOO-RMSE failed for {label}: {e}")
            traceback.print_exc()
            loo_rmse_str = "err"

        print(f"{label:<15} {rmse_str:>10} {loo_rmse_str:>10} {pp:>20} {pr:>20}")

        # Free memory
        del mdl
        import gc
        gc.collect()

    print(f"{'='*110}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        help='Dataset name or "all" for all datasets')
    args = parser.parse_args()

    if args.dataset == 'all':
        for ds in ['grit', 'tma', 'rwa', 'npi', 'wpi']:
            try:
                eval_dataset(ds)
            except Exception as e:
                print(f"ERROR on {ds}: {e}")
    else:
        eval_dataset(args.dataset)


if __name__ == '__main__':
    main()

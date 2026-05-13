#!/usr/bin/env python
"""Scoring-time evaluation of the three-way imputed ensemble.

Computes RMSE and point-ELPD for the Imputed variant by blending baseline
IRT (MCMC posterior mean), shared-discrimination GRM (MCMC posterior mean),
and pairwise stacking predictions via per-item Yao weights.

No imputed-calibration MCMC refit is performed: the imputed PMFs are built
at evaluation time from the three pre-fitted components.
"""

import os
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import argparse
import importlib
import inspect
import json
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from run_single_notebook import DATASET_CONFIGS, make_data_dict
from run_marginal_mcmc import load_shared_disc_model


def _load_baseline_grm_with_mcmc(work_dir, item_keys, num_people):
    """Build a GRModel whose surrogate_sample matches the baseline MCMC posterior."""
    from bayesianquilts.irt.grm import GRModel

    model_dir = work_dir / 'grm_baseline'
    if not model_dir.exists():
        model_dir = work_dir / 'grm_mcmc_baseline'
    model = GRModel.load_from_disk(str(model_dir))

    baseline_npz = work_dir / 'mcmc_samples' / 'mcmc_baseline.npz'
    bnpz = np.load(str(baseline_npz))

    excluded = {'eap', 'psd', 'standardize_mu', 'standardize_sigma'}
    surrogate_sample = {}
    for k in bnpz.files:
        if k in excluded:
            continue
        a = np.asarray(bnpz[k])
        flat = a.reshape(-1, *a.shape[2:])
        surrogate_sample[k] = jnp.asarray(flat)

    if 'eap' in bnpz.files:
        eap = np.asarray(bnpz['eap']).astype(np.float64)
        surrogate_sample['abilities'] = jnp.asarray(
            eap[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        )

    model.surrogate_sample = surrogate_sample
    model.calibrated_expectations = {
        k: jnp.mean(v, axis=0) for k, v in surrogate_sample.items()
    }
    return model


def eval_imputed_dataset(dataset_name):
    config = DATASET_CONFIGS[dataset_name]
    mod = importlib.import_module(config['module'])
    item_keys = mod.item_keys
    K = mod.response_cardinality
    I = len(item_keys)

    work_dir = Path(__file__).parent / dataset_name
    os.chdir(work_dir)

    print(f"=== {dataset_name.upper()} ===")
    print(f"  Items: {I}, K: {K}")

    get_data_kwargs = {'polars_out': True}
    if 'reorient' in inspect.signature(mod.get_data).parameters:
        get_data_kwargs['reorient'] = True
    df, num_people = mod.get_data(**get_data_kwargs)
    base_data = make_data_dict(df)
    print(f"  People: {num_people}")

    irt_model = _load_baseline_grm_with_mcmc(work_dir, item_keys, num_people)
    print(f"  Baseline IRT loaded (S={irt_model.surrogate_sample['discriminations'].shape[0]})")

    shared_disc_npz = work_dir / 'mcmc_samples' / 'mcmc_shared_disc.npz'
    if shared_disc_npz.exists():
        shared_disc_model = load_shared_disc_model(
            item_keys, num_people, K, shared_disc_npz)
        print(f"  Shared-disc loaded")
    else:
        shared_disc_model = None
        print(f"  Shared-disc NPZ absent -- using 2-component (pairwise + IRT) fallback")

    from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel
    stacking_path = work_dir / 'pairwise_stacking_model.yaml'
    pairwise_model = PairwiseOrdinalStackingModel.load(str(stacking_path))
    print(f"  Pairwise stacking loaded")

    def _data_factory():
        yield base_data

    from bayesianquilts.imputation.three_way import ThreeWayImputationModel
    print(f"  Building three-way Yao stack...")
    three_way = ThreeWayImputationModel(
        irt_model=irt_model,
        shared_disc_model=shared_disc_model,
        mice_model=pairwise_model,
        data_factory=_data_factory,
    )
    weights = three_way.weights
    w_arr = np.stack([weights[k] for k in item_keys], axis=0)
    print(f"  Yao weights: mean (mice, irt, shdisc) = "
          f"({w_arr[:, 0].mean():.3f}, {w_arr[:, 1].mean():.3f}, {w_arr[:, 2].mean():.3f})")

    obs_matrix = np.full((num_people, I), -1.0, dtype=np.float64)
    for i, key in enumerate(item_keys):
        obs = np.asarray(base_data[key], dtype=np.float64)
        obs_matrix[:, i] = np.where(
            np.isnan(obs) | (obs < 0) | (obs >= K), -1.0, obs)

    sq_errors = []
    log_p_obs = []

    print(f"  Evaluating per-observation Imputed PMFs...")
    sys.stdout.flush()

    progress_every = max(1, num_people // 20)
    for n in range(num_people):
        person_items = {}
        for k in item_keys:
            v = base_data[k][n]
            v_f = float(v)
            if not (np.isnan(v_f) or v_f < 0 or v_f >= K):
                person_items[k] = v_f

        for i, key in enumerate(item_keys):
            v = obs_matrix[n, i]
            if v < 0:
                continue
            pmf = three_way.predict_pmf(person_items, key, K, person_idx=n)
            pmf = np.maximum(np.asarray(pmf, dtype=np.float64), 1e-30)
            pmf = pmf / pmf.sum()
            log_p_obs.append(float(np.log(pmf[int(v)])))
            expected = float(np.sum(np.arange(K) * pmf))
            sq_errors.append((v - expected) ** 2)

        if (n + 1) % progress_every == 0:
            print(f"    {n+1}/{num_people} people, "
                  f"running RMSE={np.sqrt(np.mean(sq_errors)):.4f}, "
                  f"running ELPD/resp={np.mean(log_p_obs):.4f}", flush=True)

    sq_errors = np.array(sq_errors)
    log_p_obs = np.array(log_p_obs)

    rmse = float(np.sqrt(np.mean(sq_errors)))
    n_obs = len(log_p_obs)
    if n_obs > 1 and rmse > 0:
        se_mean_sq = float(np.std(sq_errors, ddof=1)) / np.sqrt(n_obs)
        rmse_se = float(se_mean_sq / (2 * rmse))
    else:
        rmse_se = float('nan')

    elpd_total = float(np.sum(log_p_obs))
    elpd_per_n = elpd_total / num_people
    elpd_per_resp = elpd_total / n_obs
    if n_obs > 1:
        elpd_se_per_resp = float(np.std(log_p_obs, ddof=1) / np.sqrt(n_obs))
        elpd_se_per_n = float(np.std(log_p_obs, ddof=1) * np.sqrt(n_obs) / num_people)
    else:
        elpd_se_per_resp = float('nan')
        elpd_se_per_n = float('nan')

    return {
        'dataset': dataset_name,
        'method': 'imputed_scoring_time',
        'rmse': rmse,
        'rmse_se': rmse_se,
        'elpd_total': elpd_total,
        'elpd_per_n': elpd_per_n,
        'elpd_se_per_n': elpd_se_per_n,
        'elpd_per_resp': elpd_per_resp,
        'elpd_se_per_resp': elpd_se_per_resp,
        'n_obs': n_obs,
        'num_people': num_people,
        'yao_weights_mean': {
            'pairwise': float(w_arr[:, 0].mean()),
            'irt': float(w_arr[:, 1].mean()),
            'shared_disc': float(w_arr[:, 2].mean()),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', default=None,
                        help='Output JSON path (default: imputed_eval_<dataset>.json)')
    args = parser.parse_args()

    result = eval_imputed_dataset(args.dataset)
    print("\n=== Result ===")
    print(json.dumps(result, indent=2))

    out_path = args.output or f'imputed_eval_{args.dataset}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()

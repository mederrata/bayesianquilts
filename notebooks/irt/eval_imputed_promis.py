#!/usr/bin/env python
"""Scoring-time Imputed eval for PROMIS Wave 1 + Substance Use banks.

Per-bank version of eval_imputed.py. Uses the same loaders as
eval_promis_new.py to handle the W1/SU bank-specific data format.

Builds a 2-component ThreeWayImputationModel (shared_disc=None) per bank
and reports RMSE + per-response point-ELPD.
"""

import os
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import argparse
import json
import inspect
import numpy as np
import jax.numpy as jnp
from pathlib import Path

IRT_DIR = Path(__file__).parent.resolve()
PYDIR = IRT_DIR.parent.parent / 'python'
sys.path.insert(0, str(IRT_DIR))
sys.path.insert(0, str(PYDIR))

from eval_promis_new import (
    get_dataset_config,
    load_data_for_dataset,
)


def _load_baseline_grm(work_dir, item_keys):
    """Load baseline GRModel and attach MCMC posterior samples + EAP."""
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


def eval_promis_bank(ds_name):
    config = get_dataset_config()
    cfg = config[ds_name]
    work_dir = IRT_DIR / ds_name

    print(f"=== {ds_name.upper()} ===")
    batch, item_keys, num_people, K = load_data_for_dataset(ds_name, cfg)
    I = len(item_keys)
    print(f"  Items: {I}, K: {K}, N: {num_people}")

    irt_model = _load_baseline_grm(work_dir, item_keys)
    print(f"  Baseline IRT loaded (S={irt_model.surrogate_sample['discriminations'].shape[0]})")

    from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel
    stacking_path = work_dir / 'pairwise_stacking_model.yaml'
    pairwise_model = PairwiseOrdinalStackingModel.load(str(stacking_path))
    print(f"  Pairwise stacking loaded")

    base_data = batch  # already a dict of arrays
    def _data_factory():
        yield base_data

    from bayesianquilts.imputation.three_way import ThreeWayImputationModel
    print(f"  Building three-way Yao stack (2-component fallback)...")
    three_way = ThreeWayImputationModel(
        irt_model=irt_model,
        shared_disc_model=None,
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

    sq_errors, log_p_obs = [], []

    print(f"  Evaluating per-observation Imputed PMFs...")
    sys.stdout.flush()
    progress_every = max(1, num_people // 10)
    for n in range(num_people):
        person_items = {}
        for k in item_keys:
            v_f = float(base_data[k][n])
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
    rmse_se = float((np.std(sq_errors, ddof=1) / np.sqrt(n_obs)) / (2 * rmse)) if rmse > 0 else float('nan')
    elpd_total = float(np.sum(log_p_obs))
    return {
        'dataset': ds_name,
        'method': 'imputed_scoring_time_2way',
        'rmse': rmse,
        'rmse_se': rmse_se,
        'elpd_total': elpd_total,
        'elpd_per_n': elpd_total / num_people,
        'elpd_se_per_n': float(np.std(log_p_obs, ddof=1) * np.sqrt(n_obs) / num_people),
        'elpd_per_resp': elpd_total / n_obs,
        'elpd_se_per_resp': float(np.std(log_p_obs, ddof=1) / np.sqrt(n_obs)),
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
    parser.add_argument('--dataset', required=True,
                        help='e.g. promis_w1__anger or promis_su__bank2')
    args = parser.parse_args()
    result = eval_promis_bank(args.dataset)
    print("\n=== Result ===")
    print(json.dumps(result, indent=2))

    work_dir = IRT_DIR / args.dataset
    out_path = work_dir / f'imputed_eval_{args.dataset}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Serialize bayesianquilts ThreeWayImputationModel components for libfabulouscatpy.

For a given dataset, this script:
  1. Loads the baseline GRM + MCMC posterior samples and computes posterior
     mean item parameters.
  2. Loads the shared-disc MCMC samples (if present).
  3. Loads the PairwiseOrdinalStackingModel from pairwise_stacking_model.yaml.
  4. Constructs the bayesianquilts ThreeWayImputationModel to obtain per-item
     Yao weights.
  5. Extracts the pairwise PMF tables by calling predict_pmf across all
     (target, predictor, response) combinations.
  6. Writes:
       /home/josh/workspace/libfabulouscatpy/examples/<d>/threeway_imputation.json
       /home/josh/workspace/libfabulouscatpy/examples/<d>/<d>_grm_params.npz

Usage:
    uv run --project /home/josh/workspace/bayesianquilts python \
        /home/josh/workspace/bayesianquilts/notebooks/irt/serialize_threeway_for_libfab.py \
        --dataset grit

    # All 8 datasets sequentially:
    uv run --project /home/josh/workspace/bayesianquilts python \
        /home/josh/workspace/bayesianquilts/notebooks/irt/serialize_threeway_for_libfab.py \
        --dataset all
"""

import argparse
import importlib
import inspect
import json
import os
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

IRT_DIR = Path(__file__).parent
LIBFAB_EXAMPLES = Path('/home/josh/workspace/libfabulouscatpy/examples')

sys.path.insert(0, str(IRT_DIR))
sys.path.insert(0, '/home/josh/workspace/bayesianquilts/python')

from run_single_notebook import DATASET_CONFIGS, make_data_dict
from run_marginal_mcmc import load_shared_disc_model

DATASETS_8 = ['scs', 'gcbs', 'grit', 'rwa', 'npi', 'tma', 'wpi', 'eqsq']


# ---------------------------------------------------------------------------
# GRM baseline loading (mirrors eval_imputed.py _load_baseline_grm_with_mcmc)
# ---------------------------------------------------------------------------

def _load_baseline_grm(work_dir: Path, item_keys, num_people):
    """Load baseline GRM and attach MCMC posterior samples."""
    from bayesianquilts.irt.grm import GRModel
    model_dir = work_dir / 'grm_baseline'
    if not model_dir.exists():
        model_dir = work_dir / 'grm_mcmc_baseline'
    model = GRModel.load_from_disk(str(model_dir))

    npz_path = work_dir / 'mcmc_samples' / 'mcmc_baseline.npz'
    bnpz = np.load(str(npz_path))

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
            eap[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis])

    model.surrogate_sample = surrogate_sample
    model.calibrated_expectations = {
        k: jnp.mean(v, axis=0) for k, v in surrogate_sample.items()
    }
    return model, bnpz


def _extract_grm_params(bnpz):
    """Return (slope, calibration) posterior means from a baseline MCMC NPZ.

    slope : (I,)
    calibration : (I, K-1)  -- the cumulative threshold array the libfab
        GRM expects (diff0 + cumsum(ddifficulties, axis)).

    For binary items (K=2), ddifficulties is absent and calibration has
    shape (I, 1) containing only difficulties0.
    """
    disc_raw = np.asarray(bnpz['discriminations'])   # (C, S, 1, 1, I, 1)
    disc_flat = disc_raw.reshape(-1, *disc_raw.shape[2:])  # (C*S, 1, 1, I, 1)
    slope = disc_flat.mean(0).ravel()                      # (I,)

    diff0_raw = np.asarray(bnpz['difficulties0'])      # (C, S, 1, 1, I, 1)
    diff0_flat = diff0_raw.reshape(-1, *diff0_raw.shape[2:])
    diff0_mean = diff0_flat.mean(0).ravel()             # (I,)

    if 'ddifficulties' in bnpz.files:
        ddiff_raw = np.asarray(bnpz['ddifficulties'])  # (C, S, 1, 1, I, K-2)
        ddiff_flat = ddiff_raw.reshape(-1, *ddiff_raw.shape[2:])
        ddiff_mean = ddiff_flat.mean(0)                 # (1, 1, I, K-2)
        ddiff_mean = ddiff_mean.reshape(-1, ddiff_raw.shape[-1])  # (I, K-2)
        # calibration[i, 0] = diff0_mean[i]
        # calibration[i, j] = diff0_mean[i] + sum(ddiff_mean[i, :j])  for j >= 1
        calibration = np.concatenate(
            [diff0_mean[:, np.newaxis],
             diff0_mean[:, np.newaxis] + np.cumsum(ddiff_mean, axis=1)],
            axis=1)  # (I, K-1)
    else:
        # Binary case: K=2, calibration is just (I, 1)
        calibration = diff0_mean[:, np.newaxis]  # (I, 1)

    return slope, calibration


def _extract_shared_params(shared_npz_path: Path, item_keys, K):
    """Return (slope, calibration) posterior means from a shared-disc MCMC NPZ.

    The shared-disc model has a single scalar discrimination that broadcasts
    to all items, so slope has shape (I,) with all values equal.
    """
    npz = np.load(str(shared_npz_path))

    disc_raw = np.asarray(npz['discriminations'])
    disc_flat = disc_raw.reshape(-1, *disc_raw.shape[2:])
    disc_mean_raw = disc_flat.mean(0).ravel()
    # May be (1,) or (I,); broadcast to (I,)
    I = len(item_keys)
    if disc_mean_raw.size == 1:
        slope = np.full(I, float(disc_mean_raw[0]))
    else:
        slope = disc_mean_raw[:I]

    diff0_raw = np.asarray(npz['difficulties0'])
    diff0_flat = diff0_raw.reshape(-1, *diff0_raw.shape[2:])
    diff0_mean = diff0_flat.mean(0).ravel()
    if diff0_mean.size == 1:
        diff0_mean = np.full(I, float(diff0_mean[0]))
    else:
        diff0_mean = diff0_mean[:I]

    if 'ddifficulties' in npz.files:
        ddiff_raw = np.asarray(npz['ddifficulties'])
        ddiff_flat = ddiff_raw.reshape(-1, *ddiff_raw.shape[2:])
        ddiff_mean = ddiff_flat.mean(0).reshape(-1, ddiff_raw.shape[-1])  # (1 or I, K-2)
        if ddiff_mean.shape[0] == 1:
            ddiff_mean = np.tile(ddiff_mean, (I, 1))
        else:
            ddiff_mean = ddiff_mean[:I]
        calibration = np.concatenate(
            [diff0_mean[:, np.newaxis],
             diff0_mean[:, np.newaxis] + np.cumsum(ddiff_mean, axis=1)],
            axis=1)  # (I, K-1)
    else:
        # Binary case: K=2, calibration is just (I, 1)
        calibration = diff0_mean[:, np.newaxis]  # (I, 1)

    return slope, calibration


# ---------------------------------------------------------------------------
# Pairwise PMF table extraction
# ---------------------------------------------------------------------------

def _extract_pairwise_pmf_tables(pairwise_model, item_keys, K):
    """Pull PMF tables out of a fitted PairwiseOrdinalStackingModel.

    Returns
    -------
    pmf_tables : dict[target -> predictor -> resp_val -> list[float]]
    stacking_weights : dict[target -> predictor -> float]
    """
    print("  Extracting pairwise PMF tables...", flush=True)
    pmf_tables = {}
    stacking_weights = {}

    for target in item_keys:
        if target not in pairwise_model.variable_names:
            continue
        pmf_tables[target] = {}
        stacking_weights[target] = {}
        for predictor in item_keys:
            if predictor == target:
                continue
            if predictor not in pairwise_model.variable_names:
                continue
            pmf_tables[target][predictor] = {}
            # Evaluate PMF for each possible predictor response
            for resp_val in range(K):
                observed = {predictor: float(resp_val)}
                try:
                    pmf = pairwise_model.predict_pmf(
                        observed, target=target, n_categories=K)
                    pmf_tables[target][predictor][resp_val] = [
                        float(x) for x in np.asarray(pmf)]
                except Exception:
                    pmf_tables[target][predictor][resp_val] = [
                        1.0 / K] * K

            # Stacking weight = uniform 1.0 (the pairwise model internally
            # re-weights; we store equal weights so PairwiseImputationModel
            # fallback also works correctly)
            stacking_weights[target][predictor] = 1.0

    return pmf_tables, stacking_weights


# ---------------------------------------------------------------------------
# Main per-dataset serialization
# ---------------------------------------------------------------------------

def serialize_dataset(dataset_name: str):
    print(f"\n{'='*60}")
    print(f"Serializing: {dataset_name.upper()}")
    print(f"{'='*60}")

    config = DATASET_CONFIGS[dataset_name]
    mod = importlib.import_module(config['module'])

    get_data_kwargs = {'polars_out': True}
    if 'reorient' in inspect.signature(mod.get_data).parameters:
        get_data_kwargs['reorient'] = True
    df, num_people = mod.get_data(**get_data_kwargs)
    base_data = make_data_dict(df)

    # Re-read item_keys after get_data (some modules set them lazily)
    item_keys = mod.item_keys
    K = mod.response_cardinality
    I = len(item_keys)
    print(f"  Items: {I}, K: {K}, People: {num_people}")

    work_dir = IRT_DIR / dataset_name
    out_dir = LIBFAB_EXAMPLES / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. Load baseline GRM --
    irt_model, bnpz = _load_baseline_grm(work_dir, item_keys, num_people)
    slope, calibration = _extract_grm_params(bnpz)
    print(f"  IRT slope range: [{slope.min():.3f}, {slope.max():.3f}]")

    # -- 2. Write refreshed NPZ --
    npz_path = out_dir / f'{dataset_name}_grm_params.npz'
    np.savez(str(npz_path),
             slope=slope,
             calibration=calibration,
             item_keys=np.array(item_keys),
             response_cardinality=K)
    print(f"  Wrote: {npz_path}")

    # -- 3. Load shared-disc params (optional) --
    shared_npz_path = work_dir / 'mcmc_samples' / 'mcmc_shared_disc.npz'
    if shared_npz_path.exists():
        shared_model = load_shared_disc_model(
            item_keys, num_people, K, shared_npz_path)
        shared_slope, shared_calibration = _extract_shared_params(
            shared_npz_path, item_keys, K)
        print(f"  Shared-disc slope: {shared_slope[0]:.3f} (broadcast to all items)")
    else:
        shared_model = None
        shared_slope = None
        shared_calibration = None
        print(f"  Shared-disc absent -- 2-way fallback")

    # -- 4. Load pairwise stacking model --
    from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel
    stacking_path = work_dir / 'pairwise_stacking_model.yaml'
    pairwise_model = PairwiseOrdinalStackingModel.load(str(stacking_path))
    print(f"  Pairwise stacking loaded ({len(pairwise_model.variable_names)} vars)")

    # -- 5. Build bayesianquilts ThreeWayImputationModel for Yao weights --
    def _data_factory():
        yield base_data

    from bayesianquilts.imputation.three_way import ThreeWayImputationModel
    print(f"  Building ThreeWayImputationModel for Yao weight optimization...")
    three_way = ThreeWayImputationModel(
        irt_model=irt_model,
        shared_disc_model=shared_model,
        mice_model=pairwise_model,
        data_factory=_data_factory,
    )
    yao_weights = three_way.weights   # dict[item -> (w_m, w_i, w_s) ndarray]
    w_arr = np.stack([yao_weights[k] for k in item_keys], axis=0)
    print(f"  Yao weights mean: mice={w_arr[:, 0].mean():.3f}, "
          f"irt={w_arr[:, 1].mean():.3f}, "
          f"shared={w_arr[:, 2].mean():.3f}")

    # -- 6. Extract pairwise PMF tables --
    pmf_tables, pw_stacking_weights = _extract_pairwise_pmf_tables(
        pairwise_model, item_keys, K)

    # -- 7. Build output JSON --
    out = {
        "format": "threeway_imputation_v1",
        "n_categories": K,
        "scale_name": dataset_name.upper(),
        "item_keys": item_keys,
        "pairwise_pmfs": pmf_tables,
        "pairwise_stacking_weights": pw_stacking_weights,
        "yao_weights": {k: [float(x) for x in yao_weights[k]] for k in item_keys},
        "irt_slope": [float(x) for x in slope],
        "irt_calibration": [[float(x) for x in row] for row in calibration],
        "shared_slope": ([float(x) for x in shared_slope]
                         if shared_slope is not None else None),
        "shared_calibration": ([[float(x) for x in row] for row in shared_calibration]
                                if shared_calibration is not None else None),
    }

    json_path = out_dir / 'threeway_imputation.json'
    with open(str(json_path), 'w') as fh:
        json.dump(out, fh, separators=(',', ':'))
    print(f"  Wrote: {json_path}")

    return {
        'dataset': dataset_name,
        'n_items': I,
        'K': K,
        'npz': str(npz_path),
        'json': str(json_path),
        'yao_weights_mean': {
            'mice': float(w_arr[:, 0].mean()),
            'irt': float(w_arr[:, 1].mean()),
            'shared': float(w_arr[:, 2].mean()),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        help='Dataset name or "all"')
    args = parser.parse_args()

    if args.dataset == 'all':
        datasets = DATASETS_8
    else:
        datasets = [args.dataset]

    results = {}
    for d in datasets:
        if d not in DATASET_CONFIGS:
            print(f"Unknown dataset: {d}")
            continue
        try:
            results[d] = serialize_dataset(d)
        except Exception as exc:
            import traceback
            print(f"ERROR: {d}: {exc}")
            traceback.print_exc()

    print("\n=== Summary ===")
    for d, r in results.items():
        if r:
            w = r['yao_weights_mean']
            print(f"  {d:10s}: I={r['n_items']}, K={r['K']}, "
                  f"mice={w['mice']:.3f}, irt={w['irt']:.3f}, "
                  f"shared={w['shared']:.3f}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Compute ELPD-LOO post-hoc from saved GRM models.

Loads fitted baseline/mice_only/imputed GRM models from a results directory
and computes PSIS-LOO for each, without refitting.

Usage:
    python compute_elpd_loo.py --results-dir ./results_v3 --dataset grit
    python compute_elpd_loo.py --results-dir ./results_v3 --dataset all
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import jax
import jax.numpy as jnp


DATASETS = ['grit', 'rwa', 'tma', 'wpi', 'npi', 'eqsq', 'bouldering']


def make_data_factory(data_dict, batch_size, num_people):
    """Create a data factory that yields uniform-sized batches."""
    steps_per_epoch = int(np.ceil(num_people / batch_size))

    def data_factory():
        indices = np.arange(num_people)
        np.random.shuffle(indices)
        n_needed = steps_per_epoch * batch_size
        if n_needed > num_people:
            indices = np.concatenate([
                indices,
                np.random.choice(num_people, n_needed - num_people, replace=True),
            ])
        for start in range(0, n_needed, batch_size):
            idx_batch = indices[start:start + batch_size]
            yield {k: v[idx_batch] for k, v in data_dict.items()}
    return data_factory


def calibrate_manually(model, n_samples=32, seed=42):
    """Calibrate a model by sampling from its surrogate posterior."""
    try:
        surrogate = model.surrogate_distribution_generator(model.params)
        key = jax.random.PRNGKey(seed)
        samples = surrogate.sample(n_samples, seed=key)
        expectations = {k: jnp.mean(v, axis=0) for k, v in samples.items()}
        model.calibrated_expectations = expectations
        model.surrogate_sample = samples
    except KeyError as e:
        print(f"  Warning: surrogate sampling failed ({e}), using point estimates")


def compute_elpd_for_dataset(dataset_name, results_dir, batch_size=256,
                             n_samples=100, seed=42):
    """Compute ELPD-LOO for all GRM models in a dataset results directory."""
    from common.pipeline import load_dataset
    from bayesianquilts.irt.grm import GRModel

    ds_dir = Path(results_dir) / dataset_name

    # Load the synthetic data (need to regenerate from saved abilities + neural model)
    # Instead, we reconstruct from the saved abilities.npz and the neural model
    abilities_path = ds_dir / 'abilities.npz'
    if not abilities_path.exists():
        print(f"SKIP {dataset_name}: no abilities.npz found")
        return None

    # Load original dataset to get item_keys and response_cardinality
    data_dict, item_keys, response_cardinality, num_people = load_dataset(dataset_name)

    # Load true abilities and regenerate synthetic data
    ab = np.load(abilities_path)
    true_abilities = ab['true']

    # Regenerate synthetic data from NeuralGRM
    neural_dir = ds_dir / 'neural_grm'
    if not neural_dir.exists():
        print(f"SKIP {dataset_name}: no neural_grm directory")
        return None

    from bayesianquilts.irt.neural_grm import NeuralGRModel
    from common.pipeline import generate_synthetic_data

    neural_model = NeuralGRModel.load_from_disk(neural_dir)
    calibrate_manually(neural_model, n_samples=32, seed=101)

    synth_data = generate_synthetic_data(
        neural_model, item_keys, response_cardinality,
        abilities=true_abilities,
        missingness_rate=0.25,
        missing_respondent_frac=0.4,
        seed=42,
    )

    factory_fn = make_data_factory(synth_data, batch_size, num_people)

    elpd_results = {}
    model_dirs = {
        'baseline': ds_dir / 'grm_baseline',
        'mice_only': ds_dir / 'grm_mice_only',
        'imputed': ds_dir / 'grm_imputed',
    }

    for label, model_dir in model_dirs.items():
        if not model_dir.exists():
            print(f"  {label}: not found, skipping")
            continue

        print(f"\n  Loading {label} GRM from {model_dir}...")
        model = GRModel.load_from_disk(model_dir)
        calibrate_manually(model, n_samples=32, seed=101 + hash(label) % 100)

        print(f"  Computing ELPD-LOO for {label}...")
        factory = make_data_factory(synth_data, batch_size, num_people)
        model._compute_elpd_loo(factory, n_samples=n_samples, seed=seed + 100, use_ais=True)

        elpd_results[label] = {
            'elpd_loo': model.elpd_loo,
            'elpd_loo_se': model.elpd_loo_se,
            'elpd_loo_per_obs': model.elpd_loo_per_obs,
            'n_obs': getattr(model, 'elpd_loo_n_obs', None),
            'max_khat': float(np.max(model.elpd_loo_khat)),
            'mean_khat': float(np.mean(model.elpd_loo_khat)),
        }

    # Save ELPD results
    out_path = ds_dir / 'elpd_loo.json'
    with open(out_path, 'w') as f:
        json.dump(elpd_results, f, indent=2)
    print(f"\n  ELPD-LOO results saved to {out_path}")

    # Also update results.json if it exists
    results_path = ds_dir / 'results.json'
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        results['elpd_loo'] = elpd_results
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Updated {results_path} with ELPD-LOO")

    return elpd_results


def main():
    parser = argparse.ArgumentParser(
        description="Compute ELPD-LOO post-hoc from saved GRM models."
    )
    parser.add_argument("--results-dir", required=True, help="Results directory")
    parser.add_argument("--dataset", required=True,
                        help=f"Dataset name or 'all'. Choices: {DATASETS}")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Posterior samples for PSIS-LOO (default 100)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    all_results = {}

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"ELPD-LOO: {ds.upper()}")
        print(f"{'='*60}")

        result = compute_elpd_for_dataset(
            ds, args.results_dir,
            batch_size=args.batch_size,
            n_samples=args.n_samples,
            seed=args.seed,
        )
        if result:
            all_results[ds] = result

    # Summary table
    if all_results:
        print(f"\n{'='*90}")
        print("ELPD-LOO SUMMARY")
        print(f"{'='*90}")
        header = (f"{'Dataset':<12} {'Base ELPD/obs':>14} {'MICE ELPD/obs':>14} "
                  f"{'Mixed ELPD/obs':>14} {'Base k̂':>8} {'MICE k̂':>8} {'Mixed k̂':>8}")
        print(header)
        print("-" * 90)
        for ds, r in all_results.items():
            def _e(label):
                e = r.get(label)
                return f"{e['elpd_loo_per_obs']:>14.4f}" if e else f"{'N/A':>14}"
            def _k(label):
                e = r.get(label)
                return f"{e['max_khat']:>8.3f}" if e else f"{'N/A':>8}"
            print(f"{ds:<12} {_e('baseline')} {_e('mice_only')} {_e('imputed')} "
                  f"{_k('baseline')} {_k('mice_only')} {_k('imputed')}")


if __name__ == "__main__":
    main()

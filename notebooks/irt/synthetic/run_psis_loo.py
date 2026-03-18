"""
Compute PSIS-LOO k-hat diagnostics for baseline GRM models.

For each completed dataset, loads the baseline GRM, draws posterior samples,
computes per-person log-likelihoods, runs PSIS-LOO, and reports the number
of observations with k-hat > 0.7.

Usage:
    python run_psis_loo.py [--results-dir ./results_v3] [--n-samples 100]
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from common.pipeline import load_dataset, make_data_factory, calibrate_model


DATASETS = ['grit', 'rwa', 'tma', 'wpi', 'npi', 'eqsq']


def run_psis_loo(dataset_name, results_dir, n_samples=100, batch_size=256, seed=42):
    """Load a fitted baseline GRM and compute PSIS-LOO diagnostics."""
    from bayesianquilts.irt.grm import GRModel
    from bayesianquilts.metrics.nppsis import psisloo

    model_dir = results_dir / dataset_name / "grm_baseline"
    if not (model_dir / "params.h5").exists():
        print(f"  Skipping {dataset_name}: no baseline GRM found at {model_dir}")
        return None

    # Load dataset to get the synthetic data
    synth_abilities_path = results_dir / dataset_name / "abilities.npz"
    if not synth_abilities_path.exists():
        print(f"  Skipping {dataset_name}: no abilities.npz found")
        return None

    # Load the real dataset metadata
    _, item_keys, response_cardinality, num_people = load_dataset(dataset_name)

    # Load the synthetic data that was used to fit the baseline
    # We need to regenerate it from the NeuralGRM + true abilities
    # But we can also just load the baseline model and use the same data factory
    # approach as _compute_elpd_loo

    # Load the neural GRM to regenerate synthetic data
    from bayesianquilts.irt.neural_grm import NeuralGRModel
    from common.pipeline import generate_synthetic_data, sample_abilities

    neural_dir = results_dir / dataset_name / "neural_grm"
    if not (neural_dir / "params.h5").exists():
        print(f"  Skipping {dataset_name}: no NeuralGRM found")
        return None

    print(f"\n  Loading NeuralGRM from {neural_dir}")
    neural_model = NeuralGRModel.load_from_disk(neural_dir)
    calibrate_model(neural_model)

    # Regenerate the same synthetic data (same seed as evaluate_synthetic.py)
    true_abilities = sample_abilities(num_people, dim=1, seed=42)
    synth_data = generate_synthetic_data(
        neural_model, item_keys, response_cardinality,
        abilities=true_abilities, missingness_rate=0.25,
        missing_respondent_frac=0.4, seed=42,
    )

    # Use all data (baseline model was fit on all data)
    baseline_data = {'person': np.arange(num_people, dtype=np.float64)}
    for key in item_keys:
        baseline_data[key] = synth_data[key]

    print(f"  Using all {num_people} people")

    # Load baseline GRM
    print(f"  Loading baseline GRM from {model_dir}")
    model = GRModel(
        item_keys=item_keys,
        num_people=num_people,
        dim=1,
        response_cardinality=response_cardinality,
        dtype=jnp.float64,
    )
    array_data = GRModel._load_arrays_hdf5(model_dir)
    for name, val in array_data.items():
        setattr(model, name, val)
    calibrate_model(model)

    # Draw posterior samples
    print(f"  Drawing {n_samples} posterior samples...")
    surrogate = model.surrogate_distribution_generator(model.params)
    key = jax.random.PRNGKey(seed)
    samples = surrogate.sample(n_samples, seed=key)
    if hasattr(model, 'transform'):
        samples = model.transform(samples)

    # Compute per-person log-likelihood matrix: (n_samples, n_fully_observed)
    factory = make_data_factory(baseline_data, batch_size, num_people)
    log_lik_matrix = np.full((n_samples, num_people), np.nan, dtype=np.float64)

    print(f"  Computing log-likelihoods...")
    for batch in factory():
        people = np.asarray(batch['person'], dtype=np.int32)
        pred = model.predictive_distribution(batch, **samples)
        log_lik_matrix[:, people] = np.array(pred['log_likelihood'])

    # Check coverage
    visited = ~np.isnan(log_lik_matrix[0])
    n_obs = int(np.sum(visited))
    if n_obs < num_people:
        print(f"  Warning: only {n_obs}/{num_people} people visited")
        log_lik_matrix = log_lik_matrix[:, visited]

    # Run PSIS-LOO (no adaptation)
    print(f"  Running PSIS-LOO ({n_obs} observations, {n_samples} samples)...")
    loo, loos, ks = psisloo(log_lik_matrix)

    # Count k-hat diagnostics
    n_bad = int(np.sum(ks > 0.7))
    n_marginal = int(np.sum((ks > 0.5) & (ks <= 0.7)))
    n_good = int(np.sum(ks <= 0.5))

    results = {
        'dataset': dataset_name,
        'n_observations': n_obs,
        'n_samples': n_samples,
        'elpd_loo': float(loo),
        'elpd_loo_se': float(np.std(loos) * np.sqrt(n_obs)),
        'elpd_loo_per_obs': float(loo / n_obs),
        'khat_max': float(np.max(ks)),
        'khat_mean': float(np.mean(ks)),
        'khat_median': float(np.median(ks)),
        'n_khat_good': n_good,       # k <= 0.5
        'n_khat_marginal': n_marginal, # 0.5 < k <= 0.7
        'n_khat_bad': n_bad,          # k > 0.7
        'pct_khat_bad': float(n_bad / n_obs * 100),
    }

    print(f"  ELPD-LOO: {loo:.2f} (per obs: {loo/n_obs:.4f})")
    print(f"  k-hat: max={np.max(ks):.3f}, mean={np.mean(ks):.3f}, "
          f"median={np.median(ks):.3f}")
    print(f"  k-hat > 0.7: {n_bad}/{n_obs} ({n_bad/n_obs*100:.1f}%)")
    print(f"  k-hat 0.5-0.7: {n_marginal}/{n_obs} ({n_marginal/n_obs*100:.1f}%)")
    print(f"  k-hat <= 0.5: {n_good}/{n_obs} ({n_good/n_obs*100:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute PSIS-LOO k-hat diagnostics for baseline GRM models."
    )
    parser.add_argument("--results-dir", default="./results_v3",
                        help="Results directory")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of posterior samples for PSIS")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for log-likelihood computation")
    parser.add_argument("--dataset", default="all",
                        help="Dataset name or 'all'")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    all_results = {}

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"PSIS-LOO for {ds.upper()} (baseline GRM)")
        print(f"{'='*60}")

        result = run_psis_loo(
            ds, results_dir,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
        )
        if result is not None:
            all_results[ds] = result

    # Summary table
    if all_results:
        print(f"\n{'='*80}")
        print("PSIS-LOO SUMMARY (Baseline GRM)")
        print(f"{'='*80}")
        print(f"{'Dataset':<10} {'ELPD/n':>10} {'k_max':>8} {'k_mean':>8} "
              f"{'k>0.7':>8} {'%bad':>8} {'N':>8}")
        print("-" * 80)
        for ds, r in all_results.items():
            print(f"{ds:<10} {r['elpd_loo_per_obs']:>10.4f} "
                  f"{r['khat_max']:>8.3f} {r['khat_mean']:>8.3f} "
                  f"{r['n_khat_bad']:>8d} {r['pct_khat_bad']:>7.1f}% "
                  f"{r['n_observations']:>8d}")

        # Save results
        out_path = results_dir / 'psis_loo_baseline.json'
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

"""
Evaluate synthetic data pipeline for psychometric datasets.

Fits a Neural GRM on real data, generates synthetic responses (with MCAR
missingness), fits standard GRM (baseline + imputed) on the synthetic data,
and compares how well each preserves the true ability ordering.

Usage:
    python evaluate_synthetic.py --dataset npi [--epochs 500] [--lr 2e-4]
    python evaluate_synthetic.py --dataset all [--output-dir ./results]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run_pipeline(dataset_name, output_dir, epochs=500, lr=2e-4, grm_lr=None,
                 batch_size=256, missingness_rate=0.2,
                 dim=1, kappa_scale=0.5, eta_scale=0.1, patience=10,
                 lr_decay_factor=0.9, clip_norm=1.0,
                 reload_neural_grm=False):
    """Run the full synthetic evaluation pipeline for one dataset.

    Steps:
        1. Load real data
        2. Fit NeuralGRM on real data
        3. Extract true abilities
        4. Generate synthetic data with MCAR missingness
        5. Fit MICEBayesianLOO imputation model on synthetic data
        6. Fit baseline GRM on synthetic data
        7. Fit imputed GRM on synthetic data
        8. Compare ability ordering
        9. Generate plots

    Returns:
        Dict with comparison metrics.
    """
    from common.pipeline import (
        load_dataset,
        fit_neural_grm,
        generate_synthetic_data,
        sample_abilities,
        fit_grm_baseline,
        fit_grm_imputed,
        calibrate_model,
        compare_ability_ordering,
        make_comparison_plots,
    )
    from bayesianquilts.imputation.mice_loo import MICEBayesianLOO

    output_dir = Path(output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default GRM learning rate to the same as NeuralGRM if not specified
    if grm_lr is None:
        grm_lr = lr

    # 1. Load real data
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")
    data_dict, item_keys, response_cardinality, num_people = load_dataset(dataset_name)
    print(f"  People: {num_people}, Items: {len(item_keys)}, K: {response_cardinality}")

    # 2. Fit Neural GRM on real data
    print(f"\n--- Fitting NeuralGRM on real data ---")
    neural_model = fit_neural_grm(
        data_dict, item_keys, response_cardinality, num_people,
        save_dir=output_dir / "neural_grm",
        dim=dim,
        batch_size=batch_size,
        num_epochs=epochs,
        learning_rate=lr,
        patience=patience,
        kappa_scale=kappa_scale,
        eta_scale=eta_scale,
        lr_decay_factor=lr_decay_factor,
        clip_norm=clip_norm,
        reload=reload_neural_grm,
    )

    # 3. Sample fresh abilities from N(0,1) as ground truth
    print(f"\n--- Sampling ground-truth abilities ---")
    true_abilities = sample_abilities(num_people, dim=dim, seed=42)
    print(f"  True abilities shape: {true_abilities.shape}")
    print(f"  Range: [{true_abilities.min():.3f}, {true_abilities.max():.3f}]")

    # 4. Generate synthetic data from NeuralGRM item params + sampled abilities
    print(f"\n--- Generating synthetic data (missingness={missingness_rate:.0%}) ---")
    synth_data = generate_synthetic_data(
        neural_model, item_keys, response_cardinality,
        abilities=true_abilities,
        missingness_rate=missingness_rate, seed=42,
    )
    n_bad = sum(
        np.sum(np.isnan(synth_data[k]) | (synth_data[k] < 0) | (synth_data[k] >= response_cardinality))
        for k in item_keys
    )
    print(f"  Synthetic data: {num_people} people, {n_bad} missing values")

    # 5. Fit imputation model on synthetic data
    print(f"\n--- Fitting MICEBayesianLOO on synthetic data ---")
    synth_df = pd.DataFrame({k: synth_data[k] for k in item_keys})
    synth_df = synth_df.replace(-1, np.nan)

    mice_loo = MICEBayesianLOO(
        random_state=42,
        prior_scale=1.0,
        pathfinder_num_samples=100,
        pathfinder_maxiter=50,
        batch_size=512,
        verbose=True,
    )
    mice_loo.fit_loo_models(
        X_df=synth_df,
        n_top_features=min(len(item_keys), 40),
        n_jobs=-1,
        fit_zero_predictors=True,
        seed=42,
    )
    # Override variable types: IRT items are binary (K=2) or ordinal (K>2)
    for idx in range(len(item_keys)):
        vtype = 'binary' if response_cardinality == 2 else 'ordinal'
        mice_loo.variable_types[idx] = vtype
    mice_loo.save_to_disk(output_dir / "imputation_model")
    print(f"  Imputation model saved ({len(mice_loo.univariate_results)} univariate models)")

    # 6. Fit baseline GRM on synthetic data
    print(f"\n--- Fitting baseline GRM on synthetic data ---")
    baseline_model = fit_grm_baseline(
        synth_data, item_keys, response_cardinality, num_people,
        save_dir=output_dir / "grm_baseline",
        dim=dim, batch_size=batch_size, num_epochs=epochs,
        learning_rate=grm_lr, patience=patience, kappa_scale=kappa_scale,
        lr_decay_factor=lr_decay_factor, clip_norm=clip_norm,
    )

    # 7. Fit imputed GRM on synthetic data
    print(f"\n--- Fitting imputed GRM on synthetic data ---")
    imputed_model = fit_grm_imputed(
        synth_data, item_keys, response_cardinality, num_people,
        save_dir=output_dir / "grm_imputed",
        imputation_model=mice_loo,
        dim=dim, batch_size=batch_size, num_epochs=epochs,
        learning_rate=grm_lr, patience=patience, kappa_scale=kappa_scale,
        lr_decay_factor=lr_decay_factor, clip_norm=clip_norm,
    )

    # 8. Compare ability ordering preservation
    print(f"\n--- Comparing ability orderings ---")
    baseline_abilities = np.array(baseline_model.calibrated_expectations['abilities'])
    imputed_abilities = np.array(imputed_model.calibrated_expectations['abilities'])

    baseline_metrics = compare_ability_ordering(true_abilities, baseline_abilities)
    imputed_metrics = compare_ability_ordering(true_abilities, imputed_abilities)

    # Add ELPD-LOO metrics from the fitted models
    for label, model_obj, metrics in [
        ('baseline', baseline_model, baseline_metrics),
        ('imputed', imputed_model, imputed_metrics),
    ]:
        if hasattr(model_obj, 'elpd_loo'):
            metrics['elpd_loo'] = model_obj.elpd_loo
            metrics['elpd_loo_se'] = model_obj.elpd_loo_se
            metrics['elpd_loo_per_obs'] = model_obj.elpd_loo_per_obs
            metrics['elpd_loo_se_per_obs'] = model_obj.elpd_loo_se_per_obs
            metrics['elpd_loo_n_obs'] = model_obj.elpd_loo_n_obs
            metrics['elpd_loo_khat_max'] = float(np.max(model_obj.elpd_loo_khat))
            metrics['elpd_loo_khat_mean'] = float(np.mean(model_obj.elpd_loo_khat))

    results = {
        'dataset': dataset_name,
        'num_people': num_people,
        'num_items': len(item_keys),
        'response_cardinality': response_cardinality,
        'missingness_rate': missingness_rate,
        'baseline': baseline_metrics,
        'imputed': imputed_metrics,
        'hyperparameters': {
            'epochs': epochs,
            'lr': lr,
            'grm_lr': grm_lr,
            'batch_size': batch_size,
            'dim': dim,
            'kappa_scale': kappa_scale,
            'eta_scale': eta_scale,
            'patience': patience,
            'lr_decay_factor': lr_decay_factor,
            'clip_norm': clip_norm,
        },
    }

    print(f"\n  Baseline:  Spearman r = {baseline_metrics['spearman_r']:.4f}, "
          f"Kendall tau = {baseline_metrics['kendall_tau']:.4f}, "
          f"RMSE = {baseline_metrics['rmse']:.4f}")
    print(f"  Imputed:   Spearman r = {imputed_metrics['spearman_r']:.4f}, "
          f"Kendall tau = {imputed_metrics['kendall_tau']:.4f}, "
          f"RMSE = {imputed_metrics['rmse']:.4f}")
    if 'elpd_loo' in baseline_metrics:
        print(f"  Baseline ELPD-LOO: {baseline_metrics['elpd_loo']:.2f} "
              f"(per obs: {baseline_metrics['elpd_loo_per_obs']:.4f})")
        print(f"  Imputed  ELPD-LOO: {imputed_metrics['elpd_loo']:.2f} "
              f"(per obs: {imputed_metrics['elpd_loo_per_obs']:.4f})")

    # 9. Generate plots
    print(f"\n--- Generating plots ---")
    make_comparison_plots(
        true_abilities, baseline_abilities, imputed_abilities,
        dataset_name, output_dir / "plots",
    )

    # Save results JSON
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")

    # Save abilities for manuscript figure generation
    np.savez(
        output_dir / 'abilities.npz',
        true=true_abilities,
        baseline=baseline_abilities,
        imputed=imputed_abilities,
    )
    print(f"Abilities saved to {output_dir / 'abilities.npz'}")

    return results


DATASETS = ['grit', 'rwa', 'tma', 'wpi', 'npi', 'eqsq']


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic data pipeline for psychometric datasets."
    )
    parser.add_argument(
        "--dataset", required=True,
        help=f"Dataset name or 'all'. Choices: {DATASETS}"
    )
    parser.add_argument("--output-dir", default="./results", help="Output directory")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="NeuralGRM learning rate")
    parser.add_argument("--grm-lr", type=float, default=None,
                        help="GRM learning rate (defaults to --lr if not set)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--missingness", type=float, default=0.2,
                        help="MCAR missingness rate")
    parser.add_argument("--dim", type=int, default=1, help="Latent dimension")
    parser.add_argument("--kappa-scale", type=float, default=0.5,
                        help="Kappa scale for horseshoe prior")
    parser.add_argument("--eta-scale", type=float, default=0.1,
                        help="Eta scale for horseshoe prior (local shrinkage)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--lr-decay-factor", type=float, default=0.9,
                        help="LR decay factor on plateau (default 0.9)")
    parser.add_argument("--clip-norm", type=float, default=1.0,
                        help="Gradient clipping norm (default 1.0)")
    parser.add_argument("--reload-neural-grm", action="store_true",
                        help="Reload saved NeuralGRM instead of re-training")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    all_results = {}

    for ds in datasets:
        if ds not in DATASETS:
            print(f"Warning: unknown dataset '{ds}', skipping.")
            continue

        results = run_pipeline(
            ds,
            output_dir=args.output_dir,
            epochs=args.epochs,
            lr=args.lr,
            grm_lr=args.grm_lr,
            batch_size=args.batch_size,
            missingness_rate=args.missingness,
            dim=args.dim,
            kappa_scale=args.kappa_scale,
            eta_scale=args.eta_scale,
            patience=args.patience,
            lr_decay_factor=args.lr_decay_factor,
            clip_norm=args.clip_norm,
            reload_neural_grm=args.reload_neural_grm,
        )
        all_results[ds] = results

    # Summary table
    if len(all_results) > 1:
        has_elpd = 'elpd_loo_per_obs' in next(iter(all_results.values()))['baseline']
        print(f"\n{'='*90}")
        print("SUMMARY")
        print(f"{'='*90}")
        header = (f"{'Dataset':<10} {'Base Spearman':>14} {'Imp Spearman':>14} "
                  f"{'Base RMSE':>10} {'Imp RMSE':>10}")
        if has_elpd:
            header += f" {'Base ELPD/n':>12} {'Imp ELPD/n':>12}"
        print(header)
        print("-" * 90)
        for ds, r in all_results.items():
            line = (f"{ds:<10} {r['baseline']['spearman_r']:>14.4f} "
                    f"{r['imputed']['spearman_r']:>14.4f} "
                    f"{r['baseline']['rmse']:>10.4f} "
                    f"{r['imputed']['rmse']:>10.4f}")
            if has_elpd:
                line += (f" {r['baseline']['elpd_loo_per_obs']:>12.4f}"
                         f" {r['imputed']['elpd_loo_per_obs']:>12.4f}")
            print(line)

        # Save combined results
        output_dir = Path(args.output_dir)
        with open(output_dir / 'all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()

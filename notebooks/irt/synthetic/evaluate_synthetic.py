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
                 batch_size=256, missingness_rate=0.25,
                 missing_respondent_frac=0.4,
                 dim=1, kappa_scale=0.5, eta_scale=0.1, patience=10,
                 lr_decay_factor=0.9, clip_norm=1.0,
                 reload_neural_grm=False,
                 noisy_dim=False,
                 noisy_dim_eta_scale=0.01,
                 noisy_dim_ability_scale=2.0,
                 sample_size=64):
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
        make_data_factory,
    )
    from bayesianquilts.imputation.mice_loo import MICEBayesianLOO
    from bayesianquilts.imputation.mixed import IrtMixedImputationModel

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
        noisy_dim=noisy_dim,
        noisy_dim_eta_scale=noisy_dim_eta_scale,
        noisy_dim_ability_scale=noisy_dim_ability_scale,
        sample_size=sample_size,
    )

    # 3. Sample fresh abilities from N(0,1) as ground truth
    print(f"\n--- Sampling ground-truth abilities ---")
    true_abilities = sample_abilities(num_people, dim=dim, seed=42)
    print(f"  True abilities shape: {true_abilities.shape}")
    print(f"  Range: [{true_abilities.min():.3f}, {true_abilities.max():.3f}]")

    # 4. Generate synthetic data from NeuralGRM item params + sampled abilities
    print(f"\n--- Generating synthetic data (missingness={missingness_rate:.0%} "
          f"for {missing_respondent_frac:.0%} of respondents) ---")
    synth_data = generate_synthetic_data(
        neural_model, item_keys, response_cardinality,
        abilities=true_abilities,
        missingness_rate=missingness_rate,
        missing_respondent_frac=missing_respondent_frac,
        seed=42,
    )
    n_bad = sum(
        np.sum(np.isnan(synth_data[k]) | (synth_data[k] < 0) | (synth_data[k] >= response_cardinality))
        for k in item_keys
    )
    print(f"  Synthetic data: {num_people} people, {n_bad} missing values")

    # 5. Count missingness stats
    has_missing = np.zeros(num_people, dtype=bool)
    for key in item_keys:
        has_missing |= (synth_data[key] < 0)
    n_fully_observed = int(np.sum(~has_missing))
    print(f"  Fully observed respondents: {n_fully_observed}/{num_people} "
          f"({n_fully_observed/num_people*100:.1f}%)")

    # 6. Fit imputation model on full synthetic data (all respondents)
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
        n_jobs=1,
        fit_zero_predictors=True,
        seed=42,
    )
    # Override variable types: IRT items are binary (K=2) or ordinal (K>2)
    for idx in range(len(item_keys)):
        vtype = 'binary' if response_cardinality == 2 else 'ordinal'
        mice_loo.variable_types[idx] = vtype
    mice_loo.save(str(output_dir / "mice_loo_model.yaml"))
    print(f"  Imputation model saved ({len(mice_loo.univariate_results)} univariate models)")

    # 7. Fit baseline GRM on all data (no imputation — missing responses
    #    are marginalized out, contributing 0 to the log-likelihood)
    snapshot_epoch = 50  # save early checkpoint for warm-starting imputed model
    print(f"\n--- Fitting baseline GRM on all data ({num_people} people, no imputation) ---")
    baseline_model, snapshot_params = fit_grm_baseline(
        synth_data, item_keys, response_cardinality, num_people,
        save_dir=output_dir / "grm_baseline",
        dim=dim, batch_size=batch_size, num_epochs=epochs,
        learning_rate=grm_lr, patience=patience, kappa_scale=kappa_scale,
        lr_decay_factor=lr_decay_factor, clip_norm=clip_norm,
        snapshot_epoch=snapshot_epoch, sample_size=sample_size,
    )
    if snapshot_params is not None:
        print(f"  Using baseline epoch-{snapshot_epoch} snapshot to warm-start imputed model")
    else:
        print(f"  Warning: no snapshot at epoch {snapshot_epoch} (baseline may have stopped earlier)")

    # 8. Fit MICE-only GRM (imputation from MICE alone, no IRT blending)
    print(f"\n--- Fitting MICE-only GRM on all data ({num_people} people) ---")
    mice_only_model = fit_grm_imputed(
        synth_data, item_keys, response_cardinality, num_people,
        save_dir=output_dir / "grm_mice_only",
        imputation_model=mice_loo,
        dim=dim, batch_size=batch_size, num_epochs=epochs,
        learning_rate=grm_lr, patience=patience, kappa_scale=kappa_scale,
        lr_decay_factor=lr_decay_factor, clip_norm=clip_norm,
        initial_values=snapshot_params, sample_size=sample_size,
    )

    # 9. Build mixed imputation model (blends MICE + IRT baseline via per-item WAIC)
    print(f"\n--- Building IrtMixedImputationModel ---")
    factory = make_data_factory(synth_data, batch_size, num_people)
    mixed_imputation = IrtMixedImputationModel(
        irt_model=baseline_model,
        mice_model=mice_loo,
        data_factory=factory,
        irt_elpd_batch_size=4,
    )
    print(mixed_imputation.summary())

    # Save diagnostics to HDF5 and free pointwise LOO arrays
    mixed_imputation.save_diagnostics(str(output_dir / "mixed_diagnostics.h5"))
    mixed_imputation._irt_elpd_loo_per_obs = {}
    print(f"  Diagnostics saved to {output_dir / 'mixed_diagnostics.h5'}")

    # 10. Fit mixed-imputed GRM on all data
    #     Uses weighted Rao-Blackwellization: each missing cell
    #     contributes w_mice * log[sum_k q_mice(k) * p(Y=k|theta)]
    #     When w_mice=0, contribution is 0 (ignorability).
    print(f"\n--- Fitting mixed-imputed GRM on all data ({num_people} people) ---")
    imputed_model = fit_grm_imputed(
        synth_data, item_keys, response_cardinality, num_people,
        save_dir=output_dir / "grm_imputed",
        imputation_model=mixed_imputation,
        dim=dim, batch_size=batch_size, num_epochs=epochs,
        learning_rate=grm_lr, patience=patience, kappa_scale=kappa_scale,
        lr_decay_factor=lr_decay_factor, clip_norm=clip_norm,
        initial_values=snapshot_params, sample_size=sample_size,
    )

    # 11. Compare ability ordering preservation
    print(f"\n--- Comparing ability orderings ---")
    baseline_abilities = np.array(baseline_model.calibrated_expectations['abilities'])
    mice_only_abilities = np.array(mice_only_model.calibrated_expectations['abilities'])
    imputed_abilities = np.array(imputed_model.calibrated_expectations['abilities'])

    # true_abilities shape: (N, 1, 1, 1) — single ground-truth dimension
    # estimated shape: (N, D, 1, 1) where D=2 if noisy_dim
    true_flat = true_abilities[:, 0, 0, 0]  # (N,)

    def _extract_dim(abil, d):
        """Extract dimension d from (N, D, 1, 1) abilities."""
        a = np.array(abil)
        if a.ndim == 4 and a.shape[1] > d:
            return a[:, d, 0, 0]
        elif d == 0:
            return a.flatten()
        else:
            return None

    def _extract_norm(abil):
        """Compute L2 norm across latent dimensions: (N, D, 1, 1) -> (N,)."""
        a = np.array(abil)
        if a.ndim == 4:
            return np.sqrt(np.sum(a[:, :, 0, 0] ** 2, axis=1))
        return np.abs(a.flatten())

    # Primary dimension (dim 0) comparisons
    baseline_metrics = compare_ability_ordering(true_flat, _extract_dim(baseline_abilities, 0))
    mice_only_metrics = compare_ability_ordering(true_flat, _extract_dim(mice_only_abilities, 0))
    imputed_metrics = compare_ability_ordering(true_flat, _extract_dim(imputed_abilities, 0))

    results = {
        'dataset': dataset_name,
        'num_people': num_people,
        'num_fully_observed': n_fully_observed,
        'num_items': len(item_keys),
        'response_cardinality': response_cardinality,
        'missingness_rate': missingness_rate,
        'missing_respondent_frac': missing_respondent_frac,
        'baseline': baseline_metrics,
        'mice_only': mice_only_metrics,
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
            'noisy_dim': noisy_dim,
            'noisy_dim_eta_scale': noisy_dim_eta_scale,
            'noisy_dim_ability_scale': noisy_dim_ability_scale,
        },
    }

    # Noisy dimension and norm comparisons (when noisy_dim is used)
    if noisy_dim:
        for label, abil in [('baseline', baseline_abilities),
                            ('mice_only', mice_only_abilities),
                            ('imputed', imputed_abilities)]:
            dim1 = _extract_dim(abil, 1)
            norm_abil = _extract_norm(abil)
            if dim1 is not None:
                results[f'{label}_dim1'] = compare_ability_ordering(true_flat, dim1)
            results[f'{label}_norm'] = compare_ability_ordering(true_flat, norm_abil)

    def _fmt_ci(m, key):
        """Format metric with 95% CI."""
        val = m[key]
        ci = m.get(f'{key}_ci', None)
        if ci:
            return f"{val:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
        return f"{val:.4f}"

    print(f"\n  Primary dimension (dim 0) vs ground truth:")
    for label, m in [('Baseline', baseline_metrics),
                     ('MICE-only', mice_only_metrics),
                     ('Mixed', imputed_metrics)]:
        print(f"  {label:<12} ρ = {_fmt_ci(m, 'spearman_r')}, "
              f"τ = {_fmt_ci(m, 'kendall_tau')}, "
              f"RMSE = {_fmt_ci(m, 'rmse')}")

    if noisy_dim:
        print(f"\n  Noisy dimension (dim 1) vs ground truth:")
        for label in ['baseline', 'mice_only', 'imputed']:
            m = results.get(f'{label}_dim1', {})
            if m:
                print(f"  {label:<12} ρ = {_fmt_ci(m, 'spearman_r')}, "
                      f"τ = {_fmt_ci(m, 'kendall_tau')}")

        print(f"\n  Vector norm (L2) vs ground truth:")
        for label in ['baseline', 'mice_only', 'imputed']:
            m = results.get(f'{label}_norm', {})
            if m:
                print(f"  {label:<12} ρ = {_fmt_ci(m, 'spearman_r')}, "
                      f"τ = {_fmt_ci(m, 'kendall_tau')}")

    # 12. Generate plots
    print(f"\n--- Generating plots ---")
    make_comparison_plots(
        true_abilities, baseline_abilities, mice_only_abilities,
        imputed_abilities, dataset_name, output_dir / "plots",
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
        mice_only=mice_only_abilities,
        imputed=imputed_abilities,
    )
    print(f"Abilities saved to {output_dir / 'abilities.npz'}")

    return results


DATASETS = ['grit', 'rwa', 'tma', 'wpi', 'npi', 'eqsq', 'bouldering']


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
    parser.add_argument("--missingness", type=float, default=0.25,
                        help="MCAR missingness rate for selected respondents")
    parser.add_argument("--missing-respondent-frac", type=float, default=0.4,
                        help="Fraction of respondents with any missing data (default 0.4)")
    parser.add_argument("--dim", type=int, default=1, help="Latent dimension")
    parser.add_argument("--kappa-scale", type=float, default=0.5,
                        help="Kappa scale for horseshoe prior")
    parser.add_argument("--eta-scale", type=float, default=0.1,
                        help="Eta scale for horseshoe prior (local shrinkage)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--lr-decay-factor", type=float, default=0.975,
                        help="LR decay factor on plateau (default 0.975)")
    parser.add_argument("--clip-norm", type=float, default=1.0,
                        help="Gradient clipping norm (default 1.0)")
    parser.add_argument("--reload-neural-grm", action="store_true",
                        help="Reload saved NeuralGRM instead of re-training")
    parser.add_argument("--noisy-dim", action="store_true",
                        help="Add a loosely-coupled noisy second latent dimension")
    parser.add_argument("--noisy-dim-eta-scale", type=float, default=0.01,
                        help="Discrimination scale for noisy dimension (default 0.01)")
    parser.add_argument("--noisy-dim-ability-scale", type=float, default=2.0,
                        help="Ability prior scale for noisy dimension (default 2.0)")
    parser.add_argument("--sample-size", type=int, default=64,
                        help="MC samples per ADVI gradient step (default 64)")
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
            missing_respondent_frac=args.missing_respondent_frac,
            dim=args.dim,
            kappa_scale=args.kappa_scale,
            eta_scale=args.eta_scale,
            patience=args.patience,
            lr_decay_factor=args.lr_decay_factor,
            clip_norm=args.clip_norm,
            reload_neural_grm=args.reload_neural_grm,
            noisy_dim=args.noisy_dim,
            noisy_dim_eta_scale=args.noisy_dim_eta_scale,
            noisy_dim_ability_scale=args.noisy_dim_ability_scale,
            sample_size=args.sample_size,
        )
        all_results[ds] = results

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*100}")
        print("SUMMARY")
        print(f"{'='*100}")
        header = (f"{'Dataset':<10} {'Base ρ':>10} {'MICE ρ':>10} {'Mixed ρ':>10} "
                  f"{'Base RMSE':>10} {'MICE RMSE':>10} {'Mixed RMSE':>10}")
        print(header)
        print("-" * 100)
        for ds, r in all_results.items():
            line = (f"{ds:<10} {r['baseline']['spearman_r']:>10.4f} "
                    f"{r['mice_only']['spearman_r']:>10.4f} "
                    f"{r['imputed']['spearman_r']:>10.4f} "
                    f"{r['baseline']['rmse']:>10.4f} "
                    f"{r['mice_only']['rmse']:>10.4f} "
                    f"{r['imputed']['rmse']:>10.4f}")
            print(line)

        # Save combined results
        output_dir = Path(args.output_dir)
        with open(output_dir / 'all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()

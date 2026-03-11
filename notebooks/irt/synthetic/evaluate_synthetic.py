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
import os
import sys
from pathlib import Path

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import pandas as pd


def run_pipeline(dataset_name, output_dir, epochs=500, lr=1e-3, grm_lr=None,
                 neural_lr=None, neural_batch_size=None,
                 batch_size=256, missingness_rate=0.25,
                 missing_respondent_frac=0.4,
                 dim=1, kappa_scale=0.5, eta_scale=0.1, patience=10,
                 lr_decay_factor=0.9, clip_norm=1.0,
                 reload_neural_grm=False,
                 noisy_dim=True,
                 noisy_dim_eta_scale=0.1,
                 noisy_dim_ability_scale=1.0,
                 sample_size=32,
                 seed=42,
                 parameterization="log_scale",
                 pathfinder_init=False,
                 qmc=False,
                 kl_anneal_epochs=0,
                 compute_elpd_loo=False,
                 nn_hidden_sizes=4,
                 nn_prior_scale=0.5,
                 discrimination_prior="horseshoe",
                 discrimination_prior_scale=None):
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
    # NeuralGRM defaults: smaller LR and larger batches for stability
    if neural_lr is None:
        neural_lr = lr * 0.5  # half the base LR
    if neural_batch_size is None:
        neural_batch_size = max(batch_size, 512)

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
        batch_size=neural_batch_size,
        num_epochs=epochs,
        learning_rate=neural_lr,
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
        seed=seed,
        parameterization=parameterization,
        pathfinder_init=pathfinder_init,
        qmc=qmc,
        kl_anneal_epochs=kl_anneal_epochs,
        nn_hidden_sizes=nn_hidden_sizes,
        nn_prior_scale=nn_prior_scale,
    )

    # 3. Use the neural model's posterior abilities as ground truth.
    #    The ICCs are calibrated to this ability range; using N(0,1) would
    #    produce floor/ceiling effects outside the model's operating range.
    print(f"\n--- Extracting ground-truth abilities from neural model posterior ---")
    cal_abilities = np.array(neural_model.calibrated_expectations['abilities'])
    # Extract only the primary dimension(s) for ground truth; simulate_data
    # will re-append a noisy dimension if noisy_dim=True.
    if noisy_dim and cal_abilities.shape[1] > dim:
        true_abilities = cal_abilities[:, :dim, :, :]
        print(f"  Extracted primary dim(s) from {cal_abilities.shape} -> {true_abilities.shape}")
    else:
        true_abilities = cal_abilities
    print(f"  True abilities shape: {true_abilities.shape}")
    print(f"  Range: [{true_abilities.min():.3f}, {true_abilities.max():.3f}]")
    print(f"  Std: {true_abilities.std():.4f}")

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
        seed=seed, parameterization=parameterization,
        pathfinder_init=pathfinder_init,
        qmc=qmc, kl_anneal_epochs=kl_anneal_epochs,
        compute_elpd_loo=compute_elpd_loo,
        discrimination_prior=discrimination_prior,
        discrimination_prior_scale=discrimination_prior_scale,
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
        seed=seed + 1, parameterization=parameterization,
        pathfinder_init=pathfinder_init,
        qmc=qmc, kl_anneal_epochs=kl_anneal_epochs,
        compute_elpd_loo=compute_elpd_loo,
        discrimination_prior=discrimination_prior,
        discrimination_prior_scale=discrimination_prior_scale,
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
        seed=seed + 2, parameterization=parameterization,
        pathfinder_init=pathfinder_init,
        qmc=qmc, kl_anneal_epochs=kl_anneal_epochs,
        compute_elpd_loo=compute_elpd_loo,
        discrimination_prior=discrimination_prior,
        discrimination_prior_scale=discrimination_prior_scale,
    )

    # 11. Compare ability ordering preservation (only on fully observed respondents)
    print(f"\n--- Comparing ability orderings (fully observed respondents only) ---")
    baseline_abilities = np.array(baseline_model.calibrated_expectations['abilities'])
    mice_only_abilities = np.array(mice_only_model.calibrated_expectations['abilities'])
    imputed_abilities = np.array(imputed_model.calibrated_expectations['abilities'])

    # true_abilities shape: (N, 1, 1, 1) — single ground-truth dimension
    # estimated shape: (N, D, 1, 1) where D=2 if noisy_dim
    true_flat = true_abilities[:, 0, 0, 0]  # (N,)

    # Only evaluate on fully observed respondents for fair comparison
    obs_mask = ~has_missing  # (N,) boolean — True for fully observed
    true_obs = true_flat[obs_mask]
    print(f"  Evaluating on {int(obs_mask.sum())}/{num_people} fully observed respondents")

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

    # Primary dimension (dim 0) comparisons — fully observed only
    baseline_metrics = compare_ability_ordering(
        true_obs, _extract_dim(baseline_abilities, 0)[obs_mask])
    mice_only_metrics = compare_ability_ordering(
        true_obs, _extract_dim(mice_only_abilities, 0)[obs_mask])
    imputed_metrics = compare_ability_ordering(
        true_obs, _extract_dim(imputed_abilities, 0)[obs_mask])

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
            'nn_hidden_sizes': nn_hidden_sizes,
            'nn_prior_scale': nn_prior_scale,
        },
    }

    # ELPD-LOO comparison (if computed)
    if compute_elpd_loo:
        results['elpd_loo'] = {}
        for label, model in [('baseline', baseline_model),
                              ('mice_only', mice_only_model),
                              ('imputed', imputed_model)]:
            if hasattr(model, 'elpd_loo') and model.elpd_loo is not None:
                results['elpd_loo'][label] = {
                    'elpd_loo': model.elpd_loo,
                    'elpd_loo_se': model.elpd_loo_se,
                    'elpd_loo_per_obs': model.elpd_loo_per_obs,
                    'n_obs': getattr(model, 'elpd_loo_n_obs', None),
                    'max_khat': float(np.max(model.elpd_loo_khat)),
                    'mean_khat': float(np.mean(model.elpd_loo_khat)),
                }

    # Noisy dimension and norm comparisons (when noisy_dim is used)
    if noisy_dim:
        for label, abil in [('baseline', baseline_abilities),
                            ('mice_only', mice_only_abilities),
                            ('imputed', imputed_abilities)]:
            dim1 = _extract_dim(abil, 1)
            norm_abil = _extract_norm(abil)
            if dim1 is not None:
                results[f'{label}_dim1'] = compare_ability_ordering(
                    true_obs, dim1[obs_mask])
            results[f'{label}_norm'] = compare_ability_ordering(
                true_obs, norm_abil[obs_mask])

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

    # ELPD-LOO summary
    if compute_elpd_loo and 'elpd_loo' in results:
        print(f"\n  ELPD-LOO comparison:")
        for label in ['baseline', 'mice_only', 'imputed']:
            loo = results['elpd_loo'].get(label)
            if loo:
                print(f"  {label:<12} ELPD-LOO = {loo['elpd_loo']:.2f} "
                      f"(SE: {loo['elpd_loo_se']:.2f}), "
                      f"per obs = {loo['elpd_loo_per_obs']:.4f}, "
                      f"max k-hat = {loo['max_khat']:.3f}")

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
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (per-datum ELBO, default 1e-3)")
    parser.add_argument("--grm-lr", type=float, default=None,
                        help="GRM learning rate (defaults to --lr if not set)")
    parser.add_argument("--neural-lr", type=float, default=None,
                        help="NeuralGRM learning rate (defaults to lr*0.5)")
    parser.add_argument("--neural-batch-size", type=int, default=None,
                        help="NeuralGRM batch size (defaults to max(batch_size, 512))")
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
    parser.add_argument("--noisy-dim", action="store_true", default=True,
                        help="Add a loosely-coupled noisy second latent dimension")
    parser.add_argument("--no-noisy-dim", action="store_false", dest="noisy_dim",
                        help="Disable the noisy second latent dimension")
    parser.add_argument("--noisy-dim-eta-scale", type=float, default=0.1,
                        help="Discrimination scale for noisy dimension (default 0.1)")
    parser.add_argument("--noisy-dim-ability-scale", type=float, default=1.0,
                        help="Ability prior scale for noisy dimension (default 1.0)")
    parser.add_argument("--sample-size", type=int, default=32,
                        help="MC samples per ADVI gradient step (default 32)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible ADVI training (default 42)")
    parser.add_argument("--parameterization", default="log_scale",
                        choices=["softplus", "log_scale", "natural"],
                        help="ADVI scale parameterization (default log_scale)")
    parser.add_argument("--pathfinder-init", action="store_true",
                        help="Use Pathfinder to initialize ADVI parameters")
    parser.add_argument("--qmc", action="store_true",
                        help="Use quasi-Monte Carlo (Sobol) sampling for ~2x variance reduction")
    parser.add_argument("--kl-anneal-epochs", type=int, default=0,
                        help="Number of epochs to linearly ramp KL weight from 0 to 1 (default 0)")
    parser.add_argument("--elpd-loo", action="store_true",
                        help="Compute PSIS-LOO ELPD for each fitted GRM model")
    parser.add_argument("--nn-hidden-sizes", type=int, default=4,
                        help="Number of mixture-of-logits components in NeuralGRM (default 4)")
    parser.add_argument("--nn-prior-scale", type=float, default=0.5,
                        help="Prior scale for NN params (smaller = more regularization, default 0.5)")
    parser.add_argument("--discrimination-prior", default="horseshoe",
                        choices=["horseshoe", "half_normal", "half_cauchy"],
                        help="Discrimination prior type (default horseshoe)")
    parser.add_argument("--discrimination-prior-scale", type=float, default=None,
                        help="Scale for half_normal/half_cauchy discrimination prior")
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
            neural_lr=args.neural_lr,
            neural_batch_size=args.neural_batch_size,
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
            seed=args.seed,
            parameterization=args.parameterization,
            pathfinder_init=args.pathfinder_init,
            qmc=args.qmc,
            kl_anneal_epochs=args.kl_anneal_epochs,
            compute_elpd_loo=args.elpd_loo,
            nn_hidden_sizes=args.nn_hidden_sizes,
            nn_prior_scale=args.nn_prior_scale,
            discrimination_prior=args.discrimination_prior,
            discrimination_prior_scale=args.discrimination_prior_scale,
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

        # ELPD-LOO summary table (if computed)
        has_elpd = any('elpd_loo' in r for r in all_results.values())
        if has_elpd:
            print(f"\n{'='*100}")
            print("ELPD-LOO COMPARISON")
            print(f"{'='*100}")
            header = (f"{'Dataset':<10} {'Base ELPD':>14} {'MICE ELPD':>14} "
                      f"{'Mixed ELPD':>14} {'Base k-hat':>10} "
                      f"{'MICE k-hat':>10} {'Mixed k-hat':>10}")
            print(header)
            print("-" * 100)
            for ds, r in all_results.items():
                loo = r.get('elpd_loo', {})
                def _elpd_str(label):
                    e = loo.get(label)
                    if e:
                        return f"{e['elpd_loo_per_obs']:>14.4f}"
                    return f"{'N/A':>14}"
                def _khat_str(label):
                    e = loo.get(label)
                    if e:
                        return f"{e['max_khat']:>10.3f}"
                    return f"{'N/A':>10}"
                print(f"{ds:<10} {_elpd_str('baseline')} {_elpd_str('mice_only')} "
                      f"{_elpd_str('imputed')} {_khat_str('baseline')} "
                      f"{_khat_str('mice_only')} {_khat_str('imputed')}")

        # Save combined results
        output_dir = Path(args.output_dir)
        with open(output_dir / 'all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()

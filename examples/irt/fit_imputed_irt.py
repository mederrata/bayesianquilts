#!/usr/bin/env python
"""Manuscript imputed IRT pipeline: three-way Yao + optional IPW + gofluttercat export.

Self-contained mirror of ``notebooks/irt/run_marginal_mcmc.py``'s
``imputed`` variant (the one used in the M-open subset bias paper):

  1. Load dataset (with optional sum-score-stratified IPW)
  2. Fit ``PairwiseOrdinalStackingModel`` (MICE imputation)
  3. Fit joint-ADVI baseline GRM (IRT-baseline component for the stack)
  4. Standardize baseline -> N(0,1) abilities
  5. Fit shared-discrimination GRM via marginal MCMC (third stacking
     component; skipped with ``--skip-shared-disc`` for a 2-way stack)
  6. Build ``ThreeWayImputationModel`` (per-item simplex over MICE,
     IRT-baseline, shared-disc)
  7. Marginal MCMC on the baseline GRM with three-way PMFs attached
  8. Standardize MCMC -> N(0,1), compute EAP
  9. (Optional) BCM training: ``BCMConditional`` joblib + per-J isotonic
     ``BCMSet`` JSON on top of the same (subset, gold) triples
 10. Export ``converged/`` bundle via ``bayesianquilts.io.converged.
     export_artifact`` so libfab and gofluttercat can consume the
     fitted IRT + imputation + BCM artifacts directly.

Usage:
    uv run python fit_imputed_irt.py --dataset gcbs
    uv run python fit_imputed_irt.py --dataset eqsq --use-ipw --dense-mass
    uv run python fit_imputed_irt.py --dataset scs --skip-shared-disc
"""
import argparse
import gc
import importlib
import inspect
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')

from pathlib import Path
from typing import Dict

import numpy as np
import jax
import jax.numpy as jnp


DATASET_CONFIGS = {
    'scs': {'module': 'bayesianquilts.data.scs', 'n_top_features': 10},
    'gcbs': {'module': 'bayesianquilts.data.gcbs', 'n_top_features': 15},
    'grit': {'module': 'bayesianquilts.data.grit', 'n_top_features': 12},
    'rwa': {'module': 'bayesianquilts.data.rwa', 'n_top_features': 22},
    'npi': {'module': 'bayesianquilts.data.npi', 'n_top_features': 40},
    'tma': {'module': 'bayesianquilts.data.tma', 'n_top_features': 14},
    'wpi': {'module': 'bayesianquilts.data.wpi', 'n_top_features': 20},
    'eqsq': {'module': 'bayesianquilts.data.eqsq', 'n_top_features': 30},
}


def make_data_dict(df, num_people):
    data = {}
    for col in df.columns:
        data[col] = df[col].to_numpy().astype(np.float32)
    data['person'] = np.arange(num_people, dtype=np.float32)
    return data


def compute_ipw_weights(pandas_df, n_groups=3):
    """Sum-score quantile-stratified IPW weights, normalized to sum=N."""
    total = pandas_df.sum(axis=1, skipna=True).values
    valid = ~np.isnan(total)
    quantiles = np.quantile(total[valid],
                            np.linspace(0, 1, n_groups + 1)[1:-1])
    groups = np.digitize(total, bins=quantiles)
    counts = np.bincount(groups, minlength=n_groups)
    w = np.array([1.0 / max(counts[g], 1) for g in groups], dtype=np.float32)
    w *= len(w) / w.sum()
    ess = 1.0 / np.sum((w / w.sum()) ** 2)
    return w, groups, ess


def calibrate_model(model, seed=101, n_samples=32):
    surrogate = model.surrogate_distribution_generator(model.params)
    samples = surrogate.sample(n_samples, seed=jax.random.PRNGKey(seed))
    model.surrogate_sample = samples
    model.calibrated_expectations = {
        k: jnp.mean(v, axis=0) for k, v in samples.items()
    }


def _jsonify_weights(value):
    """Coerce numpy weight arrays to plain Python lists for YAML/JSON."""
    if isinstance(value, np.ndarray):
        return [float(x) for x in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def main():
    parser = argparse.ArgumentParser(
        description='Manuscript imputed IRT pipeline: three-way Yao '
                    '(pairwise + IRT-baseline + shared-disc) with optional '
                    'IPW and converged-bundle export.')
    parser.add_argument('--dataset', default='gcbs',
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--use-ipw', action='store_true',
                        help='Apply sum-score quantile-stratified IPW weights')
    parser.add_argument('--advi-epochs', type=int, default=2000)
    parser.add_argument('--learning-rate', type=float, default=0.005,
                        help='Joint-ADVI learning rate (lower if NaNs)')

    parser.add_argument('--skip-shared-disc', action='store_true',
                        help='Skip the shared-disc third component '
                             '(2-way stack instead of three-way)')
    parser.add_argument('--shared-disc-num-warmup', type=int, default=2000)
    parser.add_argument('--shared-disc-num-samples', type=int, default=500)
    parser.add_argument('--shared-disc-num-chains', type=int, default=2)
    parser.add_argument('--shared-disc-step-size', type=float, default=0.005)

    parser.add_argument('--num-chains', type=int, default=4)
    parser.add_argument('--num-warmup', type=int, default=2000)
    parser.add_argument('--num-samples', type=int, default=2000)
    parser.add_argument('--step-size', type=float, default=0.005)
    parser.add_argument('--target-accept', type=float, default=0.85,
                        help='NUTS target acceptance; lower (0.7) for '
                             'stiff posteriors that hang in warmup')
    parser.add_argument('--dense-mass', action='store_true',
                        help='Use dense mass matrix during warmup')

    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip the baseline (no-imputation) MCMC '
                             'fit; report metrics for imputed variant only')
    parser.add_argument('--skip-bcm', action='store_true')
    parser.add_argument('--bcm-subset-sizes', type=int, nargs='+',
                        default=[5, 10, 20, 40])
    parser.add_argument('--bcm-n-subsets', type=int, default=100)
    parser.add_argument('--bcm-max-respondents', type=int, default=200)

    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    from bayesianquilts.irt.grm import GRModel
    from bayesianquilts.irt.shared_disc_grm import SharedDiscGRModel
    from bayesianquilts.imputation.pairwise_stacking import (
        PairwiseOrdinalStackingModel,
    )
    from bayesianquilts.imputation.three_way import ThreeWayImputationModel
    from bayesianquilts.io.converged import export_artifact

    config = DATASET_CONFIGS[args.dataset]
    mod = importlib.import_module(config['module'])
    item_keys = mod.item_keys
    response_cardinality = mod.response_cardinality

    output_dir = Path(args.output_dir or args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Manuscript imputed IRT: {args.dataset.upper()}")
    print(f"  Items: {len(item_keys)}, K: {response_cardinality}, "
          f"IPW: {args.use_ipw}, shared-disc: {not args.skip_shared_disc}")
    print(f"{'='*60}")

    # ---- Load data ----
    import pandas as pd
    kw = {'polars_out': True}
    if 'reorient' in inspect.signature(mod.get_data).parameters:
        kw['reorient'] = True
    df, num_people = mod.get_data(**kw)
    pandas_df = pd.DataFrame({
        k: np.where(df[k].to_numpy() == -1, np.nan,
                    df[k].to_numpy().astype(float))
        for k in item_keys
    })
    base_data = make_data_dict(df, num_people)
    print(f"  Respondents: {num_people}")

    if args.use_ipw:
        weights, groups, ess = compute_ipw_weights(pandas_df)
        base_data['sample_weights'] = weights
        print(f"  IPW: {len(set(groups))} groups, ESS: {ess:.0f}")

    # ---- Step 1: pairwise stacking imputation ----
    print(f"\n{'─'*60}\nStep 1: pairwise stacking imputation\n{'─'*60}")
    stacking_path = output_dir / 'pairwise_stacking_model.yaml'
    if stacking_path.exists():
        print(f"  loading from {stacking_path}")
        pairwise_model = PairwiseOrdinalStackingModel.load(str(stacking_path))
    else:
        pairwise_model = PairwiseOrdinalStackingModel(
            prior_scale=1.0, pathfinder_num_samples=100,
            pathfinder_maxiter=50, batch_size=512, verbose=True,
        )
        pairwise_model.fit(pandas_df, n_top_features=config['n_top_features'],
                           n_jobs=1, seed=args.seed)
        pairwise_model.save(str(stacking_path))
        print(f"  saved to {stacking_path}")

    # ---- Step 2: joint-ADVI baseline GRM (IRT-baseline component) ----
    print(f"\n{'─'*60}\nStep 2: joint-ADVI baseline GRM\n{'─'*60}")
    baseline_path = output_dir / 'grm_baseline'
    if (baseline_path / 'params.h5').exists():
        print(f"  loading from {baseline_path}")
        baseline_model = GRModel.load_from_disk(str(baseline_path))
    else:
        baseline_model = GRModel(
            item_keys=item_keys, num_people=num_people,
            response_cardinality=response_cardinality, dim=1,
            dtype=jnp.float64,
        )

        def factory():
            yield base_data

        baseline_model.fit(factory, dataset_size=num_people,
                           batch_size=num_people, num_epochs=args.advi_epochs,
                           learning_rate=args.learning_rate)
        baseline_model.save_to_disk(str(baseline_path))
        print(f"  saved to {baseline_path}")
    calibrate_model(baseline_model, seed=args.seed + 1)

    # Standardize baseline -> N(0,1) before wiring it into the three-way stack
    base_std = baseline_model.standardize_abilities()
    print(f"  standardized baseline: "
          f"mu={float(jnp.mean(base_std['mu'])):.4f}, "
          f"sigma={float(jnp.mean(base_std['sigma'])):.4f}")

    # ---- Step 3: shared-disc GRM (third stacking component) ----
    shared_disc_model = None
    if not args.skip_shared_disc:
        print(f"\n{'─'*60}\nStep 3: shared-disc GRM via marginal MCMC\n{'─'*60}")
        shared_disc_model = SharedDiscGRModel(
            item_keys=item_keys, num_people=num_people, dim=1,
            response_cardinality=response_cardinality, dtype=jnp.float64,
        )
        shared_disc_model.params = None  # init from prior (small param count)
        sd_mcmc = shared_disc_model.fit_marginal_mcmc(
            base_data,
            num_chains=args.shared_disc_num_chains,
            num_warmup=args.shared_disc_num_warmup,
            num_samples=args.shared_disc_num_samples,
            step_size=args.shared_disc_step_size,
            seed=args.seed + 13,
            verbose=True,
        )
        shared_disc_model.mcmc_samples = sd_mcmc
        # Standardize shared-disc abilities so all three components produce
        # PMFs on the same N(0,1) scale.
        shared_disc_model.standardize_marginal(base_data)
        # Manually populate surrogate_sample (shared-disc fit skipped ADVI
        # so there's no model.params to fit_surrogate_to_mcmc against).
        # The ThreeWayImputationModel reads surrogate_sample to compute
        # PMFs; flatten chains+samples and inject EAP for abilities.
        sd_eap = shared_disc_model.compute_eap_abilities(base_data)
        surrogate_sample = {
            k: jnp.asarray(np.asarray(v).reshape(-1, *v.shape[2:]))
            for k, v in shared_disc_model.mcmc_samples.items()
        }
        surrogate_sample['abilities'] = jnp.asarray(
            np.asarray(sd_eap['eap'])
            [np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        )
        shared_disc_model.surrogate_sample = surrogate_sample

    # ---- Step 4: three-way (or two-way) imputation ----
    print(f"\n{'─'*60}\nStep 4: ThreeWayImputationModel\n{'─'*60}")

    def make_factory():
        def factory():
            yield base_data
        return factory

    three_way = ThreeWayImputationModel(
        irt_model=baseline_model,
        shared_disc_model=shared_disc_model,
        mice_model=pairwise_model,
        data_factory=make_factory(),
    )

    # ---- Step 4b: baseline marginal MCMC (no imputation) for comparison ----
    baseline_npz_path = None
    if not args.skip_baseline:
        print(f"\n{'─'*60}\nStep 4b: baseline marginal MCMC (no imputation)\n{'─'*60}")
        baseline_mcmc_model = GRModel.load_from_disk(str(baseline_path))
        b_mcmc = baseline_mcmc_model.fit_marginal_mcmc(
            base_data,
            num_chains=args.num_chains,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            target_accept_prob=args.target_accept,
            step_size=args.step_size,
            dense_mass=args.dense_mass,
            seed=args.seed + 11,
            verbose=True,
        )
        baseline_mcmc_model.mcmc_samples = b_mcmc
        baseline_mcmc_model.standardize_marginal(base_data)
        b_eap = baseline_mcmc_model.compute_eap_abilities(base_data)
        b_save = {k: np.asarray(v) for k, v in b_mcmc.items()}
        b_save['eap'] = np.asarray(b_eap['eap'])
        b_save['psd'] = np.asarray(b_eap['psd'])
        baseline_npz_path = output_dir / 'mcmc_baseline.npz'
        np.savez(str(baseline_npz_path), **b_save)
        print(f"  raw MCMC -> {baseline_npz_path}")
        del baseline_mcmc_model, b_mcmc
        gc.collect()

    # ---- Step 5: marginal MCMC on baseline with three-way PMFs ----
    print(f"\n{'─'*60}\nStep 5: marginal MCMC on imputed posterior\n{'─'*60}")
    baseline_model.imputation_model = three_way
    data_with_pmfs = dict(base_data)
    pmfs, weights = baseline_model._compute_batch_pmfs(data_with_pmfs)
    if pmfs is not None:
        data_with_pmfs['_imputation_pmfs'] = pmfs
        if weights is not None:
            data_with_pmfs['_imputation_weights'] = weights
    print("  three-way imputation PMFs attached")

    mcmc_samples = baseline_model.fit_marginal_mcmc(
        data_with_pmfs,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        target_accept_prob=args.target_accept,
        step_size=args.step_size,
        dense_mass=args.dense_mass,
        seed=args.seed + 17,
        verbose=True,
    )
    baseline_model.mcmc_samples = mcmc_samples
    stats = baseline_model.standardize_marginal(data_with_pmfs)
    baseline_model.fit_surrogate_to_mcmc()
    calibrate_model(baseline_model, seed=args.seed + 23)
    eap_result = baseline_model.compute_eap_abilities(data_with_pmfs)
    print(f"  Post-std EAP: mu={float(jnp.mean(eap_result['eap'])):.4f}, "
          f"sigma={float(jnp.std(eap_result['eap'])):.4f}")

    save_dict = {k: np.asarray(v) for k, v in mcmc_samples.items()}
    save_dict['eap'] = np.asarray(eap_result['eap'])
    save_dict['psd'] = np.asarray(eap_result['psd'])
    save_dict['standardize_mu'] = stats['mu']
    save_dict['standardize_sigma'] = stats['sigma']
    npz_path = output_dir / 'mcmc_imputed.npz'
    np.savez(str(npz_path), **save_dict)
    print(f"  raw MCMC -> {npz_path}")

    # ---- Manuscript metrics: PSIS-LOO RMSE + ELPD per variant ----
    print(f"\n{'─'*60}\nManuscript metrics (PSIS-LOO)\n{'─'*60}")
    from _eval_metrics import compute_metrics_from_npz, print_metrics_table
    metrics: Dict[str, Dict[str, float]] = {}
    if baseline_npz_path is not None:
        print(f"  computing baseline metrics...")
        metrics['baseline'] = compute_metrics_from_npz(
            baseline_npz_path, base_data, item_keys, response_cardinality,
            num_people)
    print(f"  computing imputed metrics...")
    metrics['imputed'] = compute_metrics_from_npz(
        npz_path, base_data, item_keys, response_cardinality, num_people)
    print_metrics_table(metrics)

    # ---- Step 6+7: BCM (optional) + converged bundle ----
    bundle_dir = output_dir / 'converged'
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_bcm:
        print(f"\n{'─'*60}\nStep 6: BCM training\n{'─'*60}")
        from _bcm_triples import (
            build_bcm_triples, extract_item_params_from_mcmc,
        )
        from libfabulouscatpy.biascorrection import (
            BCMConditional, fit_bcm_set,
        )

        bcm_item_params = extract_item_params_from_mcmc(baseline_model)
        rng_bcm = np.random.default_rng(args.seed + 7)
        subset_scores, indicators_mat, golds, _ = build_bcm_triples(
            model=baseline_model,
            base_data=base_data,
            item_keys=item_keys,
            subset_sizes=args.bcm_subset_sizes,
            n_subsets_per_size=args.bcm_n_subsets,
            max_respondents=args.bcm_max_respondents,
            rng=rng_bcm,
            item_params=bcm_item_params,
        )
        print(f"  n_triples = {subset_scores.size}")

        bcm_cond = BCMConditional.fit(
            subset_scores, indicators_mat, golds,
            item_keys=item_keys, scale_name=args.dataset,
            n_folds=5, seed=args.seed,
            max_iter=200, learning_rate=0.05, max_depth=4,
        )
        l2_naive = float(np.sqrt(np.mean((subset_scores - golds) ** 2)))
        l2_bcm = float(np.sqrt(np.mean(
            (bcm_cond.oof_predictions - golds) ** 2)))
        print(f"  L2(subset - gold) = {l2_naive:.4f}; "
              f"L2(BCMcond - gold) = {l2_bcm:.4f}")

        bcm_cond_path = bundle_dir / f'bcm_{args.dataset}_conditional.joblib'
        bcm_cond.save(str(bcm_cond_path))
        print(f"  BCMConditional -> {bcm_cond_path}")

        js = indicators_mat.sum(axis=1).astype(int)
        cells = {}
        for j in np.unique(js):
            if j < 1:
                continue
            mask = js == j
            if mask.sum() < 2:
                continue
            cells[int(j)] = (subset_scores[mask], golds[mask])
        bcm_set = fit_bcm_set(cells, scale=args.dataset)
        bcm_set_path = bundle_dir / f'bcm_{args.dataset}.json'
        bcm_set.save(str(bcm_set_path))
        print(f"  BCMSet -> {bcm_set_path}")

    print(f"\n{'─'*60}\nStep {'7' if args.skip_bcm else '8'}: converged bundle\n{'─'*60}")
    three_way_weights = {str(k): _jsonify_weights(v)
                         for k, v in three_way._weights.items()}
    export_artifact(
        irt_model=baseline_model,
        imputation_model=pairwise_model,
        out_dir=str(bundle_dir),
        scale_names=[args.dataset],
        fit_method=('marginal_mcmc_three_way' if shared_disc_model
                    else 'marginal_mcmc_two_way'),
        source_script=os.path.basename(__file__),
        extra_manifest={
            'dataset': args.dataset,
            'pipeline': 'fit_imputed_irt.py',
            'use_ipw': args.use_ipw,
            'shared_disc': shared_disc_model is not None,
            'standardized': True,
            'three_way_weights': three_way_weights,
            'mcmc': {
                'chains': args.num_chains,
                'warmup': args.num_warmup,
                'samples': args.num_samples,
                'step_size': args.step_size,
                'target_accept': args.target_accept,
                'dense_mass': args.dense_mass,
            },
        },
    )
    print(f"  -> {bundle_dir}")
    print(f"\nDone. Artifacts in {output_dir}/")


if __name__ == '__main__':
    main()

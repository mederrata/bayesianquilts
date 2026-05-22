#!/usr/bin/env python
"""End-to-end: fit imputation, fit baseline IRT, fit BCM with imputation-blended scoring.

This is the only example in the directory that chains all four pieces of
the M-open scoring pipeline together:

  1. Fit ``PairwiseOrdinalStackingModel`` on the response matrix.
  2. Fit a baseline ``GRModel`` (GRM) via marginal ADVI.
  3. Build an ``IrtMixedImputationModel`` from (pairwise stacking + baseline GRM)
     and attach it to the fitted GRM.
  4. Build (subset_score, item_indicator, gold_score) training triples by
     repeatedly masking random subsets of items to "missing" and recomputing
     the EAP score under the imputation-blended likelihood. Fit
     ``BCMConditional`` (gradient-boosted regressor with item-indicator
     features) on the resulting triples.

At scoring time, the fitted BCM is applied to any new (subset_score,
item_indicator) pair to produce a bias-corrected score that closes the
residual gap between the imputed subset EAP and the imputed full-bank EAP.

Companion examples in libfabulouscatpy fit the BCM on top of pre-fitted GRM
parameters and use *naive* (no-imputation) scoring for both the subset and
gold scores; this script demonstrates the alternative regime in which the
underlying scoring model itself uses imputation.

Usage:
    uv run python fit_bcm_with_imputation.py --dataset gcbs
    uv run python fit_bcm_with_imputation.py --dataset scs --n-subsets 200
    uv run python fit_bcm_with_imputation.py --dataset eqsq --max-respondents 500
"""

import argparse
import gc
import inspect
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')

from pathlib import Path

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


def make_data_dict(dataframe, num_people):
    data = {}
    for col in dataframe.columns:
        data[col] = dataframe[col].to_numpy().astype(np.float32)
    data['person'] = np.arange(num_people, dtype=np.float32)
    return data


def calibrate_model(model, seed=101, n_samples=32):
    surrogate = model.surrogate_distribution_generator(model.params)
    key = jax.random.PRNGKey(seed)
    samples = surrogate.sample(n_samples, seed=key)
    model.surrogate_sample = samples
    model.calibrated_expectations = {
        k: jnp.mean(v, axis=0) for k, v in samples.items()
    }


def attach_imputation_pmfs(model, data):
    """Compute imputation PMFs for the (possibly masked) data and attach
    them under the keys ``_imputation_pmfs`` and ``_imputation_weights`` so
    that ``compute_eap_abilities`` integrates over them for missing items."""
    out = dict(data)
    pmfs, weights = model._compute_batch_pmfs(out)
    if pmfs is not None:
        out['_imputation_pmfs'] = pmfs
        if weights is not None:
            out['_imputation_weights'] = weights
    return out


def score_subset(model, base_data, item_keys, subset_mask, item_params):
    """Return EAP scores for every respondent given an item-subset mask.

    Non-subset items are set to -1 (missing) so the imputation model attached
    to ``model`` fills them in via Rao-Blackwellized PMFs.
    """
    masked = {}
    for k in item_keys:
        arr = np.asarray(base_data[k], dtype=np.float32).copy()
        if k not in subset_mask:
            arr[:] = -1.0
        masked[k] = arr
    masked['person'] = base_data['person']
    masked = attach_imputation_pmfs(model, masked)
    eap = model.compute_eap_abilities(masked, item_params=item_params)
    return np.asarray(eap['eap'])


def extract_item_params(model):
    """Pull mean item parameters from the joint-ADVI surrogate.

    ``_item_var_list()`` excludes ``abilities`` and ``mu`` (the latter is a
    location shift that ``compute_eap_abilities`` absorbs via the data, not a
    feature of the GRM kernel itself).
    """
    keys = model._item_var_list()
    return {k: model.calibrated_expectations[k] for k in keys
            if k in model.calibrated_expectations}


def stratify_respondents(base_data, item_keys, max_respondents, rng):
    """Pick a sum-score-stratified sample of respondents for tutorial speed."""
    n = int(base_data['person'].shape[0])
    if n <= max_respondents:
        return np.arange(n)
    sum_score = np.zeros(n, dtype=np.float64)
    for k in item_keys:
        arr = np.asarray(base_data[k], dtype=np.float64)
        sum_score += np.where(arr >= 0, arr, 0.0)
    sum_score += rng.normal(0, 1e-9, sum_score.shape)
    sort_order = np.argsort(sum_score)
    pick = np.linspace(0, n - 1, max_respondents).astype(int)
    return sort_order[pick]


def subsample_data(base_data, item_keys, indices):
    """Return a new data dict with only the selected respondents."""
    sub = {}
    for k in item_keys:
        sub[k] = np.asarray(base_data[k])[indices]
    sub['person'] = np.arange(len(indices), dtype=np.float32)
    if 'sample_weights' in base_data:
        sub['sample_weights'] = np.asarray(base_data['sample_weights'])[indices]
    return sub


def main():
    parser = argparse.ArgumentParser(
        description='End-to-end imputation + IRT + BCM fitting')
    parser.add_argument('--dataset', default='gcbs',
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--subset-sizes', type=int, nargs='+',
                        default=[5, 10],
                        help='Item subset sizes for BCM training')
    parser.add_argument('--n-subsets', type=int, default=100,
                        help='Random subset draws per size')
    parser.add_argument('--max-respondents', type=int, default=200,
                        help='Stratified subsample size for tutorial speed')
    parser.add_argument('--advi-epochs', type=int, default=2000)
    parser.add_argument('--advi-rank', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=0.005,
                        help='Initial ADVI learning rate (lower if NaNs)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='BCMConditional cross-validation folds')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    import importlib
    from bayesianquilts.irt.grm import GRModel
    from bayesianquilts.imputation.pairwise_stacking import (
        PairwiseOrdinalStackingModel,
    )
    from bayesianquilts.imputation.mixed import IrtMixedImputationModel

    try:
        from libfabulouscatpy.biascorrection import BCMConditional
    except ImportError as e:
        sys.exit(
            "libfabulouscatpy not importable. BCMConditional lives in "
            "libfabulouscatpy.biascorrection. Install/symlink "
            "libfabulouscatpy first.\n"
            f"  ({e})"
        )

    config = DATASET_CONFIGS[args.dataset]
    mod = importlib.import_module(config['module'])
    item_keys = mod.item_keys
    response_cardinality = mod.response_cardinality

    output_dir = Path(args.output_dir or args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*60}")
    print(f"BCM with imputation: {args.dataset.upper()}")
    print(f"  Items: {len(item_keys)}, K: {response_cardinality}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    get_data_kwargs = {'polars_out': True}
    if 'reorient' in inspect.signature(mod.get_data).parameters:
        get_data_kwargs['reorient'] = True
    df, num_people = mod.get_data(**get_data_kwargs)
    import pandas as pd
    pandas_df = pd.DataFrame({
        k: np.where(df[k].to_numpy() == -1, np.nan,
                    df[k].to_numpy().astype(float))
        for k in item_keys
    })
    base_data = make_data_dict(df, num_people)
    print(f"  Respondents: {num_people}")

    # ------------------------------------------------------------------
    # Step 1: Pairwise stacking imputation
    # ------------------------------------------------------------------
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
        pairwise_model.fit(pandas_df,
                           n_top_features=config['n_top_features'],
                           n_jobs=1, seed=args.seed)
        pairwise_model.save(str(stacking_path))
        print(f"  saved to {stacking_path}")

    # ------------------------------------------------------------------
    # Step 2: Baseline GRM (ADVI)
    # ------------------------------------------------------------------
    print(f"\n{'─'*60}\nStep 2: baseline GRM via ADVI\n{'─'*60}")
    baseline_path = output_dir / 'grm_baseline'
    if (baseline_path / 'params.h5').exists():
        print(f"  loading from {baseline_path}")
        model = GRModel.load_from_disk(str(baseline_path))
    else:
        model = GRModel(
            item_keys=item_keys, num_people=num_people,
            response_cardinality=response_cardinality, dim=1,
            dtype=jnp.float64,
        )

        def data_factory():
            yield base_data

        model.fit(data_factory, dataset_size=num_people,
                  batch_size=num_people, num_epochs=args.advi_epochs,
                  learning_rate=args.learning_rate)
        model.save_to_disk(str(baseline_path))
        print(f"  saved to {baseline_path}")
    calibrate_model(model, seed=args.seed + 1)

    # ------------------------------------------------------------------
    # Step 3: Build mixed (pairwise + IRT-baseline) imputation, attach
    # ------------------------------------------------------------------
    print(f"\n{'─'*60}\nStep 3: mixed imputation model\n{'─'*60}")

    def mixed_data_factory():
        yield base_data

    mixed_imputation = IrtMixedImputationModel(
        irt_model=model, mice_model=pairwise_model,
        data_factory=mixed_data_factory,
    )
    model.imputation_model = mixed_imputation
    print("  IrtMixedImputationModel attached to baseline GRM")

    # Extract item parameters from the joint-ADVI surrogate; scoring uses
    # these explicitly so we sidestep the marginal-ADVI rebuild (which can
    # trip the GRM ``mu`` location-shift in the prior chain).
    item_params = extract_item_params(model)
    print(f"  using joint-ADVI item params: {list(item_params.keys())}")

    # ------------------------------------------------------------------
    # Step 4: Build BCM training triples on a stratified respondent
    #         subsample. For each random subset draw, mask non-subset
    #         items to -1 and re-score under the imputation-blended
    #         likelihood. Gold = same model, no masking.
    # ------------------------------------------------------------------
    print(f"\n{'─'*60}\nStep 4: BCM training triples\n{'─'*60}")
    chosen_idx = stratify_respondents(
        base_data, item_keys, args.max_respondents, rng)
    sub_data = subsample_data(base_data, item_keys, chosen_idx)
    n_sub = len(chosen_idx)
    print(f"  stratified respondents: {n_sub}")

    print("  computing gold scores (full bank + imputation)...")
    gold = score_subset(model, sub_data, item_keys, set(item_keys),
                        item_params)
    print(f"    gold range: [{gold.min():.3f}, {gold.max():.3f}]")

    print("  computing subset scores per (size, draw)...")
    I = len(item_keys)
    key_to_idx = {k: i for i, k in enumerate(item_keys)}
    subset_scores, indicators, golds = [], [], []
    from math import comb
    for size in args.subset_sizes:
        n_draws = min(args.n_subsets, comb(I, size))
        print(f"    J={size}: {n_draws} draws")
        for d in range(n_draws):
            subset = rng.choice(item_keys, size=size, replace=False)
            subset_set = set(subset)
            scores = score_subset(model, sub_data, item_keys, subset_set,
                                  item_params)
            indicator = np.zeros(I, dtype=np.float32)
            for k in subset:
                indicator[key_to_idx[k]] = 1.0
            for pi in range(n_sub):
                subset_scores.append(scores[pi])
                indicators.append(indicator)
                golds.append(gold[pi])
        gc.collect()

    subset_scores = np.asarray(subset_scores, dtype=np.float64)
    indicators_mat = np.stack(indicators, axis=0).astype(np.float64)
    golds = np.asarray(golds, dtype=np.float64)
    print(f"  n_triples = {subset_scores.size}")

    # ------------------------------------------------------------------
    # Step 5: Fit BCMConditional
    # ------------------------------------------------------------------
    print(f"\n{'─'*60}\nStep 5: BCMConditional (5-fold CV)\n{'─'*60}")
    bcm = BCMConditional.fit(
        subset_scores, indicators_mat, golds,
        item_keys=item_keys, scale_name=args.dataset,
        n_folds=args.n_folds, seed=args.seed,
        max_iter=200, learning_rate=0.05, max_depth=4,
    )
    bcm_path = output_dir / f'bcm_{args.dataset}_imputed.joblib'
    bcm.save(str(bcm_path))
    print(f"  saved to {bcm_path}")

    # ------------------------------------------------------------------
    # Step 6: Bias summary (out-of-fold)
    # ------------------------------------------------------------------
    naive_bias = subset_scores - golds
    bcm_bias = bcm.oof_predictions - golds
    print(f"\n{'─'*60}\nStep 6: bias summary (5-fold held-out)\n{'─'*60}")
    l2_naive = float(np.sqrt(np.mean(naive_bias ** 2)))
    l2_bcm = float(np.sqrt(np.mean(bcm_bias ** 2)))
    reduction = 100.0 * (l2_naive - l2_bcm) / l2_naive if l2_naive > 0 else 0.0
    print(f"  L2(imputed subset - gold) = {l2_naive:.4f}")
    print(f"  L2(BCM(imputed)   - gold) = {l2_bcm:.4f}")
    print(f"  reduction                 = {reduction:.1f}%")
    print(f"  mean bias before          = {naive_bias.mean():+.4f} "
          f"(sd {naive_bias.std():.4f})")
    print(f"  mean bias after           = {bcm_bias.mean():+.4f} "
          f"(sd {bcm_bias.std():.4f})")

    # ------------------------------------------------------------------
    # Scoring-time demo
    # ------------------------------------------------------------------
    print(f"\n{'─'*60}\nScoring-time application on 5 random rows\n{'─'*60}")
    demo_idx = np.random.default_rng(0).choice(subset_scores.size, 5,
                                               replace=False)
    demo_corrected = bcm.apply(subset_scores[demo_idx],
                               indicators_mat[demo_idx])
    print(f"  {'imputed':>9s} {'BCM':>9s} {'gold':>9s}")
    for s, c, g in zip(subset_scores[demo_idx], demo_corrected,
                       golds[demo_idx]):
        print(f"  {s:>9.3f} {c:>9.3f} {g:>9.3f}")

    print(f"\nDone. Artifacts in {output_dir}/")


if __name__ == '__main__':
    main()

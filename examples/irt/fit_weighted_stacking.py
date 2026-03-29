#!/usr/bin/env python
"""Fit a PairwiseOrdinalStackingModel with optional survey weights.

Usage:
    python fit_weighted_stacking.py \
        --data responses.csv \
        --items item_0 item_1 ... item_19 \
        --group-col cohort \
        --group-weights '{"general": 0.97, "clinical": 0.03}' \
        --output-dir stacking_results/
"""

import argparse
import json
import os

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import pandas as pd

from pathlib import Path


def run(
    df, item_keys,
    groups=None, group_weights=None,
    output_dir='stacking_results',
    n_top_features=20, n_jobs=1, seed=42,
    batch_size=512,
):
    from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    responses = df[item_keys].copy()
    responses = responses.replace(-1, np.nan)

    n_people = len(responses)
    n_items = len(item_keys)
    n_missing = responses.isna().sum().sum()
    print(f"Respondents: {n_people}")
    print(f"Items: {n_items}")
    print(f"Missing cells: {n_missing} ({100*n_missing/(n_people*n_items):.1f}%)")

    if groups is not None:
        unique_groups, counts = np.unique(groups, return_counts=True)
        print(f"\nGroups:")
        for g, c in zip(unique_groups, counts):
            target = group_weights.get(g, group_weights.get(str(g), '?'))
            print(f"  {g}: n={c} ({100*c/n_people:.1f}%), target={target}")

        w = np.ones(n_people, dtype=np.float64)
        for g, W_g in group_weights.items():
            g_mask = groups == g
            n_g = g_mask.sum()
            if n_g > 0:
                w[g_mask] = W_g * n_people / n_g
        n_eff = w.sum()**2 / (w**2).sum()
        print(f"  Effective sample size: {n_eff:.0f} (of {n_people})")
        print(f"  Weight range: [{w.min():.3f}, {w.max():.3f}]")

    # Fit
    model = PairwiseOrdinalStackingModel(
        pathfinder_num_samples=100,
        pathfinder_maxiter=50,
        batch_size=batch_size,
        verbose=True,
    )

    model.fit(
        responses,
        n_top_features=n_top_features,
        n_jobs=n_jobs,
        seed=seed,
        groups=groups,
        group_weights=group_weights,
    )

    # Compute optimal stacking weights (Yao et al. 2018)
    print("\nComputing optimal stacking weights...")
    model.compute_optimal_stacking_weights()

    # Save both formats
    model.save(str(out / 'pairwise_stacking_model.yaml'))
    model.save_to_disk(str(out / 'pairwise_stacking_model'))
    print(f"\nModel saved to {out}")

    # Summary
    summary = model.summary()
    print(f"\nRegression marginal: {summary['n_reg_zero_converged']}/{summary['n_reg_zero_total']}")
    print(f"Regression univariate: {summary['n_reg_univariate_converged']}/{summary['n_reg_univariate_total']}")
    print(f"DM marginal: {summary['n_dm_zero_converged']}/{summary['n_dm_zero_total']}")
    print(f"DM pairwise: {summary['n_dm_pairwise_converged']}/{summary['n_dm_pairwise_total']}")

    # Prediction test
    print(f"\nPrediction test:")
    test_items = {item_keys[0]: 2.0}
    for target in item_keys[1:4]:
        try:
            result = model.predict(test_items, target, return_details=True)
            n_models = len(result['weights'])
            print(f"  {item_keys[0]}=2 -> {target}: "
                  f"pred={result['prediction']:.3f}, "
                  f"models={n_models}")
        except Exception as e:
            print(f"  {item_keys[0]}=2 -> {target}: failed ({e})")

    print(f"\nAll artifacts saved to {out}/")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fit PairwiseOrdinalStackingModel with optional survey weights')
    parser.add_argument('--data', required=True, help='CSV with item responses')
    parser.add_argument('--items', nargs='+', required=True, help='Item column names')
    parser.add_argument('--group-col', default=None)
    parser.add_argument('--group-weights', default=None,
                        help='JSON dict, e.g. \'{"general": 0.97, "clinical": 0.03}\'')
    parser.add_argument('--output-dir', default='stacking_results')
    parser.add_argument('--n-top-features', type=int, default=20)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    groups = df[args.group_col].to_numpy() if args.group_col else None
    gw = json.loads(args.group_weights) if args.group_weights else None

    run(
        df, args.items,
        groups=groups, group_weights=gw,
        output_dir=args.output_dir,
        n_top_features=args.n_top_features,
        n_jobs=args.n_jobs, seed=args.seed,
        batch_size=args.batch_size,
    )

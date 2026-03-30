#!/usr/bin/env python
"""Fit a unidimensional GRM with abilities marginalized out.

Demonstrates the marginal inference pipeline for a single-scale GRM:
1. Fit pairwise stacking imputation model
2. Fit item parameters via marginal ADVI (abilities integrated out)
3. Refine item parameters via marginal MCMC (BlackJAX NUTS)
4. Recover EAP abilities from fitted item parameters

Supports optional IPW weights for stratified calibration data.

Usage:
    uv run python fit_marginal_irt.py

The script uses synthetic groups to demonstrate IPW weighting.
To use real survey weights, replace the group construction with
your actual stratification.
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import jax
import jax.numpy as jnp


def main():
    from bayesianquilts.data.scs import (
        get_data, item_keys, response_cardinality
    )
    from bayesianquilts.irt.grm import GRModel
    from bayesianquilts.imputation.pairwise_stacking import (
        PairwiseOrdinalStackingModel
    )

    # ---- Load data ----
    df, num_people = get_data(polars_out=True)
    pandas_df = df.select(item_keys).to_pandas().replace(-1, np.nan)

    data = {}
    for col in df.columns:
        data[col] = df[col].to_numpy().astype(np.float32)
    data['person'] = np.arange(num_people, dtype=np.float32)

    print(f"Dataset: SCS")
    print(f"  Items: {len(item_keys)}, K: {response_cardinality}")
    print(f"  People: {num_people}")

    # ---- Optional: IPW weights ----
    # For demonstration, create synthetic groups based on total score
    use_ipw = True
    if use_ipw:
        total_score = pandas_df.sum(axis=1, skipna=True).values
        groups = np.digitize(
            total_score,
            bins=np.quantile(total_score[~np.isnan(total_score)], [0.33, 0.67])
        )
        # Suppose the population has equal groups but our sample over-represents
        # high scorers: population weights inversely proportional to sample size
        group_counts = np.bincount(groups, minlength=3)
        population_weights = {
            g: 1.0 / max(group_counts[g], 1) for g in range(3)
        }
        sample_weights = np.array(
            [population_weights[g] for g in groups], dtype=np.float32
        )
        # Normalize to effective sample size
        sample_weights *= len(sample_weights) / sample_weights.sum()
        data['sample_weights'] = sample_weights
        print(f"  IPW: {len(set(groups))} groups, "
              f"ESS: {1.0 / np.sum((sample_weights / sample_weights.sum())**2):.0f}")

    # ---- Step 1: Pairwise stacking imputation model ----
    print(f"\n--- Step 1: Pairwise stacking imputation ---")
    imputation_model = PairwiseOrdinalStackingModel(
        prior_scale=1.0,
        pathfinder_num_samples=100,
        pathfinder_maxiter=50,
        batch_size=512,
        verbose=True,
    )
    imputation_model.fit(
        pandas_df,
        n_top_features=len(item_keys),
        n_jobs=1,
        seed=42,
    )

    # ---- Step 2: Create GRM and attach imputation ----
    print(f"\n--- Step 2: Create marginal GRM ---")
    model = GRModel(
        item_keys=item_keys,
        num_people=num_people,
        response_cardinality=response_cardinality,
        dim=1,
        imputation_model=imputation_model,
        dtype=jnp.float64,
    )

    # Compute and attach imputation PMFs
    pmfs, weights = model._compute_batch_pmfs(data)
    if pmfs is not None:
        data['_imputation_pmfs'] = pmfs
        if weights is not None:
            data['_imputation_weights'] = weights
        print(f"  Imputation PMFs attached")

    # ---- Step 3: Marginal ADVI ----
    print(f"\n--- Step 3: Marginal ADVI (rank=0, mean-field) ---")
    losses_mf, params_mf = model.fit_marginal_advi(
        data,
        num_samples=10,
        num_epochs=1000,
        learning_rate=0.01,
        rank=0,
        verbose=True,
    )

    print(f"\n--- Step 3b: Marginal ADVI (rank=2, low-rank) ---")
    losses_lr, params_lr = model.fit_marginal_advi(
        data,
        num_samples=10,
        num_epochs=1000,
        learning_rate=0.01,
        rank=2,
        verbose=True,
    )

    # ---- Step 4: Marginal MCMC ----
    print(f"\n--- Step 4: Marginal MCMC ---")
    mcmc_samples = model.fit_marginal_mcmc(
        data,
        num_chains=2,
        num_warmup=200,
        num_samples=300,
        verbose=True,
    )

    # ---- Step 5: Recover EAP abilities ----
    print(f"\n--- Step 5: EAP abilities ---")
    eap_result = model.compute_eap_abilities(data)
    print(f"  EAP mean: {float(jnp.mean(eap_result['eap'])):.4f}")
    print(f"  EAP std:  {float(jnp.std(eap_result['eap'])):.4f}")
    print(f"  Mean PSD: {float(jnp.mean(eap_result['psd'])):.4f}")

    # ---- Save ----
    os.makedirs('marginal_output', exist_ok=True)
    np.savez(
        'marginal_output/scs_marginal.npz',
        eap=np.array(eap_result['eap']),
        psd=np.array(eap_result['psd']),
        **{f'mcmc_{k}': np.array(v) for k, v in mcmc_samples.items()},
    )
    print(f"\nSaved to marginal_output/scs_marginal.npz")


if __name__ == '__main__':
    main()

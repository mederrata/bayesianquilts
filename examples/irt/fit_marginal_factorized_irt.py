#!/usr/bin/env python
"""Fit a factorized (multi-scale) GRM with abilities marginalized out.

Demonstrates the marginal inference pipeline for a multi-dimensional GRM
where each scale (subscale) has its own ability dimension. Each scale is
fitted independently via fit_dim, which creates a GRModel per scale and
runs marginal MCMC on each.

Pipeline:
1. Fit shared pairwise stacking imputation model
2. For each scale:
   a. Create GRModel for that scale's items
   b. Fit via marginal MCMC (abilities integrated out)
   c. Recover EAP abilities for that dimension
3. Assemble multi-dimensional ability profiles

Usage:
    uv run python fit_marginal_factorized_irt.py
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import jax
import jax.numpy as jnp


def main():
    # Use RWA as example — it has a natural two-factor structure
    # (or we can split items into artificial subscales for any dataset)
    from bayesianquilts.data.scs import (
        get_data, item_keys, response_cardinality
    )
    from bayesianquilts.irt.grm import GRModel
    from bayesianquilts.irt.factorizedgrm import FactorizedGRModel
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

    n_items = len(item_keys)
    print(f"Dataset: SCS")
    print(f"  Items: {n_items}, K: {response_cardinality}")
    print(f"  People: {num_people}")

    # Split into two subscales (first half / second half)
    mid = n_items // 2
    scale_indices = [list(range(mid)), list(range(mid, n_items))]
    print(f"  Scale 0: items {scale_indices[0]}")
    print(f"  Scale 1: items {scale_indices[1]}")

    # ---- Step 1: Pairwise stacking imputation (shared) ----
    print(f"\n--- Step 1: Shared imputation model ---")
    imputation_model = PairwiseOrdinalStackingModel(
        prior_scale=1.0,
        pathfinder_num_samples=100,
        pathfinder_maxiter=50,
        batch_size=512,
        verbose=True,
    )
    imputation_model.fit(
        pandas_df,
        n_top_features=n_items,
        n_jobs=1,
        seed=42,
    )

    # ---- Step 2: Create factorized model ----
    print(f"\n--- Step 2: Create factorized GRM ---")
    factorized_model = FactorizedGRModel(
        scale_indices=scale_indices,
        item_keys=item_keys,
        num_people=num_people,
        response_cardinality=response_cardinality,
        dim=1,
        imputation_model=imputation_model,
        dtype=jnp.float64,
    )

    # ---- Step 3: Fit each scale via marginal MCMC ----
    all_eap = {}
    all_mcmc = {}

    for dim_idx in range(len(scale_indices)):
        print(f"\n--- Step 3.{dim_idx}: Scale {dim_idx} "
              f"({len(scale_indices[dim_idx])} items) ---")

        # Create a batched data factory for this scale
        scale_item_keys = [item_keys[i] for i in scale_indices[dim_idx]]

        def make_data_factory(keys=scale_item_keys):
            def factory():
                yield data
            return factory

        # fit_dim creates a GRModel for this scale
        uni_model, losses, params = factorized_model.fit_dim(
            make_data_factory(),
            dim=dim_idx,
            dataset_size=num_people,
            batch_size=num_people,
            num_epochs=2000,
            learning_rate=0.01,
            imputation_model=imputation_model,
        )

        # Prepare data for this scale's items
        scale_data = {k: data[k] for k in scale_item_keys}
        scale_data['person'] = data['person']
        if 'sample_weights' in data:
            scale_data['sample_weights'] = data['sample_weights']

        # Compute imputation PMFs for the scale model
        pmfs, weights = uni_model._compute_batch_pmfs(scale_data)
        if pmfs is not None:
            scale_data['_imputation_pmfs'] = pmfs
            if weights is not None:
                scale_data['_imputation_weights'] = weights

        # Marginal MCMC on item params
        print(f"  Running marginal MCMC for scale {dim_idx}...")
        mcmc = uni_model.fit_marginal_mcmc(
            scale_data,
            num_chains=2,
            num_warmup=200,
            num_samples=300,
            verbose=True,
        )
        all_mcmc[dim_idx] = mcmc

        # Recover EAP abilities for this dimension
        eap = uni_model.compute_eap_abilities(scale_data)
        all_eap[dim_idx] = eap
        print(f"  Scale {dim_idx} EAP: mean={float(jnp.mean(eap['eap'])):.4f}, "
              f"std={float(jnp.std(eap['eap'])):.4f}, "
              f"PSD={float(jnp.mean(eap['psd'])):.4f}")

    # ---- Step 4: Assemble multi-dimensional profiles ----
    print(f"\n--- Step 4: Multi-dimensional ability profiles ---")
    abilities = np.column_stack([
        np.array(all_eap[d]['eap']) for d in range(len(scale_indices))
    ])
    print(f"  Ability matrix: {abilities.shape}")
    print(f"  Per-dimension means: {np.mean(abilities, axis=0)}")
    print(f"  Per-dimension SDs:   {np.std(abilities, axis=0)}")
    print(f"  Correlation:         {np.corrcoef(abilities.T)[0, 1]:.4f}")

    # ---- Save ----
    os.makedirs('marginal_output', exist_ok=True)
    save_dict = {'abilities': abilities}
    for d in range(len(scale_indices)):
        save_dict[f'eap_dim{d}'] = np.array(all_eap[d]['eap'])
        save_dict[f'psd_dim{d}'] = np.array(all_eap[d]['psd'])
    np.savez('marginal_output/scs_factorized_marginal.npz', **save_dict)
    print(f"\nSaved to marginal_output/scs_factorized_marginal.npz")


if __name__ == '__main__':
    main()

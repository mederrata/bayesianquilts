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
4. Produce per-scale forest plots, ability histograms, and stats

Usage:
    uv run python fit_marginal_factorized_irt.py
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


COLORS = {'Scale 0': '#4477AA', 'Scale 1': '#228833', 'Scale 2': '#EE6677'}


def compute_max_rhat(mcmc_samples, prefix="  "):
    max_rhat_overall = 0.0
    for var_name, samples in mcmc_samples.items():
        if samples.shape[0] > 1:
            chain_means = np.mean(np.array(samples), axis=1)
            between_var = np.var(chain_means, axis=0, ddof=1)
            within_var = np.mean(
                np.var(np.array(samples), axis=1, ddof=1), axis=0)
            n = samples.shape[1]
            r_hat = np.sqrt(
                ((n - 1) / n * within_var + between_var) /
                np.maximum(within_var, 1e-30)
            )
            max_rhat = float(np.max(r_hat))
            max_rhat_overall = max(max_rhat_overall, max_rhat)
            print(f"{prefix}{var_name} R-hat: "
                  f"mean={np.mean(r_hat):.4f}, max={max_rhat:.4f}")
    print(f"{prefix}Max R-hat (overall): {max_rhat_overall:.4f}")
    return max_rhat_overall


def predictive_rmse(model, data_dict, scale_item_keys, K):
    ce = model.calibrated_expectations
    categories = jnp.arange(K, dtype=jnp.float64)
    probs = model.grm_model_prob_d(
        ce['abilities'], ce['discriminations'],
        ce['difficulties0'], ce.get('ddifficulties'))
    expected = jnp.sum(probs * categories[None, :], axis=-1)
    se_sum, count = 0.0, 0
    for i, key_name in enumerate(scale_item_keys):
        obs = np.array(data_dict[key_name], dtype=np.float64)
        pred_i = np.array(expected[:, i])
        valid = ~np.isnan(obs) & (obs >= 0) & (obs < K)
        se_sum += np.sum((obs[valid] - pred_i[valid]) ** 2)
        count += int(np.sum(valid))
    return float(np.sqrt(se_sum / count))


def _tufte(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=3, width=0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)


def plot_forest_single(scale_item_keys, mcmc_samples, param_key,
                       xlabel, out_path):
    """Forest plot for a single scale using MCMC posterior samples."""
    n_items = len(scale_item_keys)
    fig, ax = plt.subplots(figsize=(6, max(4, n_items * 0.3)))
    y_pos = np.arange(n_items)

    flat = np.asarray(mcmc_samples[param_key]).reshape(-1, n_items)
    means = np.mean(flat, axis=0)
    stds = np.std(flat, axis=0)

    ax.errorbar(means, y_pos, xerr=stds,
                fmt='o', capsize=2, markersize=4, elinewidth=1,
                color='#4477AA', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scale_item_keys,
                       fontsize=max(5, 9 - n_items // 20))
    ax.set_xlabel(xlabel)
    if param_key == 'discriminations':
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3,
                   linewidth=0.5)
    ax.invert_yaxis()
    _tufte(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ability_histogram(eap_abilities, scale_name, out_path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(eap_abilities, bins=40, color='#4477AA', alpha=0.7,
            edgecolor='white', linewidth=0.5)
    ax.set_xlabel(f'EAP ability ({scale_name})')
    ax.set_ylabel('Count')
    _tufte(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ability_scatter_2d(abilities, scale_names, out_path):
    """Scatter of scale 0 vs scale 1 abilities."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(abilities[:, 0], abilities[:, 1], alpha=0.3, s=5,
               color='#4477AA')
    ax.set_xlabel(f'Ability ({scale_names[0]})')
    ax.set_ylabel(f'Ability ({scale_names[1]})')
    r = np.corrcoef(abilities[:, 0], abilities[:, 1])[0, 1]
    ax.set_title(f'r = {r:.3f}', fontsize=10)
    _tufte(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # EQSQ has a natural 2-scale structure: E items and S items
    from bayesianquilts.data.eqsq import (
        get_data, item_keys, response_cardinality
    )
    from bayesianquilts.irt.grm import GRModel
    from bayesianquilts.irt.factorizedgrm import FactorizedGRModel  # used via factorized_model.fit_dim
    from bayesianquilts.imputation.pairwise_stacking import (
        PairwiseOrdinalStackingModel
    )

    output_dir = 'marginal_output'
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load data ----
    df, num_people = get_data(polars_out=True)
    pandas_df = df.select(item_keys).to_pandas().replace(-1, np.nan)

    data = {}
    for col in df.columns:
        data[col] = df[col].to_numpy().astype(np.float32)
    data['person'] = np.arange(num_people, dtype=np.float32)

    n_items = len(item_keys)
    print(f"Dataset: EQSQ")
    print(f"  Items: {n_items}, K: {response_cardinality}")
    print(f"  People: {num_people}")

    # Natural split: E items (first 60) and S items (last 60)
    e_items = [k for k in item_keys if k.startswith('E')]
    s_items = [k for k in item_keys if k.startswith('S')]
    scale_indices = [
        [item_keys.index(k) for k in e_items],
        [item_keys.index(k) for k in s_items],
    ]
    scale_names = ['Empathy', 'Systemizing']
    print(f"  {scale_names[0]}: {len(e_items)} items")
    print(f"  {scale_names[1]}: {len(s_items)} items")

    # ---- Step 1: Pairwise stacking imputation (shared) ----
    print(f"\n--- Step 1: Shared imputation model ---")
    stacking_path = os.path.join(output_dir, 'pairwise_stacking_model.yaml')
    if os.path.exists(stacking_path):
        print(f"  Loading from {stacking_path}")
        imputation_model = PairwiseOrdinalStackingModel.load(stacking_path)
    else:
        imputation_model = PairwiseOrdinalStackingModel(
            prior_scale=1.0,
            pathfinder_num_samples=100,
            pathfinder_maxiter=50,
            batch_size=512,
            verbose=True,
        )
        imputation_model.fit(
            pandas_df,
            n_top_features=30,
            n_jobs=1,
            seed=42,
        )
        imputation_model.save(stacking_path)
        print(f"  Saved to {stacking_path}")

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
    all_models = {}
    all_scale_data = {}

    for dim_idx in range(len(scale_indices)):
        scale_item_keys = [item_keys[i] for i in scale_indices[dim_idx]]
        scale_dir = os.path.join(output_dir, f'scale_{dim_idx}')
        os.makedirs(scale_dir, exist_ok=True)

        print(f"\n--- Step 3.{dim_idx}: {scale_names[dim_idx]} "
              f"({len(scale_indices[dim_idx])} items) ---")

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
        print(f"  Running marginal MCMC for {scale_names[dim_idx]}...")
        mcmc_samples = uni_model.fit_marginal_mcmc(
            scale_data,
            num_chains=2,
            num_warmup=200,
            num_samples=300,
            verbose=True,
        )

        # R-hat check — resume if convergence is poor
        max_rhat = compute_max_rhat(mcmc_samples, prefix="    ")
        resume_round = 0
        while max_rhat > 1.05 and resume_round < 3:
            resume_round += 1
            print(f"    Max R-hat {max_rhat:.4f} > 1.05, "
                  f"extending chains (round {resume_round}/3)...")
            mcmc_samples = uni_model.fit_marginal_mcmc(
                scale_data,
                num_samples=300,
                seed=42 + dim_idx * 10 + resume_round,
                verbose=True,
                resume=True,
            )
            max_rhat = compute_max_rhat(mcmc_samples, prefix="    ")

        all_mcmc[dim_idx] = mcmc_samples

        # Recover EAP abilities for this dimension
        eap = uni_model.compute_eap_abilities(scale_data)
        all_eap[dim_idx] = eap
        all_models[dim_idx] = uni_model
        all_scale_data[dim_idx] = scale_data
        print(f"  {scale_names[dim_idx]} EAP: "
              f"mean={float(jnp.mean(eap['eap'])):.4f}, "
              f"std={float(jnp.std(eap['eap'])):.4f}, "
              f"PSD={float(jnp.mean(eap['psd'])):.4f}")

        # ---- Per-scale plots ----
        plot_forest_single(
            scale_item_keys, mcmc_samples, 'discriminations',
            'Discrimination',
            os.path.join(scale_dir, 'forest_discriminations.png'))
        print(f"  Saved scale_{dim_idx}/forest_discriminations.png")

        plot_forest_single(
            scale_item_keys, mcmc_samples, 'difficulties0',
            'Difficulty (first threshold)',
            os.path.join(scale_dir, 'forest_difficulties.png'))
        print(f"  Saved scale_{dim_idx}/forest_difficulties.png")

        plot_ability_histogram(
            np.array(eap['eap']), scale_names[dim_idx],
            os.path.join(scale_dir, 'ability_histogram.png'))
        print(f"  Saved scale_{dim_idx}/ability_histogram.png")

    # ---- Step 4: Assemble multi-dimensional profiles ----
    print(f"\n--- Step 4: Multi-dimensional ability profiles ---")
    abilities = np.column_stack([
        np.array(all_eap[d]['eap']) for d in range(len(scale_indices))
    ])
    print(f"  Ability matrix: {abilities.shape}")
    print(f"  Per-dimension means: {np.mean(abilities, axis=0)}")
    print(f"  Per-dimension SDs:   {np.std(abilities, axis=0)}")
    if abilities.shape[1] >= 2:
        r = np.corrcoef(abilities.T)[0, 1]
        print(f"  Correlation:         {r:.4f}")
        plot_ability_scatter_2d(
            abilities, scale_names,
            os.path.join(output_dir, 'ability_scatter_2d.png'))
        print(f"  Saved ability_scatter_2d.png")

    # ---- Step 5: Compute LOO-RMSE per scale ----
    print(f"\n--- Step 5: Per-scale evaluation ---")
    print(f"\n{'Scale':<15} {'RMSE':>8}")
    print(f"{'─'*25}")
    for dim_idx in range(len(scale_indices)):
        scale_item_keys = [item_keys[i] for i in scale_indices[dim_idx]]
        uni_model = all_models[dim_idx]
        scale_data = all_scale_data[dim_idx]

        # Need calibrated_expectations for RMSE
        uni_model.fit_surrogate_to_mcmc()
        eap_arr = np.array(all_eap[dim_idx]['eap'])
        uni_model.surrogate_sample['abilities'] = jnp.array(
            eap_arr[:, np.newaxis, np.newaxis, np.newaxis]
        )[np.newaxis, ...]
        uni_model.calibrated_expectations = {
            k: jnp.mean(v, axis=0)
            for k, v in uni_model.surrogate_sample.items()
        }

        try:
            rmse = predictive_rmse(
                uni_model, scale_data, scale_item_keys, response_cardinality)
            print(f"{scale_names[dim_idx]:<15} {rmse:>8.4f}")
        except Exception as e:
            print(f"{scale_names[dim_idx]:<15} {'err':>8}  ({e})")

    # ---- Save ----
    save_dict = {'abilities': abilities}
    for d in range(len(scale_indices)):
        save_dict[f'eap_dim{d}'] = np.array(all_eap[d]['eap'])
        save_dict[f'psd_dim{d}'] = np.array(all_eap[d]['psd'])
    np.savez(os.path.join(output_dir, 'eqsq_factorized_marginal.npz'),
             **save_dict)
    print(f"\nSaved to {output_dir}/eqsq_factorized_marginal.npz")
    print(f"All plots saved to {output_dir}/")


if __name__ == '__main__':
    main()

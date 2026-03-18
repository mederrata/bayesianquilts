#!/usr/bin/env python3
"""Run NBR (zero-inflated) AIS evaluation for Table 1."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pandas as pd
import numpy as np
import os
import pickle
import gc
import importlib.resources
from pathlib import Path
from tqdm.auto import tqdm

from bayesianquilts.metrics.ais import AdaptiveImportanceSampler, MixIS
from bayesianquilts.predictors.regression.negbin import (
    NegativeBinomialRegression,
    NegativeBinomialRegressionLikelihood,
)
from bayesianquilts.metrics.gramis import GRAMISSampler

# --- Load data ---
try:
    data_path = str(importlib.resources.files('bayesianquilts.data').joinpath('roachdata.csv'))
except Exception:
    data_path = '../../python/bayesianquilts/data/roachdata.csv'

df = pd.read_csv(data_path)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

y_data = df['y'].values.astype(np.float32)
X_df = df.drop(columns=['y'])
X_data = X_df.values.astype(np.float32)

zero_pct = 100 * np.mean(y_data == 0)
use_zero_inflated = zero_pct > 20
print(f'Using zero-inflated model: {use_zero_inflated} ({zero_pct:.1f}% zeros)')

# --- Model ---
model = NegativeBinomialRegression(
    input_dim=X_data.shape[1],
    zero_inflated=use_zero_inflated,
    dtype=jnp.float64,
)

data_jax = {'X': jnp.array(X_data, dtype=jnp.float64), 'y': jnp.array(y_data, dtype=jnp.float64)}

# --- Load cached MCMC ---
cache_file = Path('.cache') / 'negbin_mcmc_samples.pkl'
with open(cache_file, 'rb') as f:
    mcmc_samples = pickle.load(f)
model.mcmc_samples = {k: jnp.array(v) for k, v in mcmc_samples.items()}
print(f'Loaded cached samples. Keys: {list(mcmc_samples.keys())}')
for k, v in mcmc_samples.items():
    print(f'  {k}: {v.shape}')

# --- Setup ---
likelihood_fn = NegativeBinomialRegressionLikelihood(model)
ais_sampler = AdaptiveImportanceSampler(likelihood_fn=likelihood_fn)
mixis = MixIS(likelihood_fn)

def prior_log_prob_fn(params):
    import tensorflow_probability.substrates.jax.distributions as tfd_
    lp = jnp.sum(tfd_.Normal(0., 10.0).log_prob(params['beta']), axis=-1)
    lp += jnp.sum(tfd_.Normal(0., 10.0).log_prob(params['intercept']), axis=-1)
    lp += jnp.sum(tfd_.Normal(0., 2.0).log_prob(params['log_concentration']), axis=-1)
    if 'zero_logit' in params:
        lp += jnp.sum(tfd_.Normal(0., 5.0).log_prob(params['zero_logit']), axis=-1)
    return lp

gramis_sampler = GRAMISSampler(
    likelihood_fn=likelihood_fn,
    prior_log_prob_fn=prior_log_prob_fn,
)

n_simulations = 25
n_samples = 1000
rhos = [2**-r for r in range(-2, 12)]

base_transform = ['identity']
other_transforms = ['ll', 'kl', 'var', 'mm1', 'mm2', 'pmm1', 'pmm2']

model_type_str = 'ZINB' if model.zero_inflated else 'NB'
print(f"\nRunning {n_simulations} simulations with s={n_samples} samples per run.")
print(f"Using {model_type_str} GLM likelihood")

N_obs = data_jax['y'].shape[0]
simulation_records = []

for i in tqdm(range(n_simulations), desc="Simulations"):
    params = model.sample_mcmc(num_samples=n_samples)

    # --- Identity baseline ---
    results_base = ais_sampler.adaptive_is_loo(
        data=data_jax,
        params=params,
        rhos=rhos,
        variational=False,
        transformations=base_transform
    )

    khat_identity = np.array(results_base['identity']['khat'])
    idx_bad = np.where(khat_identity >= 0.7)[0]

    print(f"Sim {i+1}: Found {len(idx_bad)} / {N_obs} data points needing adaptation.")

    del results_base
    gc.collect()

    method_khats = {'identity': khat_identity}
    for m in other_transforms + ['gramis', 'mixis']:
        method_khats[m] = np.array(khat_identity.copy())

    # --- AIS transforms + GRAMIS + MixIS on bad points only ---
    if len(idx_bad) > 0:
        batch_size = 16
        num_batches = int(np.ceil(len(idx_bad) / batch_size))

        for b in range(num_batches):
            batch_idx = idx_bad[b * batch_size: (b + 1) * batch_size]
            data_subset = {
                'X': data_jax['X'][batch_idx],
                'y': data_jax['y'][batch_idx]
            }

            # AIS transforms
            results_subset = ais_sampler.adaptive_is_loo(
                data=data_subset,
                params=params,
                rhos=rhos,
                variational=False,
                transformations=other_transforms
            )

            for base_method in other_transforms:
                khat_arrays = []
                for key, res in results_subset.items():
                    if key == 'best':
                        continue
                    if key == base_method or key.startswith(base_method + '_'):
                        khat_arrays.append(res['khat'])

                if khat_arrays:
                    min_khat_subset = np.array(np.min(np.stack(khat_arrays), axis=0))
                    method_khats[base_method][batch_idx] = np.minimum(
                        method_khats[base_method][batch_idx], min_khat_subset
                    )

            # GRAMIS
            gramis_results = gramis_sampler.gramis_loo(
                data=data_subset,
                params=params,
                n_proposals=20,
                n_samples_per_proposal=50,
                n_iterations=10,
                repulsion_G0=0.05,
                verbose=False,
                rng_key=jax.random.PRNGKey(i * 100 + b),
            )
            method_khats['gramis'][batch_idx] = np.minimum(
                method_khats['gramis'][batch_idx],
                np.array(gramis_results['khat'])
            )

            del results_subset, gramis_results
            gc.collect()

        # MixIS on all bad points at once
        log_ell = likelihood_fn.log_likelihood(data_jax, params)
        bad_data = {'X': data_jax['X'][idx_bad], 'y': data_jax['y'][idx_bad]}
        bad_log_ell = log_ell[:, idx_bad]
        mixis_res = mixis(
            max_iter=1, params=params, theta=None,
            data=bad_data, log_ell=bad_log_ell,
            log_ell_original=log_ell,
            rng_key=jax.random.PRNGKey(i + 5000),
        )
        method_khats['mixis'][idx_bad] = np.minimum(
            method_khats['mixis'][idx_bad],
            np.array(mixis_res['khat'])
        )

    # --- Aggregate results ---
    groups = {
        'Base': ['identity'],
        'PMM1': ['pmm1'],
        'PMM2': ['pmm2'],
        'KL': ['kl'],
        'Var': ['var'],
        'LL': ['ll'],
        'Ours_Combined': ['pmm1', 'pmm2', 'kl', 'var', 'll'],
        'MM1': ['mm1'],
        'MM2': ['mm2'],
        'GRAMIS': ['gramis'],
        'MixIS': ['mixis'],
        'Full': other_transforms + ['gramis', 'mixis'] + base_transform,
    }

    sim_counts = {}
    for group_name, methods in groups.items():
        grouped_khats = [method_khats[m] for m in methods if m in method_khats]
        if grouped_khats:
            best_group_khat = np.min(np.stack(grouped_khats), axis=0)
            sim_counts[group_name] = np.sum(best_group_khat > 0.7)
        else:
            sim_counts[group_name] = np.nan

    simulation_records.append(sim_counts)

df_sims = pd.DataFrame(simulation_records)
stats = df_sims.agg(['mean', 'std'])

print(f"\n--- Table Metrics: Unsuccessful Adaptations (Roaches/{model_type_str} GLM) ---")
print(stats.round(1))

print("\nLaTeX Format (Mean +/- Std):")
for col in df_sims.columns:
    m = stats.loc['mean', col]
    s = stats.loc['std', col]
    print(f"{col}: ${m:.1f}\\pm{s:.1f}$")

#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import tensorflow_probability.substrates.jax as tfp
import matplotlib.pyplot as plt
import arviz as az
import os
import importlib.resources

tfd = tfp.distributions
tfb = tfp.bijectors

from bayesianquilts.model import BayesianModel
from bayesianquilts.vi.minibatch import minibatch_fit_surrogate_posterior
from bayesianquilts.metrics.ais import AdaptiveImportanceSampler
from bayesianquilts.predictors.regression.poisson import PoissonRegressionLikelihood
from bayesianquilts.predictors.regression.poisson import PoissonRegression


# In[2]:


# Load the dataset
try:
    data_path = str(importlib.resources.files('bayesianquilts.data').joinpath('roachdata.csv'))
except ImportError:
    # Fallback if package not installed
    data_path = '../../bayesianquilts/data/roachdata.csv'
    if not os.path.exists(data_path):
        data_path = 'bayesianquilts/data/roachdata.csv'

print(f'Loading data from {data_path}')
df = pd.read_csv(data_path)
print(df.head())

# Preprocessing
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Target
y_data = df['y'].values

# Features
# Drop target and any index columns
X_df = df.drop(columns=['y'])
X_data = X_df.values.astype(np.float32)
y_data = y_data.astype(np.float32)

print(f'X shape: {X_data.shape}, y shape: {y_data.shape}')


# In[4]:


# Instantiate model (was missing)
model = PoissonRegression(input_dim=X_data.shape[1])

# Data dictionary for MCMC
data_dict = {'X': X_data, 'y': y_data}

# Check init
print('Checking initialization...')
try:
    seed = jax.random.PRNGKey(42)
    params = model.surrogate_distribution.sample(2, seed=seed)
    print('Log prob check:', model.unormalized_log_prob({'X': X_data, 'y': y_data}, **params))
except Exception as e:
    print(f'Initialization failed: {e}')
    pass

# Fit with MCMC (NUTS) - using Stan-like defaults
print('Fitting model with MCMC (NUTS)...')
mcmc_samples = model.fit_mcmc(
    data=data_dict,
    num_chains=4,       # Stan default
    num_warmup=3000,    # Increased warmup as requested
    num_samples=2000,   # Standard  
    target_accept_prob=0.8,
    max_tree_depth=10,
    init_strategy="random",
    verbose=True
)

# Print summary (R-hats are printed by fit_mcmc)
print('Parameter Summary (details above):')
for var in model.var_list:
    samples = mcmc_samples[var].reshape(-1, *mcmc_samples[var].shape[2:])
    print(f'  {var}: mean={jnp.mean(samples, axis=0)}, std={jnp.std(samples, axis=0)}')


# 

# In[ ]:


# Full Simulation for Table Metrics (Optimized for Memory)
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import copy
import gc

# Initialize Likelihood and Sampler
likelihood_fn = PoissonRegressionLikelihood()
ais_sampler = AdaptiveImportanceSampler(likelihood_fn=likelihood_fn)

n_simulations = 100
n_samples = 1000
rhos = [ 4**-r for r in range(11) ]

# Split transformations
base_transform = ['identity']
other_transforms = ['ll', 'kl', 'var', 'mm1', 'mm2', 'pmm1', 'pmm2']

print(f"Running {n_simulations} simulations with s={n_samples} samples per run.")
print(f"Rhos: {rhos}")
print(f"Base Transform: {base_transform}")
print(f"Other Transforms: {other_transforms}")

data_jax = {'X': jnp.array(X_data), 'y': jnp.array(y_data)}
N_obs = data_jax['y'].shape[0]

simulation_records = []

for i in tqdm(range(n_simulations), desc="Simulations"):
    # 1. Sample Parameters
    params = model.sample_mcmc(num_samples=n_samples)

    # 2. Run AIS with Identity FIRST
    results_base = ais_sampler.adaptive_is_loo(
        data=data_jax,
        params=params,
        rhos=rhos,
        variational=False, # MCMC samples
        transformations=base_transform
    )

    # Extract identity khat (convert to numpy for mutability)
    khat_identity = np.array(results_base['identity']['khat'])

    # Identify problematic points
    idx_bad = np.where(khat_identity >= 0.7)[0]

    print(f"Sim {i+1}: Found {len(idx_bad)} / {N_obs} data points needing adaptation.")

    # Clean up results_base immediately to free memory
    del results_base
    gc.collect()

    # Prepare storage for this simulation's khats per method
    # Initialize with identity khat for all (Hybrid approach: default to identity)
    method_khats = {}
    method_khats['identity'] = khat_identity
    for m in other_transforms:
        method_khats[m] = np.array(khat_identity.copy())

    # 3. Process problematic points with other methods
    if len(idx_bad) > 0:
        batch_size = 16
        num_batches = int(np.ceil(len(idx_bad) / batch_size))

        for b in range(num_batches):
            batch_idx = idx_bad[b*batch_size : (b+1)*batch_size]

            # Create data subset for BAD points in this batch
            # JAX arrays can be indexed by numpy arrays
            data_subset = {
                'X': data_jax['X'][batch_idx],
                'y': data_jax['y'][batch_idx]
            }

            # Run AIS on subset
            results_subset = ais_sampler.adaptive_is_loo(
                data=data_subset,
                params=params,
                rhos=rhos,
                variational=False,
                transformations=other_transforms
            )

            # Update method stats
            for base_method in other_transforms:
                # Find best khat for this base method (across rhos)
                khat_arrays = []
                for key, res in results_subset.items():
                    if key == 'best': continue
                    if key == base_method or key.startswith(base_method + '_'):
                        khat_arrays.append(res['khat'])

                if khat_arrays:
                    # Min over rhos for this method on the SUBSET
                    min_khat_subset = np.array(np.min(np.stack(khat_arrays), axis=0))

                    # Update the main array at the bad indices for this batch
                    method_khats[base_method][batch_idx] = min_khat_subset

            # Clean up batch results
            del results_subset
            gc.collect()

    # Define Groups
    groups = {
        'Base': ['identity'],
        'PMM1': ['pmm1'],
        'PMM2': ['pmm2'],
        'KL': ['kl'],
        'Var': ['var'],
        'Ours_Combined': ['pmm1', 'pmm2', 'kl', 'var'],
        'LL': ['ll'],
        'MM1': ['mm1'],
        'MM2': ['mm2'],
        'Full': other_transforms + base_transform
    }

    sim_counts = {}
    for group_name, methods in groups.items():
        grouped_khats = []
        for m in methods:
             if m in method_khats:
                 grouped_khats.append(method_khats[m])

        if grouped_khats:
            # Best khat across ANY method in the group for each obs
            best_group_khat = np.min(np.stack(grouped_khats), axis=0)
            # Count FAILURES (khat > 0.7)
            n_failures = np.sum(best_group_khat > 0.7)
            sim_counts[group_name] = n_failures
        else:
            sim_counts[group_name] = np.nan

    simulation_records.append(sim_counts)

# 4. Aggregate Statistics across simulations
df_sims = pd.DataFrame(simulation_records)
stats = df_sims.agg(['mean', 'std'])

print("\n--- Table Metrics: Unsuccessful Adaptations (Roaches/PR) ---")
print(stats.round(1))

# Optional: Format for LaTeX
print("\nLaTeX Format (Mean \pm Std):")
for col in df_sims.columns:
    m = stats.loc['mean', col]
    s = stats.loc['std', col]
    print(f"{col}: {m:.1f} \pm {s:.1f}")


# In[ ]:





# In[ ]:


# Compare khat across methods
print('Comparing khat across methods...')
khats = []
method_names = []
for method, k_vals in method_khats.items():
    khats.append(k_vals)
    method_names.append(method)

khats = np.array(khats) # Shape: (n_methods, n_data)
min_khats = np.min(khats, axis=0)

# We don't have all methods' full khats for all points (only identity/bad points updated),
# but method_khats holds the "effective" khat per method (identity fallback).
# Actually method_khats has [observations] size for each method.

print(f'Min khat per data point (min/max): {min_khats.min():.3f}/{min_khats.max():.3f}')

# Create a scatter plot of min khat
plt.figure(figsize=(10, 6))
plt.scatter(range(len(min_khats)), min_khats, c='blue', alpha=0.6, label='Min Khat')
plt.axhline(y=0.5, color='orange', linestyle='--', label='Threshold 0.5 (Good)')
plt.axhline(y=0.7, color='red', linestyle='--', label='Threshold 0.7 (Warning)')
plt.xlabel('Data Point Index')
plt.ylabel('Minimum Pareto k')
plt.title('Minimum Pareto k for each Data Point across AIS Transformations')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# In[ ]:


# Conditional plot: Min khat where identity khat >= 0.7
if 'identity' in method_khats:
    identity_khat = method_khats['identity']
    # Filter indices
    high_khat_indices = np.where(identity_khat >= 0.7)[0]

    if len(high_khat_indices) > 0:
        print(f"Found {len(high_khat_indices)} points with identity khat >= 0.7")
        min_khats_filtered = min_khats[high_khat_indices]

        plt.figure(figsize=(10, 6))
        plt.scatter(high_khat_indices, min_khats_filtered, c='red', alpha=0.6, label='Min Khat (identity >= 0.7)')
        plt.axhline(y=0.5, color='orange', linestyle='--', label='Threshold 0.5 (Good)')
        plt.axhline(y=0.7, color='red', linestyle='--', label='Threshold 0.7 (Warning)')
        plt.xlabel('Data Point Index')
        plt.ylabel('Minimum Pareto k')
        plt.title('Minimum Pareto k for Data Points with High Identity Khat (>= 0.7)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("No points found with identity khat >= 0.7")


# In[ ]:





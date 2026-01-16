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
import pickle
import pkg_resources

tfd = tfp.distributions
tfb = tfp.bijectors

from bayesianquilts.model import BayesianModel
from bayesianquilts.vi.minibatch import minibatch_fit_surrogate_posterior
from bayesianquilts.metrics.ais import AdaptiveImportanceSampler
from bayesianquilts.predictors.nn.poisson import NeuralPoissonRegression
from bayesianquilts.predictors.nn.poisson import NeuralPoissonLikelihood


# In[2]:


# Load the dataset
try:
    data_path = pkg_resources.resource_filename('bayesianquilts.data', 'roachdata.csv')
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

# Normalize features
mean = X_data.mean(axis=0)
std = X_data.std(axis=0)
X_data = (X_data - mean) / std
print('Features normalized (mean=0, std=1)')
y_data = y_data.astype(np.float32)

print(f'X shape: {X_data.shape}, y shape: {y_data.shape}')


# In[3]:


# Instantiate Neural Model with ZERO-INFLATED NEGATIVE BINOMIAL Likelihood
# 2 hidden layers of size 4
# Network outputs 3 values: zero-inflation prob, mean, concentration

# First, check the scale of the outcome
print(f'Target variable (y) statistics:')
print(f'  Min: {y_data.min():.2f}')
print(f'  Max: {y_data.max():.2f}')
print(f'  Mean: {y_data.mean():.2f}')
print(f'  Std: {y_data.std():.2f}')
print(f'  Zeros: {(y_data == 0).sum()} / {len(y_data)} ({100*(y_data == 0).mean():.1f}%)')
if (y_data > 0).any():
    print(f'  y[y>0] mean: {y_data[y_data > 0].mean():.2f}')

# Use the mean as the output scale
output_mean = float(y_data.mean())

model = NeuralPoissonRegression(
    dim_regressors=X_data.shape[1],
    hidden_size=4,
    depth=2,
    output_scale=output_mean,  # Scale the mean by y.mean()
    prior_scale=0.1,
    dtype=jnp.float32
)

print(f'\nModel: NeuralPoissonRegression')
print(f'  - Likelihood: Poisson')
print(f'  - Output scale: {output_mean:.2f} (mean of y)')
print(f'  - Network outputs: 1 (log_rate)')
print(f'  - Hidden layers: 2 × {4} neurons')
print(f'  - Activation: ReLU')
print(f'  - Prior: Horseshoe (sparse)')

# Data dictionary
data_dict = {'X': X_data, 'y': y_data}
data_jax = {'X': jnp.array(X_data), 'y': jnp.array(y_data)}

# Check init
print('\nChecking initialization...')
try:
    seed = jax.random.PRNGKey(42)
    params = model.prior_distribution.sample(2, seed=seed)
    print('Initial sample keys:', params.keys())
except Exception as e:
    print(f'Initialization failed: {e}')

# ============================================================================
# PATHFINDER INITIALIZATION FOR MCMC
# ============================================================================

print('\n' + '='*70)
print('PATHFINDER INITIALIZATION STRATEGY (Conservative Settings)')
print('='*70)

import jax.flatten_util
import blackjax.vi.pathfinder as pathfinder

def pathfinder_initialization(model, data, num_chains=4, num_samples=200, maxiter=100, 
                              ftol=1e-6, gtol=1e-9, verbose=True):
    """
    Use Pathfinder variational inference to initialize MCMC chains.

    Pathfinder (Zhang et al. 2022) is state-of-the-art for MCMC initialization:
    - Finds posterior mode using L-BFGS (quasi-Newton method with line search)
    - Builds multivariate normal approximation using inverse Hessian
    - Importance samples to get diverse, high-quality initial states

    Args:
        model: BayesianModel instance
        data: Data dictionary
        num_chains: Number of MCMC chains to initialize
        num_samples: Importance samples (higher = better diversity, default 200)
        maxiter: Max L-BFGS iterations (default 100)
        ftol: Function tolerance for L-BFGS (smaller = more conservative, default 1e-6)
        gtol: Gradient tolerance for L-BFGS (smaller = more conservative, default 1e-9)
        verbose: Print progress

    Returns:
        initial_states: Dict[str, Array] with shape (num_chains, ...)
    """
    if verbose:
        print('\nStep 1: Running Pathfinder variational inference...')
        print(f'  L-BFGS settings: ftol={ftol}, gtol={gtol} (conservative)')

    # Setup parameter flattening
    key = jax.random.PRNGKey(0)
    prior_sample = model.prior_distribution.sample(1, seed=key)
    template = {var: prior_sample[var][0] for var in model.var_list}
    flat_template, unflatten_fn = jax.flatten_util.ravel_pytree(template)
    param_dim = flat_template.shape[0]

    if verbose:
        print(f'  Parameter space dimension: {param_dim}')

    # Define log probability for Pathfinder
    def logprob_fn_flat(params_flat):
        params_dict = unflatten_fn(params_flat)
        return model.unormalized_log_prob(data=data, **params_dict)

    # Run Pathfinder with conservative tolerances
    initial_position = jax.random.normal(jax.random.PRNGKey(42), (param_dim,)) * 0.1

    state, info = pathfinder.approximate(
        rng_key=jax.random.PRNGKey(123),
        logdensity_fn=logprob_fn_flat,
        initial_position=initial_position,
        num_samples=num_samples,
        maxiter=maxiter,
        ftol=ftol,  # Conservative function tolerance
        gtol=gtol,  # Conservative gradient tolerance
    )

    if verbose:
        print(f'  ✓ Pathfinder converged! ELBO: {float(state.elbo):.3f}')

    # Sample diverse initial states
    if verbose:
        print(f'\nStep 2: Sampling {num_chains} diverse initial states...')

    sample_key = jax.random.PRNGKey(456)
    samples_result = pathfinder.sample(sample_key, state, num_samples=num_chains)
    samples_flat = samples_result[0] if isinstance(samples_result, tuple) else samples_result

    if verbose:
        print(f'  ✓ Sampled from approximate posterior')

    # Unflatten and organize by parameter
    chain_inits = {var: [] for var in model.var_list}

    for i in range(num_chains):
        sample_dict = unflatten_fn(samples_flat[i])
        for var_name in model.var_list:
            chain_inits[var_name].append(sample_dict[var_name])

    # Stack into (num_chains, ...) format
    for var_name in model.var_list:
        chain_inits[var_name] = jnp.stack(chain_inits[var_name], axis=0)
        if verbose:
            print(f'  {var_name}: shape {chain_inits[var_name].shape}')

    if verbose:
        print('  ✓ Initial states ready for MCMC')

    return chain_inits

def check_rhat_and_save(model, cache_dir, threshold=1.05):
    """Check R-hat convergence with strict threshold."""
    import tensorflow_probability.substrates.jax.mcmc as tfmcmc
    print("\nChecking R-hat convergence...")
    all_good = True
    max_rhat_overall = 0.0

    for var, samples in model.mcmc_samples.items():
        samples_transposed = jnp.swapaxes(samples, 0, 1)
        rhat = tfmcmc.potential_scale_reduction(samples_transposed)
        rhat = jnp.where(jnp.isnan(rhat), 1.0, rhat)
        max_r = float(jnp.max(rhat))
        mean_r = float(jnp.mean(rhat))
        max_rhat_overall = max(max_rhat_overall, max_r)

        if max_r > threshold:
            all_good = False
            print(f"  ✗ {var:10s}: max R-hat {max_r:.3f} (mean {mean_r:.3f}) > {threshold}")
        else:
            print(f"  ✓ {var:10s}: max R-hat {max_r:.3f} (mean {mean_r:.3f})")

    print(f"\nOverall max R-hat: {max_rhat_overall:.3f}")

    if all_good:
        print(f"✓ EXCELLENT: All R-hat < {threshold}! Saving model to {cache_dir}...")
        model.save_to_disk(cache_dir)
        return True
    else:
        print(f"✗ Convergence not achieved. Model NOT saved.")
        print(f"  Recommendation: Increase num_warmup to 15000-20000")
        return False

# Main fitting logic with Pathfinder
cache_dir = '/tmp/neural_regression_zinb'

if os.path.exists(os.path.join(cache_dir, 'config.yaml')):
    print(f"\nLoading fitted model from {cache_dir}...")
    try:
        model = NeuralNegativeBinomialRegression.load_from_disk(cache_dir)
        print("✓ Model loaded from cache!")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("  Will refit with Pathfinder...")

        # Run Pathfinder with conservative settings
        chain_inits = pathfinder_initialization(
            model, data_dict, 
            num_chains=4, 
            num_samples=200,
            maxiter=500,
            ftol=1e-8,  # Conservative L-BFGS tolerance
            gtol=1e-10   # Conservative gradient tolerance
        )

        # Run MCMC with conservative settings
        print('\nStep 3: Running MCMC with Pathfinder initialization...')
        try:
            model.fit_mcmc(
                data=data_dict,
                num_samples=2000,
                num_burnin_steps=30000,
                num_results=2000,
                num_chains=4,
                target_accept_prob=0.9995,
                step_size=1e-4,  # Conservative initial step size
                initial_states=chain_inits
            )
            print("✓ MCMC Complete.")
            check_rhat_and_save(model, cache_dir, threshold=1.05)
        except Exception as e:
            print(f"✗ MCMC failed: {e}")
            import traceback
            traceback.print_exc()
else:
    print('\nNo cached model found. Starting fresh fit with Pathfinder...')

    # Run Pathfinder initialization with conservative settings
    chain_inits = pathfinder_initialization(
        model, data_dict,
        num_chains=4,
        num_samples=200,    # More importance samples = better diversity
        maxiter=500,        # More iterations = better convergence
        ftol=1e-8,          # Conservative function tolerance for L-BFGS
        gtol=1e-10           # Conservative gradient tolerance for L-BFGS
    )

    # Run MCMC with Pathfinder initialization and conservative settings
    print('\nStep 3: Running MCMC with Pathfinder-initialized chains...')
    try:
        model.fit_mcmc(
            data=data_dict,
            num_samples=2000,       # Post-warmup samples
            num_warmup=30000,       # Long warmup for adaptation
            num_chains=4,           # 4 chains for convergence detection
            target_accept_prob=0.95,# Higher for difficult posteriors
            step_size=1e-4,         # Conservative initial step size (will adapt)
            initial_states=chain_inits  # Pathfinder initialization!
        )
        print("✓ MCMC Complete.")

        # Check convergence
        converged = check_rhat_and_save(model, cache_dir, threshold=1.05)

        if not converged:
            print("\n" + "!"*70)
            print("WARNING: Chains did not fully converge!")
            print("\nRecommendations:")
            print("  1. Increase num_warmup to 15000-20000")
            print("  2. Run more chains (6-8) for better mixing")
            print("  3. Try higher target_accept_prob (0.90-0.95)")
            print("  4. Current settings are already very conservative")
            print("!"*70)

    except Exception as e:
        print(f"✗ MCMC failed: {e}")
        import traceback
        traceback.print_exc()

print('\n' + '='*70)
print('FITTING COMPLETE')
print('='*70)
print('\nPathfinder + NUTS MCMC Summary (Zero-Inflated NegBin):')
print(f'  - Likelihood: Zero-Inflated Negative Binomial')
print(f'  - Output scale: {output_mean:.2f}')
print(f'  - Network outputs: [zero_logit, log_mean, log_concentration]')
print('  - Pathfinder L-BFGS: ftol=1e-6, gtol=1e-9 (conservative)')
print('  - Initial states: Diverse samples from approximate posterior')
print('  - MCMC: 4 chains × 2000 samples with 10000 warmup')
print('  - MCMC step size: 1e-4 (conservative, will adapt during warmup)')
print('  - Total time: ~30-60 minutes first run, <1s cached')
print('='*70)


# In[4]:


# Full Simulation for Table Metrics (Optimized for Memory)
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import copy
import gc

# Initialize Likelihood and Sampler for POISSON regression
likelihood_fn = NeuralPoissonLikelihood(model)
ais_sampler = AdaptiveImportanceSampler(likelihood_fn=likelihood_fn)

n_simulations = 100
n_samples = 1000
rhos = [ 2**-r for r in range(-2, 11) ]

# Split transformations
base_transform = ['identity']
other_transforms = ['ll', 'kl', 'var', 'mm1', 'mm2', 'pmm1', 'pmm2']

print(f"Running {n_simulations} simulations with s={n_samples} samples per run.")
print(f"Using POISSON likelihood")
print(f"  output_scale={model.output_scale:.2f}")
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
            # Take minimum khat (best adaptation)
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

print("\n--- Table Metrics: Unsuccessful Adaptations (Roaches/Poisson) ---")
print(stats.round(1))

# Optional: Format for LaTeX
print("\nLaTeX Format (Mean ± Std):")
for col in df_sims.columns:
    m = stats.loc['mean', col]
    s = stats.loc['std', col]
    print(f"{col}: {m:.1f} ± {s:.1f}")


# In[ ]:


# Full Simulation for Table Metrics (Optimized for Memory)
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import copy
import gc

# Initialize Likelihood and Sampler
likelihood_fn = NeuralPoissonLikelihood(model)
ais_sampler = AdaptiveImportanceSampler(likelihood_fn=likelihood_fn)

n_simulations = 100
n_samples = 1000
rhos = [ 2**-r for r in range(-2, 11) ]

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
            # Take minimum khat (best adaptation)
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
print("\nLaTeX Format (Mean \u005cpm Std):")
for col in df_sims.columns:
    m = stats.loc['mean', col]
    s = stats.loc['std', col]
    print(f"{col}: {m:.1f} \u005cpm {s:.1f}")


# In[ ]:





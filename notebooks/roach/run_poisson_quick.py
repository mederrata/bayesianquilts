"""Quick Poisson regression AIS test - 10 simulations to verify LL fix."""
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import os
import gc
jax.config.update("jax_enable_x64", True)

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from bayesianquilts.predictors.regression.poisson import PoissonRegression, PoissonRegressionLikelihood
from bayesianquilts.metrics.ais import AdaptiveImportanceSampler

# Load data
try:
    import importlib.resources
    data_path = str(importlib.resources.files('bayesianquilts.data').joinpath('roachdata.csv'))
except Exception:
    data_path = '../../python/bayesianquilts/data/roachdata.csv'

df = pd.read_csv(data_path)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

y_data = df['y'].values.astype(np.float32)
X_df = df[['roach1', 'treatment', 'senior']].copy()
X_df['roach1_treatment'] = X_df['roach1'] * X_df['treatment']
X_df['roach1_senior'] = X_df['roach1'] * X_df['senior']
X_df['treatment_senior'] = X_df['treatment'] * X_df['senior']
X_data = X_df.values.astype(np.float32)
offset_data = np.log(df['exposure2'].values).astype(np.float32)

print(f'X shape: {X_data.shape}, y shape: {y_data.shape}')

# Model
model = PoissonRegression(
    input_dim=X_data.shape[1],
    prior_scale_beta=2.5,
    prior_scale_intercept=5.0,
    dtype=jnp.float64
)

data_dict = {
    'X': jnp.array(X_data),
    'y': jnp.array(y_data),
    'offset': jnp.array(offset_data)
}

# Pathfinder init + MCMC
import jax.flatten_util
import blackjax.vi.pathfinder as pathfinder

key = jax.random.PRNGKey(0)
prior_sample = model.prior_distribution.sample(1, seed=key)
template = {var: prior_sample[var][0] for var in model.var_list}
flat_template, unflatten_fn = jax.flatten_util.ravel_pytree(template)

def logprob_fn_flat(params_flat):
    params_dict = unflatten_fn(params_flat)
    return model.unormalized_log_prob(data=data_dict, **params_dict)

initial_position = jax.random.normal(jax.random.PRNGKey(42), (flat_template.shape[0],)) * 0.1
state, info = pathfinder.approximate(
    rng_key=jax.random.PRNGKey(123),
    logdensity_fn=logprob_fn_flat,
    initial_position=initial_position,
    num_samples=200, maxiter=100, ftol=1e-6, gtol=1e-9,
)
print(f'Pathfinder ELBO: {float(state.elbo):.3f}')

sample_key = jax.random.PRNGKey(456)
samples_result = pathfinder.sample(sample_key, state, num_samples=4)
samples_flat = samples_result[0] if isinstance(samples_result, tuple) else samples_result

chain_inits = {var: [] for var in model.var_list}
for i in range(4):
    sample_dict = unflatten_fn(samples_flat[i])
    for var_name in model.var_list:
        chain_inits[var_name].append(sample_dict[var_name])
for var_name in model.var_list:
    chain_inits[var_name] = jnp.stack(chain_inits[var_name], axis=0)

print('Running MCMC...')
mcmc_samples = model.fit_mcmc(
    data=data_dict, num_chains=4, num_warmup=5000, num_samples=2000,
    target_accept_prob=0.995, max_tree_depth=12,
    initial_states=chain_inits, verbose=True
)

# AIS simulation - just 10 runs for a quick check
likelihood_fn = PoissonRegressionLikelihood(dtype=jnp.float64)
ais_sampler = AdaptiveImportanceSampler(likelihood_fn=likelihood_fn)

n_simulations = 10
n_samples = 1000
rhos = [3**-r for r in range(1, 6)]

base_transform = ['identity']
other_transforms = ['ll', 'kl', 'var', 'mm1', 'mm2', 'pmm1', 'pmm2']

data_jax = {'X': jnp.array(X_data, dtype=jnp.float64), 'y': jnp.array(y_data, dtype=jnp.float64)}
N_obs = data_jax['y'].shape[0]

simulation_records = []

for i in range(n_simulations):
    params = model.sample_mcmc(num_samples=n_samples)

    results_base = ais_sampler.adaptive_is_loo(
        data=data_jax, params=params, rhos=rhos,
        variational=False, transformations=base_transform
    )
    khat_identity = np.array(results_base['identity']['khat'])
    idx_bad = np.where(khat_identity >= 0.7)[0]
    print(f"Sim {i+1}: {len(idx_bad)} / {N_obs} points need adaptation.")
    del results_base; gc.collect()

    method_khats = {'identity': khat_identity}
    for m in other_transforms:
        method_khats[m] = np.array(khat_identity.copy())

    if len(idx_bad) > 0:
        batch_size = 16
        for b in range(int(np.ceil(len(idx_bad) / batch_size))):
            batch_idx = idx_bad[b*batch_size:(b+1)*batch_size]
            data_subset = {'X': data_jax['X'][batch_idx], 'y': data_jax['y'][batch_idx]}

            results_subset = ais_sampler.adaptive_is_loo(
                data=data_subset, params=params, rhos=rhos,
                variational=False, transformations=other_transforms
            )

            for base_method in other_transforms:
                khat_arrays = []
                for key, res in results_subset.items():
                    if key == 'best': continue
                    if key == base_method or key.startswith(base_method + '_'):
                        khat_arrays.append(res['khat'])
                if khat_arrays:
                    min_khat_subset = np.array(np.min(np.stack(khat_arrays), axis=0))
                    method_khats[base_method][batch_idx] = min_khat_subset
            del results_subset; gc.collect()

    groups = {
        'Base': ['identity'],
        'PMM1': ['pmm1'], 'PMM2': ['pmm2'],
        'KL': ['kl'], 'Var': ['var'],
        'Ours_Combined': ['pmm1', 'pmm2', 'kl', 'var'],
        'LL': ['ll'],
        'MM1': ['mm1'], 'MM2': ['mm2'],
        'Full': other_transforms + base_transform
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

print("\n" + "="*70)
print("Table Metrics: Unsuccessful Adaptations (Roaches/PR) - 10 sims")
print("="*70)
print(stats.round(1))
print("\nLaTeX Format (Mean ± Std):")
for col in df_sims.columns:
    m = stats.loc['mean', col]
    s = stats.loc['std', col]
    print(f"  {col}: {m:.1f} ± {s:.1f}")

#!/usr/bin/env python3
"""Time each AIS transformation method on the ovarian LR dataset."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pickle
import time
import pandas as pd
from pathlib import Path
from bayesianquilts.metrics.ais import (
    AdaptiveImportanceSampler, LogisticRegressionLikelihood, MixIS
)
from bayesianquilts.metrics.gramis import GRAMISSampler

base = Path.home() / 'workspace' / 'bayesianquilts'

# Load data
X_ = pd.read_csv(base / 'python' / 'bayesianquilts' / 'data' / 'overianx.csv', header=None)
y_ = pd.read_table(base / 'python' / 'bayesianquilts' / 'data' / 'overiany.csv', header=None)
X_scaled = ((X_ - X_.mean()) / X_.std()).fillna(0).to_numpy()
y_ov = y_.to_numpy()[:, 0]

with open(base / 'notebooks' / 'ovarian' / '.cache' / 'ovarian_lr_stan_samples.pkl', 'rb') as f:
    cached = pickle.load(f)

likelihood = LogisticRegressionLikelihood(dtype=jnp.float64)
sampler = AdaptiveImportanceSampler(likelihood_fn=likelihood)
mixis = MixIS(likelihood)
gramis = GRAMISSampler(likelihood_fn=likelihood, prior_log_prob_fn=None)

data = {'X': jnp.array(X_scaled, dtype=jnp.float64),
        'y': jnp.array(y_ov, dtype=jnp.float64)}

n_samples = 1000
rhos = [2**-r for r in range(-2, 12)]

# Sample params
ndx = np.random.choice(cached['beta0'].shape[0], size=n_samples, replace=False)
params = {
    'beta': jnp.array(cached['beta'][ndx], dtype=jnp.float64),
    'intercept': jnp.array(cached['beta0'][ndx, None], dtype=jnp.float64),
}

# Warmup JIT
print("Warming up JIT...")
_ = sampler.adaptive_is_loo(data=data, params=params, rhos=rhos[:2],
                             variational=False, transformations=['identity'])
_ = sampler.adaptive_is_loo(data=data, params=params, rhos=rhos[:2],
                             variational=False, transformations=['ll'])

# Get identity khat to find bad points
results_id = sampler.adaptive_is_loo(data=data, params=params, rhos=rhos,
                                      variational=False, transformations=['identity'])
khat_id = np.array(results_id['identity']['khat'])
idx_bad = np.where(khat_id >= 0.7)[0]
n_bad = len(idx_bad)
print(f"Points needing adaptation: {n_bad} / {data['y'].shape[0]}")

data_bad = {'X': data['X'][idx_bad], 'y': data['y'][idx_bad]}

# Time each method on just the bad points
methods = ['pmm1', 'pmm2', 'kl', 'var', 'll', 'mm1', 'mm2']
timings = {}

for method in methods:
    # Warmup
    _ = sampler.adaptive_is_loo(data=data_bad, params=params, rhos=rhos[:2],
                                 variational=False, transformations=[method])

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = sampler.adaptive_is_loo(data=data_bad, params=params, rhos=rhos,
                                     variational=False, transformations=[method])
        jax.block_until_ready(_)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = np.mean(times)
    std = np.std(times)
    timings[method] = (avg, std)
    print(f"{method:>6s}: {avg:.2f} ± {std:.2f} s ({n_bad} obs, {len(rhos)} rhos)")

# GRAMIS
_ = gramis.gramis_loo(data=data_bad, params=params, n_proposals=20,
                        n_samples_per_proposal=50, n_iterations=10,
                        repulsion_G0=0.05, verbose=False,
                        rng_key=jax.random.PRNGKey(0))
times = []
for rep in range(5):
    t0 = time.perf_counter()
    _ = gramis.gramis_loo(data=data_bad, params=params, n_proposals=20,
                            n_samples_per_proposal=50, n_iterations=10,
                            repulsion_G0=0.05, verbose=False,
                            rng_key=jax.random.PRNGKey(rep))
    jax.block_until_ready(_)
    t1 = time.perf_counter()
    times.append(t1 - t0)
avg, std = np.mean(times), np.std(times)
timings['gramis'] = (avg, std)
print(f"gramis: {avg:.2f} ± {std:.2f} s ({n_bad} obs)")

# MixIS
log_ell = likelihood.log_likelihood(data, params)
_ = mixis(max_iter=1, params=params, theta=None, data=data_bad,
          log_ell=log_ell[:, idx_bad], log_ell_original=log_ell,
          rng_key=jax.random.PRNGKey(0))
times = []
for rep in range(5):
    t0 = time.perf_counter()
    _ = mixis(max_iter=1, params=params, theta=None, data=data_bad,
              log_ell=log_ell[:, idx_bad], log_ell_original=log_ell,
              rng_key=jax.random.PRNGKey(rep))
    jax.block_until_ready(_)
    t1 = time.perf_counter()
    times.append(t1 - t0)
avg, std = np.mean(times), np.std(times)
timings['mixis'] = (avg, std)
print(f" mixis: {avg:.2f} ± {std:.2f} s ({n_bad} obs)")

print("\n=== SUMMARY (Ovarian LR) ===")
print(f"{'Method':>8s}  {'Time (s)':>12s}  {'Per obs (ms)':>14s}")
for m, (avg, std) in timings.items():
    per_obs = 1000 * avg / n_bad
    print(f"{m:>8s}  {avg:>5.2f} ± {std:.2f}  {per_obs:>8.1f}")

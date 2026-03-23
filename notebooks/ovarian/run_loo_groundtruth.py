#!/usr/bin/env python3
"""
Compute ground truth LOO for ovarian LR by refitting Stan model 54 times.
Then compare IS-LOO (with AIS transformations) against the ground truth.

Outputs:
  - ovarian_loo_groundtruth.pkl: per-fold Stan samples and elpd
  - Comparison of elpd and AUC/ROC between IS-LOO and ground truth
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import pickle
import importlib.resources
from pathlib import Path
from cmdstanpy import CmdStanModel
from tqdm import tqdm

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp

# ============================================================================
# Load data
# ============================================================================
with importlib.resources.path("bayesianquilts.data", "overianx.csv") as xpath:
    X = pd.read_csv(xpath, header=None)
with importlib.resources.path("bayesianquilts.data", "overiany.csv") as ypath:
    y = pd.read_csv(ypath, header=None)

X_scaled = (X - X.mean()) / X.std()
X_scaled = X_scaled.fillna(0)
X_np = X_scaled.to_numpy(dtype=float)
y_np = y.to_numpy(dtype=float).flatten().astype(int)

n, p = X_np.shape
print(f"Data: n={n}, p={p}")

# Prior hyperparameters (same as main fit)
guessnumrelevcov = n / 10
scale_global = guessnumrelevcov / ((p - guessnumrelevcov) * np.sqrt(n))

stan_data_template = {
    "d": p,
    "scale_icept": 5.0,
    "scale_global": scale_global,
    "nu_global": 1.0,
    "nu_local": 1.0,
    "slab_scale": 2.5,
    "slab_df": 1.0,
}

# ============================================================================
# Compile Stan model
# ============================================================================
stan_file = os.path.join(os.path.dirname(__file__), "ovarian_model.stan")
sm = CmdStanModel(stan_file=stan_file)
print("Stan model compiled")

# ============================================================================
# Output paths
# ============================================================================
cache_dir = Path(os.path.dirname(__file__)) / ".cache"
cache_dir.mkdir(exist_ok=True)
output_file = cache_dir / "ovarian_loo_groundtruth.pkl"

# Check for existing results (resume support)
if output_file.exists():
    with open(output_file, "rb") as f:
        results = pickle.load(f)
    print(f"Loaded {len(results['folds'])} existing folds")
else:
    results = {"folds": {}, "full_data_fit": None}

# ============================================================================
# Fit full-data model (fold 0)
# ============================================================================
if results["full_data_fit"] is None:
    print("Fitting full-data model...")
    stan_data = {**stan_data_template, "N": n, "x": X_np, "y": y_np.tolist()}
    fit_full = sm.sample(
        data=stan_data,
        chains=4,
        iter_warmup=5000,
        iter_sampling=2000,
        adapt_delta=0.99,
        max_treedepth=12,
        seed=42,
        show_progress=True,
    )
    results["full_data_fit"] = {
        k: fit_full.stan_variable(k) for k in ["beta0", "beta", "log_lik"]
    }
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print("Full-data fit saved")
else:
    print("Full-data fit already cached")

# ============================================================================
# Fit 54 LOO models
# ============================================================================
print(f"\nFitting {n} LOO models...")
for i in tqdm(range(n), desc="LOO folds"):
    if i in results["folds"]:
        continue

    # Leave out observation i
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    X_loo = X_np[mask]
    y_loo = y_np[mask].tolist()

    stan_data = {**stan_data_template, "N": n - 1, "x": X_loo, "y": y_loo}

    try:
        fit_loo = sm.sample(
            data=stan_data,
            chains=4,
            iter_warmup=5000,
            iter_sampling=2000,
            adapt_delta=0.99,
            max_treedepth=12,
            seed=42 + i + 1,
            show_progress=False,
        )
        beta = fit_loo.stan_variable("beta")  # (S, p)
        beta0 = fit_loo.stan_variable("beta0")  # (S,)

        results["folds"][i] = {"beta": beta, "beta0": beta0}

        # Save periodically
        if (i + 1) % 5 == 0:
            with open(output_file, "wb") as f:
                pickle.dump(results, f)
            print(f"  Saved checkpoint at fold {i+1}")

    except Exception as e:
        print(f"  Fold {i} FAILED: {e}")
        results["folds"][i] = None

# Final save
with open(output_file, "wb") as f:
    pickle.dump(results, f)
print(f"\nAll {n} LOO folds complete. Saved to {output_file}")

# ============================================================================
# Compute ground truth LOO elpd and predictions
# ============================================================================
print("\n" + "=" * 70)
print("Computing ground truth LOO statistics")
print("=" * 70)

gt_elpd = np.zeros(n)
gt_prob = np.zeros(n)  # P(y_i=1 | D_{-i})

for i in range(n):
    fold = results["folds"].get(i)
    if fold is None:
        gt_elpd[i] = np.nan
        gt_prob[i] = np.nan
        continue

    beta = fold["beta"]  # (S, p)
    beta0 = fold["beta0"]  # (S,)
    xi = X_np[i]
    yi = y_np[i]

    # Logits for observation i under LOO posterior
    logits = beta0 + beta @ xi  # (S,)

    # Log-likelihood of y_i under LOO posterior samples
    ll_i = yi * logits - np.logaddexp(0, logits)  # (S,)

    # LOO elpd: log E_{theta|D_{-i}}[p(y_i|theta)]
    gt_elpd[i] = float(logsumexp(jnp.array(ll_i)) - np.log(len(ll_i)))

    # LOO predicted probability: E_{theta|D_{-i}}[p(y_i=1|theta)]
    gt_prob[i] = float(jnp.mean(jax.nn.sigmoid(jnp.array(logits))))

gt_total_elpd = np.nansum(gt_elpd)
print(f"Ground truth total LOO-elpd: {gt_total_elpd:.3f}")
print(f"Ground truth per-obs elpd: mean={np.nanmean(gt_elpd):.4f}, "
      f"std={np.nanstd(gt_elpd):.4f}")

# AUC/ROC
from sklearn.metrics import roc_auc_score, roc_curve
gt_auc = roc_auc_score(y_np, gt_prob)
print(f"Ground truth LOO-AUC: {gt_auc:.4f}")

# ============================================================================
# Now compute IS-LOO using the cached full-data posterior
# ============================================================================
print("\n" + "=" * 70)
print("Computing IS-LOO with AIS transformations")
print("=" * 70)

from bayesianquilts.metrics.ais import AdaptiveImportanceSampler, LogisticRegressionLikelihood
from bayesianquilts.metrics import psis

# Load cached full-data Stan samples
with open(cache_dir / "ovarian_lr_stan_samples.pkl", "rb") as f:
    stan_samples = pickle.load(f)

# Use pre-transformed beta (not the horseshoe reparameterization)
n_stan = stan_samples["beta"].shape[0]
n_use = 1000
rng = np.random.default_rng(42)
idx = rng.choice(n_stan, size=n_use, replace=False)

beta_samples = jnp.array(stan_samples["beta"][idx], dtype=jnp.float64)
beta0_samples = jnp.array(stan_samples["beta0"][idx, None], dtype=jnp.float64)

params = {
    "beta": beta_samples,
    "intercept": beta0_samples,
}

data_jax = {
    "X": jnp.array(X_np, dtype=jnp.float64),
    "y": jnp.array(y_np, dtype=jnp.float64),
}

likelihood_fn = LogisticRegressionLikelihood(dtype=jnp.float64)
ais_sampler = AdaptiveImportanceSampler(likelihood_fn=likelihood_fn)

rhos = [2**-r for r in range(1, 9)]
transforms = ["identity", "ll", "kl", "var", "pmm1", "pmm2", "nkl", "nll"]

print(f"Running AIS with {len(transforms)} transforms, {len(rhos)} step sizes, "
      f"s={n_use} samples")

# Run AIS on all 54 observations
results_ais = ais_sampler.adaptive_is_loo(
    data=data_jax,
    params=params,
    rhos=rhos,
    variational=False,
    transformations=transforms,
)

# Extract per-method results
is_loo_results = {}
for method_key, res in results_ais.items():
    if method_key == "best" or method_key == "timings":
        continue
    if not isinstance(res, dict) or "khat" not in res:
        continue

    khat = np.array(res["khat"])
    p_loo = np.array(res.get("p_loo_psis", np.full(n, np.nan)))
    ll_loo = np.array(res.get("ll_loo_psis", np.full(n, np.nan)))

    is_loo_results[method_key] = {
        "khat": khat,
        "p_loo_psis": p_loo,
        "ll_loo_psis": ll_loo,
    }

# For each method, find the best rho per observation
method_bases = set()
for key in is_loo_results:
    base = key.split("_rho")[0]
    method_bases.add(base)

best_per_method = {}
for base in sorted(method_bases):
    matching = {k: v for k, v in is_loo_results.items()
                if k == base or k.startswith(base + "_rho")}
    if not matching:
        continue

    # For each observation, pick the rho with lowest khat
    khats = np.stack([v["khat"] for v in matching.values()])
    best_idx = np.argmin(khats, axis=0)
    keys_list = list(matching.keys())

    best_khat = np.min(khats, axis=0)
    best_ll_loo = np.zeros(n)
    best_p_loo = np.zeros(n)
    for i_obs in range(n):
        best_key = keys_list[best_idx[i_obs]]
        best_ll_loo[i_obs] = matching[best_key]["ll_loo_psis"][i_obs]
        best_p_loo[i_obs] = matching[best_key]["p_loo_psis"][i_obs]

    best_per_method[base] = {
        "khat": best_khat,
        "ll_loo_psis": best_ll_loo,
        "p_loo_psis": best_p_loo,
    }

# ============================================================================
# Comparison: IS-LOO vs Ground Truth
# ============================================================================
print("\n" + "=" * 70)
print("COMPARISON: IS-LOO vs Ground Truth")
print("=" * 70)

print(f"\n{'Method':<20s} {'Total elpd':>12s} {'Diff vs GT':>12s} "
      f"{'RMSE elpd':>12s} {'LOO-AUC':>10s}")
print("-" * 70)
print(f"{'Ground Truth':<20s} {gt_total_elpd:>12.2f} {'---':>12s} "
      f"{'---':>12s} {gt_auc:>10.4f}")

comparison_records = []
for method, res in sorted(best_per_method.items()):
    # ll_loo_psis contains p(y_i|D_{-i}) (predictive probability), not log
    p_loo = np.clip(res["ll_loo_psis"], 1e-100, None)
    elpd_vec = np.log(p_loo)
    total_elpd = np.sum(elpd_vec)
    diff = total_elpd - gt_total_elpd
    rmse = np.sqrt(np.mean((elpd_vec - gt_elpd) ** 2))

    # p_loo_psis = p(y_i|D_{-i}) = predictive density at observed y_i
    # For Bernoulli: p_loo = sigma(mu)^y * (1-sigma(mu))^(1-y)
    # So P(y=1|D_{-i}) = p_loo when y=1, and 1-p_loo when y=0
    prob_y1 = np.where(y_np == 1, p_loo, 1 - p_loo)
    prob_y1 = np.clip(prob_y1, 1e-10, 1 - 1e-10)

    try:
        auc = roc_auc_score(y_np, prob_y1)
    except:
        auc = np.nan

    n_bad = int(np.sum(res["khat"] >= 0.7))

    print(f"{method:<20s} {total_elpd:>12.2f} {diff:>+12.2f} "
          f"{rmse:>12.4f} {auc:>10.4f}  (khat>=0.7: {n_bad})")

    comparison_records.append({
        "method": method,
        "total_elpd": total_elpd,
        "diff_vs_gt": diff,
        "rmse_elpd": rmse,
        "auc": auc,
        "n_bad_khat": n_bad,
    })

# Also compute "best across all methods" combined
all_methods_ours = ["pmm1", "pmm2", "ll", "kl", "nkl", "nll", "var"]
available = [m for m in all_methods_ours if m in best_per_method]
if available:
    khats_all = np.stack([best_per_method[m]["khat"] for m in available])
    best_method_idx = np.argmin(khats_all, axis=0)

    combined_elpd = np.zeros(n)
    combined_ploo = np.zeros(n)
    combined_khat = np.min(khats_all, axis=0)
    for i_obs in range(n):
        m = available[best_method_idx[i_obs]]
        combined_elpd[i_obs] = best_per_method[m]["ll_loo_psis"][i_obs]
        combined_ploo[i_obs] = best_per_method[m]["p_loo_psis"][i_obs]

    combined_ploo_clipped = np.clip(combined_ploo, 1e-100, None)
    combined_elpd_log = np.log(combined_ploo_clipped)
    total_elpd = np.sum(combined_elpd_log)
    diff = total_elpd - gt_total_elpd
    rmse = np.sqrt(np.mean((combined_elpd_log - gt_elpd) ** 2))
    prob_y1 = np.where(y_np == 1, combined_ploo_clipped, 1 - combined_ploo_clipped)
    prob_y1 = np.clip(prob_y1, 1e-10, 1 - 1e-10)
    auc = roc_auc_score(y_np, prob_y1)
    n_bad = int(np.sum(combined_khat >= 0.7))

    print(f"{'Ours Combined':<20s} {total_elpd:>12.2f} {diff:>+12.2f} "
          f"{rmse:>12.4f} {auc:>10.4f}  (khat>=0.7: {n_bad})")

# Save comparison
comparison_file = cache_dir / "ovarian_loo_comparison.pkl"
with open(comparison_file, "wb") as f:
    pickle.dump({
        "gt_elpd": gt_elpd,
        "gt_prob": gt_prob,
        "gt_auc": gt_auc,
        "gt_total_elpd": gt_total_elpd,
        "is_loo": best_per_method,
        "comparison": comparison_records,
    }, f)
print(f"\nComparison saved to {comparison_file}")

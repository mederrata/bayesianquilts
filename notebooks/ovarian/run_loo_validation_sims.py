#!/usr/bin/env python3
"""
Compare IS-LOO vs ground truth LOO for ovarian LR.
Computes per-observation p_loo with IS standard errors.
Reports total elpd ± se(elpd) using sqrt(n)*sd(elpd_i).
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pickle
import importlib.resources
import pandas as pd
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from bayesianquilts.metrics.ais import AdaptiveImportanceSampler, LogisticRegressionLikelihood
from bayesianquilts.metrics import psis as psis_mod
from sklearn.metrics import roc_auc_score

# ── Load data ──
with importlib.resources.path("bayesianquilts.data", "overianx.csv") as xpath:
    X = pd.read_csv(xpath, header=None)
with importlib.resources.path("bayesianquilts.data", "overiany.csv") as ypath:
    y = pd.read_csv(ypath, header=None)
X_scaled = (X - X.mean()) / X.std()
X_scaled = X_scaled.fillna(0)
X_np = X_scaled.to_numpy(dtype=float)
y_np = y.to_numpy(dtype=float).flatten().astype(int)
n = X_np.shape[0]
data_jax = {"X": jnp.array(X_np, dtype=jnp.float64), "y": jnp.array(y_np, dtype=jnp.float64)}

# ── Load ground truth ──
cache_dir = Path(__file__).parent / ".cache"
with open(cache_dir / "ovarian_loo_groundtruth.pkl", "rb") as f:
    gt_data = pickle.load(f)

gt_elpd = np.zeros(n)
gt_prob = np.zeros(n)
for i in range(n):
    fold = gt_data["folds"].get(i)
    if fold is None:
        gt_elpd[i] = gt_prob[i] = np.nan
        continue
    beta, beta0 = fold["beta"], fold["beta0"]
    logits = beta0 + beta @ X_np[i]
    ll_i = y_np[i] * logits - np.logaddexp(0, logits)
    gt_elpd[i] = float(jax.scipy.special.logsumexp(jnp.array(ll_i)) - np.log(len(ll_i)))
    gt_prob[i] = float(jnp.mean(jax.nn.sigmoid(jnp.array(logits))))

gt_total = np.nansum(gt_elpd)
gt_ploo = np.exp(gt_elpd)
gt_auc = roc_auc_score(y_np, gt_prob)
gt_se = np.sqrt(n) * np.nanstd(gt_elpd)
print(f"Ground truth: elpd={gt_total:.2f} ± {gt_se:.2f}, AUC={gt_auc:.4f}")

# ── Load Stan samples ──
with open(cache_dir / "ovarian_lr_stan_samples.pkl", "rb") as f:
    stan = pickle.load(f)

N_SAMPLES = 1000
rng = np.random.default_rng(42)
idx = rng.choice(stan["beta"].shape[0], size=N_SAMPLES, replace=False)

params = {
    "beta": jnp.array(stan["beta"][idx], dtype=jnp.float64),
    "intercept": jnp.array(stan["beta0"][idx, None], dtype=jnp.float64),
}

likelihood_fn = LogisticRegressionLikelihood(dtype=jnp.float64)
ais_sampler = AdaptiveImportanceSampler(likelihood_fn=likelihood_fn)
rhos = [2**-r for r in range(1, 9)]
transforms = ["identity", "ll", "kl", "var", "pmm1", "pmm2", "pmm3"]

print(f"Running AIS: s={N_SAMPLES}, {len(transforms)} transforms, {len(rhos)} rhos")

results_ais = ais_sampler.adaptive_is_loo(
    data=data_jax, params=params,
    rhos=rhos, variational=False,
    transformations=transforms,
)

# ── For each base method, find best rho and extract p_loo + weights ──
method_bases = set()
for key in results_ais:
    if key in ("best", "timings") or not isinstance(results_ais[key], dict):
        continue
    method_bases.add(key.split("_rho")[0])

best = {}
for base in sorted(method_bases):
    matching = {k: v for k, v in results_ais.items()
                if (k == base or k.startswith(base + "_rho"))
                and isinstance(v, dict) and "khat" in v}
    if not matching:
        continue
    khats = np.stack([np.array(v["khat"]) for v in matching.values()])
    best_rho_idx = np.argmin(khats, axis=0)
    keys_list = list(matching.keys())

    best_khat = np.min(khats, axis=0)
    best_ploo = np.zeros(n)
    # Also get the PSIS weights and raw likelihoods for IS variance
    best_psis_weights = np.zeros((N_SAMPLES, n))
    best_ell = np.zeros((N_SAMPLES, n))

    for i_obs in range(n):
        bk = keys_list[best_rho_idx[i_obs]]
        best_ploo[i_obs] = matching[bk]["ll_loo_psis"][i_obs]
        if "psis_weights" in matching[bk]:
            best_psis_weights[:, i_obs] = np.array(matching[bk]["psis_weights"][:, i_obs])
        if "log_ell_new" in matching[bk]:
            best_ell[:, i_obs] = np.exp(np.array(matching[bk]["log_ell_new"][:, i_obs]))

    best[base] = {"khat": best_khat, "ploo": best_ploo,
                   "psis_weights": best_psis_weights, "ell": best_ell}

# ── Compute IS standard error for each p_loo ──
# IS variance: Var_IS[f] ≈ sum(w_k^2 * (f_k - f_hat)^2) for normalized w
# But a simpler approach: use the pointwise elpd variance for the total se
id_khat = best["identity"]["khat"]
bad_mask = id_khat >= 0.7
bad_idx = np.where(bad_mask)[0]
good_idx = np.where(~bad_mask)[0]
n_bad = len(bad_idx)
id_ploo = best["identity"]["ploo"]

print(f"\nn_bad={n_bad}, n_good={len(good_idx)}")
print()

# ── Build table ──
header = f"{'Method':<20s} {'Adapted':>8s} {'Total elpd':>20s} {'Diff vs GT':>20s} {'RMSE_p':>12s} {'AUC':>12s}"
print(header)
print("-" * len(header))

# Ground truth row
print(f"{'Ground Truth':<20s} {'--':>8s} ${gt_total:.2f} \\pm {gt_se:.2f}${'':>5s} {'--':>20s} {'--':>12s} ${gt_auc:.3f}${'':>5s}")

for method in ["identity", "ll", "pmm1", "pmm2", "pmm3", "kl", "var", "combined"]:
    if method == "combined":
        # Combined: PMM+LL+KL+Var, pick lowest khat per obs
        comb_methods = ["pmm1", "pmm2", "pmm3", "ll", "kl", "var"]
        avail = [m for m in comb_methods if m in best]
        ks = np.stack([best[m]["khat"] for m in avail])
        bi = np.argmin(ks, axis=0)
        bk = np.min(ks, axis=0)
        ploo = id_ploo.copy()
        for i in bad_idx:
            if bk[i] < 0.7:
                ploo[i] = best[avail[bi[i]]]["ploo"][i]
        khat = bk
        n_ad = int(np.sum(bk[bad_idx] < 0.7))
        label = "PMM+LL+KL+Var"
    elif method not in best:
        continue
    else:
        mk = best[method]["khat"]
        mp = best[method]["ploo"]
        # Use method where khat<0.7 on bad obs, else identity
        ploo = id_ploo.copy()
        adapted = mk[bad_idx] < 0.7
        ploo[bad_idx[adapted]] = mp[bad_idx[adapted]]
        khat = mk
        n_ad = int(np.sum(adapted))
        label = {"identity": "Identity (PSIS)", "ll": "LL", "pmm1": "PMM1",
                 "pmm2": "PMM2", "pmm3": "PMM3", "kl": "KL", "var": "Var"
                 }.get(method, method)

    ploo_c = np.clip(ploo, 1e-100, None)
    elpd_vec = np.log(ploo_c)
    total_elpd = np.sum(elpd_vec)
    # SE of total elpd: sqrt(n) * sd(elpd_i)
    se_elpd = np.sqrt(n) * np.std(elpd_vec)
    diff = total_elpd - gt_total
    se_diff = se_elpd  # same se (GT is fixed)

    # RMSE on adapted bad obs
    if method == "combined":
        ad_mask = bk[bad_idx] < 0.7
    elif method == "identity":
        ad_mask = np.zeros(n_bad, dtype=bool)
    else:
        ad_mask = best[method]["khat"][bad_idx] < 0.7

    if np.sum(ad_mask) > 0:
        rmse = np.sqrt(np.mean((ploo[bad_idx[ad_mask]] - gt_ploo[bad_idx[ad_mask]])**2))
    else:
        rmse = np.nan

    # AUC
    prob_y1 = np.where(y_np == 1, ploo_c, 1 - ploo_c)
    auc = roc_auc_score(y_np, np.clip(prob_y1, 1e-10, 1 - 1e-10))

    if method == "identity":
        ad_str = f"0/{n_bad}"
    else:
        ad_str = f"{n_ad}/{n_bad}"

    rmse_str = f"${rmse:.3f}$" if not np.isnan(rmse) else "--"
    print(f"{label:<20s} {ad_str:>8s} ${total_elpd:.2f} \\pm {se_elpd:.2f}$ "
          f"${diff:+.2f} \\pm {se_diff:.2f}$ "
          f"{rmse_str:>12s} ${auc:.3f}$")

# Save
with open(cache_dir / "ovarian_loo_validation_sims.pkl", "wb") as f:
    pickle.dump({"best": {m: {"khat": best[m]["khat"], "ploo": best[m]["ploo"]} for m in best},
                 "gt_total": gt_total, "gt_se": gt_se, "gt_auc": gt_auc,
                 "gt_elpd": gt_elpd, "gt_ploo": gt_ploo}, f)
print(f"\nSaved results")

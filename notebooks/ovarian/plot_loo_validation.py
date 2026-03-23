#!/usr/bin/env python3
"""
Tufte-style validation: IS-LOO p(y_i|D_{-i}) vs ground truth
for observations needing adaptation (khat >= 0.7).

Each panel = one method. Black ticks = ground truth. Colored dots = IS-LOO.
Annotated with mean ± sd of the error.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score

cache_dir = Path(__file__).parent / ".cache"
fig_dir = Path("/home/josh/workspace/loo_auc/figures")

with open(cache_dir / "ovarian_loo_comparison.pkl", "rb") as f:
    comp = pickle.load(f)

gt_elpd = comp["gt_elpd"]
gt_prob = comp["gt_prob"]
gt_auc = comp["gt_auc"]
gt_total = comp["gt_total_elpd"]
is_loo = comp["is_loo"]

import importlib.resources, pandas as pd
with importlib.resources.path("bayesianquilts.data", "overiany.csv") as ypath:
    y_np = pd.read_csv(ypath, header=None).to_numpy().flatten().astype(int)

# Observations needing adaptation
identity_khat = is_loo["identity"]["khat"]
bad_mask = identity_khat >= 0.7
bad_idx = np.where(bad_mask)[0]
n_bad = len(bad_idx)

# Ground truth p(y_i|D_{-i}) at bad obs
gt_ploo_bad = np.exp(gt_elpd[bad_idx])

# Sort by GT
sort_order = np.argsort(gt_ploo_bad)
sorted_bad_idx = bad_idx[sort_order]
gt_sorted = gt_ploo_bad[sort_order]

# For each method, only use its estimate when khat < 0.7
# Otherwise fall back to identity (standard PSIS)
def get_adapted_ploo(method_name):
    """Get p_loo using method only where khat < 0.7, else identity."""
    if method_name not in is_loo:
        return is_loo["identity"]["ll_loo_psis"].copy()
    khat = is_loo[method_name]["khat"]
    ploo = is_loo[method_name]["ll_loo_psis"].copy()
    # Where khat >= 0.7, fall back to identity
    fallback = khat >= 0.7
    ploo[fallback] = is_loo["identity"]["ll_loo_psis"][fallback]
    return ploo

# Combined: for each obs, pick method with lowest khat (among khat<0.7)
ours = ["pmm1", "pmm2", "ll", "kl", "nkl", "nll", "var"]
available = [m for m in ours if m in is_loo]
khats_all = np.stack([is_loo[m]["khat"] for m in available])
best_method_idx = np.argmin(khats_all, axis=0)
best_khat = np.min(khats_all, axis=0)

combined_ploo = np.zeros(len(gt_elpd))
for i in range(len(gt_elpd)):
    if best_khat[i] < 0.7:
        combined_ploo[i] = is_loo[available[best_method_idx[i]]]["ll_loo_psis"][i]
    else:
        # No method achieved khat < 0.7; use best available anyway
        combined_ploo[i] = is_loo[available[best_method_idx[i]]]["ll_loo_psis"][i]

methods = ["identity", "ll", "pmm1", "kl", "combined"]
labels = {"identity": "Identity (PSIS)", "ll": "LL", "pmm1": "PMM1",
          "kl": "KL", "combined": "Ours Combined"}
colors = {"identity": "#888888", "ll": "#0072B2", "pmm1": "#D55E00",
          "kl": "#CC79A7", "combined": "#E69F00"}

# ---- Figure ----
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8,
    "axes.linewidth": 0.4, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150,
})

n_panels = len(methods)
fig, axes = plt.subplots(n_panels, 1, figsize=(5, 0.75 * n_panels + 0.8),
                          sharex=True, gridspec_kw={"hspace": 0.08})

y_pos = np.arange(n_bad)

for ax_idx, method in enumerate(methods):
    ax = axes[ax_idx]

    if method == "combined":
        is_ploo = combined_ploo[sorted_bad_idx]
        method_khat = best_khat[sorted_bad_idx]
    elif method == "identity":
        is_ploo = is_loo[method]["ll_loo_psis"][sorted_bad_idx]
        method_khat = is_loo[method]["khat"][sorted_bad_idx]
    else:
        is_ploo = is_loo[method]["ll_loo_psis"][sorted_bad_idx]
        method_khat = is_loo[method]["khat"][sorted_bad_idx]

    color = colors[method]
    label = labels[method]

    # Mask: only show estimates where this method has khat < 0.7
    good = method_khat < 0.7
    n_good = int(np.sum(good))

    # GT as short horizontal tick marks
    for i in range(n_bad):
        ax.plot([gt_sorted[i], gt_sorted[i]], [i - 0.35, i + 0.35],
                color="0.25", linewidth=0.6, zorder=3, solid_capstyle="butt")

    # IS-LOO: solid dots where khat < 0.7, faded x where not
    if n_good > 0:
        ax.scatter(is_ploo[good], y_pos[good], s=14, color=color, zorder=4,
                   marker="o", edgecolors="none", alpha=0.85)
    if np.sum(~good) > 0:
        ax.scatter(is_ploo[~good], y_pos[~good], s=8, color=color, zorder=2,
                   marker="x", linewidths=0.4, alpha=0.25)

    # Error stats only for observations where khat < 0.7
    if n_good > 0:
        err = is_ploo[good] - gt_sorted[good]
        mean_err = np.mean(err)
        rmse = np.sqrt(np.mean(err**2))
    else:
        mean_err = np.nan
        rmse = np.nan

    # Annotation
    ax.text(0.02, 0.92, f"{label}", transform=ax.transAxes, fontsize=7.5,
            va="top", ha="left", color=color, fontweight="bold")
    if n_good > 0 and not np.isnan(rmse):
        ax.text(0.02, 0.50,
                f"adapted: {n_good}/{n_bad}\n"
                f"bias: {mean_err:+.3f}  rmse: {rmse:.3f}",
                transform=ax.transAxes, fontsize=5.5, va="top", ha="left",
                color="0.4", family="monospace")
    else:
        ax.text(0.02, 0.50, f"adapted: 0/{n_bad}",
                transform=ax.transAxes, fontsize=5.5, va="top", ha="left",
                color="0.4", family="monospace")

    ax.set_yticks([])
    ax.set_xlim(-0.05, 1.05)
    ax.spines["left"].set_visible(False)
    if ax_idx < n_panels - 1:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel(r"$p(y_i \mid \mathcal{D}_{-i})$", fontsize=9)
    ax.tick_params(axis="x", length=2, labelsize=7)

fig.suptitle(
    f"IS-LOO vs exact LOO ($n={n_bad}$ obs with "
    + r"$\hat{k}\geq 0.7$" + ")",
    fontsize=9, y=1.01,
)

# Legend
axes[0].plot([], [], color="0.25", linewidth=1, label="Ground truth")
axes[0].scatter([], [], s=14, color=colors["ll"], marker="o", label=r"IS-LOO ($\hat{k}<0.7$)")
axes[0].scatter([], [], s=8, color=colors["ll"], marker="x", linewidths=0.4,
                alpha=0.3, label=r"IS-LOO ($\hat{k}\geq 0.7$)")
axes[0].legend(loc="lower right", fontsize=6, frameon=False,
               handletextpad=0.3, borderpad=0.2)

out = fig_dir / "loo_validation_forest.pdf"
fig.savefig(out, bbox_inches="tight", dpi=300)
print(f"Saved: {out}")
plt.close()

# ---- Summary table ----
print(f"\n{'Method':<20s} {'elpd':>8s} {'diff':>8s} {'RMSE_p':>8s} {'AUC':>6s}")
print("-" * 55)
print(f"{'Ground Truth':<20s} {gt_total:>8.2f} {'--':>8s} {'--':>8s} {gt_auc:>6.4f}")

for method in methods:
    if method == "combined":
        ploo = combined_ploo
    else:
        ploo = is_loo[method]["ll_loo_psis"]
    ploo_c = np.clip(ploo, 1e-100, None)
    elpd = np.sum(np.log(ploo_c))
    diff = elpd - gt_total
    rmse_p = np.sqrt(np.mean((ploo_c[bad_idx] - np.exp(gt_elpd[bad_idx]))**2))
    prob_y1 = np.where(y_np == 1, ploo_c, 1 - ploo_c)
    auc = roc_auc_score(y_np, np.clip(prob_y1, 1e-10, 1 - 1e-10))
    print(f"{labels[method]:<20s} {elpd:>8.2f} {diff:>+8.2f} {rmse_p:>8.4f} {auc:>6.4f}")

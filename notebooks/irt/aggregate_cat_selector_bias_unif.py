#!/usr/bin/env python3
"""Aggregate CAT selector bias trajectories under B_i^unif (uniform-on-[-3,+3] integration).

This is a sibling of aggregate_cat_selector_bias.py.  The original aggregator
reads pre-aggregated mean_B from the per-dataset JSONs, which were computed under
the empirical theta distribution.  This script reads the raw per-respondent item
trajectories from the NPZ files and looks up B_i^unif from the bias_leverage CSVs,
so the headline metric rho is under the uniform prior on [-3, +3].

Key metric change:
  OLD: rho^emp  = mean_B_admin^emp  / bank_mean_B^emp
  NEW: rho^unif = mean_B_admin^unif / bank_mean_B^unif

For each (dataset, selector, length):
  1. Open <dataset>_trajectories.npz and read <sel>_item_idx  (n_resp x max_length).
  2. Take the first `length` columns as the administered subset per respondent.
  3. Look up B_i^unif for each item from <dataset>_B_unif.csv (column B_i_unif,
     indexed by item name via item_keys).
  4. Compute per-respondent mean_B_unif_admin = mean(B_i^unif) over administered items.
  5. Aggregate across respondents: mean_B_unif = mean over respondents.
  6. bank_mean_B_unif = simple mean of B_i^unif over all items in the bank.
  7. rho_unif = mean_B_unif / bank_mean_B_unif.
  8. pct_admin_unif = ECDF_bank(mean_B_unif) -- percentile of mean_B_unif against
     the bank B_i^unif ECDF.

Writes:
  notebooks/irt/cat_selector_bias_results_unif.json
  /tmp/topk_under_B_unif.md

Usage:
    python aggregate_cat_selector_bias_unif.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

IRT_DIR = Path(__file__).parent.resolve()
TRAJ_DIR = IRT_DIR / "cat_selector_bias"
BIAS_DIR = IRT_DIR / "bias_leverage"
OUT_JSON = IRT_DIR / "cat_selector_bias_results_unif.json"
OUT_MD = Path("/tmp/topk_under_B_unif.md")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SELECTOR_LABELS = {
    "fisher":               "Fisher-info",
    "bayesian_fisher":      "Bayes-Fisher",
    "global_info":          "Global-info",
    "bayesian_variance":    "Bayes-Var",
    "entropy":              "Entropy (k=1)",
    "stochastic_entropy":   "Stoch-Entropy (k=inf)",
    "stoch_entropy_top3":   "Stoch-Entropy (k=3)",
    "stoch_entropy_top5":   "Stoch-Entropy (k=5)",
    "stoch_entropy_top10":  "Stoch-Entropy (k=10)",
}

ALL_SELECTORS = list(SELECTOR_LABELS.keys())

ALL_DATASETS = [
    "promis_np__global_health",
    "promis_w1__fatigue_experience",
    "promis_w1__fatigue_impact",
    "promis_w1__physical_function_a",
    "promis_substance_use",
    # Tankie datasets (synced via rsync)
    "promis_w1__alcohol_use",
    "promis_w1__anger",
    "promis_w1__anxiety",
    "promis_w1__depression",
    # Pain datasets (run locally -- long, started after main batch)
    "promis_np__pain_behavior",
    "promis_np__pain_interference",
]


# ---------------------------------------------------------------------------
# ECDF utility
# ---------------------------------------------------------------------------

def ecdf_percentile(value, bank_values):
    """Return fraction of bank_values that are <= value (standard ECDF)."""
    return float(np.mean(bank_values <= value))


# ---------------------------------------------------------------------------
# Per-dataset computation
# ---------------------------------------------------------------------------

def compute_dataset(dataset, lengths=(4, 6, 8, 10)):
    """Compute rho_unif and pct_admin_unif for all available (selector, length) pairs."""
    traj_path = TRAJ_DIR / f"{dataset}_trajectories.npz"
    bias_path = BIAS_DIR / f"{dataset}_B_unif.csv"

    if not traj_path.exists():
        print(f"  MISSING trajectories: {traj_path}")
        return []
    if not bias_path.exists():
        print(f"  MISSING B_unif CSV: {bias_path}")
        return []

    # Use allow_pickle=False -- only numeric arrays, no object arrays needed
    npz = np.load(str(traj_path), allow_pickle=True)
    df_bias = pd.read_csv(str(bias_path))

    # Build item_name -> B_i_unif lookup
    item_keys = list(npz["item_keys"])          # list of str, length n_items
    n_items = len(item_keys)

    bi_by_name = dict(zip(df_bias["item_name"], df_bias["B_i_unif"]))
    b_unif = np.array([bi_by_name[k] for k in item_keys], dtype=float)

    bank_mean_b_unif = float(np.mean(b_unif))
    bank_b_unif = b_unif.copy()

    rows = []
    for sel in ALL_SELECTORS:
        idx_key = f"{sel}_item_idx"
        if idx_key not in npz:
            continue

        item_idx = np.asarray(npz[idx_key], dtype=int)  # (n_resp, max_length)
        n_resp = item_idx.shape[0]

        for length in lengths:
            if length > item_idx.shape[1]:
                continue

            # Administered items: first `length` columns
            admin_idx = item_idx[:, :length]     # (n_resp, length)
            admin_b = b_unif[admin_idx]          # (n_resp, length) via advanced indexing
            per_resp_mean_b = admin_b.mean(axis=1)   # (n_resp,)
            mean_b_unif = float(per_resp_mean_b.mean())

            rho_unif = (mean_b_unif / bank_mean_b_unif
                        if bank_mean_b_unif > 1e-12 else float("nan"))
            pct_admin_unif = ecdf_percentile(mean_b_unif, bank_b_unif)

            rows.append({
                "dataset": dataset,
                "selector": sel,
                "length": length,
                "mean_B_unif": mean_b_unif,
                "bank_mean_B_unif": bank_mean_b_unif,
                "rho_unif": rho_unif,
                "pct_admin_unif": pct_admin_unif,
                "n_items": n_items,
                "n_resp": n_resp,
            })

    return rows


# ---------------------------------------------------------------------------
# Cross-dataset summary
# ---------------------------------------------------------------------------

def build_summary(all_rows, lengths=(4, 6, 8, 10)):
    """Aggregate rows to (selector, length) level by averaging across datasets."""
    summary = {}
    for sel in ALL_SELECTORS:
        for length in lengths:
            subset = [
                rw for rw in all_rows
                if rw["selector"] == sel and rw["length"] == length
            ]
            if not subset:
                continue
            rhos = [rw["rho_unif"] for rw in subset if np.isfinite(rw["rho_unif"])]
            pcts = [rw["pct_admin_unif"] for rw in subset if np.isfinite(rw["pct_admin_unif"])]
            summary[(sel, length)] = {
                "avg_rho_unif": float(np.mean(rhos)) if rhos else float("nan"),
                "avg_pct_unif": float(np.mean(pcts)) if pcts else float("nan"),
                "n_datasets": len(subset),
                "datasets": [rw["dataset"] for rw in subset],
            }
    return summary


# ---------------------------------------------------------------------------
# Markdown headline table
# ---------------------------------------------------------------------------

def format_headline_md(all_rows, summary):
    """Format the headline Markdown table at t=4 and t=10 with rho_unif."""
    lines = []
    lines.append("# CAT Selector Top-k Sweep: rho under B_i^unif")
    lines.append("")
    lines.append(
        "**rho^unif** = mean_B_admin^unif / bank_mean_B^unif, where B_i^unif "
        "is computed by integrating the per-item bias discrepancy over a "
        "uniform prior on theta in [-3, +3] (not the empirical theta distribution)."
    )
    lines.append("")
    lines.append("Verdict criterion: rho^unif >= 1.0 AND adequate exposure.")
    lines.append("")

    for headline_t in [4, 10]:
        lines.append(f"## Headline table at t={headline_t}")
        lines.append("")
        lines.append(
            "| Selector | avg rho^unif | avg B_i-pct^unif | N datasets | rho>=1.0 |"
        )
        lines.append("| --- | ---: | ---: | ---: | :---: |")

        for sel in ALL_SELECTORS:
            lbl = SELECTOR_LABELS.get(sel, sel)
            key = (sel, headline_t)
            if key not in summary:
                lines.append(f"| {lbl} | -- | -- | -- | -- |")
                continue
            s = summary[key]
            rho = s["avg_rho_unif"]
            pct = s["avg_pct_unif"]
            n = s["n_datasets"]
            rho_str = f"{rho:.3f}" if np.isfinite(rho) else "--"
            pct_str = f"{pct:.2f}" if np.isfinite(pct) else "--"
            meets = "YES" if (np.isfinite(rho) and rho >= 1.0) else "no"
            lines.append(f"| {lbl} | {rho_str} | {pct_str} | {n} | {meets} |")

        lines.append("")

    # Per-length rho^unif table
    lines.append("## rho^unif by length (all datasets averaged)")
    lines.append("")
    lines.append("rho^unif = 1.0 corresponds to a uniform random draw from the bank.")
    lines.append("")
    lines.append("| Selector | t=4 | t=6 | t=8 | t=10 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for sel in ALL_SELECTORS:
        lbl = SELECTOR_LABELS.get(sel, sel)
        cells = []
        for tl in [4, 6, 8, 10]:
            key = (sel, tl)
            if key in summary and np.isfinite(summary[key]["avg_rho_unif"]):
                cells.append(f"{summary[key]['avg_rho_unif']:.3f}")
            else:
                cells.append("--")
        lines.append(f"| {lbl} | " + " | ".join(cells) + " |")
    lines.append("")

    # Per-dataset detail at t=4 and t=10
    for headline_t in [4, 10]:
        lines.append(f"## Per-dataset detail at t={headline_t}")
        lines.append("")
        lines.append(
            "| Dataset | n_items | Selector | mean_B_unif | bank_mean_B_unif"
            " | rho^unif | pct^unif |"
        )
        lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
        t_rows = [rw for rw in all_rows if rw["length"] == headline_t]
        for ds in ALL_DATASETS:
            for sel in ALL_SELECTORS:
                rw = next(
                    (r for r in t_rows if r["dataset"] == ds and r["selector"] == sel),
                    None,
                )
                if rw is None:
                    continue
                lbl = SELECTOR_LABELS.get(sel, sel)
                lines.append(
                    f"| {ds} | {rw['n_items']} | {lbl} | "
                    f"{rw['mean_B_unif']:.4f} | {rw['bank_mean_B_unif']:.4f} | "
                    f"{rw['rho_unif']:.3f} | {rw['pct_admin_unif']:.2f} |"
                )
        lines.append("")

    # Verdict paragraph slot
    lines.append("## Verdict note")
    lines.append("")

    top_k_sels = ["stoch_entropy_top3", "stoch_entropy_top5", "stoch_entropy_top10"]
    meets_at_4 = {}
    meets_at_10 = {}
    for sel in top_k_sels:
        r4 = summary.get((sel, 4), {}).get("avg_rho_unif", float("nan"))
        r10 = summary.get((sel, 10), {}).get("avg_rho_unif", float("nan"))
        meets_at_4[sel] = (np.isfinite(r4) and r4 >= 1.0)
        meets_at_10[sel] = (np.isfinite(r10) and r10 >= 1.0)

    any_meets_4 = any(meets_at_4.values())
    any_meets_10 = any(meets_at_10.values())

    if any_meets_4:
        winners_4 = ", ".join(SELECTOR_LABELS[s] for s in top_k_sels if meets_at_4[s])
        lines.append(
            f"Top-k stochastic-entropy family meets rho^unif >= 1.0 at t=4: "
            f"YES ({winners_4})."
        )
    else:
        lines.append(
            "Top-k stochastic-entropy family meets rho^unif >= 1.0 at t=4: "
            "NO (all below 1.0)."
        )

    if any_meets_10:
        winners_10 = ", ".join(SELECTOR_LABELS[s] for s in top_k_sels if meets_at_10[s])
        lines.append(
            f"Top-k stochastic-entropy family meets rho^unif >= 1.0 at t=10: "
            f"YES ({winners_10})."
        )
    else:
        lines.append(
            "Top-k stochastic-entropy family meets rho^unif >= 1.0 at t=10: "
            "NO (all below 1.0)."
        )

    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_rows = []
    for ds in ALL_DATASETS:
        print(f"Processing {ds} ...")
        rows = compute_dataset(ds, lengths=[4, 6, 8, 10])
        print(f"  -> {len(rows)} (selector, length) pairs computed")
        all_rows.extend(rows)

    summary = build_summary(all_rows, lengths=[4, 6, 8, 10])

    # Save full JSON
    out = {
        "datasets": ALL_DATASETS,
        "rows": all_rows,
        "summary": {
            f"{sel}_t{length}": v
            for (sel, length), v in summary.items()
        },
    }
    with open(str(OUT_JSON), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Full JSON -> {OUT_JSON}")

    # Write headline Markdown
    md = format_headline_md(all_rows, summary)
    with open(str(OUT_MD), "w") as f:
        f.write(md)
    print(f"Headline table -> {OUT_MD}")

    # Print summary to stdout
    print()
    print("=" * 70)
    print("HEADLINE: rho^unif at t=4 and t=10 (averaged across datasets)")
    print("=" * 70)
    for headline_t in [4, 10]:
        print(f"\n  t={headline_t}:")
        print(f"  {'Selector':<30} {'rho^unif':>10} {'pct^unif':>10} {'N':>4} {'>=1.0':>6}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*4} {'-'*6}")
        for sel in ALL_SELECTORS:
            lbl = SELECTOR_LABELS.get(sel, sel)
            key = (sel, headline_t)
            if key not in summary:
                print(f"  {lbl:<30} {'--':>10} {'--':>10} {'--':>4} {'--':>6}")
                continue
            s = summary[key]
            rho = s["avg_rho_unif"]
            pct = s["avg_pct_unif"]
            n = s["n_datasets"]
            rho_str = f"{rho:.3f}" if np.isfinite(rho) else "--"
            pct_str = f"{pct:.2f}" if np.isfinite(pct) else "--"
            meets = "YES" if (np.isfinite(rho) and rho >= 1.0) else "no"
            print(f"  {lbl:<30} {rho_str:>10} {pct_str:>10} {n:>4} {meets:>6}")

    print()
    print("VERDICT (top-k family under B_i^unif):")
    top_k_sels = ["stoch_entropy_top3", "stoch_entropy_top5", "stoch_entropy_top10"]
    for sel in top_k_sels:
        lbl = SELECTOR_LABELS[sel]
        vals = []
        for tl in [4, 6, 8, 10]:
            key = (sel, tl)
            rho = summary.get(key, {}).get("avg_rho_unif", float("nan"))
            vals.append(f"t={tl}: {rho:.3f}" if np.isfinite(rho) else f"t={tl}: --")
        print(f"  {lbl}: " + ", ".join(vals))


if __name__ == "__main__":
    main()

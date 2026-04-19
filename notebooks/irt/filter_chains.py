#!/usr/bin/env python
"""Filter MCMC chains that are stuck in outlier local modes.

Greedy: iteratively drop the chain whose removal most reduces max R-hat, until
max R-hat <= threshold (default 1.1) or only 1 chain remains.

Saves filtered NPZ as {stem}_filtered.npz and prints a report. Pass --overwrite
to replace the original NPZ.

Usage:
    uv run python filter_chains.py --path notebooks/irt/grit/mcmc_samples/mcmc_baseline.npz
    uv run python filter_chains.py --path '...' --overwrite
"""

import argparse
import os
from pathlib import Path

import numpy as np


SKIP_KEYS = {'eap', 'psd', 'standardize_mu', 'standardize_sigma',
             'eap_standardized', 'psd_standardized'}


def rhat(s):
    flat = s.reshape(s.shape[0], s.shape[1], -1)
    cm = flat.mean(axis=1)
    bv = cm.var(axis=0, ddof=1)
    wv = flat.var(axis=1, ddof=1).mean(axis=0)
    n = flat.shape[1]
    return np.sqrt(((n - 1) / n * wv + bv) / np.maximum(wv, 1e-30))


def max_rhat_of(keys, subset_npz, chain_idx):
    max_r = 0.0
    for k in keys:
        s = subset_npz[k][chain_idx]
        if s.shape[0] < 2:
            continue
        r = rhat(s)
        max_r = max(max_r, float(np.max(r)))
    return max_r


def filter_chains(path: Path, threshold: float = 1.1, overwrite: bool = False):
    z = dict(np.load(str(path)))
    param_keys = [k for k in z
                  if k not in SKIP_KEYS
                  and hasattr(z[k], 'shape') and len(z[k].shape) >= 3]
    if not param_keys:
        print(f"  {path}: no param samples found, skipping")
        return

    num_chains = z[param_keys[0]].shape[0]
    remaining = list(range(num_chains))

    # Initial R-hat
    init_r = max_rhat_of(param_keys, z, remaining)
    print(f"  {path.relative_to(path.parents[2]) if len(path.parents)>=2 else path}")
    print(f"    {num_chains} chains, initial max R-hat = {init_r:.3f}")

    if init_r <= threshold:
        print(f"    Already converged, no filtering needed.")
        return

    dropped = []
    while len(remaining) > 1:
        # Try removing each remaining chain and see which removal gives smallest max R-hat
        best_r = float('inf')
        best_drop = None
        for c in remaining:
            subset = [x for x in remaining if x != c]
            r = max_rhat_of(param_keys, z, subset)
            if r < best_r:
                best_r = r
                best_drop = c
        dropped.append(best_drop)
        remaining.remove(best_drop)
        print(f"    Drop chain {best_drop} → max R-hat = {best_r:.3f} (keeping {len(remaining)} chains)")
        if best_r <= threshold:
            break

    if max_rhat_of(param_keys, z, remaining) > threshold:
        print(f"    !! Cannot achieve R-hat <= {threshold} even with 1 chain; something else is wrong.")
        return

    # Save filtered NPZ
    filtered = {}
    for k, v in z.items():
        if k in param_keys:
            filtered[k] = v[remaining]
        else:
            filtered[k] = v

    if overwrite:
        out = path
    else:
        out = path.with_name(path.stem + '_filtered.npz')
    np.savez(str(out), **filtered)
    print(f"    Kept chains {remaining}, dropped {dropped}")
    print(f"    Saved to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--path', required=True)
    p.add_argument('--threshold', type=float, default=1.1)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    filter_chains(Path(args.path), args.threshold, args.overwrite)


if __name__ == '__main__':
    main()

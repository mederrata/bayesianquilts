#!/usr/bin/env python
"""Re-center an ADVI grm_baseline's surrogate 'loc' fields at a baseline
MCMC posterior mean.

This is a pre-processing wrapper: it takes an existing ``grm_baseline/``
(ADVI-fit model directory with ``params.h5``) and a baseline MCMC NPZ
(output of ``run_marginal_mcmc.py --variants baseline``), and writes a
new model directory whose ``params.h5`` has each surrogate ``\\loc``
parameter re-centered at the per-element posterior mean.  Surrogate
``\\scale`` parameters are shrunk (default 0.1x) so that a subsequent
MCMC run perturbs minimally around this re-centered point.

The downstream ``run_marginal_mcmc.py --model-dir <output_dir>`` consumes
the new directory unchanged — no CLI changes required.

Usage:
    uv run python init_from_baseline_posterior.py \\
        --dataset rwa \\
        --baseline-npz rwa/mcmc_samples/mcmc_baseline.npz \\
        --output-dir rwa/grm_baseline_bposterior
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


# Bijectors that may appear in the param key path.  The inverse maps
# constrained-space values (as produced by MCMC) back to the
# unconstrained-space ``loc`` parameter of the surrogate.
def _inv_bijector(name, x):
    if name == 'identity':
        return x
    if name == 'softplus':
        # softplus inverse: log(exp(x)-1).  Use numerically-stable form
        # log(expm1(x)) — but guard against x <= 0 which MCMC won't
        # produce for softplus-transformed variables.
        x = np.clip(x, 1e-12, None)
        # For x >= ~35, softplus(u) ≈ u, so inv ≈ x.
        # For x < ~35, use log(expm1(x)).
        return np.where(x > 35.0, x, np.log(np.expm1(x)))
    # Unknown — leave unchanged and warn.
    print(f"  WARNING: unknown bijector '{name}', leaving value unchanged",
          file=sys.stderr)
    return x


def _parse_key(key):
    """Parse a surrogate params key of form '<var>\\<bij>\\<dist>\\<field>'.

    Returns (var, bijector, dist, field) or None if the key doesn't
    conform.
    """
    parts = key.split('\\')
    if len(parts) != 4:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def recenter_params(output_dir, baseline_npz_path, scale_shrink=0.1):
    """Re-center loc fields in ``output_dir/params.h5`` at the posterior
    mean of the given baseline MCMC NPZ.  Also shrinks scale fields.
    """
    output_dir = Path(output_dir)
    params_path = output_dir / 'params.h5'
    npz = np.load(baseline_npz_path)

    # Index NPZ arrays by var name.  MCMC samples have shape
    # (num_chains, num_samples, ...prior_shape...).
    npz_vars = {k: np.asarray(npz[k]) for k in npz.files}

    updated = []
    shrunk = []
    skipped = []

    with h5py.File(params_path, 'r+') as f:
        # Surrogate parameters live under the 'params' group.
        if 'params' not in f:
            raise ValueError(f"{params_path}: no 'params' group")
        grp = f['params']
        keys = list(grp.keys())

        # First pass: compute per-var posterior means (in unconstrained
        # space) once, so both loc and scale passes can use them.
        for key in keys:
            parsed = _parse_key(key)
            if parsed is None:
                continue
            var, bij, _dist, field = parsed
            if field != 'loc':
                continue
            if var not in npz_vars:
                skipped.append((var, 'no NPZ samples'))
                continue
            samples = npz_vars[var]
            if samples.ndim < 3:
                skipped.append((var, f'bad NPZ ndim {samples.ndim}'))
                continue
            # Collapse chain + sample axes → per-element posterior mean
            # in constrained space.
            flat = samples.reshape(-1, *samples.shape[2:])
            mean_constrained = np.mean(flat, axis=0)
            mean_unconstrained = _inv_bijector(bij, mean_constrained)

            current = grp[key][...]
            if mean_unconstrained.shape != current.shape:
                skipped.append((var,
                                f'shape mismatch {mean_unconstrained.shape} '
                                f'vs current {current.shape}'))
                continue

            mean_unconstrained = mean_unconstrained.astype(current.dtype)
            del grp[key]
            grp.create_dataset(key, data=mean_unconstrained)
            updated.append((var, mean_unconstrained.shape))

        # Second pass: shrink scale fields for updated vars.
        updated_vars = {v for v, _ in updated}
        for key in keys:
            parsed = _parse_key(key)
            if parsed is None:
                continue
            var, _bij, _dist, field = parsed
            if field not in ('scale', 'log_scale'):
                continue
            if var not in updated_vars:
                continue
            current = grp[key][...]
            if field == 'log_scale':
                new = current + np.log(scale_shrink)
            else:
                new = current * scale_shrink
            new = new.astype(current.dtype)
            del grp[key]
            grp.create_dataset(key, data=new)
            shrunk.append((var, field))

    print(f"Re-centered loc fields ({len(updated)}):")
    for var, shape in updated:
        print(f"  {var}: shape={shape}")
    if skipped:
        print(f"Skipped ({len(skipped)}):")
        for var, why in skipped:
            print(f"  {var}: {why}")
    print(f"Shrunk scale fields (by {scale_shrink}x, {len(shrunk)}):")
    for var, field in shrunk:
        print(f"  {var}\\{field}")


def main():
    parser = argparse.ArgumentParser(
        description='Re-center ADVI surrogate loc at baseline MCMC '
                    'posterior mean.'
    )
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (used only for default paths)')
    parser.add_argument('--baseline-npz', required=True,
                        help='Path to baseline MCMC NPZ '
                             '(e.g. rwa/mcmc_samples/mcmc_baseline.npz)')
    parser.add_argument('--output-dir', required=True,
                        help='Destination model dir (will be created, '
                             'overwritten if present)')
    parser.add_argument('--source-dir', default=None,
                        help='Source ADVI model dir to copy from. '
                             'Defaults to <dataset>/grm_baseline.')
    parser.add_argument('--scale-shrink', type=float, default=0.1,
                        help='Factor to shrink surrogate scale by (default 0.1)')
    args = parser.parse_args()

    source_dir = args.source_dir
    if source_dir is None:
        source_dir = os.path.join(args.dataset, 'grm_baseline')
    source_dir = Path(source_dir)
    output_dir = Path(args.output_dir)

    if not source_dir.exists():
        print(f"ERROR: source dir {source_dir} not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.baseline_npz):
        print(f"ERROR: baseline NPZ {args.baseline_npz} not found",
              file=sys.stderr)
        sys.exit(1)

    if output_dir.exists():
        print(f"Removing existing {output_dir}")
        shutil.rmtree(output_dir)
    print(f"Copying {source_dir} -> {output_dir}")
    shutil.copytree(source_dir, output_dir)

    print(f"Re-centering {output_dir}/params.h5 from {args.baseline_npz}")
    recenter_params(output_dir, args.baseline_npz,
                    scale_shrink=args.scale_shrink)
    print(f"\nDone. New model dir: {output_dir}")
    print(f"Use with: run_marginal_mcmc.py --model-dir {output_dir}")


if __name__ == '__main__':
    main()

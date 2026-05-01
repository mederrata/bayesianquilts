#!/usr/bin/env python
"""Refit ADVI baseline GRM with reorient=True, preserving the exact
hyperparameters already saved in grm_baseline/config.yaml.

Why this is necessary:
  The existing grm_baseline/ ADVI fits predate the reorient commits, so
  the surrogate posterior was trained against *unreversed* data. The
  MCMC pipeline now loads reoriented data, producing a warm-start
  mismatch (all NUTS transitions diverge).

What this script does:
  1. Load existing config.yaml (hyperparams, priors, scales).
  2. Reload the data with reorient=True (if supported by the loader).
  3. Refit ADVI using the *same* hyperparameters on the reoriented data.
  4. Save back to grm_baseline/.

Datasets without a reorient argument (grit) are a no-op: the original
file is left in place unless --force is given.

Usage:
    uv run python refit_advi_baseline.py --dataset grit
    uv run python refit_advi_baseline.py --dataset all
"""

import argparse
import inspect
import os
import sys
import shutil

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

from pathlib import Path

import numpy as np
import yaml


DATASETS = ['grit', 'tma', 'rwa', 'npi', 'wpi', 'eqsq',
            'promis_sleep', 'promis_substance_use']


def load_existing_config(baseline_dir: Path):
    cfg_path = baseline_dir / 'config.yaml'
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def refit(dataset_name, force=False,
          lr=5e-4, epochs=500, patience=30, batch_size=256):
    import importlib
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    from bayesianquilts.irt.grm import GRModel

    base_dir = Path.home() / 'workspace/bayesianquilts/notebooks/irt' / dataset_name
    baseline_dir = base_dir / 'grm_baseline'

    mod = importlib.import_module(f'bayesianquilts.data.{dataset_name}')
    item_keys = mod.item_keys
    K = mod.response_cardinality

    sig = inspect.signature(mod.get_data).parameters
    has_reorient = 'reorient' in sig
    if not has_reorient and not force:
        print(f"=== {dataset_name.upper()}: no reorient support, skipping (use --force to override)")
        return

    kw = {'polars_out': True}
    if has_reorient:
        kw['reorient'] = True
    df, N = mod.get_data(**kw)

    data = {col: df[col].to_numpy().astype(np.float32) for col in df.columns}
    data['person'] = np.arange(N, dtype=np.float32)

    existing = load_existing_config(baseline_dir)
    if existing is None:
        print(f"ERROR: no existing config at {baseline_dir}")
        return

    # Preserve the *exact* hyperparameters from the saved config
    model_kwargs = dict(
        item_keys=item_keys,
        num_people=N,
        dim=existing.get('dimensions', 1),
        response_cardinality=K,
        dtype=jnp.float64,
        discrimination_prior=existing.get('discrimination_prior', 'half_normal'),
        discrimination_prior_scale=existing.get('discrimination_prior_scale', 2.0),
        eta_scale=existing.get('eta_scale', 0.01),
        slab_df=existing.get('slab_df', 4),
        slab_scale=existing.get('slab_scale', 2.0),
        positive_discriminations=existing.get('positive_discriminations', True),
        parameterization=existing.get('parameterization', 'softplus'),
        weight_exponent=existing.get('weight_exponent', 1.0),
        full_rank=existing.get('full_rank', False),
        include_independent=existing.get('include_independent', False),
    )
    ks = existing.get('kappa_scale')
    if ks is not None:
        model_kwargs['kappa_scale'] = np.array(ks)
    print(f"\n=== {dataset_name.upper()}: N={N}  items={len(item_keys)}  K={K}  reorient={has_reorient}")
    print(f"    prior={model_kwargs['discrimination_prior']}"
          f"({model_kwargs['discrimination_prior_scale']})  "
          f"eta={model_kwargs['eta_scale']}  slab={model_kwargs['slab_scale']}")
    sys.stdout.flush()

    # Back up the existing (un-oriented) baseline so we can restore if needed
    backup = baseline_dir.parent / 'grm_baseline.preorient.bak'
    if baseline_dir.exists() and not backup.exists():
        shutil.copytree(str(baseline_dir), str(backup))
        print(f"    Backed up original to {backup}")

    model = GRModel(**model_kwargs)

    steps_per_epoch = int(np.ceil(N / batch_size))

    def factory():
        idx = np.arange(N)
        np.random.shuffle(idx)
        need = steps_per_epoch * batch_size
        if need > N:
            idx = np.concatenate([idx,
                np.random.choice(N, need - N, replace=True)])
        for s in range(0, need, batch_size):
            b = idx[s:s+batch_size]
            yield {k: v[b] for k, v in data.items()}

    res = model.fit(
        factory,
        batch_size=batch_size,
        dataset_size=N,
        num_epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        learning_rate=lr,
        patience=patience,
        lr_decay_factor=0.975,
        clip_norm=1.0,
        zero_nan_grads=True,
        max_nan_recoveries=50,
        seed=42,
    )
    losses = res[0]
    print(f"    ADVI final loss: {losses[-1]:.2f}  epochs_ran={len(losses)}")

    # Inspect the surrogate discriminations for pathological values
    disc_loc = model.params[
        'discriminations\\softplus\\normal\\loc']
    print(f"    disc_loc range: [{float(np.min(disc_loc)):.3f}, "
          f"{float(np.max(disc_loc)):.3f}]  mean={float(np.mean(disc_loc)):.3f}")

    baseline_dir.mkdir(parents=True, exist_ok=True)
    model.save_to_disk(str(baseline_dir))
    np.save(str(baseline_dir / 'losses.npy'), np.array(losses))
    print(f"    Saved to {baseline_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=DATASETS + ['all'])
    p.add_argument('--force', action='store_true',
                   help='Refit even for datasets without reorient support')
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    args = p.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    for ds in datasets:
        try:
            refit(ds, force=args.force, lr=args.lr,
                  epochs=args.epochs, patience=args.patience,
                  batch_size=args.batch_size)
        except Exception as e:
            print(f"ERROR on {ds}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

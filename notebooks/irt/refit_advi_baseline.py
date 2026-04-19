#!/usr/bin/env python
"""Refit ADVI baseline GRM with reorient=True for a dataset.

The existing grm_baseline/ ADVI fits predate the reorient commits, so the
variational posterior points to item params oriented for the *unreversed*
data. The MCMC pipeline now loads reoriented data, producing a mismatch.
This script refits ADVI on reoriented data so downstream MCMC + IS
reweighting have a consistent warm-start.

Usage:
    uv run python refit_advi_baseline.py --dataset grit
    uv run python refit_advi_baseline.py --dataset all
"""

import argparse
import inspect
import os
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

from pathlib import Path

import numpy as np


DATASETS = ['grit', 'tma', 'rwa', 'npi', 'wpi', 'eqsq',
            'promis_sleep', 'promis_substance_use']

# Per-dataset training hyperparameters (from results.md + empirical tuning)
CONFIGS = {
    'grit':                 dict(lr=1e-3, batch_size=256, epochs=500, patience=20,
                                 discrimination_prior='half_cauchy', discrimination_prior_scale=1.0),
    'tma':                  dict(lr=1e-3, batch_size=256, epochs=500, patience=20,
                                 discrimination_prior='half_normal', discrimination_prior_scale=2.0),
    'rwa':                  dict(lr=1e-3, batch_size=256, epochs=500, patience=20,
                                 discrimination_prior='half_cauchy', discrimination_prior_scale=1.0),
    'npi':                  dict(lr=1e-3, batch_size=256, epochs=500, patience=20,
                                 discrimination_prior='half_normal', discrimination_prior_scale=2.0),
    'wpi':                  dict(lr=5e-4, batch_size=256, epochs=500, patience=20,
                                 discrimination_prior='half_normal', discrimination_prior_scale=2.0),
    'eqsq':                 dict(lr=5e-4, batch_size=256, epochs=500, patience=20,
                                 discrimination_prior='half_normal', discrimination_prior_scale=1/np.sqrt(2)),
    'promis_sleep':         dict(lr=5e-4, batch_size=256, epochs=500, patience=20,
                                 discrimination_prior='half_normal', discrimination_prior_scale=2.0),
    'promis_substance_use': dict(lr=1e-3, batch_size=256, epochs=500, patience=20,
                                 discrimination_prior='half_normal', discrimination_prior_scale=2.0),
}


def refit(dataset_name, output_root=None):
    import importlib
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    from bayesianquilts.irt.grm import GRModel

    cfg = CONFIGS[dataset_name]
    mod = importlib.import_module(f'bayesianquilts.data.{dataset_name}')
    item_keys = mod.item_keys
    K = mod.response_cardinality

    kw = {'polars_out': True}
    if 'reorient' in inspect.signature(mod.get_data).parameters:
        kw['reorient'] = True
    df, N = mod.get_data(**kw)

    data = {col: df[col].to_numpy().astype(np.float32) for col in df.columns}
    data['person'] = np.arange(N, dtype=np.float32)

    print(f"\n=== {dataset_name.upper()}: N={N}  items={len(item_keys)}  K={K}")
    print(f"    reorient={'reorient' in kw}  lr={cfg['lr']}  "
          f"prior={cfg['discrimination_prior']}({cfg['discrimination_prior_scale']:.3f})")
    sys.stdout.flush()

    model = GRModel(
        item_keys=item_keys,
        num_people=N,
        dim=1,
        response_cardinality=K,
        dtype=jnp.float64,
        discrimination_prior=cfg['discrimination_prior'],
        discrimination_prior_scale=cfg['discrimination_prior_scale'],
    )

    batch_size = cfg['batch_size']
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
        num_epochs=cfg['epochs'],
        steps_per_epoch=steps_per_epoch,
        learning_rate=cfg['lr'],
        patience=cfg['patience'],
        lr_decay_factor=0.9,
        clip_norm=1.0,
        zero_nan_grads=True,
        max_nan_recoveries=50,
        seed=42,
    )
    losses = res[0]
    print(f"    ADVI final loss: {losses[-1]:.2f}  epochs_ran={len(losses)}")

    # Save
    if output_root is None:
        output_root = Path.home() / 'workspace/bayesianquilts/notebooks/irt' / dataset_name
    out = Path(output_root) / 'grm_baseline'
    out.mkdir(parents=True, exist_ok=True)
    model.save_to_disk(str(out))
    np.save(str(out / 'losses.npy'), np.array(losses))
    print(f"    Saved to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                   choices=DATASETS + ['all'])
    p.add_argument('--output-root', default=None,
                   help='Parent dir. Default: notebooks/irt')
    args = p.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    for ds in datasets:
        try:
            refit(ds, output_root=(
                Path(args.output_root) / ds if args.output_root else None))
        except Exception as e:
            print(f"ERROR on {ds}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

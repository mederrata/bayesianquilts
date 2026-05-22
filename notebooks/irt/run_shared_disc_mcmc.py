"""Fit + sample a shared-discrimination GRM (Rasch-like) on a dataset.

Writes mcmc_samples/mcmc_shared_disc.npz with the same key schema as
mcmc_baseline.npz so eval_mcmc.py / eval_promis_new.py can consume it.

Usage:
    python run_shared_disc_mcmc.py --dataset scs --num-warmup 3000 --num-samples 500
"""
import argparse
import gc
import importlib
import inspect
import os
import sys
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'python'))

from bayesianquilts.irt.shared_disc_grm import SharedDiscGRModel
from run_single_notebook import DATASET_CONFIGS, make_data_dict
from synthetic.common.pipeline import make_data_factory


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS.keys()))
    p.add_argument('--num-chains', type=int, default=4)
    p.add_argument('--num-warmup', type=int, default=3000)
    p.add_argument('--num-samples', type=int, default=500)
    p.add_argument('--step-size', type=float, default=5e-3)
    p.add_argument('--num-advi-epochs', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    mod = importlib.import_module(cfg['module'])
    item_keys = mod.item_keys
    K = mod.response_cardinality

    kw = {'polars_out': True}
    if 'reorient' in inspect.signature(mod.get_data).parameters:
        kw['reorient'] = True
    df, num_people = mod.get_data(**kw)
    data_dict = make_data_dict(df)

    out_dir = Path(__file__).parent / args.dataset / 'mcmc_samples'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / 'mcmc_shared_disc.npz'

    print(f"\n=== Shared-disc GRM fit: {args.dataset.upper()} ===")
    print(f"  N={num_people}, I={len(item_keys)}, K={K}")
    print(f"  Chains={args.num_chains}, warmup={args.num_warmup}, samples={args.num_samples}")
    print(f"  Output: {out_npz}")
    sys.stdout.flush()

    model = SharedDiscGRModel(
        item_keys=item_keys,
        num_people=num_people,
        dim=1,
        response_cardinality=K,
        dtype=jnp.float64,
    )

    # Skip ADVI; MCMC will initialize each chain from prior samples (works
    # well for shared-disc because the parameter count is small).
    print("  Skipping ADVI init; initializing chains from prior.", flush=True)
    model.params = None

    # MCMC
    print("\n  Marginal MCMC...", flush=True)
    mcmc_samples = model.fit_marginal_mcmc(
        data_dict,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        step_size=args.step_size,
        seed=args.seed,
        verbose=True,
    )

    # Compute EAP for downstream evaluation
    print("\n  Computing EAP abilities...", flush=True)
    model.mcmc_samples = mcmc_samples
    eap_result = model.compute_eap_abilities(data_dict)
    eap = np.asarray(eap_result['eap']).flatten()

    save_dict = {var: np.asarray(s) for var, s in mcmc_samples.items()}
    save_dict['eap'] = eap
    np.savez(str(out_npz), **save_dict)
    print(f"  Saved {len(save_dict)} arrays to {out_npz}", flush=True)


if __name__ == '__main__':
    main()

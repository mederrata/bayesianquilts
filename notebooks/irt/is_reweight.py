#!/usr/bin/env python
"""IS reweighting for completed baseline MCMC domains.

Uses the built-in importance_reweight() method to compute imputation
posteriors from baseline MCMC samples, without re-running MCMC.

Usage (from a dataset directory):
    python ../is_reweight.py depression
    python ../is_reweight.py --all
"""

import argparse
import gc
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["TQDM_DISABLE"] = "1"
sys.path.insert(0, os.path.dirname(os.getcwd()))

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path


def detect_dataset():
    """Detect which PROMIS dataset we're in based on cwd."""
    cwd = Path.cwd().name
    if "copd" in cwd:
        return "copd"
    elif "neuropathic" in cwd:
        return "neuropathic_pain"
    elif "sleep" in cwd:
        return "sleep"
    elif "substance" in cwd:
        return "substance_use"
    raise ValueError(f"Unknown dataset directory: {cwd}")


def load_data(dataset, domain=None):
    """Load data for the given dataset/domain."""
    if dataset == "copd":
        from bayesianquilts.data.promis_copd import (
            get_data, item_keys, response_cardinality,
        )
        df, num_people = get_data(polars_out=True, domain=domain)
    elif dataset == "neuropathic_pain":
        from bayesianquilts.data.promis_neuropathic_pain import (
            get_data, item_keys, response_cardinality,
        )
        df, num_people = get_data(polars_out=True, domain=domain)
    elif dataset == "sleep":
        from bayesianquilts.data.promis_sleep import (
            get_data, item_keys, response_cardinality,
        )
        df, num_people = get_data(polars_out=True)
    elif dataset == "substance_use":
        from bayesianquilts.data.promis_substance_use import (
            get_data, item_keys, response_cardinality,
        )
        df, num_people = get_data(polars_out=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    data = {}
    for col in df.columns:
        data[col] = df[col].to_numpy().astype(np.float64)
    data["person"] = np.arange(num_people, dtype=np.float64)
    return data, list(item_keys), num_people, response_cardinality


def run_is(domain: str):
    """Run IS reweighting for a completed domain."""
    from bayesianquilts.irt.grm import GRModel
    from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel

    dataset = detect_dataset()
    print(f"\n{'='*60}")
    print(f"  IS Reweighting — {dataset}/{domain}")
    print(f"{'='*60}\n")

    # Check prerequisites
    mcmc_dir = f"mcmc_{domain}"
    mcmc_npz = f"mcmc_samples/mcmc_{domain}.npz"
    pairwise_path = Path("pairwise_stacking_model.yaml")

    if not Path(mcmc_dir).exists():
        print(f"  ERROR: {mcmc_dir}/ not found")
        return False
    if not Path(mcmc_npz).exists():
        print(f"  ERROR: {mcmc_npz} not found")
        return False
    if not pairwise_path.exists():
        print(f"  ERROR: pairwise_stacking_model.yaml not found")
        return False

    # Load data, model, samples
    data, domain_items, num_people, response_cardinality = load_data(dataset, domain)
    model = GRModel.load_from_disk(mcmc_dir)
    mcmc_data = np.load(mcmc_npz)
    print(f"  Loaded model and {num_people} people, {len(domain_items)} items")

    # Reconstruct mcmc_samples dict with (chains, samples, ...) shape
    # The model.mcmc_samples should already be set from load, but
    # let's reconstruct from the npz just in case
    mcmc_samples = {}
    for var in ("difficulties0", "discriminations", "ddifficulties"):
        if var in mcmc_data:
            mcmc_samples[var] = jnp.array(mcmc_data[var])
    print(f"  MCMC samples: {list(mcmc_samples.keys())}")
    first = list(mcmc_samples.values())[0]
    print(f"    shape: {first.shape}")

    # Set up data factory for mixed model
    batch_size = 256
    def data_factory():
        indices = np.arange(num_people)
        np.random.shuffle(indices)
        for start in range(0, num_people, batch_size):
            end = min(start + batch_size, num_people)
            yield {k: v[indices[start:end]] for k, v in data.items()}

    # Load pairwise model
    pairwise_model = PairwiseOrdinalStackingModel.load(str(pairwise_path))
    print(f"  Pairwise model: {len(pairwise_model.variable_names)} variables")

    # Pairwise IS
    print(f"\n--- Pairwise IS ---")
    is_pairwise = model.importance_reweight(
        data, mcmc_samples, pairwise_model, verbose=True,
    )
    os.makedirs("is_results", exist_ok=True)
    np.savez(f"is_results/is_pairwise_{domain}.npz",
             log_weights=is_pairwise["log_weights"],
             psis_weights=is_pairwise["psis_weights"],
             khat=is_pairwise["khat"],
             ess=is_pairwise["ess"])
    print(f"  k-hat={is_pairwise['khat']:.4f}, ESS={is_pairwise['ess']:.1f}")

    # Mixed IS
    try:
        from bayesianquilts.imputation.mixed import IrtMixedImputationModel

        surrogate = model.surrogate_distribution_generator(model.params)
        samples = surrogate.sample(32, seed=jax.random.PRNGKey(142))
        model.surrogate_sample = samples
        model.calibrated_expectations = {
            k: jnp.mean(v, axis=0) for k, v in samples.items()
        }

        mixed_model = IrtMixedImputationModel(
            irt_model=model,
            mice_model=pairwise_model,
            data_factory=data_factory,
            irt_elpd_batch_size=4,
        )

        print(f"\n--- Mixed IS ---")
        is_mixed = model.importance_reweight(
            data, mcmc_samples, mixed_model, verbose=True,
        )
        np.savez(f"is_results/is_mixed_{domain}.npz",
                 log_weights=is_mixed["log_weights"],
                 psis_weights=is_mixed["psis_weights"],
                 khat=is_mixed["khat"],
                 ess=is_mixed["ess"])
        print(f"  k-hat={is_mixed['khat']:.4f}, ESS={is_mixed['ess']:.1f}")
    except Exception as e:
        print(f"  Mixed IS skipped: {e}")
        is_mixed = None

    # Summary
    ok = is_pairwise["khat"] < 0.7
    if is_mixed is not None:
        ok = ok and is_mixed["khat"] < 0.7
    status = "OK" if ok else "CHECK k-hat"
    print(f"\n  >>> {domain}: {status} <<<\n")
    return ok


def find_completed_domains():
    """Find domains with completed MCMC."""
    completed = []
    mcmc_dir = Path("mcmc_samples")
    if mcmc_dir.exists():
        for f in mcmc_dir.glob("mcmc_*.npz"):
            domain = f.stem.replace("mcmc_", "")
            if Path(f"mcmc_{domain}").exists():
                completed.append(domain)
    return sorted(completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("domain", nargs="?", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or args.domain == "--all":
        domains = find_completed_domains()
        if not domains:
            print("No completed MCMC domains found")
            sys.exit(1)
        print(f"Found {len(domains)} completed domains: {domains}")
    elif args.domain:
        domains = [args.domain]
    else:
        parser.print_help()
        sys.exit(1)

    results = {}
    for domain in domains:
        try:
            results[domain] = run_is(domain)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[domain] = False
        gc.collect()

    print("\n" + "=" * 60)
    print("  IS REWEIGHTING SUMMARY")
    print("=" * 60)
    for domain, ok in results.items():
        status = "OK" if ok else "CHECK"
        print(f"  {domain:25s} {status}")


if __name__ == "__main__":
    main()

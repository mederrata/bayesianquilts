#!/usr/bin/env python
"""Run MCMC under pairwise imputation, warm-started from baseline MCMC.

For domains where IS reweighting fails (k-hat >> 1), this runs full
MCMC with imputation PMFs attached to the data. Initializes from
the baseline model's fitted parameters.

Usage (from a dataset directory):
    python ../mcmc_imputed.py global_health
    python ../mcmc_imputed.py --all
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


NUM_CHAINS = 2
NUM_WARMUP = 3000
NUM_SAMPLES = 500
THINNING = 5
STEP_SIZE = 0.001


def detect_dataset():
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
    if dataset == "copd":
        from bayesianquilts.data.promis_copd import (
            get_data, item_keys, response_cardinality)
        df, n = get_data(polars_out=True, domain=domain)
    elif dataset == "neuropathic_pain":
        from bayesianquilts.data.promis_neuropathic_pain import (
            get_data, item_keys, response_cardinality)
        df, n = get_data(polars_out=True, domain=domain)
    elif dataset == "sleep":
        from bayesianquilts.data.promis_sleep import (
            get_data, item_keys, response_cardinality)
        df, n = get_data(polars_out=True)
    elif dataset == "substance_use":
        from bayesianquilts.data.promis_substance_use import (
            get_data, item_keys, response_cardinality)
        df, n = get_data(polars_out=True)
    else:
        raise ValueError(dataset)

    data = {}
    for col in df.columns:
        data[col] = df[col].to_numpy().astype(np.float64)
    data["person"] = np.arange(n, dtype=np.float64)
    return data, list(item_keys), n, response_cardinality


def run_imputed_mcmc(domain):
    from bayesianquilts.irt.grm import GRModel
    from bayesianquilts.imputation.pairwise_stacking import (
        PairwiseOrdinalStackingModel)

    dataset = detect_dataset()
    print(f"\n{'='*60}")
    print(f"  Imputed MCMC — {dataset}/{domain}")
    print(f"{'='*60}\n")

    # Check prerequisites
    baseline_dir = f"mcmc_{domain}"
    baseline_npz = f"mcmc_samples/mcmc_{domain}.npz"
    pw_path = Path("pairwise_stacking_model.yaml")

    if not Path(baseline_dir).exists():
        print(f"  ERROR: {baseline_dir}/ not found — run baseline MCMC first")
        return False
    if not pw_path.exists():
        print(f"  ERROR: pairwise_stacking_model.yaml not found")
        return False

    # Load data
    data, domain_items, num_people, K = load_data(dataset, domain)
    print(f"  {num_people} people, {len(domain_items)} items, K={K}")

    batch_size = 256
    def data_factory():
        indices = np.arange(num_people)
        np.random.shuffle(indices)
        for start in range(0, num_people, batch_size):
            end = min(start + batch_size, num_people)
            yield {k: v[indices[start:end]] for k, v in data.items()}

    # Load baseline model (warm start)
    model = GRModel.load_from_disk(baseline_dir)
    print(f"  Loaded baseline model from {baseline_dir}/")

    # Attach pairwise imputation and compute PMFs
    pw = PairwiseOrdinalStackingModel.load(str(pw_path))
    model.imputation_model = pw
    print(f"  Pairwise model: {len(pw.variable_names)} variables")

    pmfs, weights = model._compute_batch_pmfs(data)
    if pmfs is not None:
        data["_imputation_pmfs"] = pmfs
        if weights is not None:
            data["_imputation_weights"] = weights
        n_imputed = np.sum(pmfs.sum(axis=-1) > 0)
        print(f"  Imputation PMFs: {n_imputed} cells imputed")
    else:
        print(f"  WARNING: No imputation PMFs computed")

    # Run MCMC (warm-started from baseline params)
    print(f"\n--- MCMC (pairwise imputed) ---")
    mcmc_samples = model.fit_marginal_mcmc(
        data,
        theta_grid=None,
        num_chains=NUM_CHAINS,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        thinning=THINNING,
        target_accept_prob=0.85,
        step_size=STEP_SIZE,
        seed=43,  # different seed from baseline
        verbose=True,
    )

    # EAP
    eap_result = model.compute_eap_abilities(data)
    eap = np.array(eap_result["eap"])
    stats = model.standardize_marginal(data)
    eap_result = model.compute_eap_abilities(data)
    eap_std = np.array(eap_result["eap"])
    print(f"  EAP: mean={np.mean(eap_std):.4f}, std={np.std(eap_std):.4f}")

    # Rhat
    print("\n--- Convergence Diagnostics ---")
    all_ok = True
    for var_name, samples in mcmc_samples.items():
        samples_np = np.array(samples)
        if samples_np.shape[0] < 2:
            continue
        chain_means = np.mean(samples_np, axis=1)
        between_var = np.var(chain_means, axis=0, ddof=1)
        within_var = np.mean(np.var(samples_np, axis=1, ddof=1), axis=0)
        n = samples_np.shape[1]
        r_hat = np.sqrt(
            ((n - 1) / n * within_var + between_var)
            / np.maximum(within_var, 1e-30)
        )
        max_rhat = np.max(r_hat)
        mean_rhat = np.mean(r_hat)
        status = "OK" if max_rhat < 1.1 else "FAIL"
        all_ok = all_ok and (max_rhat < 1.1)
        print(f"  {var_name}: mean R-hat={mean_rhat:.4f}, "
              f"max R-hat={max_rhat:.4f}  [{status}]")

    # Save
    save_dir = f"mcmc_{domain}_pairwise"
    model.save_to_disk(save_dir)

    os.makedirs("mcmc_samples", exist_ok=True)
    save_dict = {var: np.array(s) for var, s in mcmc_samples.items()}
    save_dict["eap"] = eap
    save_dict["eap_standardized"] = eap_std
    save_dict["psd"] = np.array(eap_result["psd"])
    save_dict["standardize_mu"] = stats["mu"]
    save_dict["standardize_sigma"] = stats["sigma"]
    np.savez(f"mcmc_samples/mcmc_{domain}_pairwise.npz", **save_dict)

    overall = "CONVERGED" if all_ok else "NOT CONVERGED"
    print(f"\n  >>> {domain} (pairwise): {overall} <<<\n")
    return all_ok


def find_baseline_domains():
    completed = []
    mcmc_dir = Path("mcmc_samples")
    if mcmc_dir.exists():
        for f in mcmc_dir.glob("mcmc_*.npz"):
            name = f.stem.replace("mcmc_", "")
            if "_pairwise" not in name and "_mixed" not in name:
                if Path(f"mcmc_{name}").exists():
                    completed.append(name)
    return sorted(completed)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("domain", nargs="?", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        domains = find_baseline_domains()
        print(f"Baseline domains: {domains}")
    elif args.domain:
        domains = [args.domain]
    else:
        parser.print_help()
        sys.exit(1)

    results = {}
    for domain in domains:
        try:
            results[domain] = run_imputed_mcmc(domain)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[domain] = False
        gc.collect()

    print("\n" + "=" * 60)
    print("  IMPUTED MCMC SUMMARY")
    print("=" * 60)
    for domain, ok in results.items():
        status = "CONVERGED" if ok else "NOT CONVERGED"
        print(f"  {domain:25s} {status}")


if __name__ == "__main__":
    main()

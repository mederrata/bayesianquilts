#!/usr/bin/env python
"""Per-domain marginal MCMC for PROMIS Neuropathic Pain.

Fits a single-scale GRM per PROMIS domain (instead of a joint factorized
model), then runs NUTS with Rao-Blackwellized abilities.

Usage:
    python mcmc_domain.py pain_interference
    python mcmc_domain.py pain_behavior
    python mcmc_domain.py --all          # run all qualifying domains

Available domains (with ≥10 items): pain_interference, pain_behavior,
    global_health
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

from bayesianquilts.data.promis_neuropathic_pain import (
    get_data,
    item_keys,
    response_cardinality,
    DOMAINS,
)
from bayesianquilts.irt.grm import GRModel


# MCMC settings
NUM_CHAINS = 2
NUM_WARMUP = 3000
NUM_SAMPLES = 500
THINNING = 5
STEP_SIZE = 0.001
ADVI_EPOCHS = 100

# Only run domains with enough items (matching the factorized notebook)
MIN_ITEMS = 10


def run_domain(domain: str):
    """Fit ADVI + MCMC for a single neuropathic pain domain."""
    print(f"\n{'='*60}")
    print(f"  PROMIS Neuropathic Pain — {domain}")
    print(f"{'='*60}\n")

    # 1. Load data for this domain
    df, num_people = get_data(polars_out=True, domain=domain)
    domain_items = list(item_keys)  # get_data populates the global
    print(f"  {num_people} people, {len(domain_items)} items, "
          f"{response_cardinality} categories\n")

    data = {}
    for col in df.columns:
        data[col] = df[col].to_numpy().astype(np.float64)
    data["person"] = np.arange(num_people, dtype=np.float64)

    batch_size = 256
    steps_per_epoch = int(np.ceil(num_people / batch_size))

    def data_factory():
        indices = np.arange(num_people)
        np.random.shuffle(indices)
        for start in range(0, num_people, batch_size):
            end = min(start + batch_size, num_people)
            yield {k: v[indices[start:end]] for k, v in data.items()}

    # 2. Fit ADVI
    print("--- ADVI ---")
    model = GRModel(
        item_keys=domain_items,
        num_people=num_people,
        response_cardinality=response_cardinality,
        discrimination_prior_scale=2.0,
        dtype=jnp.float64,
    )

    res = model.fit(
        data_factory,
        batch_size=batch_size,
        dataset_size=num_people,
        num_epochs=ADVI_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        learning_rate=2e-4,
        patience=10,
    )
    losses = res[0]
    print(f"  ADVI final loss: {losses[-1]:.2f}")

    # 3. Run MCMC
    print("\n--- MCMC ---")
    mcmc_samples = model.fit_marginal_mcmc(
        data,
        theta_grid=None,
        num_chains=NUM_CHAINS,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        thinning=THINNING,
        target_accept_prob=0.85,
        step_size=STEP_SIZE,
        seed=42,
        verbose=True,
    )

    # 4. Standardize and compute EAP
    eap_result = model.compute_eap_abilities(data)
    eap = np.array(eap_result["eap"])
    stats = model.standardize_marginal(data)
    eap_result = model.compute_eap_abilities(data)
    eap_std = np.array(eap_result["eap"])
    print(f"  EAP: mean={np.mean(eap_std):.4f}, std={np.std(eap_std):.4f}")

    # 5. Compute and report Rhat
    print("\n--- Convergence Diagnostics ---")
    all_ok = True
    for var_name, samples in mcmc_samples.items():
        samples_np = np.array(samples)
        if samples_np.shape[0] < 2:
            print(f"  {var_name}: only 1 chain, cannot compute R-hat")
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

    # 6. Save baseline
    save_dir = f"mcmc_{domain}"
    model.save_to_disk(save_dir)

    os.makedirs("mcmc_samples", exist_ok=True)
    save_dict = {var: np.array(s) for var, s in mcmc_samples.items()}
    save_dict["eap"] = eap
    save_dict["eap_standardized"] = eap_std
    save_dict["psd"] = np.array(eap_result["psd"])
    save_dict["standardize_mu"] = stats["mu"]
    save_dict["standardize_sigma"] = stats["sigma"]
    np.savez(f"mcmc_samples/mcmc_{domain}.npz", **save_dict)

    # 7. IS reweighting to imputation posteriors
    from pathlib import Path
    from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel

    pairwise_path = Path("pairwise_stacking_model.yaml")
    if pairwise_path.exists() and all_ok:
        pairwise_model = PairwiseOrdinalStackingModel.load(str(pairwise_path))
        print(f"\n--- IS Reweighting: Pairwise ---")

        is_pairwise = model.importance_reweight(
            data, mcmc_samples, pairwise_model, verbose=True,
        )
        os.makedirs("is_results", exist_ok=True)
        np.savez(f"is_results/is_pairwise_{domain}.npz",
                 log_weights=is_pairwise["log_weights"],
                 psis_weights=is_pairwise["psis_weights"],
                 khat=is_pairwise["khat"],
                 ess=is_pairwise["ess"])
        print(f"  Saved to is_results/is_pairwise_{domain}.npz")

        # Mixed imputation (pairwise + IRT)
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
            print(f"\n--- IS Reweighting: Mixed ---")
            is_mixed = model.importance_reweight(
                data, mcmc_samples, mixed_model, verbose=True,
            )
            np.savez(f"is_results/is_mixed_{domain}.npz",
                     log_weights=is_mixed["log_weights"],
                     psis_weights=is_mixed["psis_weights"],
                     khat=is_mixed["khat"],
                     ess=is_mixed["ess"])
            print(f"  Saved to is_results/is_mixed_{domain}.npz")
        except Exception as e:
            print(f"  Mixed IS skipped: {e}")
    elif not pairwise_path.exists():
        print("\n  No pairwise_stacking_model.yaml — skipping IS reweighting")

    overall = "CONVERGED" if all_ok else "NOT CONVERGED"
    print(f"\n  >>> {domain}: {overall} <<<\n")
    return all_ok


def _qualifying_domains():
    """Return domains with ≥ MIN_ITEMS items (matching factorized notebook)."""
    # We check column counts to match the factorized_grm.ipynb selection
    from bayesianquilts.data.promis_neuropathic_pain import _load_sas, _detect_domain_columns
    from pathlib import Path
    df_pandas = _load_sas(Path.cwd())
    all_columns = list(df_pandas.columns)
    qualifying = []
    for name, prefix in DOMAINS.items():
        cols = _detect_domain_columns(all_columns, prefix)
        if len(cols) >= MIN_ITEMS:
            qualifying.append(name)
    return qualifying


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "domain",
        nargs="?",
        default=None,
        help="Domain name (e.g. pain_interference) or --all",
    )
    parser.add_argument("--all", action="store_true", help="Run all qualifying domains")
    args = parser.parse_args()

    if args.all or args.domain == "--all":
        domains = _qualifying_domains()
        print(f"Running {len(domains)} domains: {domains}")
    elif args.domain:
        if args.domain not in DOMAINS:
            print(f"Unknown domain: {args.domain}")
            print(f"Available: {list(DOMAINS.keys())}")
            sys.exit(1)
        domains = [args.domain]
    else:
        parser.print_help()
        sys.exit(1)

    results = {}
    for domain in domains:
        try:
            results[domain] = run_domain(domain)
        except Exception as e:
            print(f"  ERROR in {domain}: {e}")
            import traceback
            traceback.print_exc()
            results[domain] = False
        gc.collect()

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for domain, ok in results.items():
        status = "CONVERGED" if ok else "FAIL"
        print(f"  {domain:25s} {status}")


if __name__ == "__main__":
    main()

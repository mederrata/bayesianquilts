#!/usr/bin/env python
"""Baseline marginal MCMC for PROMIS Substance Use (single scale GRM).

Runs NUTS on item parameters with abilities Rao-Blackwellized out.
Uses 3-phase warmup with step-size probing and shared chain parameters.

Usage:
    python mcmc_baseline.py
"""

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

from bayesianquilts.data.promis_substance_use import (
    get_data, item_keys, response_cardinality,
)
from bayesianquilts.irt.grm import GRModel

NUM_CHAINS = 2
NUM_WARMUP = 3000
NUM_SAMPLES = 500
THINNING = 5
STEP_SIZE = 0.001
ADVI_EPOCHS = 100


def main():
    print(f"\n{'='*60}")
    print(f"  PROMIS Substance Use — Baseline MCMC")
    print(f"{'='*60}\n")

    df, num_people = get_data(polars_out=True)
    domain_items = list(item_keys)
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

    # ADVI
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
    print(f"  ADVI final loss: {res[0][-1]:.2f}")

    # MCMC
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
    model.save_to_disk("mcmc_baseline")

    os.makedirs("mcmc_samples", exist_ok=True)
    save_dict = {var: np.array(s) for var, s in mcmc_samples.items()}
    save_dict["eap"] = eap
    save_dict["eap_standardized"] = eap_std
    save_dict["psd"] = np.array(eap_result["psd"])
    save_dict["standardize_mu"] = stats["mu"]
    save_dict["standardize_sigma"] = stats["sigma"]
    np.savez("mcmc_samples/mcmc_baseline.npz", **save_dict)

    overall = "CONVERGED" if all_ok else "NOT CONVERGED"
    print(f"\n  >>> Substance Use: {overall} <<<\n")


if __name__ == "__main__":
    main()

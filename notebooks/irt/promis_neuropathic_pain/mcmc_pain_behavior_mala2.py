#!/usr/bin/env python
"""NP pain_behavior: MALA with many more samples."""
import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["TQDM_DISABLE"] = "1"
sys.path.insert(0, os.path.dirname(os.getcwd()))

import numpy as np
import jax.numpy as jnp
from bayesianquilts.data.promis_neuropathic_pain import get_data, item_keys, response_cardinality
from bayesianquilts.irt.grm import GRModel

df, num_people = get_data(polars_out=True, domain='pain_behavior')
domain_items = list(item_keys)
print(f"{num_people} people, {len(domain_items)} items")

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

model = GRModel(
    item_keys=domain_items, num_people=num_people,
    response_cardinality=response_cardinality,
    discrimination_prior_scale=2.0, dtype=jnp.float64,
)
res = model.fit(data_factory, batch_size=batch_size, dataset_size=num_people,
                num_epochs=100, steps_per_epoch=steps_per_epoch,
                learning_rate=2e-4, patience=10)
print(f"ADVI final loss: {res[0][-1]:.2f}")

# MALA with much more samples and smaller step for better acceptance
mcmc_samples = model.fit_marginal_mala(
    data, theta_grid=None,
    num_chains=4, num_warmup=10000, num_samples=10000,
    step_size=5e-5, seed=42, verbose=True,
)

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
    r_hat = np.sqrt(((n-1)/n * within_var + between_var) / np.maximum(within_var, 1e-30))
    max_rhat = np.max(r_hat)
    status = "OK" if max_rhat < 1.1 else "FAIL"
    all_ok = all_ok and (max_rhat < 1.1)
    print(f"  {var_name}: mean R-hat={np.mean(r_hat):.4f}, max R-hat={max_rhat:.4f}  [{status}]")

model.save_to_disk("mcmc_pain_behavior")
os.makedirs("mcmc_samples", exist_ok=True)
save_dict = {var: np.array(s) for var, s in mcmc_samples.items()}
eap_result = model.compute_eap_abilities(data)
stats = model.standardize_marginal(data)
eap_result2 = model.compute_eap_abilities(data)
save_dict["eap"] = np.array(eap_result["eap"])
save_dict["eap_standardized"] = np.array(eap_result2["eap"])
save_dict["psd"] = np.array(eap_result2["psd"])
save_dict["standardize_mu"] = stats["mu"]
save_dict["standardize_sigma"] = stats["sigma"]
np.savez("mcmc_samples/mcmc_pain_behavior.npz", **save_dict)

overall = "CONVERGED" if all_ok else "NOT CONVERGED"
print(f"\n  >>> pain_behavior: {overall} <<<")

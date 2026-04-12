#!/usr/bin/env python
"""Per-item LOO-ELPD comparison: baseline vs imputation models.

Evaluates LOO-ELPD on OBSERVED responses only. Both baseline and
imputation posteriors are evaluated on the same observed data —
the difference is in the parameter estimates (imputation uses
additional information from missing data during fitting).

Usage (from a dataset directory):
    python ../compare_loo_elpd.py global_health
    python ../compare_loo_elpd.py --all
"""

import argparse
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
        raise ValueError(f"Unknown dataset: {dataset}")

    data = {}
    for col in df.columns:
        data[col] = df[col].to_numpy().astype(np.float64)
    data["person"] = np.arange(n, dtype=np.float64)
    return data, list(item_keys), n, response_cardinality


def compute_observed_loo(model, data, domain_items, flat_samples,
                         item_vars, S, tg, tlw, weights=None):
    """Compute per-item LOO-ELPD on observed responses only.

    Args:
        model: GRModel instance
        data: data dict
        domain_items: list of item keys
        flat_samples: dict of flattened MCMC samples
        item_vars: list of variable names
        S: number of samples
        tg, tlw: quadrature grid and log weights
        weights: IS weights (None = uniform/baseline)

    Returns:
        dict with per-person and per-item LOO-ELPD
    """
    from bayesianquilts.metrics.nppsis import psisloo

    K = model.response_cardinality
    n_items = len(domain_items)
    num_people = len(data["person"])

    # Observed responses (N, I)
    choices = np.stack([data[k] for k in domain_items], axis=-1)
    bad = (choices < 0) | (choices >= K) | np.isnan(choices)
    observed = ~bad  # (N, I) — True where observed

    # Per-person log-lik on observed items only: (S, N)
    ll_person = np.zeros((S, num_people))
    # Per-item log-lik on observed items only: (S, N, I)
    # Only nonzero where observed[n, i] is True
    ll_per_item = np.zeros((S, num_people, n_items))

    choices_int = np.where(bad, 0, choices).astype(np.int32)
    bad_jnp = jnp.array(bad)

    for s in range(S):
        params = {var: jnp.array(flat_samples[var][s]) for var in item_vars}

        # Response probabilities: (Q, I, K)
        rp = model._response_probs_grid(tg, **params)
        log_rp = jnp.log(jnp.clip(rp, 1e-30, None))

        # Gather log P(x_observed | theta_q): (Q, N, I)
        log_obs = jnp.take_along_axis(
            log_rp[:, None, :, :],
            jnp.array(choices_int)[None, :, :, None],
            axis=-1
        ).squeeze(-1)

        # Zero out missing items — observed only
        log_obs_masked = jnp.where(bad_jnp[None, :, :], 0.0, log_obs)

        # Full person marginal: sum over observed items, logsumexp over grid
        log_lik_person = jnp.sum(log_obs_masked, axis=-1)  # (Q, N)
        marginal = jax.scipy.special.logsumexp(
            log_lik_person + tlw[:, None], axis=0)  # (N,)
        ll_person[s] = np.array(marginal)

        # Per-item contribution: marginal(all) - marginal(all except item i)
        # Only compute for items with observations
        for i in range(n_items):
            mask_i = jnp.ones(n_items, dtype=bool).at[i].set(False)
            log_lik_no_i = jnp.sum(log_obs_masked[:, :, mask_i], axis=-1)
            marginal_no_i = jax.scipy.special.logsumexp(
                log_lik_no_i + tlw[:, None], axis=0)
            # Per-item contribution for each person
            ll_per_item[s, :, i] = np.array(marginal - marginal_no_i)

        if (s + 1) % 200 == 0:
            print(f"    {s+1}/{S} samples")

    # PSIS-LOO per person
    loo, loos, ks = psisloo(ll_person)

    # If IS weights provided, compute weighted ELPD
    if weights is not None:
        # Weighted per-person ELPD
        w = np.array(weights)
        weighted_elpd = np.sum(w[:, None] * ll_person, axis=0)
        total_elpd = np.sum(weighted_elpd)
        # Weighted per-item ELPD (averaged over people)
        per_item_elpd = np.zeros(n_items)
        for i in range(n_items):
            obs_mask = observed[:, i]
            if obs_mask.any():
                weighted_item = np.sum(
                    w[:, None] * ll_per_item[:, :, i], axis=0)
                per_item_elpd[i] = np.mean(weighted_item[obs_mask])
        return {
            'total_elpd': total_elpd,
            'per_item_elpd': per_item_elpd,
            'loos': loos, 'ks': ks,
        }

    # Baseline: uniform weights, per-item mean over observed people only
    per_item_elpd = np.zeros(n_items)
    for i in range(n_items):
        obs_mask = observed[:, i]
        if obs_mask.any():
            # Mean over samples, then mean over observed people
            item_ll = np.mean(ll_per_item[:, obs_mask, i], axis=0)
            per_item_elpd[i] = np.mean(item_ll)

    return {
        'total_elpd': float(loo),
        'per_item_elpd': per_item_elpd,
        'loos': loos, 'ks': ks,
    }


def compute_per_item_loo(domain):
    """Compare per-item LOO-ELPD: baseline vs pairwise."""
    from bayesianquilts.irt.grm import GRModel
    from bayesianquilts.imputation.pairwise_stacking import (
        PairwiseOrdinalStackingModel)

    dataset = detect_dataset()
    print(f"\n{'='*70}")
    print(f"  Per-Item LOO-ELPD (observed only): {dataset}/{domain}")
    print(f"{'='*70}\n")

    # Find MCMC model
    mcmc_dir = mcmc_npz = None
    for suffix in ['_v4', '_v3', '_v2', '']:
        d = f"mcmc_{domain}{suffix}" if suffix else f"mcmc_{domain}"
        n = (f"mcmc_samples/mcmc_{domain}{suffix}.npz" if suffix
             else f"mcmc_samples/mcmc_{domain}.npz")
        if Path(d).exists() and Path(n).exists():
            mcmc_dir, mcmc_npz = d, n
            break
    if not mcmc_dir:
        mcmc_dir = f"mcmc_{domain}"
        mcmc_npz = f"mcmc_samples/mcmc_{domain}.npz"
    if not Path(mcmc_dir).exists():
        print(f"  No MCMC results for {domain}")
        return

    data, domain_items, num_people, K = load_data(dataset, domain)
    model = GRModel.load_from_disk(mcmc_dir)
    mcmc = np.load(mcmc_npz)
    print(f"  {num_people} people, {len(domain_items)} items, K={K}")

    item_vars = [v for v in mcmc.files
                 if v in ('difficulties0', 'discriminations', 'ddifficulties')]
    flat_samples = {}
    first = mcmc[item_vars[0]]
    n_chains, n_samp = first.shape[:2]
    S = n_chains * n_samp
    for var in item_vars:
        flat_samples[var] = mcmc[var].reshape(-1, *mcmc[var].shape[2:])
    print(f"  {S} samples ({n_chains}×{n_samp})")

    tg, tlw = model._make_gauss_hermite_grid()

    # Missing data summary
    choices = np.stack([data[k] for k in domain_items], axis=-1)
    bad = (choices < 0) | (choices >= K) | np.isnan(choices)
    n_obs_per_item = np.sum(~bad, axis=0)

    # Baseline LOO-ELPD
    print("\n  --- Baseline (observed only) ---")
    base = compute_observed_loo(
        model, data, domain_items, flat_samples, item_vars, S, tg, tlw)
    print(f"  Total LOO-ELPD: {base['total_elpd']:.2f}")

    # IS-reweighted LOO-ELPD
    pw_path = Path("pairwise_stacking_model.yaml")
    pair = None
    if pw_path.exists():
        is_path = Path(f"is_results/is_pairwise_{domain}.npz")
        if is_path.exists():
            is_data = np.load(str(is_path))
            w = is_data['psis_weights']
            print("\n  --- Pairwise (IS-reweighted, observed only) ---")
            pair = compute_observed_loo(
                model, data, domain_items, flat_samples, item_vars,
                S, tg, tlw, weights=w)
            print(f"  Total weighted ELPD: {pair['total_elpd']:.2f}")
        else:
            print(f"\n  No IS results found — run is_reweight.py first")

    # Print comparison
    print(f"\n  {'Item':<15s} {'Obs':>5s} {'Miss%':>6s} "
          f"{'Base ELPD':>10s}", end="")
    if pair:
        print(f" {'Pair ELPD':>10s} {'Δ ELPD':>8s}", end="")
    print()
    print("  " + "-" * 60)

    for i, item in enumerate(domain_items):
        n_obs = n_obs_per_item[i]
        miss_pct = 100 * (1 - n_obs / num_people)
        line = (f"  {item:<15s} {n_obs:>5d} {miss_pct:>5.1f}% "
                f"{base['per_item_elpd'][i]:>10.4f}")
        if pair:
            delta = pair['per_item_elpd'][i] - base['per_item_elpd'][i]
            line += f" {pair['per_item_elpd'][i]:>10.4f} {delta:>+8.4f}"
            if abs(delta) > 0.01:
                line += " ***"
        print(line)

    if pair:
        delta_total = pair['total_elpd'] - base['total_elpd']
        print(f"\n  Total Δ ELPD: {delta_total:>+.2f}")
        if delta_total > 0:
            print(f"  → Imputation IMPROVES predictive performance")
        else:
            print(f"  → Imputation does not improve (or hurts) prediction")

    # Save
    os.makedirs("loo_results", exist_ok=True)
    save = {'items': domain_items, 'n_obs': n_obs_per_item,
            'base_elpd': base['per_item_elpd'],
            'base_total': base['total_elpd']}
    if pair:
        save['pair_elpd'] = pair['per_item_elpd']
        save['pair_total'] = pair['total_elpd']
    np.savez(f"loo_results/loo_{domain}.npz", **save)
    print(f"  Saved to loo_results/loo_{domain}.npz")


def find_completed():
    completed = []
    for f in Path('.').glob('mcmc_*'):
        if f.is_dir() and (f / 'config.yaml').exists():
            domain = f.name.replace('mcmc_', '')
            for s in ['_v4', '_v3', '_v2']:
                domain = domain.replace(s, '')
            if domain not in completed:
                completed.append(domain)
    return sorted(set(completed))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("domain", nargs="?", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        domains = find_completed()
        print(f"Completed domains: {domains}")
    elif args.domain:
        domains = [args.domain]
    else:
        parser.print_help()
        sys.exit(1)

    for domain in domains:
        try:
            compute_per_item_loo(domain)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

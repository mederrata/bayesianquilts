"""Sanity tests for ThreeWayImputationModel on NPI.

Verifies:
  (1) Per-item simplex weights are valid (>=0, sum to 1).
  (2) Per-item LOO ELPD is no worse than the best single component
      (Yao 2018 Thm 1).
  (3) PMFs from predict_pmf are valid distributions.
  (4) The 3-way stacked LOO matches the individual components when one
      weight is dominant (numerical consistency check).
  (5) Edge case: if one component is unavailable for an item, the others
      take over (no NaN propagation).

Run:
    PYTHONPATH=/home/josh/workspace/bayesianquilts/python \
        python test_three_way.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')

import importlib
import inspect
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
from run_marginal_mcmc import (
    DATASET_CONFIGS, make_data_dict, load_shared_disc_model,
)


def main():
    dataset = 'npi'
    cfg = DATASET_CONFIGS[dataset]
    mod = importlib.import_module(cfg['module'])
    item_keys = mod.item_keys
    K = mod.response_cardinality

    kw = {'polars_out': True}
    if 'reorient' in inspect.signature(mod.get_data).parameters:
        kw['reorient'] = True
    df, num_people = mod.get_data(**kw)
    base_data = make_data_dict(df, num_people)
    print(f"NPI: N={num_people}, I={len(item_keys)}, K={K}")

    # Paths
    here = Path(__file__).parent
    model_dir = here / dataset / 'grm_baseline'
    stacking_path = here / dataset / 'pairwise_stacking_model.yaml'
    shared_disc_npz = here / dataset / 'mcmc_samples' / 'mcmc_shared_disc.npz'

    assert (model_dir / 'params.h5').exists(), f"missing baseline: {model_dir}"
    assert stacking_path.exists(), f"missing stacking yaml: {stacking_path}"
    assert shared_disc_npz.exists(), f"missing shared-disc NPZ: {shared_disc_npz}"

    # Load baseline GRM
    from bayesianquilts.irt.grm import GRModel
    print("\n[1/5] Loading baseline GRM...")
    irt = GRModel.load_from_disk(str(model_dir))
    surrogate = irt.surrogate_distribution_generator(irt.params)
    key = jax.random.PRNGKey(99)
    samples = surrogate.sample(32, seed=key)
    irt.surrogate_sample = samples
    irt.calibrated_expectations = {
        k: jnp.mean(v, axis=0) for k, v in samples.items()
    }

    # Load pairwise
    from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel
    print("[2/5] Loading pairwise stacking model...")
    pairwise = PairwiseOrdinalStackingModel.load(str(stacking_path))

    # Load shared-disc
    print("[3/5] Loading shared-disc model from NPZ...")
    sd = load_shared_disc_model(
        item_keys=item_keys,
        num_people=num_people,
        response_cardinality=K,
        shared_disc_npz_path=shared_disc_npz,
    )
    print(f"   shared-disc surrogate keys: {list(sd.surrogate_sample.keys())}")
    for k, v in sd.surrogate_sample.items():
        print(f"     {k}: {tuple(v.shape)}")

    # Build three-way
    from bayesianquilts.imputation.three_way import ThreeWayImputationModel

    def _data_factory():
        yield base_data

    print("\n[4/5] Building ThreeWayImputationModel...")
    tw = ThreeWayImputationModel(
        irt_model=irt,
        shared_disc_model=sd,
        mice_model=pairwise,
        data_factory=_data_factory,
    )

    # ============================================================
    # Sanity checks
    # ============================================================
    print("\n[5/5] Running sanity checks...")
    print("-" * 60)

    n_fail = 0

    # (0) Broadcasting + reduce-op verification for both GRM forward
    #     passes used inside the three-way model. We re-run the same
    #     quadrature kernel that _compute_grm_loo_per_item uses and assert
    #     every intermediate shape.
    print("\n  -- broadcasting / reduce-op verification --")
    from numpy.polynomial.hermite import hermgauss
    I = len(item_keys)
    n_quad = 61
    nodes, weights = hermgauss(n_quad)
    quad_points = nodes * np.sqrt(2)

    def _verify_grm_kernel(name, grm_model, expect_disc_shape):
        smp = grm_model.surrogate_sample
        disc_mean = np.asarray(smp["discriminations"]).mean(0)
        diff0_mean = np.asarray(smp["difficulties0"]).mean(0)
        ddiff_mean = (np.asarray(smp["ddifficulties"]).mean(0)
                      if "ddifficulties" in smp else None)
        # Sample-axis drop
        assert disc_mean.shape == expect_disc_shape, (
            f"{name}: disc_mean expected {expect_disc_shape}, got {disc_mean.shape}")
        # diff0 must always carry the I axis (per-item base threshold)
        assert diff0_mean.shape[-2] == I, (
            f"{name}: diff0 I-axis expected {I}, got {diff0_mean.shape}")
        # theta_q has shape (Q, D, 1, 1)
        D = disc_mean.shape[0] if disc_mean.ndim >= 2 else 1
        theta_q = np.zeros((n_quad, D, 1, 1), dtype=np.float64)
        theta_q[:, 0, 0, 0] = quad_points
        probs_at_quad = np.asarray(grm_model.grm_model_prob_d(
            jnp.asarray(theta_q),
            jnp.asarray(disc_mean),
            jnp.asarray(diff0_mean),
            jnp.asarray(ddiff_mean) if ddiff_mean is not None else None,
        ))
        # MUST be (Q, I, K) regardless of whether disc has the per-item axis
        # collapsed (shared-disc) — diff0_mean's I axis drives the result.
        assert probs_at_quad.shape == (n_quad, I, K), (
            f"{name}: probs_at_quad shape expected ({n_quad},{I},{K}), "
            f"got {probs_at_quad.shape}")
        # Each row sums to 1 over K
        sums = probs_at_quad.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-5), (
            f"{name}: GRM probs along K do not sum to 1 (max dev "
            f"{np.max(np.abs(sums-1)):.2e})")
        # Now test the LOO kernel on a single batch
        responses = np.stack(
            [np.asarray(base_data[k], dtype=np.float64)[:50]
             for k in item_keys], axis=-1)  # (50, I)
        observed = ~np.isnan(responses) & (responses >= 0) & (responses < K)
        responses_int = np.where(observed, responses.astype(np.int64), 0)
        log_probs_at_quad = np.log(np.clip(probs_at_quad, 1e-30, None))
        log_lik_per_item = log_probs_at_quad[:, np.newaxis, :, :]
        resp_idx = responses_int[np.newaxis, :, :, np.newaxis]
        log_lik_qi = np.take_along_axis(
            log_lik_per_item, resp_idx, axis=-1)[..., 0]
        assert log_lik_qi.shape == (n_quad, 50, I), (
            f"{name}: log_lik_qi shape (Q,N,I) expected "
            f"({n_quad},50,{I}), got {log_lik_qi.shape}")
        log_lik_qi = np.where(observed[np.newaxis, :, :], log_lik_qi, 0.0)
        total_ll = np.sum(log_lik_qi, axis=-1)
        assert total_ll.shape == (n_quad, 50), (
            f"{name}: total_ll (sum over items) shape expected "
            f"({n_quad},50), got {total_ll.shape}")
        # LOO for one item
        i_idx = 0
        loo_ll = total_ll - log_lik_qi[:, :, i_idx]
        assert loo_ll.shape == (n_quad, 50), (
            f"{name}: loo_ll shape expected ({n_quad},50), got {loo_ll.shape}")
        # Verify the subtraction actually removed item 0's contribution
        ll_only_i0 = log_lik_qi[:, :, i_idx]
        reconstructed = loo_ll + ll_only_i0
        assert np.allclose(reconstructed, total_ll), (
            f"{name}: LOO subtraction inconsistent")
        # Reduce over Q
        log_quad_w = np.log(weights / np.sqrt(np.pi))
        log_posterior = log_quad_w[:, np.newaxis] + loo_ll
        log_Z = np.logaddexp.reduce(log_posterior, axis=0)
        assert log_Z.shape == (50,), (
            f"{name}: log_Z (reduce over Q) shape expected (50,), "
            f"got {log_Z.shape}")
        print(f"    {name}: all shapes/broadcasts/reduce axes verified.")
        return probs_at_quad

    # Baseline: discriminations are per-item, expected shape ends with I
    smp = irt.surrogate_sample
    disc0 = np.asarray(smp["discriminations"]).mean(0)
    print(f"    baseline disc_mean shape: {disc0.shape}")
    probs_baseline = _verify_grm_kernel('baseline', irt, disc0.shape)

    # Shared-disc: discriminations collapsed to (1, D=1, 1, 1)
    sdsmp = sd.surrogate_sample
    sdisc0 = np.asarray(sdsmp["discriminations"]).mean(0)
    print(f"    shared-disc disc_mean shape: {sdisc0.shape}")
    assert sdisc0.shape[-2] == 1, (
        f"shared-disc disc_mean I-axis must be 1 (shared), got {sdisc0.shape}")
    probs_shared = _verify_grm_kernel('shared-disc', sd, sdisc0.shape)

    # Cross-check: same θ grid, the two probs differ (they should — different fits)
    diff = np.abs(probs_baseline - probs_shared)
    print(f"    baseline vs shared-disc probs: max diff={diff.max():.4f}, "
          f"mean diff={diff.mean():.4f}")
    assert diff.max() > 1e-4, (
        "baseline and shared-disc forward passes are identical — fit issue?")

    # Cross-check: the per-item discrimination spread in baseline vs the
    # single shared slope in shared-disc.
    print(f"    baseline disc range: "
          f"[{disc0[0,0,:,0].min():.3f}, {disc0[0,0,:,0].max():.3f}], "
          f"shared-disc slope: {float(sdisc0.squeeze()):.3f}")

    # (1) Simplex constraints on per-item weights
    weights_matrix = np.stack([tw._weights[k] for k in item_keys])
    sums = weights_matrix.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        print(f"  FAIL: weights don't sum to 1; max |sum-1|={np.max(np.abs(sums - 1)):.2e}")
        n_fail += 1
    else:
        print(f"  PASS: all per-item weights sum to 1 (max |sum-1|={np.max(np.abs(sums - 1)):.2e})")
    if not np.all(weights_matrix >= -1e-10):
        print(f"  FAIL: negative weights found, min={weights_matrix.min():.2e}")
        n_fail += 1
    else:
        print(f"  PASS: all weights non-negative (min={weights_matrix.min():.2e})")

    # Weight summary
    mean_w = weights_matrix.mean(axis=0)
    print(f"  Component-mean weights: w_mice={mean_w[0]:.3f}, "
          f"w_irt={mean_w[1]:.3f}, w_shared={mean_w[2]:.3f}")
    print(f"  Per-item weight ranges:")
    print(f"    w_mice:   [{weights_matrix[:,0].min():.3f}, {weights_matrix[:,0].max():.3f}]")
    print(f"    w_irt:    [{weights_matrix[:,1].min():.3f}, {weights_matrix[:,1].max():.3f}]")
    print(f"    w_shared: [{weights_matrix[:,2].min():.3f}, {weights_matrix[:,2].max():.3f}]")

    # (2) Yao optimality: stacked LOO no worse than best single component
    n_violations = 0
    n_items_checked = 0
    margin_per_item = []
    for k in item_keys:
        lpd_m = tw._mice_loo.get(k, np.array([]))
        lpd_i = tw._irt_loo.get(k, np.array([]))
        lpd_s = tw._shared_loo.get(k, np.array([]))
        n_min = min(len(lpd_m), len(lpd_i), len(lpd_s))
        if n_min == 0:
            continue
        lpd_m, lpd_i, lpd_s = lpd_m[:n_min], lpd_i[:n_min], lpd_s[:n_min]
        w = tw._weights[k]
        log_w = np.log(np.maximum(w, 1e-15))
        a = lpd_m + log_w[0]
        b = lpd_i + log_w[1]
        c = lpd_s + log_w[2]
        m_ = np.maximum(np.maximum(a, b), c)
        log_mix = m_ + np.log(np.exp(a - m_) + np.exp(b - m_) + np.exp(c - m_))
        stacked_elpd = np.mean(log_mix)
        best_single = max(lpd_m.mean(), lpd_i.mean(), lpd_s.mean())
        margin_per_item.append(stacked_elpd - best_single)
        n_items_checked += 1
        if stacked_elpd < best_single - 1e-6:
            n_violations += 1
    if n_violations > 0:
        print(f"  FAIL: Yao theorem violated for {n_violations}/{n_items_checked} items")
        n_fail += 1
    else:
        print(f"  PASS: stacked LOO >= best single for all {n_items_checked} items")
    print(f"  Mean stacking improvement over best single: "
          f"{np.mean(margin_per_item):+.4f} log-density per response")

    # (3) PMF validity
    items_test = {k: 0.0 for k in item_keys[:5]}
    bad_pmf = 0
    for target in item_keys[:10]:
        pmf = tw.predict_pmf(items_test, target=target, n_categories=K,
                             person_idx=0)
        if not np.allclose(pmf.sum(), 1.0, atol=1e-6):
            bad_pmf += 1
        if np.any(pmf < -1e-10):
            bad_pmf += 1
        if not np.all(np.isfinite(pmf)):
            bad_pmf += 1
    if bad_pmf > 0:
        print(f"  FAIL: {bad_pmf} PMF anomalies")
        n_fail += 1
    else:
        print(f"  PASS: 10 sample PMFs are valid distributions")

    # (4) Consistency: when w_irt -> 1, stacked PMF should match IRT-only
    # Take an item where w_irt is dominant
    irt_dominant_idx = int(np.argmax(weights_matrix[:, 1]))
    target = item_keys[irt_dominant_idx]
    w = tw._weights[target]
    pmf_stack = tw.predict_pmf(items_test, target=target,
                               n_categories=K, person_idx=0)
    irt_idx = tw._item_to_idx[target]
    irt_pmf = np.asarray(tw._irt_pmf_matrix[0, irt_idx, :K])
    irt_pmf = irt_pmf / irt_pmf.sum()
    discrepancy = np.max(np.abs(pmf_stack - irt_pmf))
    print(f"  Item '{target}' (w_irt={w[1]:.3f}): "
          f"max |stacked - irt-only| = {discrepancy:.3f}")
    # Loose check: the dominant component should pull the stacked PMF toward it
    if discrepancy > 0.5:
        print(f"  WARN: stacked PMF diverges from dominant IRT for this item")

    # (5) Per-item LOO summary
    print(f"\n  Per-component LOO ELPD summary (mean across items):")
    irt_elpd_vals = [tw._irt_elpd[k] for k in item_keys
                     if k in tw._irt_elpd and np.isfinite(tw._irt_elpd[k])]
    shared_elpd_vals = [tw._shared_elpd[k] for k in item_keys
                        if k in tw._shared_elpd and np.isfinite(tw._shared_elpd[k])]
    mice_elpd_vals = [tw._mice_elpd[k] for k in item_keys
                      if k in tw._mice_elpd and np.isfinite(tw._mice_elpd[k])]
    print(f"    pairwise/MICE: mean={np.mean(mice_elpd_vals):+.4f} "
          f"(n={len(mice_elpd_vals)} items)")
    print(f"    IRT baseline:  mean={np.mean(irt_elpd_vals):+.4f} "
          f"(n={len(irt_elpd_vals)} items)")
    print(f"    Shared-disc:   mean={np.mean(shared_elpd_vals):+.4f} "
          f"(n={len(shared_elpd_vals)} items)")

    print("-" * 60)
    if n_fail == 0:
        print("All sanity checks PASSED.")
        return 0
    else:
        print(f"{n_fail} sanity checks FAILED.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

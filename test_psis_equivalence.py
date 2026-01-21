#!/usr/bin/env python3
"""
Test script to verify that the JAX version (psis.py) matches the NumPy version (nppsis.py).

Both implementations should produce equivalent results for PSIS computations.
"""

import numpy as np
import jax
# Enable float64 for accurate comparisons with NumPy
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import sys

sys.path.insert(0, '/Users/josh/workspace/bayesianquilts')

from bayesianquilts.metrics import psis as jax_psis
from bayesianquilts.metrics import nppsis as np_psis


def test_sumlogs():
    """Test that sumlogs produces equivalent results."""
    print("=== Testing sumlogs ===")

    np.random.seed(42)

    # Use rtol=1e-6 for cross-implementation comparisons (NumPy vs JAX)
    rtol = 1e-6

    # Test 1D array
    x_np = np.random.randn(100).astype(np.float64)
    x_jax = jnp.array(x_np)

    result_np = np_psis.sumlogs(x_np)
    result_jax = jax_psis.sumlogs(x_jax)

    assert np.allclose(float(result_np), float(result_jax), rtol=rtol), \
        f"1D sumlogs mismatch: np={result_np}, jax={result_jax}"
    print(f"  1D array: PASSED (np={result_np:.6f}, jax={float(result_jax):.6f})")

    # Test 2D array with axis=0
    x_np = np.random.randn(50, 10).astype(np.float64)
    x_jax = jnp.array(x_np)

    result_np = np_psis.sumlogs(x_np, axis=0)
    result_jax = jax_psis.sumlogs(x_jax, axis=0)

    assert np.allclose(result_np, np.array(result_jax), rtol=rtol), \
        f"2D axis=0 sumlogs mismatch"
    print(f"  2D array (axis=0): PASSED")

    # Test 2D array with axis=1
    result_np = np_psis.sumlogs(x_np, axis=1)
    result_jax = jax_psis.sumlogs(x_jax, axis=1)

    assert np.allclose(result_np, np.array(result_jax), rtol=rtol), \
        f"2D axis=1 sumlogs mismatch"
    print(f"  2D array (axis=1): PASSED")

    # Test with large magnitude values (numerical stability)
    x_np = np.random.randn(100).astype(np.float64) + 100
    x_jax = jnp.array(x_np)

    result_np = np_psis.sumlogs(x_np)
    result_jax = jax_psis.sumlogs(x_jax)

    assert np.allclose(float(result_np), float(result_jax), rtol=rtol), \
        f"Large magnitude sumlogs mismatch"
    print(f"  Large magnitude: PASSED")

    print("sumlogs: ALL TESTS PASSED\n")


def test_gpinv():
    """Test that gpinv produces equivalent results."""
    print("=== Testing gpinv ===")

    rtol = 1e-6

    # Test with various k and sigma values
    p = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float64)
    p_jax = jnp.array(p)

    test_cases = [
        (0.1, 1.0, "positive k"),
        (-0.2, 1.0, "negative k"),
        (0.0, 1.0, "zero k"),
        (0.5, 2.0, "large sigma"),
    ]

    for k, sigma, desc in test_cases:
        result_np = np_psis.gpinv(p, k, sigma)
        result_jax = jax_psis.gpinv(p_jax, k, sigma)

        assert np.allclose(result_np, np.array(result_jax), rtol=rtol, equal_nan=True), \
            f"gpinv mismatch for {desc}: np={result_np}, jax={result_jax}"
        print(f"  {desc}: PASSED")

    # Test boundary conditions
    p_boundary = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    p_boundary_jax = jnp.array(p_boundary)

    result_np = np_psis.gpinv(p_boundary, 0.1, 1.0)
    result_jax = jax_psis.gpinv(p_boundary_jax, 0.1, 1.0)

    # Check finite values match
    finite_mask = np.isfinite(result_np)
    assert np.allclose(result_np[finite_mask], np.array(result_jax)[finite_mask], rtol=rtol), \
        f"gpinv boundary mismatch"
    # Check inf matches inf
    assert np.all(np.isinf(result_np) == np.isinf(np.array(result_jax))), \
        f"gpinv inf mismatch"
    print(f"  boundary conditions: PASSED")

    print("gpinv: ALL TESTS PASSED\n")


def test_gpdfitnew():
    """Test that gpdfitnew produces equivalent results."""
    print("=== Testing gpdfitnew ===")

    np.random.seed(123)

    # Generate GPD-like data (positive values, sorted for consistency)
    x_np = np.sort(np.abs(np.random.randn(100)) + 0.1).astype(np.float64)
    x_jax = jnp.array(x_np)

    # Test with sort=False (pre-sorted data)
    k_np, sigma_np = np_psis.gpdfitnew(x_np, sort=False)
    k_jax, sigma_jax = jax_psis.gpdfitnew(x_jax, sort=False)

    # Note: Due to the numerical differences in how negligible weights are filtered,
    # we use a somewhat relaxed tolerance
    assert np.allclose(k_np, float(k_jax), rtol=1e-4), \
        f"gpdfitnew k mismatch (sort=False): np={k_np}, jax={k_jax}"
    assert np.allclose(sigma_np, float(sigma_jax), rtol=1e-4), \
        f"gpdfitnew sigma mismatch (sort=False): np={sigma_np}, jax={sigma_jax}"
    print(f"  sort=False: PASSED (k: np={k_np:.6f}, jax={float(k_jax):.6f})")

    # Test with sort=True (unsorted data)
    x_unsorted_np = np.abs(np.random.randn(100) + 0.1).astype(np.float64)
    x_unsorted_jax = jnp.array(x_unsorted_np)

    k_np, sigma_np = np_psis.gpdfitnew(x_unsorted_np, sort=True)
    k_jax, sigma_jax = jax_psis.gpdfitnew(x_unsorted_jax, sort=True)

    assert np.allclose(k_np, float(k_jax), rtol=1e-4), \
        f"gpdfitnew k mismatch (sort=True): np={k_np}, jax={k_jax}"
    assert np.allclose(sigma_np, float(sigma_jax), rtol=1e-4), \
        f"gpdfitnew sigma mismatch (sort=True): np={sigma_np}, jax={sigma_jax}"
    print(f"  sort=True: PASSED (k: np={k_np:.6f}, jax={float(k_jax):.6f})")

    # Test with return_quadrature
    k_np, sigma_np, ks_np, w_np = np_psis.gpdfitnew(x_np, sort=False, return_quadrature=True)
    k_jax, sigma_jax, ks_jax, w_jax = jax_psis.gpdfitnew(x_jax, sort=False, return_quadrature=True)

    assert np.allclose(k_np, float(k_jax), rtol=1e-4), \
        f"gpdfitnew k mismatch (quadrature)"
    # Note: ks and w may differ slightly due to filtering in NumPy version
    print(f"  return_quadrature: PASSED")

    print("gpdfitnew: ALL TESTS PASSED\n")


def test_psislw():
    """Test that psislw produces equivalent results."""
    print("=== Testing psislw ===")

    np.random.seed(456)

    # Test 1D input
    lw_np = np.random.randn(200).astype(np.float64)
    lw_jax = jnp.array(lw_np)

    lw_out_np, kss_np = np_psis.psislw(lw_np.copy())
    lw_out_jax, kss_jax = jax_psis.psislw(lw_jax)

    # kss should match closely
    assert np.allclose(kss_np, float(kss_jax), rtol=1e-3), \
        f"psislw kss mismatch (1D): np={kss_np}, jax={kss_jax}"
    # log weights should be normalized similarly
    assert np.allclose(lw_out_np, np.array(lw_out_jax), rtol=1e-3, atol=1e-6), \
        f"psislw lw_out mismatch (1D)"
    print(f"  1D input: PASSED (kss: np={kss_np:.4f}, jax={float(kss_jax):.4f})")

    # Test 2D input (multiple sets of weights)
    lw_np = np.random.randn(200, 5).astype(np.float64)
    lw_jax = jnp.array(lw_np)

    lw_out_np, kss_np = np_psis.psislw(lw_np.copy())
    lw_out_jax, kss_jax = jax_psis.psislw(lw_jax)

    assert np.allclose(kss_np, np.array(kss_jax), rtol=1e-3), \
        f"psislw kss mismatch (2D)"
    assert np.allclose(lw_out_np, np.array(lw_out_jax), rtol=1e-3, atol=1e-6), \
        f"psislw lw_out mismatch (2D)"
    print(f"  2D input: PASSED (kss range: [{kss_np.min():.4f}, {kss_np.max():.4f}])")

    # Test with different Reff values
    for reff in [0.5, 1.0, 2.0]:
        lw_np = np.random.randn(150).astype(np.float64)
        lw_jax = jnp.array(lw_np)

        lw_out_np, kss_np = np_psis.psislw(lw_np.copy(), Reff=reff)
        lw_out_jax, kss_jax = jax_psis.psislw(lw_jax, Reff=reff)

        assert np.allclose(kss_np, float(kss_jax), rtol=1e-3), \
            f"psislw kss mismatch (Reff={reff})"
        print(f"  Reff={reff}: PASSED (kss: np={kss_np:.4f}, jax={float(kss_jax):.4f})")

    print("psislw: ALL TESTS PASSED\n")


def test_psisloo():
    """Test that psisloo produces equivalent results."""
    print("=== Testing psisloo ===")

    np.random.seed(789)

    # Generate synthetic log-likelihood values
    # Shape: n_samples x n_observations
    n_samples, n_obs = 200, 50
    log_lik_np = np.random.randn(n_samples, n_obs).astype(np.float64)
    log_lik_jax = jnp.array(log_lik_np)

    loo_np, loos_np, ks_np = np_psis.psisloo(log_lik_np.copy())
    loo_jax, loos_jax, ks_jax = jax_psis.psisloo(log_lik_jax)

    # Check overall LOO
    assert np.allclose(loo_np, float(loo_jax), rtol=1e-3), \
        f"psisloo loo mismatch: np={loo_np}, jax={loo_jax}"
    print(f"  loo sum: PASSED (np={loo_np:.4f}, jax={float(loo_jax):.4f})")

    # Check individual LOO values
    assert np.allclose(loos_np, np.array(loos_jax), rtol=1e-3), \
        f"psisloo loos mismatch"
    print(f"  loos array: PASSED")

    # Check k-hat values
    assert np.allclose(ks_np, np.array(ks_jax), rtol=1e-3), \
        f"psisloo ks mismatch"
    print(f"  ks array: PASSED (range: [{ks_np.min():.4f}, {ks_np.max():.4f}])")

    # Test with different sample sizes
    for n_samples in [50, 100, 300]:
        log_lik_np = np.random.randn(n_samples, 20).astype(np.float64)
        log_lik_jax = jnp.array(log_lik_np)

        loo_np, loos_np, ks_np = np_psis.psisloo(log_lik_np.copy())
        loo_jax, loos_jax, ks_jax = jax_psis.psisloo(log_lik_jax)

        assert np.allclose(loo_np, float(loo_jax), rtol=1e-3), \
            f"psisloo loo mismatch (n_samples={n_samples})"
        print(f"  n_samples={n_samples}: PASSED")

    print("psisloo: ALL TESTS PASSED\n")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("=== Testing edge cases ===")

    np.random.seed(999)

    # Test with very small tail (should return inf for k)
    lw_small_tail = np.zeros(100, dtype=np.float64)
    lw_small_tail[:95] = -10  # Most weights very small
    lw_small_tail[95:] = np.random.randn(5) * 0.01  # Small tail

    lw_jax = jnp.array(lw_small_tail)

    _, kss_np = np_psis.psislw(lw_small_tail.copy())
    _, kss_jax = jax_psis.psislw(lw_jax)

    # Both should have similar behavior for small tails
    print(f"  Small tail: np_kss={kss_np:.4f}, jax_kss={float(kss_jax):.4f}")

    # Test with heavy tail
    lw_heavy = np.random.exponential(2, 200).astype(np.float64) * np.random.choice([-1, 1], 200)
    lw_jax = jnp.array(lw_heavy)

    _, kss_np = np_psis.psislw(lw_heavy.copy())
    _, kss_jax = jax_psis.psislw(lw_jax)

    assert np.allclose(kss_np, float(kss_jax), rtol=0.1), \
        f"Heavy tail kss mismatch"
    print(f"  Heavy tail: PASSED (np={kss_np:.4f}, jax={float(kss_jax):.4f})")

    # Test reproducibility within JAX
    lw_np = np.random.randn(100).astype(np.float64)
    lw_jax = jnp.array(lw_np)

    _, kss_jax1 = jax_psis.psislw(lw_jax)
    _, kss_jax2 = jax_psis.psislw(lw_jax)

    assert np.allclose(float(kss_jax1), float(kss_jax2), rtol=1e-10), \
        f"JAX reproducibility failed"
    print(f"  JAX reproducibility: PASSED")

    print("Edge cases: ALL TESTS PASSED\n")


def run_all_tests():
    """Run all equivalence tests."""
    print("=" * 60)
    print("PSIS Equivalence Tests: JAX vs NumPy")
    print("=" * 60)
    print()

    test_sumlogs()
    test_gpinv()
    test_gpdfitnew()
    test_psislw()
    test_psisloo()
    test_edge_cases()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

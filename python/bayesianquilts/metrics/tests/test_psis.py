"""Tests for Pareto Smoothed Importance Sampling (PSIS) module.

Tests cover:
- sumlogs: Log-sum-exp computation
- gpinv: Inverse generalized Pareto distribution
- gpdfitnew: GPD parameter estimation
- psislw: Pareto smoothed importance sampling weights
- psisloo: PSIS leave-one-out cross-validation
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from bayesianquilts.metrics import psis

# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


class TestSumlogs:
    """Tests for the sumlogs function (numerically stable log-sum-exp)."""

    def test_basic_sum(self):
        """Test basic log-sum-exp computation."""
        x = jnp.array([1.0, 2.0, 3.0])
        result = psis.sumlogs(x)
        expected = jnp.log(jnp.sum(jnp.exp(x)))
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_numerical_stability_large_values(self):
        """Test numerical stability with large log values."""
        x = jnp.array([1000.0, 1001.0, 1002.0])
        result = psis.sumlogs(x)
        # Direct computation would overflow, but sumlogs should handle it
        assert jnp.isfinite(result)
        # Check relative difference
        expected = 1002.0 + jnp.log(jnp.exp(-2) + jnp.exp(-1) + 1)
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_numerical_stability_small_values(self):
        """Test numerical stability with very negative log values."""
        x = jnp.array([-1000.0, -1001.0, -1002.0])
        result = psis.sumlogs(x)
        assert jnp.isfinite(result)
        expected = -1000.0 + jnp.log(1 + jnp.exp(-1) + jnp.exp(-2))
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_2d_array_axis0(self):
        """Test sumlogs along axis 0."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = psis.sumlogs(x, axis=0)
        expected = jnp.log(jnp.sum(jnp.exp(x), axis=0))
        assert result.shape == (2,)
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_2d_array_axis1(self):
        """Test sumlogs along axis 1."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = psis.sumlogs(x, axis=1)
        expected = jnp.log(jnp.sum(jnp.exp(x), axis=1))
        assert result.shape == (2,)
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_single_element(self):
        """Test sumlogs with a single element."""
        x = jnp.array([5.0])
        result = psis.sumlogs(x)
        assert jnp.allclose(result, 5.0, atol=1e-6)


class TestGpinv:
    """Tests for the inverse generalized Pareto distribution function."""

    def test_basic_quantiles(self):
        """Test basic quantile computation."""
        p = jnp.array([0.1, 0.5, 0.9])
        k = 0.5
        sigma = 1.0
        result = psis.gpinv(p, k, sigma)
        assert result.shape == (3,)
        assert jnp.all(jnp.isfinite(result))
        # Quantiles should be monotonically increasing
        assert jnp.all(jnp.diff(result) > 0)

    def test_boundary_p_zero(self):
        """Test that p=0 returns 0."""
        p = jnp.array([0.0])
        result = psis.gpinv(p, k=0.5, sigma=1.0)
        assert jnp.allclose(result, 0.0, atol=1e-10)

    def test_boundary_p_one_positive_k(self):
        """Test that p=1 with k>=0 returns inf."""
        p = jnp.array([1.0])
        result = psis.gpinv(p, k=0.5, sigma=1.0)
        assert jnp.isinf(result[0])

    def test_boundary_p_one_negative_k(self):
        """Test that p=1 with k<0 returns -sigma/k."""
        p = jnp.array([1.0])
        k = -0.5
        sigma = 1.0
        result = psis.gpinv(p, k, sigma)
        expected = -sigma / k
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_k_near_zero(self):
        """Test behavior when k is very close to zero (exponential case)."""
        p = jnp.array([0.5])
        k = 1e-15
        sigma = 1.0
        result = psis.gpinv(p, k, sigma)
        # Should approximate exponential distribution: -log(1-p)
        expected = -jnp.log(1 - p) * sigma
        assert jnp.allclose(result, expected, atol=1e-4)

    def test_negative_sigma(self):
        """Test that negative sigma returns NaN."""
        p = jnp.array([0.5])
        result = psis.gpinv(p, k=0.5, sigma=-1.0)
        assert jnp.all(jnp.isnan(result))

    def test_scalar_input(self):
        """Test with scalar probability input."""
        result = psis.gpinv(0.5, k=0.5, sigma=1.0)
        assert jnp.isfinite(result)


class TestGpdfitnew:
    """Tests for GPD parameter estimation."""

    def test_basic_fit(self):
        """Test basic GPD fitting with known parameters."""
        # Generate samples from GPD with known parameters
        key = jax.random.PRNGKey(42)
        n = 1000
        true_k = 0.3
        true_sigma = 1.0
        # Generate uniform samples and transform via inverse GPD
        u = jax.random.uniform(key, (n,))
        x = psis.gpinv(u, true_k, true_sigma)
        x = jnp.sort(x)

        k_est, sigma_est = psis.gpdfitnew(x, sort=False)

        # Estimates should be reasonably close to true values
        assert jnp.abs(k_est - true_k) < 0.2
        assert jnp.abs(sigma_est - true_sigma) < 0.5

    def test_requires_1d_input(self):
        """Test that 2D input raises ValueError."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="Invalid input array"):
            psis.gpdfitnew(x)

    def test_requires_multiple_samples(self):
        """Test that single sample raises ValueError."""
        x = jnp.array([1.0])
        with pytest.raises(ValueError, match="Invalid input array"):
            psis.gpdfitnew(x)

    def test_with_sorting(self):
        """Test fitting with internal sorting."""
        key = jax.random.PRNGKey(123)
        x = jax.random.uniform(key, (100,)) * 10
        k1, sigma1 = psis.gpdfitnew(x, sort=True)
        k2, sigma2 = psis.gpdfitnew(jnp.sort(x), sort=False)
        assert jnp.allclose(k1, k2, atol=1e-6)
        assert jnp.allclose(sigma1, sigma2, atol=1e-6)

    def test_return_quadrature(self):
        """Test returning quadrature points and weights."""
        key = jax.random.PRNGKey(456)
        x = jax.random.uniform(key, (100,)) * 10
        k, sigma, ks, w = psis.gpdfitnew(x, return_quadrature=True)
        assert jnp.isfinite(k)
        assert jnp.isfinite(sigma)
        assert len(ks) == len(w)
        assert jnp.allclose(jnp.sum(w), 1.0, atol=1e-6)


class TestPsislw:
    """Tests for Pareto smoothed importance sampling weights."""

    def test_basic_smoothing_1d(self):
        """Test basic PSIS with 1D input."""
        key = jax.random.PRNGKey(42)
        n = 500
        # Generate log weights with heavy tail
        lw = jax.random.normal(key, (n,)) * 2
        lw_out, khat = psis.psislw(lw)

        assert lw_out.shape == (n,)
        assert jnp.isfinite(khat)
        # Smoothed weights should be normalized
        assert jnp.allclose(psis.sumlogs(lw_out), 0.0, atol=1e-6)

    def test_basic_smoothing_2d(self):
        """Test basic PSIS with 2D input."""
        key = jax.random.PRNGKey(42)
        n, m = 500, 10
        lw = jax.random.normal(key, (n, m)) * 2
        lw_out, khat = psis.psislw(lw)

        assert lw_out.shape == (n, m)
        assert khat.shape == (m,)
        # Each column should be normalized
        for i in range(m):
            assert jnp.allclose(psis.sumlogs(lw_out[:, i]), 0.0, atol=1e-6)

    def test_well_behaved_weights(self):
        """Test with well-behaved weights (low k-hat)."""
        key = jax.random.PRNGKey(42)
        n = 1000
        # Generate nearly uniform weights
        lw = jax.random.normal(key, (n,)) * 0.1
        lw_out, khat = psis.psislw(lw)

        # Well-behaved weights should have low k-hat
        assert khat < 0.7

    def test_problematic_weights(self):
        """Test with problematic weights (high k-hat warning)."""
        key = jax.random.PRNGKey(42)
        n = 500
        # Generate weights with very heavy tail
        lw = jax.random.exponential(key, (n,)) * 5
        lw_out, khat = psis.psislw(lw)

        # Should still produce finite output
        assert jnp.all(jnp.isfinite(lw_out))
        # k-hat indicates problematic weights
        assert khat > 0.5

    def test_requires_multiple_weights(self):
        """Test that single weight raises ValueError."""
        lw = jnp.array([1.0])
        with pytest.raises(ValueError, match="More than one log-weight needed"):
            psis.psislw(lw)

    def test_invalid_dimensions(self):
        """Test that 3D input raises ValueError."""
        lw = jnp.ones((10, 10, 10))
        with pytest.raises(ValueError, match="must be 1 or 2 dimensional"):
            psis.psislw(lw)

    def test_reff_parameter(self):
        """Test Reff parameter affects results."""
        key = jax.random.PRNGKey(42)
        n = 500
        lw = jax.random.normal(key, (n,)) * 2

        lw_out1, khat1 = psis.psislw(lw, Reff=1.0)
        lw_out2, khat2 = psis.psislw(lw, Reff=0.5)

        # Different Reff should give different results
        assert not jnp.allclose(khat1, khat2)

    def test_overwrite_flag(self):
        """Test overwrite_lw parameter."""
        key = jax.random.PRNGKey(42)
        lw = jax.random.normal(key, (100,))

        # With overwrite=False (default), input unchanged
        lw_copy = jnp.copy(lw)
        lw_out, _ = psis.psislw(lw, overwrite_lw=False)
        # Note: JAX arrays are immutable so input is never modified

        # With overwrite=True, should still work
        lw_out2, _ = psis.psislw(lw, overwrite_lw=True)
        assert jnp.allclose(lw_out, lw_out2, atol=1e-6)


class TestPsisloo:
    """Tests for PSIS leave-one-out cross-validation."""

    def test_basic_loo(self):
        """Test basic PSIS-LOO computation."""
        key = jax.random.PRNGKey(42)
        n_samples, n_data = 500, 50
        # Generate log likelihood values
        log_lik = jax.random.normal(key, (n_samples, n_data))

        loo, loos, ks = psis.psisloo(log_lik)

        assert jnp.isfinite(loo)
        assert loos.shape == (n_data,)
        assert ks.shape == (n_data,)
        # loo should be sum of loos
        assert jnp.allclose(loo, jnp.sum(loos), atol=1e-5)

    def test_loo_with_uniform_likelihood(self):
        """Test LOO with uniform likelihood (no outliers)."""
        key = jax.random.PRNGKey(42)
        n_samples, n_data = 1000, 20
        # Nearly uniform log likelihood
        log_lik = jax.random.normal(key, (n_samples, n_data)) * 0.1

        loo, loos, ks = psis.psisloo(log_lik)

        # Uniform likelihood should have low k-hat values
        assert jnp.all(ks < 0.7)

    def test_loo_shape_consistency(self):
        """Test output shapes are consistent."""
        key = jax.random.PRNGKey(42)
        for n_data in [10, 50, 100]:
            log_lik = jax.random.normal(key, (200, n_data))
            loo, loos, ks = psis.psisloo(log_lik)

            assert loos.shape == (n_data,)
            assert ks.shape == (n_data,)


class TestNumericalStability:
    """Tests for numerical stability edge cases."""

    def test_extreme_log_weights(self):
        """Test PSIS with extreme log weight values."""
        n = 100
        # Mix of very large and very small log weights
        lw = jnp.concatenate([
            jnp.full(50, -500.0),
            jnp.full(50, 0.0)
        ])
        lw_out, khat = psis.psislw(lw)

        assert jnp.all(jnp.isfinite(lw_out))
        assert jnp.isfinite(khat)

    def test_identical_weights(self):
        """Test with all identical log weights."""
        n = 100
        lw = jnp.zeros(n)
        lw_out, khat = psis.psislw(lw)

        # Uniform weights should remain uniform
        assert jnp.all(jnp.isfinite(lw_out))
        # All weights should be equal after normalization
        weights = jnp.exp(lw_out)
        assert jnp.allclose(weights, weights[0], atol=1e-6)

    def test_float64_precision(self):
        """Test that computations use float64 for accuracy."""
        key = jax.random.PRNGKey(42)
        # Use float32 input
        lw = jax.random.normal(key, (500,), dtype=jnp.float32)
        lw_out, khat = psis.psislw(lw)

        # Output should be float64
        assert lw_out.dtype == jnp.float64
        assert khat.dtype == jnp.float64

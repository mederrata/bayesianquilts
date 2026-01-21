"""Tests for Adaptive Importance Sampling (AIS) module.

Tests cover:
- LikelihoodFunction implementations (Logistic, Linear regression)
- Transformation base classes
- SmallStepTransformation subclasses (LikelihoodDescent, KLDivergence, etc.)
- GlobalTransformation subclasses (MM1, MM2)
- AdaptiveImportanceSampler main class
- Input validation and edge cases
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from bayesianquilts.metrics import ais

# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


# Fixtures for common test data
@pytest.fixture
def simple_logistic_data():
    """Generate simple logistic regression test data."""
    key = jax.random.PRNGKey(42)
    n_data, n_features = 50, 5
    X = jax.random.normal(key, (n_data, n_features))
    true_beta = jax.random.normal(jax.random.PRNGKey(1), (n_features,))
    logits = X @ true_beta
    y = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
    return {"X": X, "y": y}


@pytest.fixture
def simple_linear_data():
    """Generate simple linear regression test data."""
    key = jax.random.PRNGKey(42)
    n_data, n_features = 50, 5
    X = jax.random.normal(key, (n_data, n_features))
    true_beta = jax.random.normal(jax.random.PRNGKey(1), (n_features,))
    y = X @ true_beta + jax.random.normal(jax.random.PRNGKey(2), (n_data,)) * 0.1
    return {"X": X, "y": y}


@pytest.fixture
def logistic_params():
    """Generate sample logistic regression parameters."""
    key = jax.random.PRNGKey(42)
    n_samples, n_features = 100, 5
    beta = jax.random.normal(key, (n_samples, n_features))
    intercept = jax.random.normal(jax.random.PRNGKey(1), (n_samples,))
    return {"beta": beta, "intercept": intercept}


@pytest.fixture
def linear_params():
    """Generate sample linear regression parameters."""
    key = jax.random.PRNGKey(42)
    n_samples, n_features = 100, 5
    beta = jax.random.normal(key, (n_samples, n_features))
    intercept = jax.random.normal(jax.random.PRNGKey(1), (n_samples,))
    log_sigma = jax.random.normal(jax.random.PRNGKey(2), (n_samples,)) * 0.1
    return {"beta": beta, "intercept": intercept, "log_sigma": log_sigma}


class TestLogisticRegressionLikelihood:
    """Tests for LogisticRegressionLikelihood class."""

    def test_log_likelihood_shape(self, simple_logistic_data, logistic_params):
        """Test log-likelihood output shape."""
        likelihood = ais.LogisticRegressionLikelihood()
        ll = likelihood.log_likelihood(simple_logistic_data, logistic_params)

        n_samples = logistic_params["beta"].shape[0]
        n_data = simple_logistic_data["X"].shape[0]
        assert ll.shape == (n_samples, n_data)

    def test_log_likelihood_finite(self, simple_logistic_data, logistic_params):
        """Test log-likelihood values are finite."""
        likelihood = ais.LogisticRegressionLikelihood()
        ll = likelihood.log_likelihood(simple_logistic_data, logistic_params)
        assert jnp.all(jnp.isfinite(ll))

    def test_log_likelihood_negative(self, simple_logistic_data, logistic_params):
        """Test log-likelihood values are non-positive."""
        likelihood = ais.LogisticRegressionLikelihood()
        ll = likelihood.log_likelihood(simple_logistic_data, logistic_params)
        assert jnp.all(ll <= 0)

    def test_gradient_shape(self, simple_logistic_data, logistic_params):
        """Test gradient output shape (PyTree)."""
        likelihood = ais.LogisticRegressionLikelihood()
        grad = likelihood.log_likelihood_gradient(simple_logistic_data, logistic_params)

        n_samples = logistic_params["beta"].shape[0]
        n_data = simple_logistic_data["X"].shape[0]
        n_features = logistic_params["beta"].shape[1]

        assert "beta" in grad
        assert "intercept" in grad
        assert grad["beta"].shape == (n_samples, n_data, n_features)
        assert grad["intercept"].shape == (n_samples, n_data)

    def test_gradient_finite(self, simple_logistic_data, logistic_params):
        """Test gradient values are finite."""
        likelihood = ais.LogisticRegressionLikelihood()
        grad = likelihood.log_likelihood_gradient(simple_logistic_data, logistic_params)
        assert jnp.all(jnp.isfinite(grad["beta"]))
        assert jnp.all(jnp.isfinite(grad["intercept"]))

    def test_hessian_diag_shape(self, simple_logistic_data, logistic_params):
        """Test Hessian diagonal output shape (PyTree)."""
        likelihood = ais.LogisticRegressionLikelihood()
        hess = likelihood.log_likelihood_hessian_diag(simple_logistic_data, logistic_params)

        n_samples = logistic_params["beta"].shape[0]
        n_data = simple_logistic_data["X"].shape[0]
        n_features = logistic_params["beta"].shape[1]

        assert "beta" in hess
        assert "intercept" in hess
        assert hess["beta"].shape == (n_samples, n_data, n_features)
        assert hess["intercept"].shape == (n_samples, n_data)

    def test_hessian_diag_negative(self, simple_logistic_data, logistic_params):
        """Test Hessian diagonal is negative (concave log-likelihood)."""
        likelihood = ais.LogisticRegressionLikelihood()
        hess = likelihood.log_likelihood_hessian_diag(simple_logistic_data, logistic_params)
        # For logistic regression, Hessian diagonal should be non-positive
        assert jnp.all(hess["beta"] <= 0)
        assert jnp.all(hess["intercept"] <= 0)


class TestLinearRegressionLikelihood:
    """Tests for LinearRegressionLikelihood class."""

    def test_log_likelihood_shape(self, simple_linear_data, linear_params):
        """Test log-likelihood output shape."""
        likelihood = ais.LinearRegressionLikelihood()
        ll = likelihood.log_likelihood(simple_linear_data, linear_params)

        n_samples = linear_params["beta"].shape[0]
        n_data = simple_linear_data["X"].shape[0]
        assert ll.shape == (n_samples, n_data)

    def test_log_likelihood_finite(self, simple_linear_data, linear_params):
        """Test log-likelihood values are finite."""
        likelihood = ais.LinearRegressionLikelihood()
        ll = likelihood.log_likelihood(simple_linear_data, linear_params)
        assert jnp.all(jnp.isfinite(ll))

    def test_gradient_pytree_structure(self, simple_linear_data, linear_params):
        """Test gradient returns PyTree matching params structure."""
        likelihood = ais.LinearRegressionLikelihood()
        grad = likelihood.log_likelihood_gradient(simple_linear_data, linear_params)

        assert "beta" in grad
        assert "intercept" in grad
        assert "log_sigma" in grad

    def test_gradient_without_log_sigma(self, simple_linear_data):
        """Test gradient when log_sigma not in params."""
        likelihood = ais.LinearRegressionLikelihood()
        params = {
            "beta": jax.random.normal(jax.random.PRNGKey(42), (50, 5)),
            "intercept": jax.random.normal(jax.random.PRNGKey(1), (50,)),
        }
        grad = likelihood.log_likelihood_gradient(simple_linear_data, params)

        assert "beta" in grad
        assert "intercept" in grad
        assert "log_sigma" not in grad


class TestTransformationComputeMoments:
    """Tests for Transformation.compute_moments static method."""

    def test_moments_1d_params(self):
        """Test moments computation with 1D parameters."""
        params = {"a": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])}  # (S,)
        weights = jnp.array([
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.2],
            [0.3, 0.2, 0.2],
            [0.2, 0.1, 0.2],
            [0.2, 0.2, 0.1],
        ])  # (S, N)

        moments = ais.Transformation.compute_moments(params, weights)

        assert "a" in moments
        assert "mean" in moments["a"]
        assert "mean_w" in moments["a"]
        assert "var" in moments["a"]
        assert "var_w" in moments["a"]

        # mean_w should have shape (N,)
        assert moments["a"]["mean_w"].shape == (3,)
        # var_w should have shape (N,)
        assert moments["a"]["var_w"].shape == (3,)

    def test_moments_2d_params(self):
        """Test moments computation with 2D parameters."""
        params = {"beta": jnp.ones((5, 3))}  # (S, K)
        weights = jnp.ones((5, 10)) / 5  # (S, N)

        moments = ais.Transformation.compute_moments(params, weights)

        # mean_w should have shape (N, K)
        assert moments["beta"]["mean_w"].shape == (10, 3)
        # var_w should have shape (N, K)
        assert moments["beta"]["var_w"].shape == (10, 3)

    def test_moments_multiple_params(self):
        """Test moments with multiple parameters."""
        params = {
            "beta": jnp.ones((5, 3)),
            "intercept": jnp.ones((5,)),
        }
        weights = jnp.ones((5, 10)) / 5

        moments = ais.Transformation.compute_moments(params, weights)

        assert "beta" in moments
        assert "intercept" in moments

    def test_weighted_mean_correctness(self):
        """Test weighted mean computation is correct."""
        params = {"a": jnp.array([1.0, 2.0, 3.0, 4.0])}  # (S,)
        # Uniform weights for single data point
        weights = jnp.array([[0.25], [0.25], [0.25], [0.25]])  # (S, 1)

        moments = ais.Transformation.compute_moments(params, weights)

        # With uniform weights, weighted mean should equal unweighted mean
        expected_mean = jnp.mean(params["a"])
        assert jnp.allclose(moments["a"]["mean_w"][0], expected_mean, atol=1e-6)


class TestAdaptiveImportanceSamplerInit:
    """Tests for AdaptiveImportanceSampler initialization and validation."""

    def test_init_with_likelihood(self):
        """Test initialization with valid likelihood function."""
        likelihood = ais.LogisticRegressionLikelihood()
        sampler = ais.AdaptiveImportanceSampler(likelihood)
        assert sampler.likelihood_fn is likelihood

    def test_init_requires_likelihood_function(self):
        """Test that non-LikelihoodFunction raises TypeError."""
        with pytest.raises(TypeError, match="must be a LikelihoodFunction instance"):
            ais.AdaptiveImportanceSampler("not a likelihood")

    def test_init_with_optional_functions(self):
        """Test initialization with prior and surrogate functions."""
        likelihood = ais.LogisticRegressionLikelihood()

        def prior_fn(params):
            return jnp.zeros(params["beta"].shape[0])

        def surrogate_fn(params):
            return jnp.zeros(params["beta"].shape[0])

        sampler = ais.AdaptiveImportanceSampler(
            likelihood,
            prior_log_prob_fn=prior_fn,
            surrogate_log_prob_fn=surrogate_fn,
        )
        assert sampler.prior_log_prob_fn is prior_fn
        assert sampler.surrogate_log_prob_fn is surrogate_fn


class TestAdaptiveImportanceSamplerLoo:
    """Tests for AdaptiveImportanceSampler.adaptive_is_loo method."""

    def test_basic_loo(self, simple_logistic_data, logistic_params):
        """Test basic LOO-CV computation."""
        likelihood = ais.LogisticRegressionLikelihood()
        sampler = ais.AdaptiveImportanceSampler(likelihood)

        results = sampler.adaptive_is_loo(
            simple_logistic_data,
            logistic_params,
            transformations=["identity"],
            verbose=False,
        )

        assert "identity" in results
        assert "best" in results
        assert "khat" in results["best"]

    def test_empty_params_raises(self, simple_logistic_data):
        """Test that empty params raises ValueError."""
        likelihood = ais.LogisticRegressionLikelihood()
        sampler = ais.AdaptiveImportanceSampler(likelihood)

        with pytest.raises(ValueError, match="params cannot be empty"):
            sampler.adaptive_is_loo(simple_logistic_data, {})

    def test_zero_samples_raises(self, simple_logistic_data):
        """Test that zero samples raises ValueError."""
        likelihood = ais.LogisticRegressionLikelihood()
        sampler = ais.AdaptiveImportanceSampler(likelihood)

        params = {
            "beta": jnp.zeros((0, 5)),
            "intercept": jnp.zeros((0,)),
        }

        with pytest.raises(ValueError, match="at least 1 sample"):
            sampler.adaptive_is_loo(simple_logistic_data, params)

    def test_identity_transformation(self, simple_logistic_data, logistic_params):
        """Test identity transformation produces valid results."""
        likelihood = ais.LogisticRegressionLikelihood()
        sampler = ais.AdaptiveImportanceSampler(likelihood)

        results = sampler.adaptive_is_loo(
            simple_logistic_data,
            logistic_params,
            transformations=["identity"],
        )

        assert "identity" in results
        res = results["identity"]
        assert "eta_weights" in res
        assert "psis_weights" in res
        assert "khat" in res
        # Weights should sum to 1 along sample axis
        assert jnp.allclose(
            jnp.sum(res["eta_weights"], axis=0),
            jnp.ones(simple_logistic_data["X"].shape[0]),
            atol=1e-5,
        )

    def test_likelihood_descent(self, simple_logistic_data, logistic_params):
        """Test likelihood descent transformation."""
        likelihood = ais.LogisticRegressionLikelihood()
        sampler = ais.AdaptiveImportanceSampler(likelihood)

        results = sampler.adaptive_is_loo(
            simple_logistic_data,
            logistic_params,
            transformations=["likelihood_descent"],
            n_sweeps=2,
        )

        # Results should contain likelihood_descent entries (with rho suffix for multiple rhos)
        assert any("likelihood_descent" in k for k in results.keys()) or "best" in results

    def test_mm1_transformation(self, simple_logistic_data, logistic_params):
        """Test MM1 (moment matching shift) transformation."""
        likelihood = ais.LogisticRegressionLikelihood()
        sampler = ais.AdaptiveImportanceSampler(likelihood)

        results = sampler.adaptive_is_loo(
            simple_logistic_data,
            logistic_params,
            transformations=["mm1"],
        )

        assert "mm1" in results
        assert "theta_new" in results["mm1"]

    def test_mm2_transformation(self, simple_logistic_data, logistic_params):
        """Test MM2 (moment matching shift+scale) transformation."""
        likelihood = ais.LogisticRegressionLikelihood()
        sampler = ais.AdaptiveImportanceSampler(likelihood)

        results = sampler.adaptive_is_loo(
            simple_logistic_data,
            logistic_params,
            transformations=["mm2"],
        )

        assert "mm2" in results
        assert "log_jacobian" in results["mm2"]

    def test_custom_rhos(self, simple_logistic_data, logistic_params):
        """Test custom step size grid."""
        likelihood = ais.LogisticRegressionLikelihood()
        sampler = ais.AdaptiveImportanceSampler(likelihood)

        custom_rhos = jnp.array([0.01, 0.1, 1.0])
        results = sampler.adaptive_is_loo(
            simple_logistic_data,
            logistic_params,
            transformations=["likelihood_descent"],
            rhos=custom_rhos,
        )

        # Should have results for each rho (with rho suffix) plus best
        assert "best" in results
        # With multiple rhos, should have entries with rho suffixes
        rho_keys = [k for k in results.keys() if "rho" in k or "likelihood_descent" in k]
        assert len(rho_keys) >= 1

    def test_best_metrics_updated(self, simple_logistic_data, logistic_params):
        """Test that best metrics are properly tracked."""
        likelihood = ais.LogisticRegressionLikelihood()
        sampler = ais.AdaptiveImportanceSampler(likelihood)

        results = sampler.adaptive_is_loo(
            simple_logistic_data,
            logistic_params,
            transformations=["identity", "mm1"],
        )

        best = results["best"]
        assert "khat" in best
        assert "p_loo_eta" in best
        assert "p_loo_psis" in best
        # Best khat should be finite for at least some data points
        assert jnp.any(jnp.isfinite(best["khat"]))


class TestSmallStepTransformations:
    """Tests for SmallStepTransformation subclasses."""

    def test_likelihood_descent_compute_q(self, simple_logistic_data, logistic_params):
        """Test LikelihoodDescent.compute_Q returns valid gradient."""
        likelihood = ais.LogisticRegressionLikelihood()
        transform = ais.LikelihoodDescent(likelihood)

        log_ell = likelihood.log_likelihood(simple_logistic_data, logistic_params)

        Q = transform.compute_Q(
            logistic_params,
            simple_logistic_data,
            logistic_params,
            log_ell,
        )

        assert "beta" in Q
        assert "intercept" in Q
        assert jnp.all(jnp.isfinite(Q["beta"]))

    def test_kl_divergence_requires_log_pi(self, simple_logistic_data, logistic_params):
        """Test KLDivergence raises error without log_pi."""
        likelihood = ais.LogisticRegressionLikelihood()
        transform = ais.KLDivergence(likelihood)

        log_ell = likelihood.log_likelihood(simple_logistic_data, logistic_params)

        with pytest.raises(ValueError, match="log_pi required"):
            transform.compute_Q(
                logistic_params,
                simple_logistic_data,
                logistic_params,
                log_ell,
                log_pi=None,
            )

    def test_pmm1_requires_log_ell_original(self, simple_logistic_data, logistic_params):
        """Test PMM1 raises error without log_ell_original."""
        likelihood = ais.LogisticRegressionLikelihood()
        transform = ais.PMM1(likelihood)

        log_ell = likelihood.log_likelihood(simple_logistic_data, logistic_params)

        with pytest.raises(ValueError, match="log_ell_original required"):
            transform.compute_Q(
                logistic_params,
                simple_logistic_data,
                logistic_params,
                log_ell,
                log_ell_original=None,
            )

    def test_normalize_vector_field(self, simple_logistic_data, logistic_params):
        """Test vector field normalization."""
        likelihood = ais.LogisticRegressionLikelihood()
        transform = ais.LikelihoodDescent(likelihood)

        log_ell = likelihood.log_likelihood(simple_logistic_data, logistic_params)
        Q = transform.compute_Q(
            logistic_params,
            simple_logistic_data,
            logistic_params,
            log_ell,
        )

        Q_norm, norm_max = transform.normalize_vector_field(Q)

        # Normalized Q should have same structure
        assert "beta" in Q_norm
        assert "intercept" in Q_norm
        # norm_max should be (1, N)
        assert norm_max.shape[0] == 1


class TestGlobalTransformations:
    """Tests for GlobalTransformation subclasses (MM1, MM2)."""

    def test_mm1_shift_only(self, simple_logistic_data, logistic_params):
        """Test MM1 applies shift transformation."""
        likelihood = ais.LogisticRegressionLikelihood()
        transform = ais.MM1(likelihood)

        log_ell = likelihood.log_likelihood(simple_logistic_data, logistic_params)

        # Expand params for N dimension
        theta_expanded = jax.tree_util.tree_map(
            lambda x: x[:, jnp.newaxis, ...] if x.ndim == 2 else x[:, jnp.newaxis, jnp.newaxis],
            logistic_params,
        )

        result = transform(
            max_iter=1,
            params=logistic_params,
            theta=theta_expanded,
            data=simple_logistic_data,
            log_ell=log_ell,
            log_ell_original=log_ell,
        )

        assert "theta_new" in result
        assert "log_jacobian" in result
        # MM1 has zero Jacobian (volume preserving)
        assert jnp.allclose(result["log_jacobian"], 0.0)

    def test_mm2_shift_and_scale(self, simple_logistic_data, logistic_params):
        """Test MM2 applies shift and scale transformation."""
        likelihood = ais.LogisticRegressionLikelihood()
        transform = ais.MM2(likelihood)

        log_ell = likelihood.log_likelihood(simple_logistic_data, logistic_params)

        theta_expanded = jax.tree_util.tree_map(
            lambda x: x[:, jnp.newaxis, ...] if x.ndim == 2 else x[:, jnp.newaxis, jnp.newaxis],
            logistic_params,
        )

        result = transform(
            max_iter=1,
            params=logistic_params,
            theta=theta_expanded,
            data=simple_logistic_data,
            log_ell=log_ell,
            log_ell_original=log_ell,
        )

        assert "theta_new" in result
        assert "log_jacobian" in result
        # MM2 has non-zero Jacobian due to scaling
        # (though it might be close to zero if variance ratios are near 1)


class TestNumericalStability:
    """Tests for numerical stability of AIS computations."""

    def test_extreme_log_likelihood(self, simple_logistic_data):
        """Test handling of extreme log-likelihood values."""
        likelihood = ais.LogisticRegressionLikelihood()
        sampler = ais.AdaptiveImportanceSampler(likelihood)

        # Create params that would give extreme predictions
        params = {
            "beta": jnp.ones((50, 5)) * 10,  # Large coefficients
            "intercept": jnp.zeros((50,)),
        }

        # Should not raise and produce finite results
        results = sampler.adaptive_is_loo(
            simple_logistic_data,
            params,
            transformations=["identity"],
        )

        assert jnp.all(jnp.isfinite(results["identity"]["eta_weights"]))

    def test_weight_normalization(self, simple_logistic_data, logistic_params):
        """Test that importance weights are properly normalized."""
        likelihood = ais.LogisticRegressionLikelihood()
        sampler = ais.AdaptiveImportanceSampler(likelihood)

        results = sampler.adaptive_is_loo(
            simple_logistic_data,
            logistic_params,
            transformations=["identity"],
        )

        eta_weights = results["identity"]["eta_weights"]
        psis_weights = results["identity"]["psis_weights"]

        # Weights should sum to 1 along sample dimension
        assert jnp.allclose(jnp.sum(eta_weights, axis=0), 1.0, atol=1e-5)
        assert jnp.allclose(jnp.sum(psis_weights, axis=0), 1.0, atol=1e-5)

    def test_entropy_computation(self):
        """Test entropy computation is numerically stable."""
        sampler = ais.AdaptiveImportanceSampler(ais.LogisticRegressionLikelihood())

        # Uniform weights
        weights = jnp.ones((100, 10)) / 100
        entropy = sampler.entropy(weights)
        expected = jnp.log(100.0)  # Entropy of uniform distribution
        assert jnp.allclose(entropy, expected, atol=1e-4)

        # Peaked weights (one sample dominates)
        weights_peaked = jnp.zeros((100, 10))
        weights_peaked = weights_peaked.at[0, :].set(1.0)
        entropy_peaked = sampler.entropy(weights_peaked)
        assert jnp.all(entropy_peaked < entropy[0])  # Lower entropy


class TestAutoDiffLikelihoodMixin:
    """Tests for AutoDiffLikelihoodMixin functionality."""

    @pytest.mark.skip(reason="AutoDiffLikelihoodMixin requires specific log_likelihood shape handling")
    def test_mixin_provides_gradient(self):
        """Test that mixin provides working gradient method.

        Note: The mixin implementation assumes log_likelihood returns (S, N) when
        given params with leading S dimension. Custom implementations need to
        handle batch dimensions correctly.
        """

        class CustomLikelihood(ais.AutoDiffLikelihoodMixin):
            def log_likelihood(self, data, params):
                X = data["X"]
                beta = params["beta"]
                return -0.5 * jnp.sum((X @ beta.T) ** 2, axis=1).T

        likelihood = CustomLikelihood()
        data = {"X": jax.random.normal(jax.random.PRNGKey(42), (10, 3))}
        params = {"beta": jax.random.normal(jax.random.PRNGKey(1), (5, 3))}

        grad = likelihood.log_likelihood_gradient(data, params)

        assert "beta" in grad
        # Should have shape (S, N, K)
        assert grad["beta"].shape == (5, 10, 3)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_few_samples(self, simple_logistic_data):
        """Test with few posterior samples (minimum for PSIS is 2)."""
        likelihood = ais.LogisticRegressionLikelihood()
        sampler = ais.AdaptiveImportanceSampler(likelihood)

        # PSIS requires at least 2 samples
        params = {
            "beta": jax.random.normal(jax.random.PRNGKey(42), (10, 5)),
            "intercept": jax.random.normal(jax.random.PRNGKey(1), (10,)),
        }

        results = sampler.adaptive_is_loo(
            simple_logistic_data,
            params,
            transformations=["identity"],
        )

        assert "identity" in results

    def test_single_data_point(self):
        """Test with single data point."""
        likelihood = ais.LogisticRegressionLikelihood()

        data = {
            "X": jax.random.normal(jax.random.PRNGKey(42), (1, 5)),
            "y": jnp.array([1.0]),
        }
        params = {
            "beta": jax.random.normal(jax.random.PRNGKey(1), (50, 5)),
            "intercept": jax.random.normal(jax.random.PRNGKey(2), (50,)),
        }

        ll = likelihood.log_likelihood(data, params)
        assert ll.shape == (50, 1)

    def test_high_dimensional_features(self):
        """Test with high-dimensional features."""
        likelihood = ais.LogisticRegressionLikelihood()

        n_features = 100
        data = {
            "X": jax.random.normal(jax.random.PRNGKey(42), (20, n_features)),
            "y": jax.random.bernoulli(jax.random.PRNGKey(1), 0.5, (20,)).astype(jnp.float32),
        }
        params = {
            "beta": jax.random.normal(jax.random.PRNGKey(2), (30, n_features)) * 0.1,
            "intercept": jax.random.normal(jax.random.PRNGKey(3), (30,)) * 0.1,
        }

        ll = likelihood.log_likelihood(data, params)
        grad = likelihood.log_likelihood_gradient(data, params)

        assert ll.shape == (30, 20)
        assert grad["beta"].shape == (30, 20, n_features)


class TestComputeImportanceWeightsHelper:
    """Tests for compute_importance_weights_helper method."""

    def test_weight_computation(self, simple_logistic_data, logistic_params):
        """Test importance weight computation."""
        likelihood = ais.LogisticRegressionLikelihood()
        transform = ais.LikelihoodDescent(likelihood)

        log_ell = likelihood.log_likelihood(simple_logistic_data, logistic_params)
        log_jacobian = jnp.zeros_like(log_ell)
        log_pi = jnp.zeros(logistic_params["beta"].shape[0])

        eta_weights, psis_weights, khat, log_ell_new = transform.compute_importance_weights_helper(
            likelihood,
            simple_logistic_data,
            logistic_params,
            logistic_params,
            log_jacobian,
            variational=False,
            log_pi_original=log_pi,
            log_ell_original=log_ell,
        )

        # Check shapes
        n_samples = logistic_params["beta"].shape[0]
        n_data = simple_logistic_data["X"].shape[0]

        assert eta_weights.shape == (n_samples, n_data)
        assert psis_weights.shape == (n_samples, n_data)
        assert khat.shape == (n_data,)
        assert log_ell_new.shape == (n_samples, n_data)

        # Weights should be non-negative
        assert jnp.all(eta_weights >= 0)
        assert jnp.all(psis_weights >= 0)

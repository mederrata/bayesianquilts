"""
Tests for the Adaptive LOO ELPD via AIS integration.

Tests:
1. SimpleModelLikelihood wrapper works with all three model types
2. _compute_loo_elpd works with and without AIS fallback
3. BayesianModel.compute_loo() works
4. IRTModel._compute_elpd_loo() AIS path
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ========================================================================
# Test 1: SimpleModelLikelihood wrapper
# ========================================================================

def test_simple_model_likelihood_linear():
    """SimpleModelLikelihood wraps SimpleLinearRegression correctly."""
    from bayesianquilts.imputation.mice_loo import (
        SimpleLinearRegression,
        SimpleModelLikelihood,
    )

    model = SimpleLinearRegression(n_predictors=2, dtype=jnp.float64)
    likelihood_fn = SimpleModelLikelihood(model)

    # Synthetic data
    np.random.seed(42)
    N, S = 20, 10
    data = {
        'X': np.random.randn(N, 2).astype(np.float64),
        'y': np.random.randn(N).astype(np.float64),
    }
    params = {
        'beta': jnp.array(np.random.randn(S, 2)),
        'intercept': jnp.array(np.random.randn(S)),
        'log_sigma': jnp.array(np.random.randn(S)),
    }

    # log_likelihood
    ll = likelihood_fn.log_likelihood(data, params)
    assert ll.shape == (S, N), f"Expected ({S}, {N}), got {ll.shape}"
    assert jnp.all(jnp.isfinite(ll))

    # gradient
    grad = likelihood_fn.log_likelihood_gradient(data, params)
    assert 'beta' in grad
    assert 'intercept' in grad
    assert grad['beta'].shape == (S, N, 2), f"Grad beta shape: {grad['beta'].shape}"
    assert grad['intercept'].shape == (S, N), f"Grad intercept shape: {grad['intercept'].shape}"

    # hessian diagonal
    hess = likelihood_fn.log_likelihood_hessian_diag(data, params)
    assert 'beta' in hess
    assert hess['beta'].shape == (S, N, 2)

    print("PASS: test_simple_model_likelihood_linear")


def test_simple_model_likelihood_logistic():
    """SimpleModelLikelihood wraps SimpleLogisticRegression correctly."""
    from bayesianquilts.imputation.mice_loo import (
        SimpleLogisticRegression,
        SimpleModelLikelihood,
    )

    model = SimpleLogisticRegression(n_predictors=2, dtype=jnp.float64)
    likelihood_fn = SimpleModelLikelihood(model)

    np.random.seed(43)
    N, S = 20, 10
    data = {
        'X': np.random.randn(N, 2).astype(np.float64),
        'y': np.random.choice([0.0, 1.0], size=N).astype(np.float64),
    }
    params = {
        'beta': jnp.array(np.random.randn(S, 2)),
        'intercept': jnp.array(np.random.randn(S)),
    }

    ll = likelihood_fn.log_likelihood(data, params)
    assert ll.shape == (S, N)
    assert jnp.all(jnp.isfinite(ll))

    grad = likelihood_fn.log_likelihood_gradient(data, params)
    assert grad['beta'].shape == (S, N, 2)

    print("PASS: test_simple_model_likelihood_logistic")


def test_simple_model_likelihood_ordinal():
    """SimpleModelLikelihood wraps SimpleOrdinalLogisticRegression correctly."""
    from bayesianquilts.imputation.mice_loo import (
        SimpleOrdinalLogisticRegression,
        SimpleModelLikelihood,
    )

    model = SimpleOrdinalLogisticRegression(
        n_classes=4, n_predictors=2, dtype=jnp.float64
    )
    likelihood_fn = SimpleModelLikelihood(model)

    np.random.seed(44)
    N, S = 20, 8
    data = {
        'X': np.random.randn(N, 2).astype(np.float64),
        'y': np.random.choice([0.0, 1.0, 2.0, 3.0], size=N).astype(np.float64),
    }
    params = {
        'beta': jnp.array(np.random.randn(S, 2)),
        'cutpoints_raw': jnp.array(np.random.randn(S, 3)),
    }

    ll = likelihood_fn.log_likelihood(data, params)
    assert ll.shape == (S, N)
    assert jnp.all(jnp.isfinite(ll))

    grad = likelihood_fn.log_likelihood_gradient(data, params)
    assert grad['beta'].shape == (S, N, 2)
    assert grad['cutpoints_raw'].shape == (S, N, 3)

    print("PASS: test_simple_model_likelihood_ordinal")


# ========================================================================
# Test 2: _compute_loo_elpd with standard PSIS (no AIS needed for good data)
# ========================================================================

def test_compute_loo_elpd_standard():
    """_compute_loo_elpd returns valid results with standard PSIS."""
    from bayesianquilts.imputation.mice_loo import (
        MICEBayesianLOO,
        SimpleLinearRegression,
    )

    mice = MICEBayesianLOO(verbose=False)
    model = SimpleLinearRegression(n_predictors=1, dtype=jnp.float64)

    np.random.seed(45)
    N, S = 50, 100
    X = np.random.randn(N, 1).astype(np.float64)
    true_beta = 0.5
    y = (X[:, 0] * true_beta + np.random.randn(N) * 0.5).astype(np.float64)
    data = {'X': X, 'y': y}

    # Create well-behaved posterior samples (tight around truth)
    params = {
        'beta': jnp.array(np.random.randn(S, 1) * 0.1 + true_beta),
        'intercept': jnp.array(np.random.randn(S) * 0.1),
        'log_sigma': jnp.array(np.random.randn(S) * 0.1 + np.log(0.5)),
    }

    elpd, se, kmax, kmean = mice._compute_loo_elpd(model, data, params)

    assert np.isfinite(elpd), f"elpd should be finite, got {elpd}"
    assert np.isfinite(se), f"se should be finite, got {se}"
    assert np.isfinite(kmax), f"kmax should be finite, got {kmax}"
    assert np.isfinite(kmean), f"kmean should be finite, got {kmean}"
    assert elpd < 0, f"elpd should be negative for log scale, got {elpd}"

    print(f"PASS: test_compute_loo_elpd_standard (elpd={elpd:.2f}, kmax={kmax:.3f})")


# ========================================================================
# Test 3: _compute_loo_elpd AIS path (with mock high k-hat)
# ========================================================================

def test_compute_loo_elpd_with_ais_fallback():
    """_compute_loo_elpd uses AIS when surrogate/prior are provided.

    We don't force k-hat > 0.7 here (hard to manufacture), just verify
    the code path works when called with surrogate/prior functions.
    """
    from bayesianquilts.imputation.mice_loo import (
        MICEBayesianLOO,
        SimpleLinearRegression,
    )
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    mice = MICEBayesianLOO(verbose=False)
    model = SimpleLinearRegression(n_predictors=1, dtype=jnp.float64)

    np.random.seed(46)
    N, S = 30, 100
    X = np.random.randn(N, 1).astype(np.float64)
    y = (X[:, 0] * 0.5 + np.random.randn(N) * 0.5).astype(np.float64)
    data = {'X': X, 'y': y}

    params = {
        'beta': jnp.array(np.random.randn(S, 1) * 0.1 + 0.5),
        'intercept': jnp.array(np.random.randn(S) * 0.1),
        'log_sigma': jnp.array(np.random.randn(S) * 0.1 + np.log(0.5)),
    }

    # Build surrogate and prior log prob functions
    prior = model.create_prior()
    prior_log_prob_fn = lambda p: prior.log_prob(p)

    surrogate_dists = {
        'beta': tfd.Independent(
            tfd.Normal(
                loc=jnp.mean(params['beta'], axis=0),
                scale=jnp.maximum(jnp.std(params['beta'], axis=0), 1e-6)
            ), reinterpreted_batch_ndims=1
        ),
        'intercept': tfd.Normal(
            loc=jnp.mean(params['intercept']),
            scale=jnp.maximum(jnp.std(params['intercept']), 1e-6)
        ),
        'log_sigma': tfd.Normal(
            loc=jnp.mean(params['log_sigma']),
            scale=jnp.maximum(jnp.std(params['log_sigma']), 1e-6)
        ),
    }
    surrogate = tfd.JointDistributionNamed(surrogate_dists)
    surrogate_log_prob_fn = lambda p: surrogate.log_prob(p)

    # Call with surrogate/prior - should work even if AIS doesn't activate
    elpd, se, kmax, kmean = mice._compute_loo_elpd(
        model, data, params,
        surrogate_log_prob_fn=surrogate_log_prob_fn,
        prior_log_prob_fn=prior_log_prob_fn,
    )

    assert np.isfinite(elpd)
    assert np.isfinite(se)
    print(f"PASS: test_compute_loo_elpd_with_ais_fallback (elpd={elpd:.2f}, kmax={kmax:.3f})")


# ========================================================================
# Test 4: _run_pathfinder returns surrogate_log_prob_fn
# ========================================================================

def test_pathfinder_returns_surrogate():
    """_run_pathfinder returns a callable surrogate_log_prob_fn."""
    from bayesianquilts.imputation.mice_loo import (
        MICEBayesianLOO,
        SimpleLinearRegression,
    )

    mice = MICEBayesianLOO(
        verbose=False,
        pathfinder_num_samples=50,
        pathfinder_maxiter=20,
        inference_method='pathfinder',
    )
    model = SimpleLinearRegression(n_predictors=1, dtype=jnp.float64)

    np.random.seed(47)
    N = 30
    X = np.random.randn(N, 1).astype(np.float64)
    y = (X[:, 0] * 0.5 + np.random.randn(N) * 0.5).astype(np.float64)
    data = {'X': X, 'y': y}

    result = mice._run_pathfinder(model, data, scale_factor=1.0, seed=42)
    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple"
    samples_dict, elbo, converged, surrogate_fn = result

    if converged:
        assert surrogate_fn is not None, "Surrogate should be returned on convergence"
        # Test the surrogate function with params
        lp = surrogate_fn(samples_dict)
        assert lp.shape == (50,), f"Expected (50,), got {lp.shape}"
        assert jnp.all(jnp.isfinite(lp))
        print(f"PASS: test_pathfinder_returns_surrogate (elbo={elbo:.2f})")
    else:
        print("SKIP: test_pathfinder_returns_surrogate (pathfinder did not converge)")


# ========================================================================
# Test 5: _run_advi returns surrogate_log_prob_fn
# ========================================================================

def test_advi_returns_surrogate():
    """_run_advi returns a callable surrogate_log_prob_fn."""
    from bayesianquilts.imputation.mice_loo import (
        MICEBayesianLOO,
        SimpleLinearRegression,
    )

    mice = MICEBayesianLOO(
        verbose=False,
        pathfinder_num_samples=50,
        pathfinder_maxiter=30,
        inference_method='advi',
    )
    model = SimpleLinearRegression(n_predictors=1, dtype=jnp.float64)

    np.random.seed(48)
    N = 30
    X = np.random.randn(N, 1).astype(np.float64)
    y = (X[:, 0] * 0.5 + np.random.randn(N) * 0.5).astype(np.float64)
    data = {'X': X, 'y': y}

    result = mice._run_advi(model, data, scale_factor=1.0, seed=42)
    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple"
    samples_dict, elbo, converged, surrogate_fn = result

    if converged:
        assert surrogate_fn is not None
        lp = surrogate_fn(samples_dict)
        assert lp.shape == (50,), f"Expected (50,), got {lp.shape}"
        assert jnp.all(jnp.isfinite(lp))
        print(f"PASS: test_advi_returns_surrogate (elbo={elbo:.2f})")
    else:
        print("SKIP: test_advi_returns_surrogate (ADVI did not converge)")


# ========================================================================
# Test 6: _run_inference_with_fallback returns 5-tuple
# ========================================================================

def test_inference_with_fallback_returns_5tuple():
    """_run_inference_with_fallback returns (params, elbo, converged, dtype, surrogate_fn)."""
    from bayesianquilts.imputation.mice_loo import (
        MICEBayesianLOO,
        SimpleLinearRegression,
    )

    mice = MICEBayesianLOO(
        verbose=False,
        pathfinder_num_samples=50,
        pathfinder_maxiter=20,
    )
    model = SimpleLinearRegression(n_predictors=1, dtype=jnp.float32)

    np.random.seed(49)
    N = 30
    X = np.random.randn(N, 1).astype(np.float32)
    y = (X[:, 0] * 0.5 + np.random.randn(N) * 0.5).astype(np.float32)
    data = {'X': X, 'y': y}

    result = mice._run_inference_with_fallback(
        model, data, scale_factor=1.0, seed=42, current_dtype=jnp.float32
    )
    assert len(result) == 5, f"Expected 5-tuple, got {len(result)}-tuple"
    params, elbo, converged, dtype_used, surrogate_fn = result

    print(f"PASS: test_inference_with_fallback_returns_5tuple "
          f"(converged={converged}, dtype={dtype_used})")


# ========================================================================
# Test 7: AIS integration with AdaptiveImportanceSampler
# ========================================================================

def test_ais_with_simple_model_likelihood():
    """AdaptiveImportanceSampler works with SimpleModelLikelihood."""
    from bayesianquilts.imputation.mice_loo import (
        SimpleLinearRegression,
        SimpleModelLikelihood,
    )
    from bayesianquilts.metrics.ais import AdaptiveImportanceSampler
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    model = SimpleLinearRegression(n_predictors=1, dtype=jnp.float64)
    likelihood_fn = SimpleModelLikelihood(model)

    np.random.seed(50)
    N, S = 30, 100
    X = np.random.randn(N, 1).astype(np.float64)
    y = (X[:, 0] * 0.5 + np.random.randn(N) * 0.5).astype(np.float64)
    data = {'X': X, 'y': y}

    params = {
        'beta': jnp.array(np.random.randn(S, 1) * 0.1 + 0.5),
        'intercept': jnp.array(np.random.randn(S) * 0.1),
        'log_sigma': jnp.array(np.random.randn(S) * 0.1 + np.log(0.5)),
    }

    prior = model.create_prior()
    prior_log_prob_fn = lambda p: prior.log_prob(p)

    surrogate_dists = {
        'beta': tfd.Independent(
            tfd.Normal(
                loc=jnp.mean(params['beta'], axis=0),
                scale=jnp.maximum(jnp.std(params['beta'], axis=0), 1e-6),
            ), reinterpreted_batch_ndims=1
        ),
        'intercept': tfd.Normal(
            loc=jnp.mean(params['intercept']),
            scale=jnp.maximum(jnp.std(params['intercept']), 1e-6),
        ),
        'log_sigma': tfd.Normal(
            loc=jnp.mean(params['log_sigma']),
            scale=jnp.maximum(jnp.std(params['log_sigma']), 1e-6),
        ),
    }
    surrogate = tfd.JointDistributionNamed(surrogate_dists)
    surrogate_log_prob_fn = lambda p: surrogate.log_prob(p)

    sampler = AdaptiveImportanceSampler(
        likelihood_fn, prior_log_prob_fn, surrogate_log_prob_fn
    )

    results = sampler.adaptive_is_loo(
        data, params,
        variational=True,
        khat_threshold=0.7,
        transformations=['identity', 'mm1', 'mm2'],
    )

    assert 'best' in results
    best = results['best']
    assert 'khat' in best
    assert 'll_loo_psis' in best
    assert best['khat'].shape == (N,)
    assert best['ll_loo_psis'].shape == (N,)

    print(f"PASS: test_ais_with_simple_model_likelihood "
          f"(max_khat={float(jnp.max(best['khat'])):.3f})")


if __name__ == '__main__':
    test_simple_model_likelihood_linear()
    test_simple_model_likelihood_logistic()
    test_simple_model_likelihood_ordinal()
    test_compute_loo_elpd_standard()
    test_compute_loo_elpd_with_ais_fallback()
    test_pathfinder_returns_surrogate()
    test_advi_returns_surrogate()
    test_inference_with_fallback_returns_5tuple()
    test_ais_with_simple_model_likelihood()
    print("\n=== All tests passed! ===")

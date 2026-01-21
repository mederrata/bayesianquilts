#!/usr/bin/env python3
"""
Test script for the generalized Adaptive Importance Sampling framework.

This demonstrates how to use the new framework with different likelihood functions.
"""

import jax
import jax.numpy as jnp
import sys
import os

# Add the bayesianquilts package to the path
sys.path.insert(0, '/Users/josh/workspace/bayesianquilts')

from bayesianquilts.metrics.ais import (
    AdaptiveImportanceSampler,
    LogisticRegressionLikelihood,
    PoissonRegressionLikelihood,
    LinearRegressionLikelihood
)


def test_logistic_regression_ais():
    """Test AIS with logistic regression."""
    print("=== Testing Logistic Regression AIS ===")

    # Generate synthetic data
    key = jax.random.PRNGKey(42)
    n_data, n_features = 100, 3
    n_samples = 50

    X = jax.random.normal(key, (n_data, n_features))
    true_beta = jnp.array([0.5, -0.3, 0.8])
    true_intercept = 0.2

    logits = X @ true_beta + true_intercept
    probs = jax.nn.sigmoid(logits)
    y = jax.random.bernoulli(key, probs).astype(jnp.float32)

    data = {"X": X, "y": y}

    # Generate posterior samples (normally these would come from MCMC/VI)
    key, subkey = jax.random.split(key)
    beta_samples = jax.random.normal(subkey, (n_samples, n_features)) * 0.1 + true_beta[jnp.newaxis, :]
    intercept_samples = jax.random.normal(key, (n_samples,)) * 0.1 + true_intercept

    params = {
        "beta": beta_samples,
        "intercept": intercept_samples
    }

    # Create likelihood function and AIS sampler
    likelihood_fn = LogisticRegressionLikelihood()
    ais_sampler = AdaptiveImportanceSampler(likelihood_fn)

    # Run AIS with different transformations
    results = ais_sampler.adaptive_is_loo(
        data, params, hbar=1.0, variational=False,
        transformations=['ll', 'kl', 'identity']
    )

    print(f"Transformations tested: {list(results.keys())}")
    for transform_name, result in results.items():
        print(f"\n{transform_name.upper()} Transformation:")
        print(f"  - k-hat range: [{jnp.min(result['khat']):.3f}, {jnp.max(result['khat']):.3f}]")
        print(f"  - Weight entropy range: [{jnp.min(result['weight_entropy']):.3f}, {jnp.max(result['weight_entropy']):.3f}]")
        print(f"  - Mean p_loo (eta): {jnp.mean(result['p_loo_eta']):.3f}")
        print(f"  - Mean p_loo (PSIS): {jnp.mean(result['p_loo_psis']):.3f}")

    return results


def test_poisson_regression_ais():
    """Test AIS with Poisson regression."""
    print("\n=== Testing Poisson Regression AIS ===")

    key = jax.random.PRNGKey(123)
    n_data, n_features = 80, 2
    n_samples = 30

    X = jax.random.normal(key, (n_data, n_features))
    true_beta = jnp.array([0.3, -0.2])
    true_intercept = 1.0

    log_rates = X @ true_beta + true_intercept
    rates = jnp.exp(log_rates)
    y = jax.random.poisson(key, rates).astype(jnp.float32)

    data = {"X": X, "y": y}

    # Generate posterior samples
    key, subkey = jax.random.split(key)
    beta_samples = jax.random.normal(subkey, (n_samples, n_features)) * 0.05 + true_beta[jnp.newaxis, :]
    intercept_samples = jax.random.normal(key, (n_samples,)) * 0.05 + true_intercept

    params = {
        "beta": beta_samples,
        "intercept": intercept_samples
    }

    # Create likelihood and sampler
    likelihood_fn = PoissonRegressionLikelihood()
    ais_sampler = AdaptiveImportanceSampler(likelihood_fn)

    # Run AIS
    results = ais_sampler.adaptive_is_loo(
        data, params, hbar=0.5, variational=False,
        transformations=['ll', 'var']
    )

    print(f"Transformations tested: {list(results.keys())}")
    for transform_name, result in results.items():
        print(f"\n{transform_name.upper()} Transformation:")
        print(f"  - k-hat range: [{jnp.min(result['khat']):.3f}, {jnp.max(result['khat']):.3f}]")
        print(f"  - Mean LL LOO (eta): {jnp.mean(result['ll_loo_eta']):.3f}")
        print(f"  - Mean LL LOO (PSIS): {jnp.mean(result['ll_loo_psis']):.3f}")

    return results


def test_linear_regression_ais():
    """Test AIS with linear regression."""
    print("\n=== Testing Linear Regression AIS ===")

    key = jax.random.PRNGKey(456)
    n_data, n_features = 60, 2
    n_samples = 40

    X = jax.random.normal(key, (n_data, n_features))
    true_beta = jnp.array([1.2, -0.8])
    true_intercept = 0.5
    true_sigma = 0.3

    y = X @ true_beta + true_intercept + jax.random.normal(key, (n_data,)) * true_sigma

    data = {"X": X, "y": y}

    # Generate posterior samples
    key, subkey1, subkey2 = jax.random.split(key, 3)
    beta_samples = jax.random.normal(subkey1, (n_samples, n_features)) * 0.1 + true_beta[jnp.newaxis, :]
    intercept_samples = jax.random.normal(subkey2, (n_samples,)) * 0.1 + true_intercept
    log_sigma_samples = jax.random.normal(key, (n_samples,)) * 0.05 + jnp.log(true_sigma)

    params = {
        "beta": beta_samples,
        "intercept": intercept_samples,
        "log_sigma": log_sigma_samples
    }

    # Create likelihood and sampler
    likelihood_fn = LinearRegressionLikelihood()
    ais_sampler = AdaptiveImportanceSampler(likelihood_fn)

    # Run AIS
    results = ais_sampler.adaptive_is_loo(
        data, params, hbar=1.5, variational=False,
        transformations=['ll', 'kl', 'var', 'identity']
    )

    print(f"Transformations tested: {list(results.keys())}")
    for transform_name, result in results.items():
        print(f"\n{transform_name.upper()} Transformation:")
        print(f"  - k-hat range: [{jnp.min(result['khat']):.3f}, {jnp.max(result['khat']):.3f}]")
        print(f"  - PSIS entropy range: [{jnp.min(result['psis_entropy']):.3f}, {jnp.max(result['psis_entropy']):.3f}]")

    return results


def test_with_priors():
    """Test AIS with prior functions."""
    print("\n=== Testing AIS with Prior Functions ===")

    key = jax.random.PRNGKey(789)
    n_data, n_features = 50, 2
    n_samples = 25

    X = jax.random.normal(key, (n_data, n_features))
    y = jax.random.bernoulli(key, 0.3, (n_data,)).astype(jnp.float32)

    data = {"X": X, "y": y}

    # Generate posterior samples
    key, subkey = jax.random.split(key)
    beta_samples = jax.random.normal(subkey, (n_samples, n_features)) * 0.5
    intercept_samples = jax.random.normal(key, (n_samples,)) * 0.5

    params = {
        "beta": beta_samples,
        "intercept": intercept_samples
    }

    # Define prior functions
    def prior_log_prob(params):
        """Simple Gaussian priors."""
        beta_prior = -0.5 * jnp.sum(params["beta"]**2, axis=-1)  # N(0, 1) prior on beta
        intercept_prior = -0.5 * params["intercept"]**2  # N(0, 1) prior on intercept
        return beta_prior + intercept_prior

    def surrogate_log_prob(params):
        """Surrogate (variational) approximation."""
        # Assume independent Gaussians with learned means/variances
        beta_mean = jnp.array([0.1, -0.1])
        beta_var = jnp.array([0.5, 0.3])
        intercept_mean = 0.0
        intercept_var = 0.4

        beta_log_prob = jnp.sum(-0.5 * jnp.log(2 * jnp.pi * beta_var) -
                               0.5 * (params["beta"] - beta_mean)**2 / beta_var, axis=-1)
        intercept_log_prob = (-0.5 * jnp.log(2 * jnp.pi * intercept_var) -
                             0.5 * (params["intercept"] - intercept_mean)**2 / intercept_var)

        return beta_log_prob + intercept_log_prob

    # Create AIS sampler with priors
    likelihood_fn = LogisticRegressionLikelihood()
    ais_sampler = AdaptiveImportanceSampler(
        likelihood_fn,
        prior_log_prob_fn=prior_log_prob,
        surrogate_log_prob_fn=surrogate_log_prob
    )

    # Test both variational and non-variational modes
    print("\nNon-variational mode (use full posterior):")
    results_full = ais_sampler.adaptive_is_loo(
        data, params, hbar=1.0, variational=False,
        transformations=['ll', 'identity']
    )

    print("\nVariational mode (trust surrogate):")
    results_var = ais_sampler.adaptive_is_loo(
        data, params, hbar=1.0, variational=True,
        transformations=['ll', 'identity']
    )

    for mode, results in [("Full", results_full), ("Variational", results_var)]:
        print(f"\n{mode} mode results:")
        for transform_name, result in results.items():
            print(f"  {transform_name}: k-hat range [{jnp.min(result['khat']):.3f}, {jnp.max(result['khat']):.3f}]")

    return results_full, results_var


def main():
    """Run all tests."""
    print("🚀 Testing Generalized Adaptive Importance Sampling Framework")
    print("=" * 60)

    try:
        # Test different likelihood functions
        logistic_results = test_logistic_regression_ais()
        poisson_results = test_poisson_regression_ais()
        linear_results = test_linear_regression_ais()

        # Test with priors
        prior_results = test_with_priors()

        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        print("\nFramework features demonstrated:")
        print("✓ Logistic regression likelihood")
        print("✓ Poisson regression likelihood")
        print("✓ Linear regression likelihood")
        print("✓ Multiple transformation strategies (ll, kl, var, identity)")
        print("✓ Prior and surrogate probability functions")
        print("✓ Both variational and non-variational modes")
        print("✓ PSIS weight computation and diagnostics")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
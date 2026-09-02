"""Correctness tests for noise-regularized affine LOO transformations."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from bayesianquilts.metrics.regularized_loo import (
    AffineMap,
    clipped_log_weight_loss,
    crossfit_regularized_pmm_loo_fold,
    fit_pmm,
    regularized_pmm_loo_fold,
    three_fold_split,
    transformed_log_weights,
)


jax.config.update("jax_enable_x64", True)


def ordinary_moments(draws):
    mean = jnp.mean(draws, axis=0)
    centered = draws - mean
    return mean, centered.T @ centered / draws.shape[0]


def weighted_moments(draws, log_weights):
    weights = jax.nn.softmax(log_weights)
    mean = jnp.sum(weights[:, None] * draws, axis=0)
    centered = draws - mean
    return mean, (centered.T * weights) @ centered


class RegularizedLOOTests(unittest.TestCase):
    def setUp(self):
        self.draws = jax.random.normal(jax.random.PRNGKey(20260902), (200, 4))
        self.log_weights = 0.35 * self.draws[:, 0] - 0.2 * self.draws[:, 1]

    def test_identity_and_full_pmm1_means(self):
        fitted = fit_pmm(self.draws, self.log_weights, method="pmm1")
        np.testing.assert_allclose(fitted.at(0.0)(self.draws), self.draws)
        transformed = fitted.at(1.0)(self.draws)
        transformed_mean, _ = ordinary_moments(transformed)
        weighted_mean, _ = weighted_moments(self.draws, self.log_weights)
        np.testing.assert_allclose(transformed_mean, weighted_mean, atol=1e-10)

    def test_full_pmm2_matches_marginal_moments(self):
        fitted = fit_pmm(
            self.draws, self.log_weights, method="pmm2", ridge=0.0
        )
        transformed = fitted.at(1.0)(self.draws)
        transformed_mean, transformed_covariance = ordinary_moments(transformed)
        weighted_mean, weighted_covariance = weighted_moments(
            self.draws, self.log_weights
        )
        np.testing.assert_allclose(transformed_mean, weighted_mean, atol=1e-10)
        np.testing.assert_allclose(
            jnp.diag(transformed_covariance),
            jnp.diag(weighted_covariance),
            atol=1e-10,
        )

    def test_full_pmm3_matches_full_moments(self):
        fitted = fit_pmm(
            self.draws, self.log_weights, method="pmm3", ridge=0.0
        )
        transformed = fitted.at(1.0)(self.draws)
        transformed_mean, transformed_covariance = ordinary_moments(transformed)
        weighted_mean, weighted_covariance = weighted_moments(
            self.draws, self.log_weights
        )
        np.testing.assert_allclose(transformed_mean, weighted_mean, atol=1e-10)
        np.testing.assert_allclose(
            transformed_covariance, weighted_covariance, atol=1e-10
        )

    def test_partial_pmm3_uses_exact_determinant(self):
        fitted = fit_pmm(self.draws, self.log_weights, method="pmm3")
        partial = fitted.at(0.37)
        expected = np.linalg.slogdet(np.asarray(partial.linear))[1]
        self.assertAlmostEqual(float(partial.log_abs_determinant), expected, places=12)

    def test_tuning_loss_is_bounded(self):
        log_weights = jnp.array([-1e6, -2.0, 0.0, 2.0, 1e6])
        loss = clipped_log_weight_loss(log_weights, center=0.0, clip=5.0)
        self.assertGreaterEqual(float(loss), 0.0)
        self.assertLessEqual(float(loss), 25.0)

    def test_exact_weight_includes_prior_ratio(self):
        base = self.draws[:, :2]
        affine = AffineMap(
            linear=jnp.array([[1.1, 0.0], [0.2, 0.9]]),
            offset=jnp.array([0.3, -0.1]),
            h=0.5,
        )

        def log_prior(theta):
            return -0.5 * jnp.sum(theta**2 / 4.0, axis=1)

        def log_likelihood(theta):
            return -0.5 * jnp.sum((theta - 1.0) ** 2, axis=1)

        def log_full(theta):
            return log_prior(theta) + log_likelihood(theta)

        transformed, actual = transformed_log_weights(
            base, affine, log_full, log_likelihood
        )
        expected = (
            log_prior(transformed)
            - log_prior(base)
            - log_likelihood(base)
            + affine.log_abs_determinant
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_heldout_regularization_avoids_multivariate_overstep(self):
        dimension = 12
        sample_size = 80
        proposal_mean = jnp.zeros(dimension)
        target_mean = jnp.full(dimension, 0.08)
        proposal_variance = 1.0
        target_variance = 1.01
        keys = jax.random.split(jax.random.PRNGKey(260902), 3)
        splits = [
            proposal_mean
            + jnp.sqrt(proposal_variance)
            * jax.random.normal(key, (sample_size, dimension))
            for key in keys
        ]

        def log_normal(theta, mean, variance):
            return -0.5 * jnp.sum((theta - mean) ** 2, axis=1) / variance

        def log_full(theta):
            return log_normal(theta, proposal_mean, proposal_variance)

        def log_target(theta):
            return log_normal(theta, target_mean, target_variance)

        def log_heldout(theta):
            return log_full(theta) - log_target(theta)

        result = regularized_pmm_loo_fold(
            *splits,
            log_full_posterior=log_full,
            log_heldout_likelihood=log_heldout,
            method="pmm3",
            h_grid=np.linspace(0.0, 1.0, 21),
            ridge=1e-10,
        )
        self.assertLess(float(result.selected_map.h), 0.8)
        self.assertTrue(np.isfinite(float(result.raw_effective_sample_size)))

        def population_kl(affine):
            candidate_mean = affine.linear @ proposal_mean + affine.offset
            candidate_covariance = (
                proposal_variance * affine.linear @ affine.linear.T
            )
            inverse = jnp.linalg.inv(candidate_covariance)
            difference = candidate_mean - target_mean
            return 0.5 * (
                target_variance * jnp.trace(inverse)
                + difference @ inverse @ difference
                - dimension
                + jnp.linalg.slogdet(candidate_covariance)[1]
                - dimension * jnp.log(target_variance)
            )

        selected_kl = population_kl(result.selected_map)
        full_kl = population_kl(result.fitted_map.at(1.0))
        self.assertLess(float(selected_kl), float(full_kl))

    def test_three_fold_crossfit_rotates_all_draws(self):
        dimension = 3
        sample_size = 90
        draws = jax.random.normal(
            jax.random.PRNGKey(17), (sample_size, dimension)
        )

        def log_full(theta):
            return -0.5 * jnp.sum(theta**2, axis=1)

        def log_heldout(theta):
            return -0.1 * jnp.sum(theta, axis=1)

        folds = three_fold_split(draws, jax.random.PRNGKey(20260902))
        result = crossfit_regularized_pmm_loo_fold(
            folds,
            log_full_posterior=log_full,
            log_heldout_likelihood=log_heldout,
            method="pmm1",
        )
        self.assertEqual(
            result.transformed_evaluation_draws.shape, (sample_size, dimension)
        )
        self.assertEqual(result.evaluation_log_weights.shape, (sample_size,))
        self.assertEqual(result.selected_steps.shape, (3,))
        self.assertEqual(len(result.fold_results), 3)

    def test_step_outside_unit_interval_is_rejected(self):
        fitted = fit_pmm(self.draws, self.log_weights, method="pmm1")
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            fitted.at(1.01)


if __name__ == "__main__":
    unittest.main()

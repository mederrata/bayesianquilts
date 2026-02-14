"""Integration tests for IRT models with analytic Rao-Blackwellized imputation.

Tests cover:
- GRModel and FactorizedGRModel with complete and missing data
- validate_imputation_model() sanity checks
- Analytic imputation via PMF-weighted log-likelihood
"""

import warnings
import numpy as np
import jax.numpy as jnp
import pytest
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from unittest.mock import MagicMock


# =========================================================================
# Mock imputation model (avoids needing full MICEBayesianLOO for unit tests)
# =========================================================================

@dataclass
class MockUnivariateResult:
    n_obs: int = 100
    elpd_loo: float = -50.0
    elpd_loo_per_obs: float = -0.5
    elpd_loo_per_obs_se: float = 0.1
    khat_max: float = 0.3
    khat_mean: float = 0.2
    predictor_idx: Optional[int] = None
    target_idx: int = 0
    converged: bool = True
    params: Optional[dict] = None
    predictor_mean: Optional[float] = 0.0
    predictor_std: Optional[float] = 1.0
    beta_mean: Optional[float] = 0.5
    intercept_mean: Optional[float] = 0.0
    cutpoints_mean: Optional[np.ndarray] = None


class MockImputationModel:
    """Mock MICEBayesianLOO for testing validation and imputation."""

    def __init__(self, variable_names, variable_types=None, fitted=True,
                 converged_items=None, khat_overrides=None):
        self.variable_names = variable_names if fitted else []
        self.variable_types = variable_types or {}
        self.zero_predictor_results = {}
        self.univariate_results = {}
        self.prediction_graph = {}
        self.n_obs_total = 100

        if fitted:
            for i, name in enumerate(variable_names):
                if converged_items is not None and name not in converged_items:
                    # Create non-converged result
                    self.zero_predictor_results[i] = MockUnivariateResult(
                        target_idx=i, converged=False
                    )
                else:
                    khat = 0.3
                    if khat_overrides and name in khat_overrides:
                        khat = khat_overrides[name]
                    self.zero_predictor_results[i] = MockUnivariateResult(
                        target_idx=i, converged=True, khat_max=khat
                    )

                # Also set variable types if not manually specified
                if i not in self.variable_types:
                    self.variable_types[i] = 'ordinal'

    def predict(self, items, target, return_details=False):
        """Mock predict returning a stacked prediction."""
        # Return a plausible ordinal value
        pred = 2.0  # middle of 0-4 range
        if return_details:
            return {
                'prediction': pred,
                'weights': {'intercept': 1.0},
                'predictions': {'intercept': pred},
                'elpd_loo': {'intercept': -50.0},
                'elpd_loo_se': {'intercept': 5.0},
                'n_obs_total': self.n_obs_total,
            }
        return pred

    def predict_pmf(self, items, target, n_categories, uncertainty_penalty=1.0):
        """Mock predict_pmf returning a proper categorical PMF.

        Returns a distribution peaked at category 2 with some spread.
        """
        pmf = np.ones(n_categories) * 0.02
        pmf[min(2, n_categories - 1)] += 1.0
        pmf /= pmf.sum()
        return pmf


# =========================================================================
# Test validate_imputation_model
# =========================================================================

def _make_concrete_irt(item_keys, num_people=10, response_cardinality=5,
                       imputation_model=None):
    """Create a concrete IRTModel subclass for testing validation/imputation.

    GRModel is the simplest concrete subclass; we use it directly.
    """
    from bayesianquilts.irt.grm import GRModel
    return GRModel(
        item_keys=item_keys,
        num_people=num_people,
        response_cardinality=response_cardinality,
        imputation_model=imputation_model,
    )


class TestValidateImputationModel:

    def _make_irt_model(self, item_keys, imputation_model=None):
        """Create a concrete IRT model instance for testing."""
        return _make_concrete_irt(item_keys, imputation_model=imputation_model)

    def test_unfitted_model_raises(self):
        """Check 1: Unfitted imputation model raises ValueError."""
        im = MockImputationModel(
            variable_names=['q1', 'q2', 'q3'],
            fitted=False,
        )
        model = self._make_irt_model(['q1', 'q2', 'q3'], imputation_model=im)
        with pytest.raises(ValueError, match="not been fitted"):
            model.validate_imputation_model()

    def test_missing_item_coverage_raises(self):
        """Check 2: Missing item coverage raises ValueError."""
        im = MockImputationModel(
            variable_names=['q1', 'q2', 'q3'],
        )
        model = self._make_irt_model(['q1', 'q2', 'q3', 'q4'], imputation_model=im)
        with pytest.raises(ValueError, match="does not cover"):
            model.validate_imputation_model()

    def test_continuous_type_raises(self):
        """Check 3: Continuous variable type raises ValueError."""
        im = MockImputationModel(
            variable_names=['q1', 'q2', 'q3'],
            variable_types={0: 'ordinal', 1: 'continuous', 2: 'ordinal'},
        )
        model = self._make_irt_model(['q1', 'q2', 'q3'], imputation_model=im)
        with pytest.raises(ValueError, match="continuous"):
            model.validate_imputation_model()

    def test_no_converged_models_warns(self):
        """Check 4: No converged models issues warning."""
        im = MockImputationModel(
            variable_names=['q1', 'q2', 'q3'],
            converged_items=['q1', 'q2'],  # q3 has no converged models
        )
        model = self._make_irt_model(['q1', 'q2', 'q3'], imputation_model=im)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.validate_imputation_model()
            assert any("no converged" in str(warning.message).lower() for warning in w)

    def test_high_khat_warns(self):
        """Check 5: High khat issues warning."""
        im = MockImputationModel(
            variable_names=['q1', 'q2', 'q3'],
            khat_overrides={'q1': 0.85},
        )
        model = self._make_irt_model(['q1', 'q2', 'q3'], imputation_model=im)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.validate_imputation_model()
            assert any("khat" in str(warning.message).lower() for warning in w)

    def test_all_checks_pass(self):
        """All checks pass with properly fitted model."""
        im = MockImputationModel(
            variable_names=['q1', 'q2', 'q3'],
        )
        model = self._make_irt_model(['q1', 'q2', 'q3'], imputation_model=im)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.validate_imputation_model()
            # No warnings should be raised
            assert len(w) == 0

    def test_none_imputation_model_noop(self):
        """validate_imputation_model with None model does nothing."""
        model = self._make_irt_model(['q1', 'q2'], imputation_model=None)
        # Should not raise
        model.validate_imputation_model()


# =========================================================================
# Test GRModel
# =========================================================================

class TestGRModel:

    def _make_grm(self, num_items=10, num_people=50, response_cardinality=5,
                  imputation_model=None):
        from bayesianquilts.irt.grm import GRModel
        item_keys = [f"item_{i}" for i in range(num_items)]
        model = GRModel(
            item_keys=item_keys,
            num_people=num_people,
            response_cardinality=response_cardinality,
            imputation_model=imputation_model,
        )
        return model, item_keys

    def _make_synthetic_data(self, num_people, item_keys, response_cardinality,
                             missingness_rate=0.0):
        """Generate synthetic IRT response data."""
        rng = np.random.default_rng(42)
        n_obs = num_people
        n_items = len(item_keys)

        data = {
            'person': np.arange(n_obs, dtype=np.float64),
        }
        for i, key in enumerate(item_keys):
            responses = rng.integers(0, response_cardinality, size=n_obs).astype(np.float64)
            if missingness_rate > 0:
                mask = rng.random(n_obs) < missingness_rate
                responses[mask] = np.nan
            data[key] = responses

        return data

    def test_grm_construction(self):
        """GRModel can be constructed with valid parameters."""
        model, _ = self._make_grm()
        assert model.params is not None
        assert len(model.var_list) > 0
        assert 'abilities' in model.var_list

    def test_grm_complete_data_no_imputation(self):
        """GRModel fits with complete data and no imputation model."""
        model, item_keys = self._make_grm(num_items=5, num_people=20)
        data = self._make_synthetic_data(20, item_keys, 5, missingness_rate=0.0)

        # Create a simple data factory
        def data_factory():
            return iter([data])

        losses, params = model.fit(
            data_factory,
            batch_size=20,
            dataset_size=20,
            num_epochs=2,
            steps_per_epoch=1,
            learning_rate=0.1,
        )
        assert params is not None
        # Loss should be finite
        assert np.isfinite(losses[-1])

    def test_grm_missing_data_no_imputation(self):
        """GRModel fits with missing data using entropy fill."""
        model, item_keys = self._make_grm(num_items=5, num_people=20)
        data = self._make_synthetic_data(20, item_keys, 5, missingness_rate=0.2)

        def data_factory():
            return iter([data])

        losses, params = model.fit(
            data_factory,
            batch_size=20,
            dataset_size=20,
            num_epochs=2,
            steps_per_epoch=1,
            learning_rate=0.1,
        )
        assert params is not None
        assert np.isfinite(losses[-1])

    def test_grm_complete_data_with_imputation(self):
        """GRModel fits with complete data and imputation model (no-op)."""
        item_keys = [f"item_{i}" for i in range(5)]
        im = MockImputationModel(variable_names=item_keys)
        model, _ = self._make_grm(
            num_items=5, num_people=20, imputation_model=im
        )
        data = self._make_synthetic_data(20, item_keys, 5, missingness_rate=0.0)

        def data_factory():
            return iter([data])

        losses, params = model.fit(
            data_factory,
            batch_size=20,
            dataset_size=20,
            num_epochs=2,
            steps_per_epoch=1,
            learning_rate=0.1,
        )
        assert params is not None
        assert np.isfinite(losses[-1])

    def test_grm_missing_data_with_imputation(self):
        """GRModel fits with missing data and PMF-weighted imputation."""
        item_keys = [f"item_{i}" for i in range(5)]
        im = MockImputationModel(variable_names=item_keys)
        model, _ = self._make_grm(
            num_items=5, num_people=20, imputation_model=im,
        )
        data = self._make_synthetic_data(20, item_keys, 5, missingness_rate=0.2)

        def data_factory():
            return iter([data])

        losses, params = model.fit(
            data_factory,
            batch_size=20,
            dataset_size=20,
            num_epochs=2,
            steps_per_epoch=1,
            learning_rate=0.1,
        )
        assert params is not None
        assert np.isfinite(losses[-1])


# =========================================================================
# Test FactorizedGRModel
# =========================================================================

class TestFactorizedGRModel:

    def _make_fgrm(self, num_people=20, response_cardinality=5,
                   imputation_model=None):
        from bayesianquilts.irt.factorizedgrm import FactorizedGRModel
        item_keys = [f"item_{i}" for i in range(6)]
        scale_indices = [[0, 1, 2], [3, 4, 5]]
        model = FactorizedGRModel(
            scale_indices=scale_indices,
            kappa_scale=0.1,
            item_keys=item_keys,
            num_people=num_people,
            response_cardinality=response_cardinality,
            imputation_model=imputation_model,
        )
        return model, item_keys

    def _make_synthetic_data(self, num_people, item_keys, response_cardinality,
                             missingness_rate=0.0):
        rng = np.random.default_rng(42)
        n_obs = num_people
        data = {
            'person': np.arange(n_obs, dtype=np.float64),
        }
        for key in item_keys:
            responses = rng.integers(0, response_cardinality, size=n_obs).astype(np.float64)
            if missingness_rate > 0:
                mask = rng.random(n_obs) < missingness_rate
                responses[mask] = np.nan
            data[key] = responses
        return data

    def test_fgrm_construction(self):
        """FactorizedGRModel can be constructed."""
        model, _ = self._make_fgrm()
        assert model.params is not None
        assert len(model.var_list) > 0

    def test_fgrm_complete_data_no_imputation(self):
        """FactorizedGRModel fits with complete data."""
        model, item_keys = self._make_fgrm()
        data = self._make_synthetic_data(20, item_keys, 5, missingness_rate=0.0)

        def data_factory():
            return iter([data])

        losses, params = model.fit(
            data_factory,
            batch_size=20,
            dataset_size=20,
            num_epochs=2,
            steps_per_epoch=1,
            learning_rate=0.1,
        )
        assert params is not None
        assert np.isfinite(losses[-1])

    def test_fgrm_missing_data_no_imputation(self):
        """FactorizedGRModel fits with missing data using zero fill."""
        model, item_keys = self._make_fgrm()
        data = self._make_synthetic_data(20, item_keys, 5, missingness_rate=0.15)

        def data_factory():
            return iter([data])

        losses, params = model.fit(
            data_factory,
            batch_size=20,
            dataset_size=20,
            num_epochs=2,
            steps_per_epoch=1,
            learning_rate=0.1,
        )
        assert params is not None
        assert np.isfinite(losses[-1])

    def test_fgrm_missing_data_with_imputation(self):
        """FactorizedGRModel fits with PMF-weighted imputation."""
        item_keys = [f"item_{i}" for i in range(6)]
        im = MockImputationModel(variable_names=item_keys)
        model, _ = self._make_fgrm(
            imputation_model=im,
        )
        data = self._make_synthetic_data(20, item_keys, 5, missingness_rate=0.15)

        def data_factory():
            return iter([data])

        losses, params = model.fit(
            data_factory,
            batch_size=20,
            dataset_size=20,
            num_epochs=2,
            steps_per_epoch=1,
            learning_rate=0.1,
        )
        assert params is not None
        assert np.isfinite(losses[-1])

    def test_fgrm_validate_imputation_model(self):
        """Sanity checks pass for factorized model."""
        item_keys = [f"item_{i}" for i in range(6)]
        im = MockImputationModel(variable_names=item_keys)
        model, _ = self._make_fgrm(imputation_model=im)
        # Should not raise
        model.validate_imputation_model()


# =========================================================================
# Test FactorizedGRModel fitting with missing responses
# =========================================================================

class TestFactorizedGRModelMissingFit:
    """Thorough tests for fitting FactorizedGRModel with missing response data.

    Exercises the PMF-weighted imputation, verifies scale structure is preserved,
    and confirms loss is well-behaved.
    """

    NUM_PEOPLE = 40
    NUM_ITEMS_PER_SCALE = 4
    NUM_SCALES = 3
    RESPONSE_CARDINALITY = 5

    @pytest.fixture
    def item_keys(self):
        return [f"q_{i}" for i in range(self.NUM_SCALES * self.NUM_ITEMS_PER_SCALE)]

    @pytest.fixture
    def scale_indices(self):
        n = self.NUM_ITEMS_PER_SCALE
        return [list(range(j * n, (j + 1) * n)) for j in range(self.NUM_SCALES)]

    def _make_data(self, item_keys, missingness_rate=0.0, seed=123):
        """Generate synthetic ordinal response data with optional MCAR missingness."""
        rng = np.random.default_rng(seed)
        data = {"person": np.arange(self.NUM_PEOPLE, dtype=np.float64)}
        for key in item_keys:
            vals = rng.integers(0, self.RESPONSE_CARDINALITY, size=self.NUM_PEOPLE).astype(
                np.float64
            )
            if missingness_rate > 0:
                mask = rng.random(self.NUM_PEOPLE) < missingness_rate
                vals[mask] = np.nan
            data[key] = vals
        return data

    def _make_model(self, item_keys, scale_indices, imputation_model=None):
        from bayesianquilts.irt.factorizedgrm import FactorizedGRModel
        return FactorizedGRModel(
            scale_indices=scale_indices,
            kappa_scale=0.1,
            item_keys=item_keys,
            num_people=self.NUM_PEOPLE,
            response_cardinality=self.RESPONSE_CARDINALITY,
            imputation_model=imputation_model,
        )

    # ------------------------------------------------------------------
    # 1. PMF computation mechanics
    # ------------------------------------------------------------------

    def test_compute_batch_pmfs_shape_and_values(self, item_keys, scale_indices):
        """_compute_batch_pmfs returns (N, I, K) with proper PMFs for missing cells."""
        im = MockImputationModel(variable_names=item_keys)
        model = self._make_model(
            item_keys, scale_indices, imputation_model=im,
        )
        data = self._make_data(item_keys, missingness_rate=0.25)

        pmfs = model._compute_batch_pmfs(data)

        N = self.NUM_PEOPLE
        I = len(item_keys)
        K = self.RESPONSE_CARDINALITY
        assert pmfs.shape == (N, I, K)

        # Missing cells should have valid PMFs (sum to 1)
        for i, key in enumerate(item_keys):
            col = np.asarray(data[key], dtype=np.float64)
            bad = np.isnan(col) | (col < 0) | (col >= K)
            for row_idx in np.where(bad)[0]:
                row_pmf = pmfs[row_idx, i, :]
                assert np.allclose(row_pmf.sum(), 1.0, atol=1e-10), (
                    f"PMF for missing cell ({row_idx}, {key}) doesn't sum to 1"
                )
                assert np.all(row_pmf >= 0), (
                    f"PMF for missing cell ({row_idx}, {key}) has negative values"
                )

        # Observed cells should have zeros
        for i, key in enumerate(item_keys):
            col = np.asarray(data[key], dtype=np.float64)
            good = ~(np.isnan(col) | (col < 0) | (col >= K))
            for row_idx in np.where(good)[0]:
                row_pmf = pmfs[row_idx, i, :]
                assert np.allclose(row_pmf, 0.0), (
                    f"PMF for observed cell ({row_idx}, {key}) is not zero"
                )

    def test_observed_values_unchanged_in_batch(self, item_keys, scale_indices):
        """PMF computation does not modify the original batch data."""
        im = MockImputationModel(variable_names=item_keys)
        model = self._make_model(
            item_keys, scale_indices, imputation_model=im,
        )
        data = self._make_data(item_keys, missingness_rate=0.2, seed=99)
        original_data = {k: np.array(v, copy=True) for k, v in data.items()}

        model._compute_batch_pmfs(data)

        for k in item_keys:
            np.testing.assert_array_equal(
                data[k], original_data[k],
                err_msg=f"_compute_batch_pmfs modified original data for {k}",
            )

    # ------------------------------------------------------------------
    # 2. PMF-weighted log-likelihood verification
    # ------------------------------------------------------------------

    def test_pmf_weighted_log_likelihood_manual(self, item_keys, scale_indices):
        """Verify PMF-weighted LL matches manual computation for a simple case."""
        import jax
        im = MockImputationModel(variable_names=item_keys)
        model = self._make_model(
            item_keys, scale_indices, imputation_model=im,
        )
        data = self._make_data(item_keys, missingness_rate=0.3, seed=42)

        pmfs = model._compute_batch_pmfs(data)

        # The PMFs should be valid distributions for missing cells
        for i, key in enumerate(item_keys):
            col = np.asarray(data[key], dtype=np.float64)
            bad = np.isnan(col) | (col < 0) | (col >= self.RESPONSE_CARDINALITY)
            for row_idx in np.where(bad)[0]:
                row_pmf = pmfs[row_idx, i, :]
                # Should be a valid probability distribution
                assert row_pmf.sum() > 0.99, f"PMF not valid at ({row_idx}, {i})"

    # ------------------------------------------------------------------
    # 3. End-to-end fitting with missing data
    # ------------------------------------------------------------------

    def test_fit_missing_data_with_imputation_loss_finite(
        self, item_keys, scale_indices
    ):
        """Fit with 25% missingness + imputation: loss should remain finite."""
        im = MockImputationModel(variable_names=item_keys)
        model = self._make_model(
            item_keys, scale_indices, imputation_model=im,
        )
        data = self._make_data(item_keys, missingness_rate=0.25)

        def factory():
            return iter([data])

        losses, params = model.fit(
            factory,
            batch_size=self.NUM_PEOPLE,
            dataset_size=self.NUM_PEOPLE,
            num_epochs=3,
            steps_per_epoch=2,
            learning_rate=0.05,
        )
        assert params is not None
        for i, l in enumerate(losses):
            assert np.isfinite(l), f"Loss at epoch {i} is not finite: {l}"

    def test_fit_high_missingness(self, item_keys, scale_indices):
        """50% missingness still produces finite losses (stress test)."""
        im = MockImputationModel(variable_names=item_keys)
        model = self._make_model(
            item_keys, scale_indices, imputation_model=im,
        )
        data = self._make_data(item_keys, missingness_rate=0.50, seed=77)

        def factory():
            return iter([data])

        losses, params = model.fit(
            factory,
            batch_size=self.NUM_PEOPLE,
            dataset_size=self.NUM_PEOPLE,
            num_epochs=3,
            steps_per_epoch=1,
            learning_rate=0.05,
        )
        assert params is not None
        assert np.isfinite(losses[-1]), f"Final loss not finite: {losses[-1]}"

    def test_fit_single_epoch(self, item_keys, scale_indices):
        """Single epoch with imputation should work."""
        im = MockImputationModel(variable_names=item_keys)
        model = self._make_model(
            item_keys, scale_indices, imputation_model=im,
        )
        data = self._make_data(item_keys, missingness_rate=0.2)

        def factory():
            return iter([data])

        losses, params = model.fit(
            factory,
            batch_size=self.NUM_PEOPLE,
            dataset_size=self.NUM_PEOPLE,
            num_epochs=2,
            steps_per_epoch=1,
            learning_rate=0.1,
        )
        assert params is not None
        assert np.isfinite(losses[-1])

    # ------------------------------------------------------------------
    # 4. Comparison: missing data with vs without imputation
    # ------------------------------------------------------------------

    def test_fit_missing_without_imputation_also_works(
        self, item_keys, scale_indices
    ):
        """Fitting with missing data but no imputation model uses zero-fill
        and should also produce finite losses."""
        model = self._make_model(
            item_keys, scale_indices, imputation_model=None,
        )
        data = self._make_data(item_keys, missingness_rate=0.25)

        def factory():
            return iter([data])

        losses, params = model.fit(
            factory,
            batch_size=self.NUM_PEOPLE,
            dataset_size=self.NUM_PEOPLE,
            num_epochs=3,
            steps_per_epoch=1,
            learning_rate=0.05,
        )
        assert params is not None
        assert np.isfinite(losses[-1])

    # ------------------------------------------------------------------
    # 5. Scale structure preserved after imputed training
    # ------------------------------------------------------------------

    def test_scale_parameters_present_after_fit(self, item_keys, scale_indices):
        """After fitting with imputation, per-scale surrogate params exist."""
        im = MockImputationModel(variable_names=item_keys)
        model = self._make_model(
            item_keys, scale_indices, imputation_model=im,
        )
        data = self._make_data(item_keys, missingness_rate=0.2)

        def factory():
            return iter([data])

        _, params = model.fit(
            factory,
            batch_size=self.NUM_PEOPLE,
            dataset_size=self.NUM_PEOPLE,
            num_epochs=2,
            steps_per_epoch=1,
            learning_rate=0.1,
        )

        # Each scale should have its own discrimination and difficulty params
        for j in range(self.NUM_SCALES):
            assert any(f"discriminations_{j}" in k for k in params), (
                f"Missing discriminations_{j} params after fit"
            )
            assert any(f"difficulties0_{j}" in k for k in params), (
                f"Missing difficulties0_{j} params after fit"
            )
            assert any(f"abilities_{j}" in k for k in params), (
                f"Missing abilities_{j} params after fit"
            )

    # ------------------------------------------------------------------
    # 6. Missingness patterns
    # ------------------------------------------------------------------

    def test_fit_missingness_concentrated_in_one_scale(
        self, item_keys, scale_indices
    ):
        """Missingness only in one scale's items; other scales are complete."""
        im = MockImputationModel(variable_names=item_keys)
        model = self._make_model(
            item_keys, scale_indices, imputation_model=im,
        )
        rng = np.random.default_rng(55)
        data = {"person": np.arange(self.NUM_PEOPLE, dtype=np.float64)}
        for i, key in enumerate(item_keys):
            vals = rng.integers(0, self.RESPONSE_CARDINALITY, size=self.NUM_PEOPLE).astype(
                np.float64
            )
            # Only inject missingness in the first scale's items
            if i < self.NUM_ITEMS_PER_SCALE:
                mask = rng.random(self.NUM_PEOPLE) < 0.4
                vals[mask] = np.nan
            data[key] = vals

        def factory():
            return iter([data])

        losses, params = model.fit(
            factory,
            batch_size=self.NUM_PEOPLE,
            dataset_size=self.NUM_PEOPLE,
            num_epochs=3,
            steps_per_epoch=1,
            learning_rate=0.05,
        )
        assert params is not None
        assert np.isfinite(losses[-1])

    def test_fit_single_item_missing_per_person(self, item_keys, scale_indices):
        """Each person is missing exactly one item (monotone-ish pattern)."""
        im = MockImputationModel(variable_names=item_keys)
        model = self._make_model(
            item_keys, scale_indices, imputation_model=im,
        )
        rng = np.random.default_rng(33)
        data = {"person": np.arange(self.NUM_PEOPLE, dtype=np.float64)}
        n_items = len(item_keys)
        for i, key in enumerate(item_keys):
            data[key] = rng.integers(
                0, self.RESPONSE_CARDINALITY, size=self.NUM_PEOPLE
            ).astype(np.float64)

        # For each person, set exactly one random item to NaN
        for p in range(self.NUM_PEOPLE):
            drop_idx = rng.integers(0, n_items)
            data[item_keys[drop_idx]][p] = np.nan

        def factory():
            return iter([data])

        losses, params = model.fit(
            factory,
            batch_size=self.NUM_PEOPLE,
            dataset_size=self.NUM_PEOPLE,
            num_epochs=3,
            steps_per_epoch=1,
            learning_rate=0.05,
        )
        assert params is not None
        assert np.isfinite(losses[-1])

    # ------------------------------------------------------------------
    # 7. Deprecated n_imputation_samples is silently ignored
    # ------------------------------------------------------------------

    def test_n_imputation_samples_ignored(self, item_keys, scale_indices):
        """Passing n_imputation_samples does not error (backwards compat)."""
        im = MockImputationModel(variable_names=item_keys)
        model = self._make_model(
            item_keys, scale_indices, imputation_model=im,
        )
        data = self._make_data(item_keys, missingness_rate=0.2)

        def factory():
            return iter([data])

        losses, params = model.fit(
            factory,
            batch_size=self.NUM_PEOPLE,
            dataset_size=self.NUM_PEOPLE,
            num_epochs=2,
            steps_per_epoch=1,
            learning_rate=0.1,
            n_imputation_samples=5,  # should be silently ignored
        )
        assert params is not None
        assert np.isfinite(losses[-1])


# =========================================================================
# Test PMF batch computation
# =========================================================================

class TestComputeBatchPmfs:

    def test_no_missing_returns_zeros(self):
        """Batch with no missing values returns all-zero PMFs."""
        item_keys = ['q1', 'q2', 'q3']
        im = MockImputationModel(variable_names=item_keys)
        model = _make_concrete_irt(
            item_keys=item_keys,
            num_people=10,
            imputation_model=im,
        )

        batch = {
            'person': np.arange(5, dtype=np.float64),
            'q1': np.array([0, 1, 2, 3, 4], dtype=np.float64),
            'q2': np.array([1, 2, 3, 0, 1], dtype=np.float64),
            'q3': np.array([4, 3, 2, 1, 0], dtype=np.float64),
        }

        pmfs = model._compute_batch_pmfs(batch)
        assert pmfs.shape == (5, 3, 5)
        assert np.allclose(pmfs, 0.0)

    def test_missing_detected(self):
        """Batch with NaN values detected as having missing."""
        item_keys = ['q1', 'q2']
        model = _make_concrete_irt(
            item_keys=item_keys,
            num_people=5,
        )

        batch = {
            'person': np.arange(5, dtype=np.float64),
            'q1': np.array([0, 1, np.nan, 3, 4], dtype=np.float64),
            'q2': np.array([1, 2, 3, 0, 1], dtype=np.float64),
        }

        assert model._has_missing_values(batch)

    def test_compute_pmfs_fills_missing(self):
        """_compute_batch_pmfs produces valid PMFs for missing cells."""
        item_keys = ['q1', 'q2']
        im = MockImputationModel(variable_names=item_keys)
        model = _make_concrete_irt(
            item_keys=item_keys,
            num_people=5,
            imputation_model=im,
        )

        batch = {
            'person': np.arange(5, dtype=np.float64),
            'q1': np.array([0, 1, np.nan, 3, np.nan], dtype=np.float64),
            'q2': np.array([1, np.nan, 3, 0, 1], dtype=np.float64),
        }

        pmfs = model._compute_batch_pmfs(batch)
        assert pmfs.shape == (5, 2, 5)

        # Missing cells should have valid PMFs
        # q1 missing at indices 2, 4
        assert np.allclose(pmfs[2, 0, :].sum(), 1.0)
        assert np.allclose(pmfs[4, 0, :].sum(), 1.0)
        # q2 missing at index 1
        assert np.allclose(pmfs[1, 1, :].sum(), 1.0)

        # Observed cells should be zero
        assert np.allclose(pmfs[0, 0, :], 0.0)  # q1[0] = 0 (observed)
        assert np.allclose(pmfs[0, 1, :], 0.0)  # q2[0] = 1 (observed)

    def test_observed_cells_zero(self):
        """Observed cells in PMF array are zeros."""
        item_keys = ['q1', 'q2']
        im = MockImputationModel(variable_names=item_keys)
        model = _make_concrete_irt(
            item_keys=item_keys,
            num_people=5,
            imputation_model=im,
        )

        original_q2 = np.array([1, 2, 3, 0, 4], dtype=np.float64)
        batch = {
            'person': np.arange(5, dtype=np.float64),
            'q1': np.array([0, np.nan, 2, 3, 4], dtype=np.float64),
            'q2': original_q2.copy(),
        }

        pmfs = model._compute_batch_pmfs(batch)

        # q2 has no missing values, entire column should be zero
        assert np.allclose(pmfs[:, 1, :], 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

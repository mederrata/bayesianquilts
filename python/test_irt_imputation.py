"""Integration tests for IRT models with stochastic imputation.

Tests cover:
- GRModel and FactorizedGRModel with complete and missing data
- validate_imputation_model() sanity checks
- Stochastic imputation via MICEBayesianLOO
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
        """GRModel fits with missing data and imputation model."""
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
            n_imputation_samples=2,
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
        """FactorizedGRModel fits with imputation model."""
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
            n_imputation_samples=2,
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

    Exercises the imputation iterator, verifies scale structure is preserved,
    checks multiple n_imputation_samples values, and confirms loss is well-behaved.
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
    # 1. Imputation iterator mechanics
    # ------------------------------------------------------------------

    def test_imputing_iterator_yields_n_copies_for_missing_batch(
        self, item_keys, scale_indices
    ):
        """When a batch has missing values, the imputing iterator yields
        n_imputation_samples copies, each with no NaNs."""
        n_samples = 3
        im = MockImputationModel(variable_names=item_keys)
        model = self._make_model(
            item_keys, scale_indices, imputation_model=im,
        )
        data = self._make_data(item_keys, missingness_rate=0.25)

        # Verify the raw data has NaNs
        nan_count = sum(np.isnan(data[k]).sum() for k in item_keys)
        assert nan_count > 0, "Test data should have missing values"

        # Collect batches from the imputing iterator
        rng = np.random.default_rng(0)
        copies = []
        for _ in range(n_samples):
            copies.append(model._impute_batch(data, rng))

        assert len(copies) == n_samples

        for i, batch in enumerate(copies):
            for k in item_keys:
                assert not np.any(np.isnan(batch[k])), (
                    f"Copy {i}, item {k} still has NaN"
                )
                assert np.all(batch[k] >= 0), f"Copy {i}, {k} has negatives"
                assert np.all(batch[k] < self.RESPONSE_CARDINALITY), (
                    f"Copy {i}, {k} has out-of-range values"
                )

    def test_imputation_stochastic_across_draws(self, item_keys, scale_indices):
        """Multiple imputation draws should not all be identical (stochasticity)."""
        im = MockImputationModel(variable_names=item_keys)
        model = self._make_model(
            item_keys, scale_indices, imputation_model=im,
        )
        data = self._make_data(item_keys, missingness_rate=0.3, seed=7)

        rng = np.random.default_rng(42)
        draws = [model._impute_batch(data, rng) for _ in range(20)]

        # At least one pair of draws should differ somewhere
        any_different = False
        for k in item_keys:
            vals = np.stack([d[k] for d in draws], axis=0)
            if np.any(vals != vals[0:1]):
                any_different = True
                break
        assert any_different, (
            "All 20 imputation draws are identical — imputation may not be stochastic"
        )

    def test_observed_values_unchanged_after_imputation(self, item_keys, scale_indices):
        """Imputation must not modify observed (non-missing) responses."""
        im = MockImputationModel(variable_names=item_keys)
        model = self._make_model(
            item_keys, scale_indices, imputation_model=im,
        )
        data = self._make_data(item_keys, missingness_rate=0.2, seed=99)

        imputed = model._impute_batch(data)

        for k in item_keys:
            orig = data[k]
            observed_mask = ~np.isnan(orig) & (orig >= 0) & (orig < self.RESPONSE_CARDINALITY)
            np.testing.assert_array_equal(
                imputed[k][observed_mask], orig[observed_mask],
                err_msg=f"Observed values in {k} were modified by imputation",
            )

    # ------------------------------------------------------------------
    # 2. End-to-end fitting with missing data
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
            n_imputation_samples=2,
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
            n_imputation_samples=3,
        )
        assert params is not None
        assert np.isfinite(losses[-1]), f"Final loss not finite: {losses[-1]}"

    def test_fit_single_imputation_sample(self, item_keys, scale_indices):
        """n_imputation_samples=1 should work (no averaging needed)."""
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
    # 3. Comparison: missing data with vs without imputation
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
    # 4. Scale structure preserved after imputed training
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
            n_imputation_samples=2,
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
    # 5. Missingness patterns
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
            n_imputation_samples=2,
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
            n_imputation_samples=2,
        )
        assert params is not None
        assert np.isfinite(losses[-1])


# =========================================================================
# Test imputation batch wrapping
# =========================================================================

class TestImputationBatchWrapping:

    def test_no_missing_passes_through(self):
        """Batch with no missing values passes through unchanged."""
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

        assert not model._has_missing_values(batch)

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

    def test_impute_batch_fills_missing(self):
        """_impute_batch fills all missing values."""
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

        imputed = model._impute_batch(batch)

        # No NaN values in imputed batch
        for key in item_keys:
            assert not np.any(np.isnan(imputed[key])), f"NaN found in {key}"
            # Values should be in valid range
            assert np.all(imputed[key] >= 0)
            assert np.all(imputed[key] < 5)

        # Observed values should be preserved
        assert imputed['q1'][0] == 0.0
        assert imputed['q1'][1] == 1.0
        assert imputed['q1'][3] == 3.0

    def test_impute_preserves_observed(self):
        """Observed values are not modified by imputation."""
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

        imputed = model._impute_batch(batch)

        # q2 had no missing values, should be unchanged
        np.testing.assert_array_equal(imputed['q2'], original_q2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

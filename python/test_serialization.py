"""Round-trip serialization tests for FactorizedGRModel."""

import tempfile
import shutil

import jax.numpy as jnp
import numpy as np
import pytest

from bayesianquilts.irt.factorizedgrm import FactorizedGRModel


BACKENDS = ["hdf5", "safetensors"]


class TestFactorizedGRModelSerialization:

    def _make_model(self, num_people=20, response_cardinality=5):
        item_keys = [f"item_{i}" for i in range(6)]
        scale_indices = [[0, 1, 2], [3, 4, 5]]
        model = FactorizedGRModel(
            scale_indices=scale_indices,
            kappa_scale=0.1,
            item_keys=item_keys,
            num_people=num_people,
            response_cardinality=response_cardinality,
        )
        return model

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_save_load_roundtrip(self, backend):
        """Config and params survive a save/load cycle."""
        model = self._make_model()
        tmpdir = tempfile.mkdtemp()
        try:
            model.save_to_disk(tmpdir, backend=backend)
            loaded = FactorizedGRModel.load_from_disk(tmpdir)

            # Check config values
            assert loaded.scale_indices == model.scale_indices
            assert loaded.kappa_scale == model.kappa_scale
            assert loaded.item_keys == model.item_keys
            assert loaded.num_people == model.num_people
            assert loaded.response_cardinality == model.response_cardinality
            assert loaded.dtype == model.dtype
            assert loaded.num_items == model.num_items
            assert loaded.dimensions == model.dimensions

            # Check params keys match
            assert set(loaded.params.keys()) == set(model.params.keys())

            # Check param values match
            for k in model.params:
                np.testing.assert_allclose(
                    np.array(loaded.params[k]),
                    np.array(model.params[k]),
                    atol=1e-7,
                    err_msg=f"Param mismatch for {k} (backend={backend})",
                )
        finally:
            shutil.rmtree(tmpdir)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_save_load_preserves_dtype(self, backend):
        """dtype is correctly round-tripped (not left as a string)."""
        model = self._make_model()
        tmpdir = tempfile.mkdtemp()
        try:
            model.save_to_disk(tmpdir, backend=backend)
            loaded = FactorizedGRModel.load_from_disk(tmpdir)
            assert loaded.dtype == jnp.float64
            assert loaded.dtype is not str
        finally:
            shutil.rmtree(tmpdir)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_save_load_different_cardinality(self, backend):
        """Model with non-default response_cardinality round-trips."""
        model = self._make_model(response_cardinality=3)
        tmpdir = tempfile.mkdtemp()
        try:
            model.save_to_disk(tmpdir, backend=backend)
            loaded = FactorizedGRModel.load_from_disk(tmpdir)
            assert loaded.response_cardinality == 3
        finally:
            shutil.rmtree(tmpdir)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_save_load_var_list(self, backend):
        """var_list is correctly reconstructed after load."""
        model = self._make_model()
        tmpdir = tempfile.mkdtemp()
        try:
            model.save_to_disk(tmpdir, backend=backend)
            loaded = FactorizedGRModel.load_from_disk(tmpdir)
            assert set(loaded.var_list) == set(model.var_list)
        finally:
            shutil.rmtree(tmpdir)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_loaded_model_can_sample(self, backend):
        """Loaded model can generate samples from surrogate."""
        model = self._make_model()
        tmpdir = tempfile.mkdtemp()
        try:
            model.save_to_disk(tmpdir, backend=backend)
            loaded = FactorizedGRModel.load_from_disk(tmpdir)
            samples = loaded.sample(batch_shape=(2,))
            assert isinstance(samples, dict)
            assert len(samples) > 0
        finally:
            shutil.rmtree(tmpdir)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_loaded_model_can_compute_log_prob(self, backend):
        """Loaded model can evaluate unormalized_log_prob on data."""
        num_people = 10
        model = self._make_model(num_people=num_people)
        rng = np.random.default_rng(42)
        data = {"person": np.arange(num_people, dtype=np.float64)}
        for key in model.item_keys:
            data[key] = rng.integers(
                0, model.response_cardinality, size=num_people
            ).astype(np.float64)

        tmpdir = tempfile.mkdtemp()
        try:
            model.save_to_disk(tmpdir, backend=backend)
            loaded = FactorizedGRModel.load_from_disk(tmpdir)
            # batch_shape=(1,) needed so transform() gets the expected 5D tensors
            samples = loaded.sample(batch_shape=(1,))
            lp_orig = model.unormalized_log_prob(data=data, **samples)
            lp_loaded = loaded.unormalized_log_prob(data=data, **samples)
            np.testing.assert_allclose(
                np.array(lp_orig), np.array(lp_loaded), atol=1e-5,
            )
        finally:
            shutil.rmtree(tmpdir)

    def test_invalid_backend_raises(self):
        """Passing an unknown backend raises ValueError."""
        model = self._make_model()
        tmpdir = tempfile.mkdtemp()
        try:
            with pytest.raises(ValueError, match="backend"):
                model.save_to_disk(tmpdir, backend="pickle")
        finally:
            shutil.rmtree(tmpdir)

    def test_safetensors_file_created(self):
        """safetensors backend creates the expected file."""
        import pathlib
        model = self._make_model()
        tmpdir = tempfile.mkdtemp()
        try:
            model.save_to_disk(tmpdir, backend="safetensors")
            p = pathlib.Path(tmpdir)
            assert (p / "config.yaml").exists()
            assert (p / "tensors.safetensors").exists()
            assert not (p / "params.h5").exists()
        finally:
            shutil.rmtree(tmpdir)

    def test_hdf5_file_created(self):
        """hdf5 backend creates the expected file."""
        import pathlib
        model = self._make_model()
        tmpdir = tempfile.mkdtemp()
        try:
            model.save_to_disk(tmpdir, backend="hdf5")
            p = pathlib.Path(tmpdir)
            assert (p / "config.yaml").exists()
            assert (p / "params.h5").exists()
            assert not (p / "tensors.safetensors").exists()
        finally:
            shutil.rmtree(tmpdir)

    def test_autodetect_backend(self):
        """load_from_disk picks the right backend from the YAML _backend key."""
        import yaml, pathlib
        model = self._make_model()

        for backend in BACKENDS:
            tmpdir = tempfile.mkdtemp()
            try:
                model.save_to_disk(tmpdir, backend=backend)

                # Verify _backend is recorded in YAML
                with open(pathlib.Path(tmpdir) / "config.yaml") as f:
                    cfg = yaml.safe_load(f)
                assert cfg["_backend"] == backend

                # load auto-detects
                loaded = FactorizedGRModel.load_from_disk(tmpdir)
                assert set(loaded.params.keys()) == set(model.params.keys())
            finally:
                shutil.rmtree(tmpdir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

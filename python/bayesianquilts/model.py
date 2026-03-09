import gzip
import inspect
import pathlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict
import os
import yaml
import h5py

import dill
import jax
import jax.numpy as jnp
import numpy as np
try:
    # arviz >= 1.0: InferenceData replaced by xarray.DataTree
    from arviz_base import convert_to_datatree, dict_to_dataset
    InferenceData = None  # not available in arviz 1.0
except ImportError:
    # arviz < 1.0
    from arviz.data import InferenceData
    from arviz.data.base import dict_to_dataset
    convert_to_datatree = None
from flax import nnx
from jax import random
from tensorflow_probability.substrates.jax import tf2jax as tf
from tqdm import tqdm

from bayesianquilts.jax.parameter import Interactions
from bayesianquilts.util import training_loop
from bayesianquilts.vi.advi import pathfinder_initialize
from bayesianquilts.vi.minibatch import minibatch_fit_surrogate_posterior
import bayesianquilts.tfp_patch

import tensorflow_probability.substrates.jax as tfp
tfmcmc = tfp.mcmc
tfd = tfp.distributions

def FactorizedDistributionMoments(dist, samples=100):
    try:
        # Try analytical moments
        mean = dist.mean()
        var = dist.variance()
        return mean, var
    except Exception:
        # Fallback to sampling
        s = dist.sample(samples)
        if isinstance(s, dict):
            mean = {k: jnp.mean(v, axis=0) for k, v in s.items()}
            var = {k: jnp.var(v, axis=0) for k, v in s.items()}
        else:
            mean = jnp.mean(s, axis=0)
            var = jnp.var(s, axis=0)
        return mean, var


class BayesianModel(nnx.Module, ABC):
    surrogate_distribution: Any = nnx.data(None)
    surrogate_sample: Any = nnx.data(None)
    prior_distribution: Any = nnx.data(None)
    data: Any = nnx.data(None)
    data_cardinality: Any = nnx.data(None)
    params: Any = nnx.data(None)
    point_estimate_vars: Any = nnx.data(None)
    var_list: list = []
    bijectors: list = []

    def __init__(
        self,
        data_transform_fn: Callable | None = None,
        dtype: jax.typing.DTypeLike = jnp.float64,
        **kwargs,
    ):
        """Instantiate Model object based on tensorflow dataset
        Arguments:
            data {tf.data or factory} -- Either a dataset or a factory to generate a dataset
        Keyword Arguments:
            data_transform_fn {[type]} -- [description] (default: {None})
            strategy {[type]} -- [description] (default: {None})
        Raises:
            AttributeError: [description]
        """
        super(BayesianModel, self).__init__()
        self.data_transform_fn = data_transform_fn

        self.dtype = dtype
        self.params = None
        self.mcmc_samples = None

    def fit(
        self,
        batched_data_factory,
        initial_values: Dict[str, jax.typing.ArrayLike] = None,
        point_estimate_vars: dict | None = None,
        **kwargs,
    ):
        # Auto-detect from self, allow override
        pe_vars = point_estimate_vars or getattr(self, 'point_estimate_vars', None)

        def infinite_data_iterator():
            while True:
                # Re-instantiate the iterator from the factory
                iterator = batched_data_factory()
                try:
                    yield from iterator
                except TypeError:
                    # If it's not iterable?
                    yield iterator

        res = self._calibrate_minibatch_advi(
            infinite_data_iterator(), initial_values=initial_values,
            point_estimate_vars=pe_vars, **kwargs
        )
        self.params = res[1]
        return res

    def compute_loo(self, data, likelihood_fn, khat_threshold=0.7):
        """Compute LOO-CV using PSIS, with AIS fallback for k-hat > threshold.

        Args:
            data: Full dataset dict
            likelihood_fn: LikelihoodFunction instance with log_likelihood,
                log_likelihood_gradient, log_likelihood_hessian_diag methods
            khat_threshold: AIS kicks in above this

        Stores on self:
            self.loo_results: dict with elpd_loo, elpd_loo_per_obs,
                elpd_loo_per_obs_se, pointwise_loo, khat, etc.
        """
        from bayesianquilts.metrics import nppsis
        from bayesianquilts.metrics.ais import AdaptiveImportanceSampler

        # Draw posterior samples
        params = self.sample(batch_shape=(200,))

        # Standard PSIS-LOO first
        log_lik = likelihood_fn.log_likelihood(data, params)  # (S, N)
        loo, loos, ks = nppsis.psisloo(np.array(log_lik))

        # If any k-hat > threshold, use AIS
        if np.max(ks) > khat_threshold:
            try:
                surrogate = self.surrogate_distribution_generator(self.params)
                surrogate_log_prob_fn = lambda p: surrogate.log_prob(p)
                prior_log_prob_fn = lambda p: self.prior_distribution.log_prob(p)

                sampler = AdaptiveImportanceSampler(
                    likelihood_fn, prior_log_prob_fn, surrogate_log_prob_fn
                )
                results = sampler.adaptive_is_loo(
                    data, params, variational=True,
                    khat_threshold=khat_threshold,
                )
                best = results['best']
                loos = np.array(best['ll_loo_psis'])
                ks = np.array(best['khat'])
                loo = float(np.sum(loos))
            except Exception as e:
                print(f"Warning: AIS fallback failed ({e}), using standard PSIS-LOO")

        n = len(loos)
        se = float(np.sqrt(n * np.var(loos)))

        self.loo_results = {
            'elpd_loo': float(loo),
            'elpd_loo_per_obs': float(loo) / n,
            'elpd_loo_per_obs_se': se / n,
            'pointwise_loo': loos,
            'khat': ks,
            'khat_max': float(np.max(ks)),
            'khat_mean': float(np.mean(ks)),
        }

    def set_data(self, data, data_transform_fn):
        self.data = data
        self.data_transform_fn = data_transform_fn

    def preprocessor(self):
        return None

    def _calibrate_advi(self):
        pass

    def _calibrate_minibatch_advi(
        self,
        batched_data_factory: Callable,
        batch_size: int,
        dataset_size: int,
        steps_per_epoch: int = 1,
        num_epochs: int = 1,
        accumulation_steps: int = 1,
        check_convergence_every: int = 1,
        sample_size=8,
        sample_batches=1,
        lr_decay_factor: float = 0.5,
        learning_rate=1.0,
        patience: int = 3,
        initial_values: Dict[str, jax.typing.ArrayLike] | None = None,
        unormalized_log_prob_fn: Callable | None = None,
        verbose: bool = True,
        zero_nan_grads: bool = False,
        snapshot_epoch: int | None = None,
        pathfinder_init: bool = False,
        pathfinder_kwargs: dict | None = None,
        point_estimate_vars: dict | None = None,
        **kwargs,
    ):
        """Calibrate using ADVI

        Args:
            data_factory (callable, required): [description]. Factory
                for generating a batch iterator.
            num_steps (int, optional): [description]. Defaults to 100.
            learning_rate (float, optional): [description]. Defaults to 0.1.
            opt ([type], optional): [description]. Defaults to None.
            abs_tol ([type], optional): [description]. Defaults to 1e-10.
            rel_tol ([type], optional): [description]. Defaults to 1e-8.
            clip_value ([type], optional): [description]. Defaults to 5..
            lr_decay_factor (float, optional): [description]. Defaults to 0.99.
            check_every (int, optional): [description]. Defaults to 25.
            set_expectations (bool, optional): [description]. Defaults to True.
            sample_size (int, optional): [description]. Defaults to 4.
        """

        if pathfinder_init and initial_values is None:
            first_batch = next(batched_data_factory)
            initial_values = pathfinder_initialize(
                log_prob_fn=unormalized_log_prob_fn or self.unormalized_log_prob,
                surrogate_initializer=self.surrogate_parameter_initializer,
                data=first_batch,
                dataset_size=dataset_size,
                batch_size=batch_size,
                **(pathfinder_kwargs or {}),
            )

        def run_approximation():
            losses = minibatch_fit_surrogate_posterior(
                target_log_prob_fn=(
                    unormalized_log_prob_fn
                    if unormalized_log_prob_fn is not None
                    else self.unormalized_log_prob
                ),
                initial_values=initial_values,
                surrogate_generator=self.surrogate_distribution_generator,
                surrogate_initializer=self.surrogate_parameter_initializer,
                dataset_size=dataset_size,
                batch_size=batch_size,
                sample_size=sample_size,
                sample_batches=sample_batches,
                learning_rate=learning_rate,
                check_convergence_every=check_convergence_every,
                patience=patience,
                lr_decay_factor=lr_decay_factor,
                steps_per_epoch=steps_per_epoch,
                num_epochs=num_epochs,
                accumulation_steps=accumulation_steps,
                data_iterator=batched_data_factory,
                verbose=verbose,
                zero_nan_grads=zero_nan_grads,
                snapshot_epoch=snapshot_epoch,
                point_estimate_vars=point_estimate_vars,
                **kwargs,
            )
            return losses

        result = run_approximation()
        losses, params = result[0], result[1]

        # Separate point-estimate params from surrogate params
        from bayesianquilts.vi.minibatch import PE_PREFIX
        surrogate_params = {}
        point_params = {}
        for k, v in params.items():
            if k.startswith(PE_PREFIX):
                point_params[k[len(PE_PREFIX):]] = v
            else:
                surrogate_params[k] = v

        self.params = surrogate_params
        if point_params:
            self.point_estimate_vars = point_params

        if len(result) > 2:
            return losses, surrogate_params, result[2]
        return losses, surrogate_params

    def set_calibration_expectations(self, samples: int = 24, variational: bool = True):
        if variational:
            # Use current params to regenerate surrogate if possible, as self.surrogate_distribution relies on init
            try:
                if hasattr(self, 'surrogate_distribution_generator') and hasattr(self, 'params') and self.params is not None:
                    dist = self.surrogate_distribution_generator(self.params)
                else:
                    dist = self.surrogate_distribution
            except Exception:
                dist = self.surrogate_distribution

            mean, var = FactorizedDistributionMoments(
                dist, samples=samples
            )
            self.calibrated_expectations = {k: v for k, v in mean.items()}
            self.calibrated_sd = {k: jnp.sqrt(v) for k, v in var.items()}
        else:
            self.calibrated_expectations = {
                k: jnp.mean(v, axis=0, keepdims=True)
                for k, v in self.surrogate_sample.items()
            }

            self.calibrated_sd = {
                k: jnp.std(v, axis=0, keepdims=True)
                for k, v in self.surrogate_sample.items()
            }

        # Point-estimate vars have zero variance by definition
        if self.point_estimate_vars:
            for k, v in self.point_estimate_vars.items():
                self.calibrated_expectations[k] = v
                self.calibrated_sd[k] = jnp.zeros_like(v)

    @abstractmethod
    def predictive_distribution(self, data, **params):
        pass

    def log_likelihood(self, data, **params):
        return self.predictive_distribution(data, return_params=False, **params)[
            "log_likelihood"
        ]

    def sample_stats(
        self, data_factory, params=None, num_samples=100, num_splits=20, data_batches=25
    ):
        likelihood_vars = inspect.getfullargspec(self.log_likelihood).args[1:]

        # split param samples
        params = self.surrogate_sample if params is None else params
        if "data" in likelihood_vars:
            likelihood_vars.remove("data")
        params = (
            self.surrogate_distribution.sample(num_samples)
            if (params is None)
            else params
        )
        if len(likelihood_vars) == 0:
            likelihood_vars = params.keys()
        if "data" in likelihood_vars:
            likelihood_vars.remove("data")
        if len(likelihood_vars) == 0:
            likelihood_vars = self.var_list
        splits = [
            tf.split(value=params[v], num_or_size_splits=num_splits)
            for v in likelihood_vars
        ]

        # reshape the splits
        splits = [
            {k: v for k, v in zip(likelihood_vars, split)} for split in zip(*splits)
        ]
        ll = []
        for batch in tqdm(data_factory()):
            # This should have shape S x N, where S is the number of param
            # samples and N is the batch size
            batch_log_likelihoods = [
                self.log_likelihood(**this_split, data=batch) for this_split in splits
            ]
            batch_log_likelihoods = jnp.concatenate(batch_log_likelihoods, axis=0)
            finite_part = jnp.where(
                jnp.isfinite(batch_log_likelihoods),
                batch_log_likelihoods,
                jnp.zeros_like(batch_log_likelihoods),
            )
            min_val = jnp.min(finite_part)
            #  batch_ll = tf.clip_by_value(batch_ll, min_val-1000, 0.)
            batch_log_likelihoods = jnp.where(
                jnp.finite.is_finite(batch_log_likelihoods),
                batch_log_likelihoods,
                jnp.ones_like(batch_log_likelihoods) * min_val * 1.01,
            )
            ll += [batch_log_likelihoods.numpy()]
        ll = np.concatenate(ll, axis=1)
        ll = np.moveaxis(ll, 0, -1)
        return {
            "log_likelihood": ll.T[np.newaxis, ...],
            "params": {k: v.numpy()[np.newaxis, ...] for k, v in params.items()},
        }

    def save(self, filename="model_save.pkl", gz=True):
        if not isinstance(filename, pathlib.Path):
            filename = pathlib.Path(filename)
        filename = filename.with_suffix(".pkl")
        if not gz:
            with open(filename, "wb") as file:
                #  dill.dump((self.__class__, self), file)
                dill.dump(self, file)
                #  dill.dump((self.__class__, self), file)
                dill.dump(self, file)

    @staticmethod
    def _clean_for_yaml(obj):
        """Convert an object to a YAML-serializable form, or None if not possible."""
        if obj is None:
            return None
        if isinstance(obj, bool):
            return bool(obj)
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if hasattr(obj, 'dtype') and isinstance(obj, type):
            return obj.__name__
        if isinstance(obj, type):
            return obj.__name__
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, (list, tuple)):
            return [BayesianModel._clean_for_yaml(x) for x in obj]
        if isinstance(obj, dict):
            cleaned = {
                str(k): BayesianModel._clean_for_yaml(v) for k, v in obj.items()
            }
            cleaned = {k: v for k, v in cleaned.items() if v is not None}
            return cleaned if cleaned else None
        if isinstance(obj, (int, float, str)):
            return obj
        if callable(obj):
            return None
        return None

    @staticmethod
    def _collect_init_arg_names(cls):
        """Return a set of all __init__ parameter names across the MRO."""
        names = set()
        for mro_cls in cls.__mro__:
            if mro_cls is object:
                continue
            init = getattr(mro_cls, '__init__', None)
            if init is None:
                continue
            try:
                argspec = inspect.getfullargspec(init)
            except TypeError:
                continue
            names.update(argspec.args[1:])
            names.update(argspec.kwonlyargs or [])
        return names

    def _discover_attribute_names(self):
        """Return all public attribute names on this instance."""
        names = set()
        # From init args across MRO
        names.update(self._collect_init_arg_names(type(self)))
        # From instance __dict__
        for k in list(getattr(self, '__dict__', {})):
            if not k.startswith('_'):
                names.add(k)
        # From class annotations (nnx.data fields, etc.)
        for cls in type(self).__mro__:
            if cls is object:
                continue
            for k in getattr(cls, '__annotations__', {}):
                if not k.startswith('_'):
                    names.add(k)
        return names

    @staticmethod
    def _is_array(val):
        return isinstance(val, (jnp.ndarray, np.ndarray))

    @staticmethod
    def _is_dict_of_arrays(val):
        return (
            isinstance(val, dict)
            and val
            and all(
                isinstance(v, (jnp.ndarray, np.ndarray)) for v in val.values()
            )
        )

    def _collect_array_and_scalar_attrs(self):
        """Partition instance attributes into array-like and scalar-like."""
        init_arg_names = self._collect_init_arg_names(type(self))
        attr_names = self._discover_attribute_names()

        array_attrs = {}   # name -> ndarray | dict[str, ndarray]
        scalar_attrs = {}  # name -> yaml-serializable value

        for name in sorted(attr_names):
            try:
                val = getattr(self, name)
            except Exception:
                continue
            if val is None:
                continue
            if callable(val) and not isinstance(val, type):
                continue

            if self._is_array(val):
                array_attrs[name] = val
                # Also keep in YAML if it is an init arg (needed for reconstruction)
                if name in init_arg_names:
                    scalar_attrs[name] = val
            elif self._is_dict_of_arrays(val):
                array_attrs[name] = val
            else:
                scalar_attrs[name] = val

        return array_attrs, scalar_attrs, init_arg_names

    def _save_config_yaml(self, path, scalar_attrs, init_arg_names, backend):
        """Write the YAML config file."""
        config = self._clean_for_yaml(scalar_attrs)
        if config is None:
            config = {}
        config = {k: v for k, v in config.items() if v is not None}
        config['_class_name'] = self.__class__.__name__
        config['_init_arg_names'] = sorted(init_arg_names & set(config.keys()))
        config['_backend'] = backend

        with open(path / "config.yaml", "w") as f:
            yaml.dump(config, f)

    @staticmethod
    def _save_arrays_hdf5(path, array_attrs):
        """Write array attributes into an HDF5 file."""
        with h5py.File(path / "params.h5", "w") as f:
            for name, val in array_attrs.items():
                if isinstance(val, dict):
                    grp = f.create_group(name)
                    for k, v in val.items():
                        arr = np.array(v)
                        if arr.size > 0:
                            grp.create_dataset(k, data=arr)
                else:
                    arr = np.array(val)
                    if arr.size > 0:
                        f.create_dataset(name, data=arr)

    @staticmethod
    def _save_arrays_safetensors(path, array_attrs):
        """Write array attributes into a safetensors file.

        Dict-of-array attributes are flattened with a ``::`` separator
        (e.g. ``params::loc``).  Standalone arrays use plain keys.
        """
        from safetensors.numpy import save_file

        flat = {}
        groups = []
        for name, val in array_attrs.items():
            if isinstance(val, dict):
                groups.append(name)
                for k, v in val.items():
                    arr = np.array(v)
                    if arr.size > 0:
                        flat[f"{name}::{k}"] = arr
            else:
                arr = np.array(val)
                if arr.size > 0:
                    flat[name] = arr

        metadata = {"_groups": ",".join(groups)} if groups else None
        save_file(flat, str(path / "tensors.safetensors"), metadata=metadata)

    def save_to_disk(self, path, backend="hdf5"):
        """Serialize model to disk.

        Saves all serializable attributes:
        - Arrays and dicts-of-arrays go into HDF5 or safetensors.
        - Scalars, strings, lists, and small dicts go into ``config.yaml``.

        The YAML also records ``_init_arg_names`` so that ``load_from_disk``
        knows which keys to pass to the constructor vs. set afterwards.

        Args:
            path: Directory to write to (created if it does not exist).
            backend: ``"hdf5"`` (default) or ``"safetensors"``.
        """
        if backend not in ("hdf5", "safetensors"):
            raise ValueError(
                f"backend must be 'hdf5' or 'safetensors', got {backend!r}"
            )

        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        array_attrs, scalar_attrs, init_arg_names = (
            self._collect_array_and_scalar_attrs()
        )

        self._save_config_yaml(path, scalar_attrs, init_arg_names, backend)

        if backend == "safetensors":
            self._save_arrays_safetensors(path, array_attrs)
        else:
            self._save_arrays_hdf5(path, array_attrs)

    _DTYPE_MAP = {
        'float16': jnp.float16,
        'float32': jnp.float32,
        'float64': jnp.float64,
        'int32': jnp.int32,
        'int64': jnp.int64,
    }

    @staticmethod
    def _load_arrays_hdf5(path):
        """Read array attributes from an HDF5 file.

        Returns a dict mapping attribute names to either a jnp array or a
        dict of jnp arrays (for groups).
        """
        result = {}
        if not (path / "params.h5").exists():
            return result
        with h5py.File(path / "params.h5", "r") as f:
            for name in f:
                if isinstance(f[name], h5py.Group):
                    result[name] = {
                        k: jnp.array(v) for k, v in f[name].items()
                    }
                else:
                    result[name] = jnp.array(f[name])
        return result

    @staticmethod
    def _load_arrays_safetensors(path):
        """Read array attributes from a safetensors file.

        Keys containing ``::`` are reassembled into dicts.
        """
        from safetensors.numpy import load_file
        from safetensors import safe_open

        st_path = path / "tensors.safetensors"
        if not st_path.exists():
            return {}

        # Read metadata to learn which prefixes are groups
        with safe_open(str(st_path), framework="numpy") as f:
            metadata = f.metadata() or {}
        group_names = set(
            g for g in metadata.get("_groups", "").split(",") if g
        )

        flat = load_file(str(st_path))

        result = {}
        for key, arr in flat.items():
            if "::" in key:
                group, subkey = key.split("::", 1)
                result.setdefault(group, {})[subkey] = jnp.array(arr)
            else:
                result[key] = jnp.array(arr)

        # Ensure all declared groups exist even if empty
        for g in group_names:
            result.setdefault(g, {})

        return result

    @classmethod
    def load_from_disk(cls, path):
        """Load a model previously written by ``save_to_disk``.

        Reconstructs the object from saved init args, then overlays every
        other saved attribute (arrays from the tensor file, scalars from YAML).

        The backend (HDF5 or safetensors) is auto-detected from the YAML
        config's ``_backend`` key, falling back to HDF5 for older saves.
        """
        path = pathlib.Path(path)
        with open(path / "config.yaml", "r") as f:
            config = yaml.safe_load(f)

        config.pop('_class_name', None)
        init_arg_names = set(config.pop('_init_arg_names', []))
        backend = config.pop('_backend', 'hdf5')

        # Convert dtype string back to actual dtype
        if 'dtype' in config and isinstance(config['dtype'], str):
            config['dtype'] = cls._DTYPE_MAP.get(config['dtype'], jnp.float64)

        # Determine which config keys are valid constructor arguments
        valid_init_args = cls._collect_init_arg_names(cls)

        # Use the stored _init_arg_names when available; fall back to MRO scan
        if init_arg_names:
            init_keys = init_arg_names & valid_init_args
        else:
            init_keys = valid_init_args

        init_kwargs = {k: v for k, v in config.items() if k in init_keys}
        extra_yaml = {k: v for k, v in config.items() if k not in init_keys}

        instance = cls(**init_kwargs)

        # Restore non-init-arg scalars from YAML
        for name, val in extra_yaml.items():
            if hasattr(instance, name):
                try:
                    setattr(instance, name, val)
                except Exception:
                    pass

        # Restore array attributes from the appropriate backend
        if backend == "safetensors":
            array_data = cls._load_arrays_safetensors(path)
        else:
            array_data = cls._load_arrays_hdf5(path)

        for name, val in array_data.items():
            setattr(instance, name, val)

        return instance


    def unormalized_log_prob_list(self, data, params, prior_weight=1.0):
        dict_params = {k: p for k, p in zip(self.var_list, params)}
        return self.unormalized_log_prob(
            data=data, prior_weight=jnp.astype(prior_weight, self.dtype), **dict_params
        )

    @abstractmethod
    def unormalized_log_prob(
        self,
        data: dict[str, jax.typing.ArrayLike] = None,
        prior_weight: jax.typing.ArrayLike | float = jnp.array(1.0),
        **params,
    ):
        """Generic method for the unormalized log probability function"""
        return

    def fit_projection(
        self, other, data_iterator, nsamples: int = 32, initial_values=None, **kwargs
    ):
        def objective(data, params):
            """Objective function for the training loop."""
            seed = random.PRNGKey(0)
            samples = self.surrogate_distribution_generator(params).sample(
                nsamples, seed=seed
            )
            this_prediction = self.predictive_distribution(data, **samples)[
                "prediction"
            ]
            other_prediction = other.predictive_distribution(data, **other.sample(nsamples))[
                "prediction"
            ]
            delta = other_prediction.kl_divergence(this_prediction)
            return jnp.mean(delta)

        if initial_values is None:
            initial_values = self.params

        return training_loop(
            loss_fn=objective,
            initial_values=initial_values,
            data_iterator=data_iterator,
            **kwargs,
        )

    @abstractmethod
    def create_distributions():
        """Create the prior and surrogate distributions for the model."""
        raise NotImplementedError(
            "This method should be implemented in subclasses to create the "
            "prior and surrogate distributions."
        )


    def transform(self, params):
        return params

    def sample(self, batch_shape=None, prior=False):
        _, sample_key = random.split(random.PRNGKey(np.random.randint(1e8)))
        surrogate = self.surrogate_distribution_generator(self.params)
        if prior:
            if batch_shape is None:
                params = self.prior_distribution.sample(seed=sample_key)
            else:
                params = self.prior_distribution.sample(batch_shape, seed=sample_key)
        elif batch_shape is None:
            params = surrogate.sample(seed=sample_key)
        else:
            params = surrogate.sample(batch_shape, seed=sample_key)
        params = self.transform(params)
        # Merge point-estimate values (no sampling, just broadcast)
        if self.point_estimate_vars and not prior:
            for k, v in self.point_estimate_vars.items():
                if batch_shape is not None:
                    params[k] = jnp.broadcast_to(v, batch_shape + v.shape)
                else:
                    params[k] = v
        return params

    # =========================================================================
    # MCMC Inference Methods
    # =========================================================================

    mcmc_samples: Any = nnx.data(None)  # Storage for MCMC samples

    def fit_mcmc(
        self,
        data: Dict[str, jax.typing.ArrayLike],
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
        step_size: float = 0.1,
        init_strategy: str = "prior",
        initial_states: Dict[str, jnp.ndarray] = None,  # NEW: accept custom initial states
        seed: int = None,
        verbose: bool = True,
        **kwargs,
    ) -> Dict[str, jnp.ndarray]:
        """Perform MCMC inference using NUTS (No-U-Turn Sampler).

        Uses Stan-like defaults for robust sampling.

        Args:
            data: Dictionary of data arrays (e.g., {'X': ..., 'y': ...})
            num_chains: Number of independent chains (Stan default: 4)
            num_warmup: Number of warmup/burn-in steps per chain (Stan default: 1000)
            num_samples: Number of post-warmup samples per chain (Stan default: 1000)
            target_accept_prob: Target acceptance probability for dual averaging
                (Stan's adapt_delta, default: 0.8, increase to 0.85-0.99 for difficult posteriors)
            max_tree_depth: Maximum tree depth for NUTS (Stan default: 10)
            step_size: Initial step size for HMC/NUTS (will be adapted during warmup)
            init_strategy: How to initialize chains - "prior" or "zero" (ignored if initial_states provided)
            initial_states: Optional dict with custom initial states of shape (num_chains, ...)
                If provided, overrides init_strategy. Useful for MAP-based initialization.
            seed: Random seed (uses random seed if None)
            verbose: Whether to print progress information

        Returns:
            Dictionary mapping parameter names to arrays of shape (num_chains, num_samples, ...)
        """
        if seed is None:
            seed = np.random.randint(0, 2**31)
        key = random.PRNGKey(seed)

        if verbose:
            print(f"Running NUTS with {num_chains} chains...")
            print(f"  Warmup: {num_warmup}, Samples: {num_samples}")
            print(f"  Target acceptance: {target_accept_prob}, Max tree depth: {max_tree_depth}")

        # Create target log probability function
        def target_log_prob_fn(*params_flat):
            params_dict = {k: v for k, v in zip(self.var_list, params_flat)}
            return self.unormalized_log_prob(data=data, prior_weight=1.0, **params_dict)

        # Initialize chains
        key, init_key = random.split(key)
        if initial_states is None:
            initial_states = self._create_initial_states(
                num_chains, init_strategy, init_key
            )
        else:
            if verbose:
                print("  Using custom initial states")

        # Run MCMC for each chain
        all_samples = {var: [] for var in self.var_list}
        all_accept_ratios = []

        for chain_idx in range(num_chains):
            key, chain_key = random.split(key)
            
            if verbose:
                print(f"\nChain {chain_idx + 1}/{num_chains}:")

            chain_initial = [initial_states[var][chain_idx] for var in self.var_list]
            
            samples, accept_ratio = self._run_nuts_chain(
                target_log_prob_fn=target_log_prob_fn,
                initial_state=chain_initial,
                num_warmup=num_warmup,
                num_samples=num_samples,
                step_size=step_size,
                target_accept_prob=target_accept_prob,
                max_tree_depth=max_tree_depth,
                seed=chain_key,
                verbose=verbose,
            )

            # Store results if chain is healthy
            if accept_ratio > 1e-6:
                for var, sample in zip(self.var_list, samples):
                    all_samples[var].append(sample)
                all_accept_ratios.append(accept_ratio)
            else:
                if verbose:
                    print(f"  Chain {chain_idx + 1} discarded (acceptance ratio {accept_ratio:.3e} too low)")

        # Stack chains: (num_valid_chains, num_samples, ...)
        result = {}
        num_valid_chains = len(all_accept_ratios)
        
        if num_valid_chains == 0:
            print("WARNING: All chains failed to converge (zero acceptance). Returning last chain to avoid crash.")
            # Fallback: keep the last chain even if bad so we have structure
            for var, sample in zip(self.var_list, samples):
                all_samples[var].append(sample)
            all_accept_ratios.append(accept_ratio)

        for var in self.var_list:
            result[var] = jnp.stack(all_samples[var], axis=0)

        if verbose:
            mean_accept = np.mean(all_accept_ratios)
            print(f"\n--- MCMC Complete ---")
            print(f"Mean acceptance ratio: {mean_accept:.3f}")
            for var in self.var_list:
                # Calculate R-hat
                # potential_scale_reduction expects shape [num_chains, num_samples, ...]
                # but TFP sometimes expects [num_samples, num_chains, ...] depending on version
                # Checking docs: TFP uses [num_samples, num_chains, ...] by default but independent_chain_ndims=1
                # lets us pass [num_chains, num_samples, ...]. 
                # Actually, simpler to just transpose to [num_samples, num_chains, ...]
                samples_transposed = jnp.swapaxes(result[var], 0, 1)
                rhat = tfmcmc.potential_scale_reduction(samples_transposed)
                
                # Check for NaNs in R-hat (can happen if variance is 0)
                rhat = jnp.where(jnp.isnan(rhat), 1.0, rhat)
                max_rhat = jnp.max(rhat)
                
                samples_flat = result[var].reshape(-1, *result[var].shape[2:])
                print(f"  {var}: mean={jnp.mean(samples_flat, axis=0)}, std={jnp.std(samples_flat, axis=0)}, max_rhat={max_rhat:.3f}")

        # Store samples for later use
        self.mcmc_samples = result
        return result

    def _create_initial_states(
        self,
        num_chains: int,
        init_strategy: str,
        key: jax.Array,
    ) -> Dict[str, jnp.ndarray]:
        """Create initial states for MCMC chains.

        Args:
            num_chains: Number of chains
            init_strategy: "prior" to sample from prior, "zero" for zeros
            key: JAX random key

        Returns:
            Dictionary mapping variable names to initial values of shape (num_chains, ...)
        """
        if init_strategy == "prior" and self.prior_distribution is not None:
            # Sample from prior
            samples = self.prior_distribution.sample(num_chains, seed=key)
            if isinstance(samples, dict):
                return samples
            else:
                return {self.var_list[0]: samples}
        elif init_strategy == "zero":
            # Initialize at zero (assumes unconstrained parameterization)
            result = {}
            # Get shapes from surrogate or prior
            if hasattr(self, 'surrogate_parameter_initializer'):
                template = self.surrogate_parameter_initializer(key=key)
                for var in self.var_list:
                    # Find the location parameter for this variable
                    loc_key = f"{var}_loc" if f"{var}_loc" in template else var
                    if loc_key in template:
                        shape = template[loc_key].shape
                        result[var] = jnp.zeros((num_chains,) + shape, dtype=self.dtype)
            return result
        else:
            # Default: small random values
            result = {}
            keys = random.split(key, len(self.var_list))
            if hasattr(self, 'surrogate_parameter_initializer'):
                template = self.surrogate_parameter_initializer(key=key)
                for i, var in enumerate(self.var_list):
                    loc_key = f"{var}_loc" if f"{var}_loc" in template else var
                    if loc_key in template:
                        shape = template[loc_key].shape
                        result[var] = random.normal(keys[i], (num_chains,) + shape, dtype=self.dtype) * 0.1
            return result

    def _run_nuts_chain(
        self,
        target_log_prob_fn: Callable,
        initial_state: list,
        num_warmup: int,
        num_samples: int,
        step_size: float,
        target_accept_prob: float,
        max_tree_depth: int,
        seed: jax.Array,
        verbose: bool = True,
    ) -> tuple:
        """Run a single NUTS chain.

        Args:
            target_log_prob_fn: Unnormalized log probability function
            initial_state: List of initial values for each parameter
            num_warmup: Number of warmup steps
            num_samples: Number of sampling steps
            step_size: Initial step size
            target_accept_prob: Target acceptance probability
            max_tree_depth: Maximum tree depth
            seed: Random key
            verbose: Print progress

        Returns:
            Tuple of (samples_list, mean_accept_ratio)
        """
        # Create NUTS kernel with step size adaptation
        nuts_kernel = tfmcmc.NoUTurnSampler(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            max_tree_depth=max_tree_depth,
        )

        # Wrap with dual averaging step size adaptation
        # Adapt for full warmup period for better convergence (was 80%)
        adaptive_kernel = tfmcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=nuts_kernel,
            num_adaptation_steps=num_warmup,
            target_accept_prob=target_accept_prob,
        )

        # Run the chain
        @jax.jit
        def run_chain(init_state, seed):
            samples, kernel_results = tfmcmc.sample_chain(
                num_results=num_samples,
                num_burnin_steps=num_warmup,
                current_state=init_state,
                kernel=adaptive_kernel,
                seed=seed,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
            )
            return samples, kernel_results

        if verbose:
            print("  Running warmup and sampling...")

        samples, is_accepted = run_chain(initial_state, seed)

        # Compute acceptance ratio
        accept_ratio = jnp.mean(is_accepted.astype(jnp.float32))

        if verbose:
            print(f"  Acceptance ratio: {accept_ratio:.3f}")

        return samples, float(accept_ratio)

    def sample_mcmc(self, num_samples: int = None) -> Dict[str, jnp.ndarray]:
        """Sample from stored MCMC results.

        Args:
            num_samples: Number of samples to return. If None, returns all.
                Samples are drawn randomly from all chains.

        Returns:
            Dictionary of parameter samples with shape (num_samples, ...)
        """
        if self.mcmc_samples is None:
            raise ValueError("No MCMC samples available. Run fit_mcmc() first.")

        result = {}
        for var, samples in self.mcmc_samples.items():
            # Flatten chains: (num_chains, num_samples, ...) -> (total, ...)
            flat = samples.reshape(-1, *samples.shape[2:])
            if num_samples is not None and num_samples < flat.shape[0]:
                # Random subsample
                key = random.PRNGKey(np.random.randint(0, 2**31))
                indices = random.choice(key, flat.shape[0], (num_samples,), replace=False)
                result[var] = flat[indices]
            else:
                result[var] = flat
        return result


    def to_arviz(self, data_factory=None):
        sample_stats = self.sample_stats(data_factory=data_factory)
        params = sample_stats["params"]

        idict = {
            "posterior": dict_to_dataset(params),
            "sample_stats": dict_to_dataset(
                {"log_likelihood": sample_stats["log_likelihood"]}
            ),
        }

        if convert_to_datatree is not None:
            # arviz >= 1.0
            return convert_to_datatree(idict)
        return InferenceData(**idict)

class QuiltedBayesianModel(BayesianModel):
    """Quilted Bayesian Model

    Initially a global model, Quilted Bayesian Models can be expanded along a given
    interaction alignment to create a larger model.

    Provides staged_fit() for progressive training by interaction order with
    sparsity-based pruning of unnecessary higher-order components.
    """

    def _get_decompositions(self) -> dict:
        """Return all Decomposed objects in this model.

        Subclasses should override if they use non-standard attribute names.
        Default implementation collects any attribute that is a Decomposed instance.
        """
        from bayesianquilts.jax.parameter import Decomposed
        decomps = {}
        for attr_name in dir(self):
            if attr_name.startswith('_'):
                continue
            try:
                val = getattr(self, attr_name)
            except Exception:
                continue
            if isinstance(val, Decomposed):
                decomps[attr_name] = val
        return decomps

    def staged_fit(
        self,
        batched_data_factory,
        max_order: int | None = None,
        sparsity_threshold: float = 0.1,
        sparsity_method: str = "relative_norm",
        epochs_per_stage: int | None = None,
        final_epochs: int | None = None,
        freeze_between_stages: bool = True,
        coarse_dtype: jnp.dtype | None = jnp.float32,
        verbose: bool = True,
        **kwargs,
    ):
        """Train progressively by interaction order with sparsity-based pruning.

        For each order 0..max_order:
        1. Train new components at this order as full Bayes (ADVI)
        2. Previously-trained non-sparse components are frozen as point estimates
        3. Assess sparsity, exclude descendants of sparse components

        This leverages the hybrid point-estimate + full Bayes system:
        frozen components pass through the likelihood as point estimates
        (no reparameterization, no KL cost), while new components get
        full surrogate posterior treatment.

        Args:
            batched_data_factory: Data factory (same as fit())
            max_order: Maximum interaction order to train. None = auto-detect.
            sparsity_threshold: Threshold for sparse_components()
            sparsity_method: Method for sparse_components()
            epochs_per_stage: Epochs per warmup stage
            final_epochs: Epochs for final joint training
            freeze_between_stages: If True, freeze trained components as point
                estimates between stages. If False, keep all as full Bayes
                (just use optimize_keys to control which get gradients).
            coarse_dtype: Dtype for early (coarse) stages. None = use model dtype.
            verbose: Whether to print stage progress
            **kwargs: Forwarded to _calibrate_minibatch_advi (batch_size,
                dataset_size, learning_rate, etc.)
        """
        from bayesianquilts.jax.parameter import Decomposed

        decompositions = self._get_decompositions()

        if max_order is None:
            max_order = max(d.max_order() for d in decompositions.values())

        num_epochs = kwargs.pop("num_epochs", 100)
        if epochs_per_stage is None:
            epochs_per_stage = max(num_epochs // (max_order + 2), 1)
        if final_epochs is None:
            final_epochs = num_epochs

        # Collect ALL component names across all decompositions
        all_component_names = set()
        for d in decompositions.values():
            all_component_names.update(d._tensor_parts.keys())

        # Auxiliary keys (horseshoe, noise) - always optimized
        all_keys = list(self.params.keys())
        aux_keys = Decomposed.non_component_keys(all_component_names, all_keys)

        # Track state
        excluded = set()       # Components pruned by sparsity
        frozen_point = {}      # Component name -> point-estimate value (posterior mean)
        active_bayes = set()   # Components currently in the surrogate

        def infinite_data_iterator():
            while True:
                iterator = batched_data_factory()
                try:
                    yield from iterator
                except TypeError:
                    yield iterator

        for order in range(max_order + 1):
            # Determine which components at this order to activate
            new_components = set()
            for d in decompositions.values():
                for name in d.components_at_order(order):
                    if name not in excluded:
                        new_components.add(name)
                        active_bayes.add(name)

            # Build optimize_keys: active Bayes components + aux keys
            opt_keys = (
                Decomposed.surrogate_keys_for_components(active_bayes, all_keys)
                + aux_keys
            )

            # Build point_estimate_vars from frozen components
            pe_vars = dict(frozen_point) if freeze_between_stages and frozen_point else None

            # Use coarse dtype for early stages, full precision for final
            stage_kwargs = dict(kwargs)
            if coarse_dtype is not None and order < max_order:
                stage_kwargs["compute_dtype"] = coarse_dtype

            if verbose:
                n_bayes = len(active_bayes)
                n_frozen = len(frozen_point)
                n_excluded = len(excluded)
                print(f"Stage order={order}: {n_bayes} Bayes, "
                      f"{n_frozen} frozen, {n_excluded} excluded")

            self._calibrate_minibatch_advi(
                infinite_data_iterator(),
                num_epochs=epochs_per_stage,
                optimize_keys=opt_keys,
                point_estimate_vars=pe_vars,
                **stage_kwargs,
            )

            # After training: assess sparsity and freeze non-sparse components
            if order < max_order:
                all_sparse = set()
                for d in decompositions.values():
                    sparse = d.sparse_components(
                        self.params, threshold=sparsity_threshold,
                        method=sparsity_method,
                    )
                    new_excl = d.hereditary_exclusions(sparse)
                    if verbose and new_excl:
                        sparse_at_order = [n for n in sparse
                                           if d.component_order(n) == order]
                        if sparse_at_order:
                            print(f"  Sparse at order {order}: {sparse_at_order}")
                            print(f"  Excluding descendants: {new_excl}")
                    excluded |= new_excl
                    all_sparse |= sparse

                # Freeze trained non-sparse components as point estimates
                if freeze_between_stages:
                    for name in list(active_bayes):
                        if name in excluded or name in all_sparse:
                            continue
                        loc_key = next(
                            (k for k in self.params
                             if k.startswith(name + "\\") and k.endswith("\\loc")),
                            None
                        )
                        if loc_key is not None:
                            frozen_point[name] = self.params[loc_key]
                            active_bayes.discard(name)

                # Zero out sparse components
                for name in all_sparse:
                    for k in all_keys:
                        if k.startswith(name + "\\") and k.endswith("\\loc"):
                            self.params[k] = jnp.zeros_like(self.params[k])

        # Final training: unfreeze everything for joint optimization
        all_active = active_bayes | set(frozen_point.keys())
        opt_keys = (
            Decomposed.surrogate_keys_for_components(all_active, all_keys)
            + aux_keys
        )
        if verbose:
            print(f"Final training: {len(all_active)} active components, "
                  f"{final_epochs} epochs (full precision)")

        res = self._calibrate_minibatch_advi(
            infinite_data_iterator(),
            num_epochs=final_epochs,
            optimize_keys=opt_keys,
            # No point estimates in final stage - full Bayes for all active
            point_estimate_vars=None,
            **kwargs,  # original kwargs, no coarse_dtype
        )

        self.active_components = all_active
        self.excluded_components = excluded
        return res

    @abstractmethod
    def expand(self, interaction: Interactions):
        pass
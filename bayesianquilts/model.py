import gzip
import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Dict

import dill
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from arviz.data import InferenceData
from arviz.data.base import dict_to_dataset
from flax import nnx
from jax import random
from tensorflow_probability.substrates.jax import tf2jax as tf
from tqdm import tqdm

from bayesianquilts.distributions import FactorizedDistributionMoments
from bayesianquilts.tf.parameter import MultiwayContingencyTable
from bayesianquilts.util import DummyObject, training_loop
from bayesianquilts.vi.minibatch import minibatch_fit_surrogate_posterior


class BayesianModel(ABC, nnx.Module):
    surrogate_distribution = None
    surrogate_sample = None
    prior_distribution = None
    data = None
    data_cardinality = None
    var_list = []
    bijectors = []

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

    def fit(
        self,
        batched_data_factory,
        initial_values: Dict[str, jax.typing.ArrayLike] = None,
        **kwargs,
    ):
        return self._calibrate_minibatch_advi(
            iter(batched_data_factory()), initial_values=initial_values, **kwargs
        )

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
        sample_size=8,
        sample_batches=1,
        lr_decay_factor: float = 0.5,
        learning_rate=1.0,
        patience: int = 3,
        initial_values: Dict[str, jax.typing.ArrayLike] | None = None,
        unormalized_log_prob_fn: Callable | None = None,
        set_expectations=True,
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
                patience=patience,
                lr_decay_factor=lr_decay_factor,
                steps_per_epoch=steps_per_epoch,
                num_epochs=num_epochs,
                accumulation_steps=accumulation_steps,
                data_iterator=batched_data_factory,
                **kwargs,
            )
            return losses

        losses, params = run_approximation()
        if set_expectations:
            if (not np.isnan(losses[-1])) and (not np.isinf(losses[-1])):
                self.surrogate_distribution = self.surrogate_distribution_generator(params)
                self.set_calibration_expectations()
        return losses, params

    def set_calibration_expectations(self, samples: int = 24, variational: bool = True):
        if variational:
            mean, var = FactorizedDistributionMoments(
                self.surrogate_distribution, samples=samples
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
        if not gz:
            with open(filename, "wb") as file:
                #  dill.dump((self.__class__, self), file)
                dill.dump(self, file)
        else:
            with gzip.open(filename + ".gz", "wb") as file:
                #  dill.dump((self.__class__, self), file)
                dill.dump(self, file)

    def __getstate__(self):
        state = self.__dict__.copy()
        keys = self.__dict__.keys()
        state["surrogate_sample"] = None
        for k in keys:
            # print(k)
            if isinstance(state[k], jax.typing.ArrayLike):
                state[k] = state[k]
            elif isinstance(state[k], dict) or isinstance(state[k], list):
                try:
                    flat = tf.nest.flatten(state[k])
                    new = []
                    for t in flat:
                        if isinstance(t, jax.typing.ArrayLike):
                            new += [t.numpy()]
                        elif hasattr(inspect.getmodule(t), "__name__"):
                            if inspect.getmodule(t).__name__.startswith("tensorflow"):
                                if not isinstance(t, jnp.dtype):
                                    new += [None]
                                else:
                                    new += [None]
                            else:
                                new += [t]
                        else:
                            new += [t]
                    state[k] = jax.tree_util.tree_unflatten(state[k], new)
                except TypeError:
                    state[k] = None
                    print(f"failed serializing {k}")
            elif hasattr(inspect.getmodule(state[k]), "__name__"):
                if inspect.getmodule(state[k]).__name__.startswith("tensorflow"):
                    if not isinstance(state[k], jnp.dtypes.DType):
                        del state[k]
        state["strategy"] = None
        return state

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
        self, other, batched_data_factory, num_steps, samples=32, **kwargs
    ):
        def objective(data):
            this_prediction = self.predictive_distribution(
                data, **self.sample(samples)
            )["rv_outcome"]
            other_prediction = other.predictive_distribution(
                data, **other.sample(samples)
            )["rv_outcome"]
            delta = other_prediction.kl_divergence(this_prediction)
            return jnp.mean(delta)

        return training_loop(
            objective,
            batched_data_factory=batched_data_factory,
            num_steps=num_steps,
            trainable_variables=self.surrogate_distribution.variables,
            **kwargs,
        )

    @abstractmethod
    def create_distributions():
        """Create the prior and surrogate distributions for the model."""
        raise NotImplementedError(
            "This method should be implemented in subclasses to create the "
            "prior and surrogate distributions."
        )

    def reconstitute(self, state):
        surrogate_params = {t.name: t for t in state["surrogate_vars"]}
        if "max_order" in state.keys():
            try:
                self.create_distributions(
                    max_order=state["max_order"], surrogate_params=surrogate_params
                )
                return
            except TypeError:
                self.create_distributions()
            try:
                for j, value in tqdm(enumerate(state["surrogate_vars"])):
                    self.surrogate_distribution.trainable_variables[j].assign(
                        jnp.astype(value, self.dtype)
                    )
            except KeyError:
                self.state = state
                print("Was unable to set vars, check self.saved_state")
        else:
            self.create_distributions()

    def transform(self, params):
        return params

    def sample(self, batch_shape=None, prior=False):
        _, sample_key = random.split(random.PRNGKey(0))
        surrogate = self.surrogate_distribution_generator(self.params)
        if prior:
            if batch_shape is None:
                params = self.prior_distribution.sample(seed=sample_key)
            else:
                params = self.prior_distribution.sample(batch_shape, seed=sample_key)
        elif batch_shape is None:
            return surrogate.sample(seed=sample_key)
        else:
            params = surrogate.sample(batch_shape, seed=sample_key)
        params = self.transform(params)
        return params

    def to_arviz(self, data_factory=None):
        sample_stats = self.sample_stats(data_factory=data_factory)
        params = sample_stats["params"]

        idict = {
            "posterior": dict_to_dataset(params),
            "sample_stats": dict_to_dataset(
                {"log_likelihood": sample_stats["log_likelihood"]}
            ),
        }

        return InferenceData(**idict)

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self.reconstitute(state)
        self.saved_state = state


def generate_distributions(
    decomposition,
    skip=None,
    table=None,
    dim_decay_factor=1.0,
    scale=1.0,
    gaussian_only=True,
    dtype=jnp.float32,
    surrogate_params=None,
    gen_surrogate=True,
):
    (
        tensors,
        vars,
        shapes,
    ) = decomposition.generate_tensors(skip=skip, dtype=dtype)

    if isinstance(table, MultiwayContingencyTable):
        N = table.lookup()
        # check if there are certain dimensions in the interactions
        # that are not covered by the contingency table
        scales = {}
        for k, v in shapes.items():
            scales[k] = (
                scale
                * dim_decay_factor ** (len([d for d in v if d > 1]))
                * np.reshape(
                    np.sqrt(table.lookup(vars[k]) / N),
                    -1,
                )
            )

    else:
        scales = {
            k: scale * dim_decay_factor ** (len([d for d in v if d > 1]))
            for k, v in shapes.items()
        }

    if decomposition._implicit:
        scales = {k: v[1:] if len(v) > 1 else v for k, v in scales.items()}
    decomposition.set_scales(scales)

    prior_dict = {}
    for label, tensor in tensors.items():
        prior_dict[label] = tfd.Independent(
            tfd.Normal(
                loc=jnp.zeros_like(jnp.astype(tensor, dtype)),
                scale=jnp.ones_like(jnp.astype(tensor, dtype)),
            ),
            reinterpreted_batch_ndims=len(tensor.shape.as_list()),
        )

    if len(prior_dict) > 0:
        prior = tfd.JointDistributionNamed(prior_dict)
    else:
        prior = DummyObject()
        prior.model = {}

    out = {
        "decomposition": decomposition,
        "prior": prior,
        "vars": vars,
        "shapes": shapes,
        "scales": scales,
    }
    if gen_surrogate:
        if len(prior_dict) > 0:
            out["surrogate"] = build_surrogate_posterior(
                prior,
                initializers=tensors,
                dtype=dtype,
                bijectors=defaultdict(tfp.bijectors.Identity),
                gaussian_only=gaussian_only,
                surrogate_params=surrogate_params,
            )
        else:
            out["surrogate"] = DummyObject()
            out["surrogate"].model = {}

    return out

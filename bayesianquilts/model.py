import gzip
import inspect
import pathlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import dill
import jax
import jax.numpy as jnp
import numpy as np
from arviz.data import InferenceData
from arviz.data.base import dict_to_dataset
from flax import nnx
from jax import random
from tensorflow_probability.substrates.jax import tf2jax as tf
from tqdm import tqdm

from bayesianquilts.jax.parameter import Interactions
from bayesianquilts.util import training_loop
from bayesianquilts.vi.minibatch import minibatch_fit_surrogate_posterior
import bayesianquilts.tfp_patch

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
    surrogate_distribution: Any = nnx.Variable(None)
    surrogate_sample: Any = nnx.Variable(None)
    prior_distribution: Any = nnx.Variable(None)
    data: Any = nnx.Variable(None)
    data_cardinality: Any = nnx.Variable(None)
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

    def fit(
        self,
        batched_data_factory,
        initial_values: Dict[str, jax.typing.ArrayLike] = None,
        **kwargs,
    ):
        res = self._calibrate_minibatch_advi(
            iter(batched_data_factory()), initial_values=initial_values, **kwargs
        )
        self.params = res[1]
        return res

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
                check_convergence_every=check_convergence_every,
                patience=patience,
                lr_decay_factor=lr_decay_factor,
                steps_per_epoch=steps_per_epoch,
                num_epochs=num_epochs,
                accumulation_steps=accumulation_steps,
                data_iterator=batched_data_factory,
                verbose=verbose,
                **kwargs,
            )
            return losses

        losses, params = run_approximation()
        self.params = params
        return losses, params

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
        else:
            filename = filename.with_suffix(".pkl.gz")
            with gzip.open(filename, "wb") as file:
                #  dill.dump((self.__class__, self), file)
                dill.dump(self, file)


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

class QuiltedBayesianModel(BayesianModel):
    """Quailted Bayesian Model

    Initially a global model, Quilted Bayesian Models can be expanded along a given
    interaction alignment to create a larger model.
    """
    @abstractmethod
    def expand(self, interaction: Interactions):
        pass
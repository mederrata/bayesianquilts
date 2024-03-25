import gzip
import inspect
import os
import tempfile
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from itertools import cycle

import arviz as az
import dill
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr
from arviz.data import InferenceData
from arviz.data.base import dict_to_dataset
from tensorflow.python.data.ops.dataset_ops import BatchDataset
from tensorflow.python.distribute.input_lib import DistributedDataset
from tensorflow_probability.python import distributions as tfd
from tqdm import tqdm

from bayesianquilts.distributions import FactorizedDistributionMoments
from bayesianquilts.tf.parameter import (
    Decomposed,
    Interactions,
    MultiwayContingencyTable,
)
from bayesianquilts.util import DummyObject, batched_minimize
from bayesianquilts.vi.advi import build_surrogate_posterior
from bayesianquilts.vi.minibatch import minibatch_fit_surrogate_posterior


class BayesianModel(ABC):
    surrogate_distribution = None
    surrogate_sample = None
    prior_distribution = None
    data = None
    data_cardinality = None
    var_list = []
    bijectors = []

    def __init__(
        self,
        data=None,
        data_transform_fn=None,
        strategy=None,
        dtype=tf.float64,
        *args,
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
        if data is not None:
            self.set_data(data, data_transform_fn)

        self.strategy = strategy
        self.dtype = dtype

    def fit(self, *args, **kwargs):
        return self._calibrate_minibatch_advi(*args, **kwargs)
    
    def set_data(self, data, data_transform_fn):
        self.data = data
        self.data_transform_fn = data_transform_fn
        
    def preprocessor(self):
        return None

    def _calibrate_advi(self):
        pass

    def _calibrate_minibatch_advi(
        self,
        batched_data_factory,
        batch_size,
        dataset_size,
        num_steps=100,
        learning_rate=0.1,
        opt=None,
        abs_tol=1e-10,
        rel_tol=1e-8,
        clip_value=5.0,
        clip_by="norm",
        max_decay_steps=25,
        lr_decay_factor=0.99,
        check_every=1,
        set_expectations=False,
        sample_size=24,
        sample_batches=1,
        trainable_variables=None,
        unormalized_log_prob_fn=None,
        accumulate_batches=False,
        batches_per_step=None,
        temp_dir=tempfile.gettempdir(),
        test_fn=None,
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
            max_decay_steps (int, optional): [description]. Defaults to 25.
            lr_decay_factor (float, optional): [description]. Defaults to 0.99.
            check_every (int, optional): [description]. Defaults to 25.
            set_expectations (bool, optional): [description]. Defaults to True.
            sample_size (int, optional): [description]. Defaults to 4.
        """
        if trainable_variables is None:
            trainable_variables = self.surrogate_distribution.variables

        def run_approximation(num_steps):
            losses = minibatch_fit_surrogate_posterior(
                target_log_prob_fn=unormalized_log_prob_fn
                if unormalized_log_prob_fn is not None
                else self.unormalized_log_prob,
                surrogate_posterior=self.surrogate_distribution,
                dataset_size=dataset_size,
                batch_size=batch_size,
                num_steps=num_steps,
                sample_size=sample_size,
                sample_batches=sample_batches,
                learning_rate=learning_rate,
                max_decay_steps=max_decay_steps,
                decay_rate=lr_decay_factor,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                clip_value=clip_value,
                clip_by=clip_by,
                check_every=check_every,
                strategy=self.strategy,
                accumulate_batches=accumulate_batches,
                batches_per_step=batches_per_step,
                trainable_variables=trainable_variables,
                batched_data_factory=batched_data_factory,
                test_fn=test_fn,
                **kwargs,
            )
            return losses

        losses = run_approximation(num_steps)
        if set_expectations:
            if (not np.isnan(losses[-1])) and (not np.isinf(losses[-1])):
                self.surrogate_sample = self.surrogate_distribution.sample(100)
                self.set_calibration_expectations()
        return losses

    def set_calibration_expectations(self, samples=24, variational=True):
        if variational:
            mean, var = FactorizedDistributionMoments(
                self.surrogate_distribution, samples=samples
            )
            self.calibrated_expectations = {k: tf.Variable(v) for k, v in mean.items()}
            self.calibrated_sd = {
                k: tf.Variable(tf.math.sqrt(v)) for k, v in var.items()
            }
        else:
            self.calibrated_expectations = {
                k: tf.Variable(tf.reduce_mean(v, axis=0, keepdims=True))
                for k, v in self.surrogate_sample.items()
            }

            self.calibrated_sd = {
                k: tf.Variable(tf.math.reduce_std(v, axis=0, keepdims=True))
                for k, v in self.surrogate_sample.items()
            }

    def calibrate_mcmc(
        self,
        batched_data_factory,
        dataset_size,
        batch_size,
        num_steps=1000,
        burnin=500,
        init_state=None,
        step_size=1e-1,
        nuts=True,
        data_batches=10,
        clip=None,
    ):
        """Calibrate using HMC/NUT
        Keyword Arguments:
            num_chains {int} -- [description] (default: {1})
        """

        if init_state is None:
            init_state = self.calibrated_expectations

        step_size = tf.cast(step_size, self.dtype)

        initial_list = [init_state[v] for v in self.var_list]
        bijectors = [self.bijectors[k] for k in self.var_list]

        dataset = batched_data_factory()

        @tf.function(autograph=True)
        def energy(*x):
            energies = [
                self.unormalized_log_prob_list(
                    batch, x, prior_weight=tf.constant(batch_size / dataset_size)
                )
                for batch in dataset
            ]

            return tf.add_n(energies)

        samples, sampler_stat = run_chain(
            init_state=initial_list,
            step_size=step_size,
            target_log_prob_fn=(
                energy if clip is None else clip_gradients(energy, clip)
            ),
            unconstraining_bijectors=bijectors,
            num_steps=num_steps,
            burnin=burnin,
        )
        self.surrogate_sample = {k: sample for k, sample in zip(self.var_list, samples)}
        self.set_calibration_expectations()

        return samples, sampler_stat

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
            batch_log_likelihoods = tf.concat(batch_log_likelihoods, axis=0)
            finite_part = tf.where(
                tf.math.is_finite(batch_log_likelihoods),
                batch_log_likelihoods,
                tf.zeros_like(batch_log_likelihoods),
            )
            min_val = tf.math.reduce_min(finite_part)
            #  batch_ll = tf.clip_by_value(batch_ll, min_val-1000, 0.)
            batch_log_likelihoods = tf.where(
                tf.math.is_finite(batch_log_likelihoods),
                batch_log_likelihoods,
                tf.ones_like(batch_log_likelihoods) * min_val * 1.01,
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
            if isinstance(state[k], tf.Tensor) or isinstance(state[k], tf.Variable):
                state[k] = state[k].numpy()
            elif isinstance(state[k], dict) or isinstance(state[k], list):
                try:
                    flat = tf.nest.flatten(state[k])
                    new = []
                    for t in flat:
                        if isinstance(t, tf.Tensor) or isinstance(t, tf.Variable):
                            new += [t.numpy()]
                        elif hasattr(inspect.getmodule(t), "__name__"):
                            if inspect.getmodule(t).__name__.startswith("tensorflow"):
                                if not isinstance(t, tf.dtypes.DType):
                                    new += [None]
                                else:
                                    new += [None]
                            else:
                                new += [t]
                        else:
                            new += [t]
                    state[k] = tf.nest.pack_sequence_as(state[k], new)
                except TypeError:
                    state[k] = None
                    print(f"failed serializing {k}")
            elif hasattr(inspect.getmodule(state[k]), "__name__"):
                if inspect.getmodule(state[k]).__name__.startswith("tensorflow"):
                    if not isinstance(state[k], tf.dtypes.DType):
                        del state[k]
        state["strategy"] = None
        return state

    def unormalized_log_prob_list(self, data, params, prior_weight=tf.constant(1)):
        dict_params = {k: p for k, p in zip(self.var_list, params)}
        return self.unormalized_log_prob(
            data=data, prior_weight=tf.cast(prior_weight, self.dtype), **dict_params
        )

    @abstractmethod
    def unormalized_log_prob(self, data, prior_weight=tf.constant(1), *args, **kwargs):
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
            return tf.reduce_mean(delta)

        return batched_minimize(
            objective,
            batched_data_factory=batched_data_factory,
            num_steps=num_steps,
            trainable_variables=self.surrogate_distribution.variables,
            **kwargs,
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
                        tf.cast(value, self.dtype)
                    )
            except KeyError:
                self.state = state
                print("Was unable to set vars, check self.saved_state")
        else:
            self.create_distributions()

    def sample(self, batch_shape=None, prior=False):
        if prior:
            if batch_shape is None:
                return self.prior_distributions.sample()
            return self.prior_distributions.sample(batch_shape)
        if batch_shape is None:
            return self.surrogate_distribution.sample()
        return self.surrogate_distribution.sample(batch_shape)

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
        #  self.dtype = tf.float64
        self.reconstitute(state)
        self.saved_state = state


def generate_distributions(
    decomposition,
    skip=None,
    table=None,
    dim_decay_factor=1.0,
    scale=1.0,
    gaussian_only=True,
    dtype=tf.float32,
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
                loc=tf.zeros_like(tf.cast(tensor, dtype)),
                scale=tf.ones_like(tf.cast(tensor, dtype)),
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


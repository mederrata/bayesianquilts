import inspect
from itertools import cycle
import tempfile
import os
import gzip

import dill
import arviz as az
from arviz.data.base import dict_to_dataset
from arviz.data import InferenceData
from abc import ABC, abstractmethod

import tensorflow as tf

import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
import xarray as xr

from tensorflow.python.data.ops.dataset_ops import BatchDataset
from tensorflow.python.distribute.input_lib import DistributedDataset
from tqdm import tqdm

from bayesianquilts.util import (
    clip_gradients,
    run_chain,
    tf_data_cardinality,
)
from bayesianquilts.vi.minibatch import minibatch_fit_surrogate_posterior

from bayesianquilts.distributions import FactorizedDistributionMoments


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

    def _calibrate_advi(self):
        pass

    def _calibrate_minibatch_advi(
        self,
        batched_data_factory,
        batch_size,
        dataset_size,
        num_epochs=100,
        learning_rate=0.1,
        opt=None,
        abs_tol=1e-10,
        rel_tol=1e-8,
        clip_value=5.0,
        max_decay_steps=25,
        lr_decay_factor=0.99,
        check_every=1,
        set_expectations=True,
        sample_size=4,
        sample_batches=1,
        trainable_variables=None,
        temp_dir=tempfile.gettempdir(),
        test_fn=None,
        **kwargs,
    ):
        """Calibrate using ADVI

        Args:
            data_factory (callable, required): [description]. Factory
                for generating a batch iterator.
            num_epochs (int, optional): [description]. Defaults to 100.
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

        def run_approximation(num_epochs):
            losses = minibatch_fit_surrogate_posterior(
                target_log_prob_fn=self.unormalized_log_prob,
                surrogate_posterior=self.surrogate_distribution,
                dataset_size=dataset_size,
                batch_size=batch_size,
                num_epochs=num_epochs,
                sample_size=sample_size,
                sample_batches=sample_batches,
                learning_rate=learning_rate,
                max_decay_steps=max_decay_steps,
                decay_rate=lr_decay_factor,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                clip_value=clip_value,
                check_every=check_every,
                strategy=self.strategy,
                trainable_variables=trainable_variables,
                batched_data_factory=batched_data_factory,
                test_fn=test_fn,
            )
            return losses

        losses = run_approximation(num_epochs)
        if set_expectations:
            if (not np.isnan(losses[-1])) and (not np.isinf(losses[-1])):
                self.surrogate_sample = self.surrogate_distribution.sample(100)
                self.set_calibration_expectations()
        return losses

    def set_calibration_expectations(self, samples=50, variational=True):
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

    def reconstitute(self, state):
        self.create_distributions()
        try:
            for j, value in enumerate(state["surrogate_vars"]):
                self.surrogate_distribution.trainable_variables[j].assign(
                    tf.cast(value, self.dtype)
                )
        except KeyError:
            self.state = state
            print("Was unable to set vars, check self.saved_state")

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
        self.set_calibration_expectations()

import functools
import inspect
import os
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow.python.data.ops.dataset_ops import BatchDataset
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops, math_ops, state_ops
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python.training import optimizer
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import softplus as softplus_lib
from tensorflow_probability.python.distributions.transformed_distribution import (
    TransformedDistribution,
)
from tensorflow_probability.python.internal import (
    dtype_util,
    prefer_static,
    tensorshape_util,
)
from tensorflow_probability.python.vi import csiszar_divergence
from tqdm import tqdm

from tensorflow_probability.python.vi import GradientEstimators
from bayesianquilts.util import (
    batched_minimize,
    TransformedVariable,
    minimize_distributed,
)
from bayesianquilts.util import _trace_variables, _trace_loss


@tf.function
def minibatch_mc_variational_loss(
    target_log_prob_fn,
    surrogate_posterior,
    dataset_size,
    batch_size,
    sample_size=1,
    sample_batches=1,
    importance_weight=False,
    seed=None,
    data=None,
    name=None,
    **kwargs,
):
    """The minibatch variational loss

    Args:
        target_log_prob_fn (_type_): log_likelihood + prior_weight*log_prior
        surrogate_posterior (_type_): _description_
        dataset_size (_type_): _description_
        batch_size (_type_): _description_
        sample_size (int, optional): _description_. Defaults to 1.
        discrepancy_fn (_type_, optional): _description_. Defaults to tfp.vi.kl_reverse.
        seed (_type_, optional): _description_. Defaults to None.
        data (_type_, optional): _description_. Defaults to None.
        name (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    elbo_samples = [tf.zeros(sample_size, tf.float64)] * sample_batches
    q_lp = [tf.zeros(sample_size, tf.float64)] * sample_batches
    penalized_like = [tf.zeros(sample_size, tf.float64)] * sample_batches
    for n in range(sample_batches):
        q_samples, q_lp_ = surrogate_posterior.experimental_sample_and_log_prob(
            [sample_size], seed=seed
        )

        penalized_like[n] += target_log_prob_fn(
            data=data, prior_weight=tf.constant(batch_size / dataset_size), **q_samples
        )
        q_lp[n] += q_lp_
        elbo_samples[n] += q_lp[n] * batch_size / dataset_size - penalized_like[n]

    q_lp__ = tf.concat(q_lp, axis=0)
    penalized_like__ = tf.concat(penalized_like, axis=0)
    elbo_samples__ = tf.concat(elbo_samples, axis=0)
    if not importance_weight:
        return tf.reduce_mean(elbo_samples__)
    max_val = tf.reduce_mean(penalized_like__ - q_lp__)
    weights = tf.exp(penalized_like__ - q_lp__ - max_val)

    weights = weights / tf.reduce_sum(weights)
    elbo = tf.reduce_sum(elbo_samples__ * weights)

    # @TODO compute importance weights
    return elbo


def minibatch_fit_surrogate_posterior(
    target_log_prob_fn,
    surrogate_posterior,
    batched_data_factory,
    batch_size,
    dataset_size,
    num_epochs=1000,
    trace_fn=_trace_loss,
    sample_size=8,
    sample_batches=1,
    check_every=1,
    decay_rate=0.9,
    learning_rate=1.0,
    clip_value=10.0,
    trainable_variables=None,
    jit_compile=None,
    abs_tol=None,
    rel_tol=None,
    strategy=None,
    name=None,
    test_fn=None,
    **kwargs,
):
    if trainable_variables is None:
        trainable_variables = surrogate_posterior.trainable_variables

    def complete_variational_loss_fn(data=None):
        """This becomes the loss function called in the
        optimization loop

        Keyword Arguments:
            data {tf.data.Datasets batch} -- [description] (default: {None})

        Returns:
            [type] -- [description]
        """
        return minibatch_mc_variational_loss(
            target_log_prob_fn,
            surrogate_posterior,
            sample_size=sample_size,
            sample_batches=sample_batches,
            data=data,
            dataset_size=dataset_size,
            batch_size=batch_size,
            strategy=strategy,
            name=name,
            **kwargs,
        )

    if strategy is None:
        return batched_minimize(
            complete_variational_loss_fn,
            batched_data_factory=batched_data_factory,
            num_epochs=num_epochs,
            trace_fn=trace_fn,
            learning_rate=learning_rate,
            trainable_variables=trainable_variables,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            clip_value=clip_value,
            decay_rate=decay_rate,
            check_every=check_every,
            test_fn=test_fn,
            **kwargs,
        )
    else:
        return minimize_distributed(
            complete_variational_loss_fn,
            data_factory=batched_data_factory,
            num_epochs=num_epochs,
            trace_fn=trace_fn,
            learning_rate=learning_rate,
            trainable_variables=trainable_variables,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            decay_rate=decay_rate,
            check_every=check_every,
            strategy=strategy,
            test_fn=test_fn,
            **kwargs,
        )

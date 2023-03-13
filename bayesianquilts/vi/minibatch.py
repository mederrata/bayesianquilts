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
from tensorflow_probability.python.vi import csiszar_divergence, kl_reverse
from tqdm import tqdm

from tensorflow_probability.python.vi import GradientEstimators
from tensorflow_probability.python import monte_carlo

from bayesianquilts.util import (
    batched_minimize,
    TransformedVariable,
    minimize_distributed,
)
from bayesianquilts.util import _trace_variables, _trace_loss


# @tf.function(autograph=False)
def minibatch_mc_variational_loss(
    target_log_prob_fn,
    surrogate_posterior,
    dataset_size,
    batch_size,
    data,
    sample_size=1,
    sample_batches=1,
    importance_sample_size=1,
    discrepancy_fn=kl_reverse,
    use_reparameterization=None,
    gradient_estimator=None,
    stopped_surrogate_posterior=None,
    seed=None,
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
    # @tf.function
    def sample_elbo():
        q_samples, q_lp = surrogate_posterior.experimental_sample_and_log_prob(
            sample_size, seed=seed
        )
        return q_samples, q_lp

    # @tf.function(autograph=False)
    batch_expectations = []

    reweighted = functools.partial(
        target_log_prob_fn,
        data=data,
        prior_weight=tf.cast(batch_size / dataset_size, tf.float64),
    )

    for _ in range(sample_batches):
        q_samples, q_lp = sample_elbo()

        batch_expectations += [
            monte_carlo.expectation(
                f=csiszar_divergence._make_importance_weighted_divergence_fn(
                    reweighted,
                    surrogate_posterior=surrogate_posterior,
                    discrepancy_fn=discrepancy_fn,
                    precomputed_surrogate_log_prob=q_lp,
                    importance_sample_size=importance_sample_size,
                    gradient_estimator=gradient_estimator,
                    stopped_surrogate_posterior=(stopped_surrogate_posterior),
                ),
                samples=q_samples,
                # Log-prob is only used if `gradient_estimator == SCORE_FUNCTION`.
                log_prob=surrogate_posterior.log_prob,
                use_reparameterization=(
                    gradient_estimator != GradientEstimators.SCORE_FUNCTION
                ),
            )
        ]
    batch_expectations = tf.reduce_mean(batch_expectations, axis=0)
    return batch_expectations


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
    clip_by="norm",
    trainable_variables=None,
    jit_compile=None,
    accumulate_batches=False,
    batches_per_epoch=None,
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
            clip_by=clip_by,
            accumulate_batches=accumulate_batches,
            batches_per_epoch=batches_per_epoch,
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
            accumulate_batches=accumulate_batches,
            check_every=check_every,
            strategy=strategy,
            test_fn=test_fn,
            **kwargs,
        )

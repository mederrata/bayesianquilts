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

from bayesianquilts.distributions import SqrtInverseGamma

tfd = tfp.distributions
tfb = tfp.bijectors


def flatten(lst):
    for el in lst:
        if isinstance(el, list):
            yield from el
        else:
            yield el


def _trace_loss(loss, grads, variables):
    return loss


build_trainable_InverseGamma_dist = None
build_trainable_normal_dist = None
build_surrogate_posterior = None
fit_surrogate_posterior = None


def _trace_variables(loss, grads, variables):
    return loss, variables


def minimize_distributed(
    loss_fn,
    data_factory,
    strategy,
    trainable_variables,
    num_epochs=100000,
    max_plateau_epochs=3,
    abs_tol=1e-4,
    rel_tol=1e-4,
    trace_fn=_trace_loss,
    max_decay_steps=8,
    learning_rate=1.0,
    check_every=25,
    decay_rate=0.95,
    checkpoint_name=None,
    max_initialization_steps=1000,
    clip_value=5.0,
    training_order=None,
    name="minimize",
    dtype=tf.float64,
    **kwargs,
):

    checkpoint_name = str(uuid.uuid4()) if checkpoint_name is None else checkpoint_name

    with strategy.scope():

        train_dist_dataset = strategy.experimental_distribute_dataset(data_factory())
        iterator = iter(train_dist_dataset)

        learning_rate = 1.0 if learning_rate is None else learning_rate

        def learning_rate_schedule_fn(step):
            return learning_rate * decay_rate**step

        decay_step = 0

        optimizer = tf.optimizers.Adam(
            learning_rate=lambda: learning_rate_schedule_fn(decay_step)
        )
        opt = tfa.optimizers.Lookahead(optimizer)

        checkpoint = tf.train.Checkpoint(
            optimizer=opt,
            **{"var_" + str(j): v for j, v in enumerate(trainable_variables)},
        )
        manager = tf.train.CheckpointManager(
            checkpoint,
            f"./.tf_ckpts/{checkpoint_name}/",
            checkpoint_name=checkpoint_name,
            max_to_keep=3,
        )
        save_path = manager.save()

        @tf.function
        def train_step(data):
            with tf.GradientTape() as tape:
                loss = loss_fn(data=data)
                gradients = tape.gradient(loss, trainable_variables)
                gradients = tf.nest.pack_sequence_as(
                    gradients,
                    tf.clip_by_global_norm(
                        [
                            tf.where(tf.math.is_finite(t), t, tf.zeros_like(t))
                            for t in tf.nest.flatten(gradients)
                        ],
                        clip_value,
                    )[0],
                )
                opt.apply_gradients(zip(gradients, trainable_variables))
                return loss

    with strategy.scope():

        @tf.function(input_signature=[train_dist_dataset.element_spec])
        def distributed_train_step(data):
            per_replica_losses = strategy.experimental_run_v2(train_step, args=(data,))
            return strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
            )

        converged = False
        results = []
        losses = []
        avg_losses = [1e10] * 3
        deviations = [1e10] * 3
        min_loss = 1e10
        min_state = None

        batches_since_checkpoint = 0
        batches_since_plateau = 0
        accepted_batches = 0
        num_resets = 0
        converged = False
        epoch = 1
        save_path = manager.save()
        print(f"Saved an initial checkpoint: {save_path}")
        for epoch in range(1, num_epochs + 1):
            if converged:
                break
            print(f"Epoch: {epoch}")
            total_loss = 0.0
            num_batches = 0

            for data in train_dist_dataset:
                total_loss += distributed_train_step(data)
                num_batches += 1
            train_loss = total_loss / num_batches
            losses += [train_loss]

            if epoch % check_every == 0:
                recent_losses = tf.convert_to_tensor(losses[-check_every:])
                avg_loss = tf.reduce_mean(recent_losses).numpy()

                if not np.isfinite(avg_loss):
                    status = "Backtracking"
                    print(status)
                    cp_status = checkpoint.restore(manager.latest_checkpoint)
                    cp_status.assert_consumed()

                    if accepted_batches == 0:
                        if epoch > max_initialization_steps:
                            converged = True
                            decay_step += 1
                            print(
                                "Failed to initialize within"
                                + f" {max_initialization_steps} steps"
                            )
                        if decay_step > max_decay_steps:
                            converged = True
                            continue
                    else:
                        decay_step += 1
                        if decay_step > max_decay_steps:
                            converged = True
                            continue

                    epoch += 1
                    cp_status.assert_consumed()
                    print(f" new learning rate: {optimizer.lr}")
                avg_losses += [avg_loss]
                # deviation = tf.math.reduce_std(recent_losses).numpy()
                deviation = np.abs(avg_losses[-1] - avg_losses[-2])
                deviations += [deviation]
                rel = np.abs(deviation / avg_loss)
                status = f"Iteration {epoch} -- loss: {losses[-1].numpy()}, "
                status += f"abs_err: {deviation}, rel_err: {rel}"
                print(status, flush=True)
                """Check for plateau
                """
                if (
                    (
                        (avg_losses[-1] > avg_losses[-3])
                        and (avg_losses[-1] > avg_losses[-2])
                    )
                    or batches_since_checkpoint > 4
                ) and batches_since_plateau > 2:
                    decay_step += 1
                    if batches_since_plateau >= max_plateau_epochs:
                        converged = True
                        print(f"We have reset {num_resets} times so quitting")
                    else:
                        status = "We are in a loss plateau"
                        status += f" learning rate: {optimizer.lr}"
                        print(status)
                        cp_status = checkpoint.restore(manager.latest_checkpoint)
                        cp_status.assert_consumed()

                        print(status)
                        batches_since_checkpoint += 1
                        batches_since_plateau = 0
                        num_resets += 1
                else:
                    if losses[-1] - min_loss < 0.0:
                        """
                        Save a checkpoint
                        """
                        min_loss = losses[-1]
                        save_path = manager.save()
                        accepted_batches += 1
                        print(f"Saved a checkpoint: {save_path}")
                        batches_since_checkpoint = 0
                    else:
                        batches_since_checkpoint += 1
                        decay_step += 1
                        status = "We are in a loss plateau"
                        status += f" learning rate: {optimizer.lr}"
                        print(status)
                        cp_status = checkpoint.restore(manager.latest_checkpoint)
                        cp_status.assert_consumed()

                        print(status)
                        batches_since_plateau = 0

                    if deviation < abs_tol:
                        print(
                            f"Converged in {epoch} iterations "
                            + "with acceptable absolute tolerance "
                            + f"{round(deviation, 3)}"
                        )
                        converged = True
                    elif rel < rel_tol:
                        print(
                            f"Converged in {epoch} iterations with "
                            + f"acceptable relative tolerance: {rel}"
                        )
                        converged = True
                    batches_since_plateau += 1
            epoch += 1
            if epoch >= num_epochs:
                print("Terminating because we are out of iterations")
        return losses


def batched_minimize(
    loss_fn,
    batched_data_factory,
    num_epochs=1000,
    max_plateau_epochs=10,
    abs_tol=1e-4,
    rel_tol=1e-4,
    trainable_variables=None,
    trace_fn=_trace_loss,
    learning_rate=1.0,
    decay_rate=0.95,
    max_decay_steps=8,
    checkpoint_name=None,
    processing_fn=None,
    name="minimize",
    check_every=1,
    clip_by="norm",
    clip_value=10.0,
    test_fn=None,
    verbose=False,
    temp_dir=os.path.join(tempfile.gettempdir(), "tfcheckpoints/"),
    **kwargs,
):

    checkpoint_name = str(uuid.uuid4()) if checkpoint_name is None else checkpoint_name
    learning_rate = 1.0 if learning_rate is None else learning_rate

    def learning_rate_schedule_fn(step):
        return learning_rate * decay_rate**step

    decay_step = 0

    optimizer = tf.optimizers.Adam(
        learning_rate=lambda: learning_rate_schedule_fn(decay_step)
    )

    opt = tfa.optimizers.Lookahead(optimizer)

    watched_variables = trainable_variables

    checkpoint = tf.train.Checkpoint(
        optimizer=opt, **{"var_" + str(j): v for j, v in enumerate(watched_variables)}
    )
    manager = tf.train.CheckpointManager(
        checkpoint,
        os.path.join(temp_dir, checkpoint_name),
        checkpoint_name=checkpoint_name,
        max_to_keep=3,
    )

    @tf.function(autograph=False)
    def train_loop_body(old_result, step, data=None):
        """Run a single optimization step."""
        if data is None:
            data = next(iter(batched_data_factory()))
        with tf.GradientTape(
            watch_accessed_variables=trainable_variables is None
        ) as tape:
            for v in trainable_variables or []:
                tape.watch(v)
            loss = loss_fn(data=data)
        watched_variables = tape.watched_variables()
        grads = tape.gradient(loss, watched_variables)
        flat_grads = tf.nest.flatten(grads)
        if clip_by == "norm":
            adjusted = tf.nest.pack_sequence_as(
                grads,
                tf.clip_by_global_norm(
                    [
                        tf.where(tf.math.is_finite(t), t, tf.zeros_like(t))
                        for t in flat_grads
                    ],
                    clip_value,
                )[0],
            )
        else:
            adjusted = (
                tf.nest.pack_sequence_as(
                    grads,
                    [
                        tf.clip_by_value(
                            tf.where(tf.math.is_finite(t), t, tf.zeros_like(t)),
                            -clip_value,
                            clip_value,
                        )
                        for t in flat_grads
                    ],
                )
            )
        train_op = opt.apply_gradients(zip(adjusted, watched_variables))
        with tf.control_dependencies([train_op]):
            state = trace_fn(
                tf.identity(loss),
                [tf.identity(g) for g in adjusted],
                [tf.identity(v) for v in watched_variables],
            )
        return state, adjusted

    with tf.name_scope(name) as name:
        # Compute the shape of the trace without executing the graph.
        concrete_loop_body = train_loop_body.get_concrete_function(
            tf.TensorSpec([]),
            tf.TensorSpec(
                [],
            ),
        )  # Inputs ignored.
        if all(
            [
                tensorshape_util.is_fully_defined(shape)
                for shape in tf.nest.flatten(concrete_loop_body.output_shapes)
            ]
        ):
            state_initializer = tf.nest.map_structure(
                lambda shape, dtype: tf.zeros(shape, dtype=dtype),
                concrete_loop_body.output_shapes,
                concrete_loop_body.output_dtypes,
            )
            initial_trace_step = None
        else:
            state_initializer = concrete_loop_body(
                tf.convert_to_tensor(0.0), tf.convert_to_tensor(0.0)
            )
            initial_trace_step = state_initializer

        converged = False
        losses = []
        avg_losses = [1e308] * 3
        deviations = [1e308] * 3
        min_loss = 1e308

        """
        # Test the first step, and make sure we can initialize safely

        loss = batch_normalized_loss(data=next(iter(data_factory())))

        if not np.isfinite(np.sum(loss.numpy())):
            # print(loss)
            print("Failed to initialize", flush=True)
            converged = True
        else:
            print(f"Initial loss: {loss}", flush=True)
        """

        step = tf.cast(0, tf.int32)
        batches_since_checkpoint = 0
        batches_since_plateau = 0
        accepted_batches = 0
        num_resets = 0
        save_path = manager.save()
        print(f"Saved a checkpoint: {save_path}", flush=True)
        while (step < num_epochs) and not converged:
            batch_losses = []
            for data in batched_data_factory():
                if processing_fn is not None:
                    data = processing_fn(data)
                batch_loss, grads = train_loop_body(state_initializer, step, data)
                if verbose:
                    for g, v in zip(grads, watched_variables):
                        tf.print(v.name, tf.reduce_max(g))
                if np.isfinite(batch_loss.numpy()):
                    batch_losses += [batch_loss.numpy()]
                else:
                    print("Batch loss NaN, skipping it for this epoch", flush=True)
                    cp_status = checkpoint.restore(manager.latest_checkpoint)
                    cp_status.assert_consumed()
                    decay_step += 1
                    if decay_step > max_decay_steps:
                        converged = True
                        continue
                if verbose:
                    if batch_losses[-1] < 0:
                        pass
                    else:
                        pass

            loss = tf.reduce_mean(batch_losses)
            avg_losses += [loss.numpy()]
            losses += [loss.numpy()]
            deviation = np.abs(avg_losses[-1] - min_loss)
            rel = np.abs(deviation / loss)
            print(
                f"Epoch {step}: average-batch loss:" + f"{loss} rel loss: {rel}",
                flush=True,
            )

            if (step > 0) and (step % check_every) == 0:
                """
                Check for convergence
                """
                if test_fn is not None:
                    test_fn()

                if not np.isfinite(loss):
                    cp_status = checkpoint.restore(manager.latest_checkpoint)
                    cp_status.assert_consumed()
                    print("Epoch loss NaN, restoring a checkpoint", flush=True)
                    decay_step += 1

                    if decay_step > max_decay_steps:
                        converged = True
                        continue
                    print(f"New learning rate: {optimizer.lr}", flush=True)
                    continue
                if losses[-1] < min_loss:
                    """
                    Save a checkpoint
                    """
                    min_loss = losses[-1]
                    save_path = manager.save()
                    accepted_batches += 1
                    print(f"Saved a checkpoint: {save_path}", flush=True)
                    batches_since_checkpoint = 0
                    if (deviation < abs_tol) and (
                        np.abs((avg_losses[2] - min_loss)) < abs_tol
                    ):
                        print(
                            f"Converged in {step} iterations "
                            + "with acceptable absolute tolerance",
                            flush=True,
                        )
                        converged = True
                    elif (rel < rel_tol) and (
                        (np.abs(avg_losses[2] - min_loss) / loss) < abs_tol
                    ):
                        print(
                            f"Converged in {step} iterations with "
                            + "acceptable relative tolerance"
                        )
                        converged = True
                    batches_since_plateau += 1
                else:
                    batches_since_checkpoint += 1
                    decay_step += 1
                    if decay_step > max_decay_steps:
                        converged = True
                        continue
                    batches_since_plateau = 0
                    print(f"New learning rate: {optimizer.lr}", flush=True)

                    if batches_since_checkpoint >= 2:
                        if batches_since_checkpoint >= max_plateau_epochs:
                            converged = True
                            print(
                                f"We have had {batches_since_checkpoint} epochs with no improvement so we give up",
                                flush=True,
                            )
                        else:
                            status = "We are in a loss plateau"
                            print(status, flush=True)
                            status = "Restoring from a checkpoint"
                            print(status, flush=True)
                            batches_since_plateau = 0

            step += 1
            if step > num_epochs:
                print("Terminating because we are out of iterations", flush=True)

        trace = tf.stack(losses)
        """
        if initial_trace_step is not None:
            trace = tf.nest.map_structure(
                lambda a, b: tf.concat(
                    [a[tf.newaxis, ...], tf.cast(b, a.dtype)], axis=0
                ),
                initial_trace_step,
                trace,
            )
        """
        cp_status = checkpoint.restore(manager.latest_checkpoint)
        cp_status.assert_consumed()
        trace.latest_checkpoint = manager.latest_checkpoint
        return trace


def clip_gradients(fn, clip_value, clip_by="norm", dtype=tf.float64):
    def wrapper(*args, **kwargs):
        @tf.custom_gradient
        def grad_wrapper(*flat_args_kwargs):
            with tf.GradientTape() as tape:
                tape.watch(flat_args_kwargs)
                new_args, new_kwargs = tf.nest.pack_sequence_as(
                    (args, kwargs), flat_args_kwargs
                )
                ret = fn(*new_args, **new_kwargs)

            def grad_fn(*dy):
                flat_grads = tape.gradient(ret, flat_args_kwargs, output_gradients=dy)

                if clip_by == "norm":
                    adjusted = tf.nest.pack_sequence_as(
                        flat_grads,
                        tf.clip_by_global_norm(
                            [
                                tf.where(tf.math.is_finite(t), t, tf.zeros_like(t))
                                for t in tf.nest.flatten(flat_grads)
                            ],
                            clip_value,
                        )[0],
                    )
                else:
                    adjusted = (
                        tf.nest.pack_sequence_as(
                            flat_grads,
                            [
                                tf.clip_by_value(
                                    tf.where(tf.math.is_finite(t), t, tf.zeros_like(t)),
                                    -clip_value,
                                    clip_value,
                                )
                                for t in tf.nest.flatten(flat_grads)
                            ],
                        ),
                    )
                flat_grads = tf.nest.map_structure(
                    lambda g: tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)),
                    flat_grads,
                )
                return adjusted

            return ret, grad_fn

        return grad_wrapper(*[tf.nest.flatten((args, kwargs))])

    return wrapper


@tf.function(autograph=False, experimental_compile=True)
def run_chain(
    init_state,
    step_size,
    target_log_prob_fn,
    unconstraining_bijectors,
    num_steps=500,
    num_leapfrog_steps=10,
    burnin=50,
):
    def trace_fn(_, pkr):
        return (
            pkr.inner_results.inner_results.target_log_prob,
            pkr.inner_results.inner_results.leapfrogs_taken,
            pkr.inner_results.inner_results.has_divergence,
            pkr.inner_results.inner_results.energy,
            pkr.inner_results.inner_results.log_accept_ratio,
        )

    kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.NoUTurnSampler(target_log_prob_fn, step_size=step_size),
        bijector=unconstraining_bijectors,
    )
    kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            num_leapfrog_steps=num_leapfrog_steps,
            step_size=0.1,
            state_gradients_are_stopped=True,
        ),
        bijector=unconstraining_bijectors,
    )
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=kernel, num_adaptation_steps=int(burnin * 0.8)
    )
    pbar = tfp.experimental.mcmc.ProgressBarReducer(num_steps)
    """
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=burnin,
        step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
            inner_results=pkr.inner_results._replace(step_size=new_step_size)
        ),
        step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
    )
    """
    kernel = tfp.experimental.mcmc.WithReductions(kernel, pbar)

    # Sampling from the chain.
    chain_state, sampler_stat = tfp.mcmc.sample_chain(
        num_results=num_steps,
        num_burnin_steps=burnin,
        current_state=init_state,
        kernel=kernel,
        trace_fn=trace_fn,
    )
    return chain_state, sampler_stat


class TransformedVariable(tfp_util.TransformedVariable):
    def __init__(
        self, initial_value, bijector, dtype=None, scope=None, name=None, **kwargs
    ):
        """Creates the `TransformedVariable` object.

        Args:
        initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
            which is the initial value for the Variable. Can also be a callable with
            no argument that returns the initial value when called. Note: if
            `initial_value` is a `TransformedVariable` then the instantiated object
            does not create a new `tf.Variable`, but rather points to the underlying
            `Variable` and chains the `bijector` arg with the underlying bijector as
            `tfb.Chain([bijector, initial_value.bijector])`.
        bijector: A `Bijector`-like instance which defines the transformations
            applied to the underlying `tf.Variable`.
        dtype: `tf.dtype.DType` instance or otherwise valid `dtype` value to
            `tf.convert_to_tensor(..., dtype)`.
            Default value: `None` (i.e., `bijector.dtype`).
        name: Python `str` representing the underlying `tf.Variable`'s name.
            Default value: `None`.
        **kwargs: Keyword arguments forward to `tf.Variable`.
        """
        # Check if `bijector` is "`Bijector`-like".
        for attr in {
            "forward",
            "forward_event_shape",
            "inverse",
            "inverse_event_shape",
            "name",
            "dtype",
        }:
            if not hasattr(bijector, attr):
                raise TypeError(
                    "Argument `bijector` missing required `Bijector` "
                    'attribute "{}".'.format(attr)
                )

        if callable(initial_value):
            initial_value = initial_value()
        initial_value = tf.convert_to_tensor(
            initial_value, dtype_hint=bijector.dtype, dtype=dtype
        )

        if scope is not None:
            with scope:
                variable = tf.Variable(
                    initial_value=bijector.inverse(initial_value),
                    name=name,
                    dtype=dtype,
                    **kwargs,
                )
        else:
            variable = tf.Variable(
                initial_value=bijector.inverse(initial_value),
                name=name,
                dtype=dtype,
                **kwargs,
            )
        super(tfp_util.TransformedVariable, self).__init__(
            pretransformed_input=variable,
            transform_fn=bijector,
            shape=initial_value.shape,
            name=bijector.name,
        )
        self._bijector = bijector


def tf_data_cardinality(tf_dataset):
    _have_cardinality = False
    up = tf_dataset
    card = -1
    root = False
    while (not _have_cardinality) and (not root):
        card = tf.data.experimental.cardinality(up)
        if card > 1:
            _have_cardinality = True
        else:
            if hasattr(up, "._input_dataset"):
                up = up._input_dataset
            else:
                root = True

    if card < 1:
        print("Getting the cardinality the slow way")
        num_elements = 0
        for _ in tqdm(up):
            num_elements += 1
        card = num_elements
    return card


def split_tensor(tensor, num_parts, axis=0):
    fn = split_tensor_factory(num_parts=tf.constant(num_parts), axis=axis)
    return fn(tensor)


def split_tensor_factory(num_parts, axis=0):
    @tf.function(experimental_relax_shapes=True)
    def split_tensor(tensor):
        max_divisor = tf.cast(tf.shape(tensor)[0] // num_parts, tf.int32)
        bulk = tensor[: max_divisor * num_parts, ...]
        remainder = tensor[max_divisor * num_parts :, ...]
        bulk = tf.split(bulk, num_parts, axis=axis)
        return bulk + [remainder]

    return split_tensor

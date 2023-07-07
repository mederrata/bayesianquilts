import functools
import inspect
import os
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import softplus as softplus_lib
from tensorflow_probability.python.internal import (
    dtype_util,
    prefer_static,
    tensorshape_util,
)
from tqdm import tqdm

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


def batched_minimize(
    loss_fn,
    batched_data_factory,
    batches_per_step=1,
    num_steps=1000,
    max_plateau_epochs=10,
    plateau_epochs_til_restore=5,
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
    clip_value=10.0,
    clip_norm=1.0,
    test_fn=None,
    verbose=False,
    debug=False,
    temp_dir=os.path.join(tempfile.gettempdir(), "tfcheckpoints/"),
    **kwargs,
):
    checkpoint_name = str(uuid.uuid4()) if checkpoint_name is None else checkpoint_name
    learning_rate = 1.0 if learning_rate is None else learning_rate

    def learning_rate_schedule_fn(step):
        return learning_rate * decay_rate**step

    decay_step = 0

    opt = tf.optimizers.Adam(
        learning_rate=lambda: learning_rate_schedule_fn(decay_step),
        clipvalue=clip_value,
        global_clipnorm=clip_norm,
    )

    # opt = tfa.optimizers.Lookahead(opt)

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

    def accumulate_grads(data, gradient_accumulation, trainable_variables):
        """Run a single optimization step."""
        if data is None:
            data = next(iter(batched_data_factory()))
        with tf.GradientTape(
            watch_accessed_variables=trainable_variables is None
        ) as tape:
            for v in trainable_variables or []:
                tape.watch(v)
            loss = loss_fn(data=data)
        # watched_variables = tape.watched_variables()
        grads = tape.gradient(loss, trainable_variables)
        flat_grads = tf.nest.flatten(grads)
        flat_grads = [
            tf.cond(tf.math.is_finite(loss), lambda: t, lambda: tf.zeros_like(t))
            for t in flat_grads
        ]

        for i, t in enumerate(flat_grads):
            t = tf.where(tf.math.is_finite(t), t, tf.zeros_like(t))
            gradient_accumulation[i].assign_add(t, read_value=False)

        state = trace_fn(
            tf.identity(loss),
            [tf.identity(g) for g in flat_grads],
            [tf.identity(v) for v in watched_variables],
        )
        return state, flat_grads

    if not debug:
        accumulate_grads = tf.function(accumulate_grads, autograph=False)

    def apply_grads(gradient_accumulation, trainable_variables):
        return opt.apply_gradients(zip(gradient_accumulation, trainable_variables))

    with tf.name_scope(name) as name:
        converged = False
        losses = []
        avg_losses = [1e308]
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

        batches_since_checkpoint = 0
        batches_since_plateau = 0
        accepted_batches = 0
        save_path = manager.save()
        batch_losses = []
        print(
            f"Running optimization for {num_steps} steps of {batches_per_step} accumulated batches, checking every {check_every} steps",
            flush=True,
        )
        print(f"Saved a checkpoint: {save_path}", flush=True)
        gradient_accumulation = None
        data_factory = batched_data_factory()
        data_factory = data_factory.repeat()

        pbar_outer = tqdm(total=num_steps, position=0)
        pbar = tqdm(total=batches_per_step, leave=False, position=1)
        test_results = []
        step = 0
        batch_loss = np.nan
        for n_batch, data in enumerate(data_factory):
            if n_batch % batches_per_step == 0:
                pbar = tqdm(total=batches_per_step, leave=False, position=1)
                # this batch is the start of a gradient step
                if step > 0:
                    pbar_outer.update(1)
                    
                    if not np.isfinite(batch_loss):
                        print(f"Step {step} has an infinite loss", flush=True)

                        #  apply the grad
                    elif gradient_accumulation is not None:
                        _ = apply_grads(gradient_accumulation, watched_variables)
                        if debug:
                            print(f"batch {n_batch} step {step}", flush=True)
                            print("applying gradient", flush=True)
                step += 1
                batch_losses += [[]]
                gradient_accumulation = [
                    tf.Variable(
                        tf.zeros_like(v),
                        trainable=False,
                        name="grad_accum_" + str(i),
                        synchronization=tf.VariableSynchronization.ON_READ,
                        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                    )
                    for i, v in enumerate(watched_variables)
                ]
                if (step > num_steps):
                    print("Terminating because we are out of iterations", flush=True)
                    break

                if converged:
                    print("Terminating because the loss converged", flush=True)
                    break

            if processing_fn is not None:
                data = processing_fn(data)

            batch_loss, grads = accumulate_grads(
                data, gradient_accumulation, watched_variables
            )
            pbar.update(1)

            if np.isfinite(batch_loss.numpy()):
                batch_losses[-1] += [batch_loss.numpy()]
            else:
                if verbose:
                    for g, v in zip(grads, watched_variables):
                        tf.print(v.name, tf.reduce_max(g))
                if loss_fn:
                    test_results += [loss_fn(data=data)]
                continue

            loss = np.mean(batch_losses[-1])
            avg_losses += [loss]
            losses += [loss]
            deviation = np.abs(avg_losses[-1] - min_loss)
            rel = np.abs(deviation / loss)

            if (
                (step > 0)
                and (n_batch % batches_per_step == batches_per_step - 1)
                and (step % check_every) == 0
            ):
                """
                Check for convergence
                """

                print(
                    f"\Step {step}: average-batch loss:" + f"{loss} rel loss: {rel}",
                    flush=True,
                )
                save_because_of_test = False
                if test_fn is not None:
                    test_results += [test_fn()]
                    if isinstance(test_results[-1], tf.Tensor):
                        test_results[-1] = test_results[-1].numpy()
                    if len(test_results) > 1:
                        if test_results[-1] > np.max(test_results[:-1]):
                            save_because_of_test = True

                if not np.isfinite(loss):
                    cp_status = checkpoint.restore(manager.latest_checkpoint)
                    cp_status.assert_consumed()
                    print("\Step loss NaN, restoring a checkpoint", flush=True)
                    decay_step += 1

                    if decay_step > max_decay_steps:
                        converged = True
                        continue
                    print(f"New learning rate: {opt.lr.numpy()}", flush=True)
                    continue
                save_because_of_loss = losses[-1] < min_loss
                save_this = save_because_of_test if test_fn else save_because_of_loss

                if save_this:
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
                    print(f"New learning rate: {opt.lr}", flush=True)

                    if batches_since_checkpoint >= plateau_epochs_til_restore:
                        if batches_since_checkpoint >= max_plateau_epochs:
                            converged = True
                            print(
                                f"We have had {batches_since_checkpoint} checks with no improvement so we give up",
                                flush=True,
                            )
                        else:
                            status = "We are in a loss plateau"
                            print(status, flush=True)
                            status = "Restoring from a checkpoint"
                            print(status, flush=True)
                            batches_since_plateau = 0

        trace = tf.stack(losses)
        if test_fn is not None:
            # take the checkpoint that had the best test result
            trace.test_eval = test_results
            return trace
        else:
            try:
                if np.isnan(losses[-1]):
                    cp_status = checkpoint.restore(manager.latest_checkpoint)
                    cp_status.assert_consumed()
                    trace.latest_checkpoint = manager.latest_checkpoint
                else:
                    #
                    if len(test_results) > 1:
                        if test_results[-1] < np.max(test_results[:-1]):
                            cp_status = checkpoint.restore(manager.latest_checkpoint)
                            cp_status.assert_consumed()
                            trace.latest_checkpoint = manager.latest_checkpoint
            except AssertionError:
                pass

        return trace


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


class IndexMapper(object):
    def __init__(self, vocab):
        self.vocab = tf.constant(vocab)
        self.N_keys = tf.shape(self.vocab)[0]
        self.vals = tf.constant(tf.range(1, self.N_keys + 1, dtype=tf.int32))
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.vocab, self.vals), 0
        )

    def map(self, x):
        return self.table.lookup(x)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["table"]
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.vocab, self.vals), 0
        )


class CountEncoder(object):
    def __init__(self, vocab, dtype=tf.int32):
        self.vocab = tf.constant(vocab)
        self.N_keys = len(vocab)
        self.vals = tf.range(1, self.N_keys + 1, dtype=tf.int32)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.vocab, self.vals), 0
        )

    def encode(self, x):
        # x = tf.strings.split(x, ",")
        shape = tf.constant([self.N_keys + 1])
        x = self.table.lookup(x)
        y, idx, count = tf.unique_with_counts(x)
        counts = tf.scatter_nd(y[..., tf.newaxis], tf.cast(count, tf.int32), shape)
        return tf.cast(counts, tf.float64)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["table"]
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.vocab, self.vals), 0
        )


class DummyObject(object):
    pass
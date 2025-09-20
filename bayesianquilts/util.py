import numbers
from typing import Any, Callable, Iterator

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.core import unfreeze
from tensorflow_probability.substrates.jax import tf2jax as tf
from tqdm import tqdm


def flatten(lst):
    for el in lst:
        if isinstance(el, list):
            yield from el
        else:
            yield el


# --- Core Training Components ---


def mk_train_step_fn(optimizer: optax.GradientTransformation):
    @jax.jit
    def train_step(params: Any, opt_state: Any, grads: Any):
        """JIT-compiled function to apply a single optimizer update."""
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    return train_step


def training_loop(
    initial_values: dict[str, Any],
    loss_fn: Callable[[Any, Any], float],
    data_iterator: Iterator,
    steps_per_epoch: int,
    num_epochs: int,
    base_optimizer_fn: Callable[[float], optax.GradientTransformation] | None = None,
    accumulation_steps: int = 1,
    check_convergence_every: int = 1,
    patience: int = 3,
    lr_decay_factor: float = 0.5,
    learning_rate: float = 0.001,
    checkpoint_dir: str = "/tmp/checkpoints",
    optimize_keys: list[str] | None = None,
    grad_clip_value: float | None = None,
):
    """
    Advanced training loop with checkpointing, early stopping, LR decay on plateau,
    and selective parameter optimization via optimize_keys.
    """
    # 1. Setup for new features

    best_loss = float("inf")
    checks_no_improve = 0
    current_lr = learning_rate
    if base_optimizer_fn is None:
        base_optimizer_fn = lambda lr: optax.adam(learning_rate=lr)
    # 2. Initial Optimizer Setup

    # Select which parameters to optimize
    if optimize_keys is None:
        optimize_keys = list(initial_values.keys())

    def filter_params(params):
        return {k: v for k, v in params.items() if k in optimize_keys}

    def merge_params(full_params, updated_subset):
        merged = dict(full_params)
        for k in updated_subset:
            merged[k] = updated_subset[k]
        return merged

    optimizer = base_optimizer_fn(current_lr)
    opt_state = optimizer.init(filter_params(initial_values))
    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=[1]))
    if checkpoint_dir is not None:
        ocp.test_utils.erase_and_create_empty(checkpoint_dir)
        checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())

    print("--- Starting Training ---")
    print(f"Patience for early stopping: {patience} epochs")
    print(f"LR decay factor on plateau: {lr_decay_factor}")
    print(f"Convergence will be checked every: {check_convergence_every} epoch(s)")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"Optimizing keys: {optimize_keys}")
    print("-------------------------")
    params = dict(initial_values)
    # 3. Main training loop
    train_step_fn = mk_train_step_fn(optimizer)
    total_steps = 0
    epoch_losses = []

    try:
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            grad_accumulator = jax.tree_util.tree_map(
                jnp.zeros_like, unfreeze(filter_params(params))
            )
            with tqdm(
                range(steps_per_epoch),
                desc=f"Epoch {epoch + 1}/{num_epochs} (LR: {current_lr:.6f})",
                unit="batch",
                leave=False,
            ) as pbar:
                for _ in pbar:
                    try:
                        batch = next(data_iterator)
                        loss_val, grads = value_and_grad_fn(batch, params)
                        epoch_loss += loss_val
                        if jnp.isnan(loss_val) or jnp.isinf(loss_val):
                            print(
                                f"Skipping batch due to NaN/Inf loss at epoch {epoch + 1}, step {total_steps + 1}."
                            )
                            current_lr *= lr_decay_factor
                            optimizer = base_optimizer_fn(current_lr)
                            opt_state = optimizer.init(
                                filter_params(params)
                            )  # Re-initialize optimizer state with new LR
                            print(f"  -> Decaying learning rate to: {current_lr:.6f}")
                            continue
                        try:
                            ave_grad = jnp.mean(
                                jnp.array([jnp.mean(x) for x in grads[0]])
                            )
                        except TypeError:
                            ave_grad = jnp.isnan(
                                jnp.mean(
                                    jnp.array([jnp.mean(x) for x in grads[0].values()])
                                )
                            )
                        if jnp.isnan(ave_grad) or jnp.isinf(ave_grad):
                            print(
                                f"Skipping batch due to NaN/Inf gradients at epoch {epoch + 1}, step {total_steps + 1}."
                            )
                            checks_no_improve += 1
                            continue

                        # Only accumulate gradients for selected keys
                        filtered_grads = filter_params(grads[0])

                        # Apply gradient clipping if specified
                        if grad_clip_value is not None:
                            filtered_grads = optax.clip_by_global_norm(
                                filtered_grads, grad_clip_value
                            )[0]

                        grad_accumulator = jax.tree_util.tree_map(
                            lambda acc, g: acc + g, grad_accumulator, filtered_grads
                        )
                        total_steps += 1
                        if (total_steps + 1) % accumulation_steps == 0:
                            tot_grads = jax.tree_util.tree_map(
                                lambda g: g, grad_accumulator
                            )
                            # Only update selected keys
                            filtered_params = filter_params(params)
                            new_filtered_params, opt_state = train_step_fn(
                                filtered_params, opt_state, tot_grads
                            )
                            # Merge updated subset back into full params
                            params = merge_params(params, new_filtered_params)
                            grad_accumulator = jax.tree_util.tree_map(
                                jnp.zeros_like, unfreeze(filter_params(params))
                            )

                        pbar.set_postfix(
                            loss=f"{loss_val:.4f}", best_loss=f"{best_loss:.4f}"
                        )
                    except KeyboardInterrupt:
                        print("\nTraining interrupted by user.")
                        return epoch_losses, params
            avg_epoch_loss = epoch_loss / steps_per_epoch
            epoch_losses += [avg_epoch_loss]
            # 4. Check for improvement, save checkpoints, and decay LR
            if (epoch + 1) % check_convergence_every == 0:
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    checks_no_improve = 0
                    # Save the best model parameters
                    if checkpoint_dir is not None:
                        ckpt = {
                            "params": params,
                            "opt_state": opt_state,
                            "epoch": epoch,
                            "best_loss": best_loss,
                        }

                        checkpointer.save(
                            f"{checkpoint_dir}/best_model_{epoch}",
                            args=ocp.args.Composite(state=ocp.args.StandardSave(ckpt)),
                        )
                    print(f"  -> New best loss found. Checkpoint saved.                    ")
                else:
                    checks_no_improve += 1
                    print(
                        f"  -> No improvement in loss for {checks_no_improve} check(s).                    "
                    )
                    # Decay learning rate
                    current_lr *= lr_decay_factor
                    optimizer = base_optimizer_fn(current_lr)
                    opt_state = optimizer.init(
                        filter_params(params)
                    )  # Re-initialize optimizer state with new LR
                    print(f"  -> Decaying learning rate to: {current_lr:.6f}")

                # 5. Early stopping check
                if checks_no_improve >= patience:
                    print(
                        f"\nEarly stopping triggered after {patience} epochs with no improvement."
                    )
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print(f"Completed {len(epoch_losses)} epochs before interruption.")
        return epoch_losses, params

    print("\n--- Training Finished ---")

    # 6. Restore from best snapshot if needed
    if checkpoint_dir is not None:
        try:
            # Find the latest best model checkpoint
            import glob

            best_checkpoints = glob.glob(f"{checkpoint_dir}/best_model_*")
            if best_checkpoints:
                latest_checkpoint = max(
                    best_checkpoints, key=lambda x: int(x.split("_")[-1])
                )
                restored_ckpt = checkpointer.restore(latest_checkpoint)
                final_params = restored_ckpt["state"]["params"]
                print(f"Restored model from checkpoint: {latest_checkpoint}")
            else:
                final_params = params
                print("No checkpoints found, using final parameters")
        except Exception as e:
            print(f"Failed to restore checkpoint: {e}")
            final_params = params
    else:
        final_params = params

    return epoch_losses, final_params


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
    def __init__(self, vocab, dtype=jnp.int32):
        self.vocab = vocab
        self.N_keys = len(vocab)
        self.vals = jnp.range(1, self.N_keys + 1).astype(dtype)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.vocab, self.vals), 0
        )

    def encode(self, x):
        # x = tf.strings.split(x, ",")
        shape = tf.constant([self.N_keys + 1])
        x = self.table.lookup(x)
        y, idx, count = jnp.unique(x, return_index=True, return_counts=True)
        count = tf.gather(x)
        counts = tf.scatteruse_nd(y[..., tf.newaxis], tf.cast(count, tf.int32), shape)
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


class PiecewiseFunction(object):
    def __init__(
        self, breakpoints, values, cadlag=True, unique_breaks=False, dtype=tf.float64
    ):
        self.dtype = dtype
        self.unique_breaks = unique_breaks
        values = values.astype(dtype)
        breakpoints = breakpoints.astype(dtype)
        last = breakpoints.shape[-1]
        if last is None:
            last = 1

        if values.ndims > breakpoints.ndims:
            breakpoints += jnp.zeros(
                values.shape[:-1] + [last],
                breakpoints.dtype,
            )
        self.breakpoints = breakpoints.astype(dtype)
        self.values = values
        self.cadlag = cadlag

    def __call__(self, value):
        value = value.astype(self.dtype)
        ndx = jnp.sum(
            self.breakpoints[..., tf.newaxis] <= value[..., tf.newaxis, :],
            axis=-2,
        ).astype(tf.int32)
        return tf.gather(
            self.values, ndx, batch_dims=len(self.values.shape.as_list()) - 1
        )

    def __add__(self, obj):
        if isinstance(obj, numbers.Number):
            return PiecewiseFunction(self.breakpoints, obj + self.values)
        left = self.breakpoints

        left_batch_shape = left.shape[:-1]
        right = obj.breakpoints.astype(self.dtype)
        right_batch_shape = right.shape[:-1]

        if len(left_batch_shape) < len(right_batch_shape):
            left += jnp.zeros(right_batch_shape + left.shape[-1:], self.dtype)
        elif len(left_batch_shape) > len(right_batch_shape):
            right += jnp.zeros(left_batch_shape + right.shape[-1:], self.dtype)

        breakpoints = jnp.concatenate([left, right], axis=-1)
        breakpoints, _ = jnp.unique(x=breakpoints, axis=[-1])
        breakpoints = jnp.sort(breakpoints, axis=-1)
        d = len(breakpoints.shape.as_list())
        breakpoints_ = jnp.pad(breakpoints, [(0, 0)] * (d - 1) + [(1, 0)])
        v1 = self(breakpoints_)
        v2 = obj(breakpoints_)
        return PiecewiseFunction(breakpoints, v1 + v2, dtype=self.dtype)


def demo():
    f = PiecewiseFunction([[1, 2, 3]], [[1, 2, 2, 3]])
    g = f + 1
    print(g(1))
    h = g + f
    print(h(1))
    print(h(5))
    pass


if __name__ == "__main__":
    demo()

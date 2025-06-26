import numbers
import os
from typing import Any, Callable, Iterator

import jax
import jax.numpy as jnp
import optax
from flax.core import unfreeze
from flax.training import checkpoints
from flax.traverse_util import path_aware_map
from tensorflow_probability.substrates.jax import tf2jax as tf
from tqdm import tqdm


def flatten(lst):
    for el in lst:
        if isinstance(el, list):
            yield from el
        else:
            yield el


# --- Core Training Components ---

def create_optimizer(base_optimizer: optax.GradientTransformation, trainable_filter: Callable[[str, Any], bool]):
    """
    Creates a partitioned optimizer that only applies updates to parameters
    matching the trainable_filter.

    Args:
        base_optimizer: The core Optax optimizer (e.g., optax.adam).
        trainable_filter: A function that returns True for parameters that should be trained.
                          It receives the parameter's path and value.

    Returns:
        An Optax multi_transform optimizer.
    """
    # Use path_aware_map to label parameters as 'trainable' or 'frozen'
    param_labels = path_aware_map(
        lambda path, _: 'trainable' if trainable_filter(path, _) else 'frozen',
        # We pass a dummy param structure here just to build the label tree.
        # The actual params will be used during the update.
        {'layer1': {'kernel': True, 'bias': True}, 'layer2': {'kernel': True, 'bias': True}}
    )

    # multi_transform applies different optimizers to different partitions of the parameters.
    # We apply the base_optimizer to 'trainable' params and set gradients for 'frozen' params to zero.
    return optax.multi_transform(
        {'trainable': base_optimizer, 'frozen': optax.set_to_zero()},
        param_labels
    )

@jax.jit
def train_step(params: Any, opt_state: Any, optimizer: optax.GradientTransformation, grads: Any):
    """JIT-compiled function to apply a single optimizer update."""
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

def training_loop(
    params: Any,
    loss_fn: Callable[[Any, Any, Any], float],
    data_iterator: Iterator,
    steps_per_epoch: int,
    num_epochs: int,
    base_optimizer_fn: Callable[[float], optax.GradientTransformation],
    trainable_filter: Callable[[str, Any], bool],
    accumulation_steps: int,
    patience: int = 3,
    lr_decay_factor: float = 0.5,
    learning_rate: float = 0.001,
    checkpoint_dir: str = './checkpoints',
):
    """
    Advanced training loop with checkpointing, early stopping, and LR decay on plateau.
    """
    # 1. Setup for new features
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float('inf')
    epochs_no_improve = 0
    current_lr = learning_rate

    # 2. Initial Optimizer Setup
    optimizer = create_optimizer(lambda: base_optimizer_fn(current_lr), trainable_filter)
    opt_state = optimizer.init(params)
    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    print("--- Starting Training ---")
    print(f"Patience for early stopping: {patience} epochs")
    print(f"LR decay factor on plateau: {lr_decay_factor}")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print("-------------------------")

    # 3. Main training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        grad_accumulator = jax.tree_util.tree_map(jnp.zeros_like, unfreeze(params))
        
        with tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{num_epochs} (LR: {current_lr:.6f})", unit="batch") as pbar:
            for step in pbar:
                x_batch, y_batch = next(data_iterator)
                loss_val, grads = value_and_grad_fn(params, x_batch, y_batch)
                epoch_loss += loss_val
                grad_accumulator = jax.tree_util.tree_map(lambda acc, g: acc + g, grad_accumulator, grads)

                if (step + 1) % accumulation_steps == 0:
                    avg_grads = jax.tree_util.tree_map(lambda g: g / accumulation_steps, grad_accumulator)
                    params, opt_state = train_step(params, opt_state, optimizer, avg_grads)
                    grad_accumulator = jax.tree_util.tree_map(jnp.zeros_like, unfreeze(params))
                
                pbar.set_postfix(loss=f"{loss_val:.4f}", best_loss=f"{best_loss:.4f}")
        
        avg_epoch_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch + 1} Summary | Average Loss: {avg_epoch_loss:.6f}")

        # 4. Check for improvement, save checkpoints, and decay LR
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            epochs_no_improve = 0
            # Save the best model parameters
            checkpoints.save_checkpoint(ckpt_dir=checkpoint_dir, target=params, step=epoch, prefix='best_model_', overwrite=True)
            print(f"  -> New best loss found. Checkpoint saved.")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement in loss for {epochs_no_improve} epoch(s).")
            # Decay learning rate
            current_lr *= lr_decay_factor
            optimizer = create_optimizer(lambda: base_optimizer_fn(current_lr), trainable_filter)
            opt_state = optimizer.init(params) # Re-initialize optimizer state with new LR
            print(f"  -> Decaying learning rate to: {current_lr:.6f}")

        # 5. Early stopping check
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            break
            
    print("\n--- Training Finished ---")

    # 6. Restore from best snapshot if needed
    final_params = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=params, prefix='best_model_')
    if final_params is not params:
         print("Restored model from the best checkpoint.")
    
    return final_params

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

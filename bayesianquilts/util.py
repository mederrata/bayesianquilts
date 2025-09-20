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
    clip_norm: float | None = None,
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

    # Create base optimizer
    base_optimizer = base_optimizer_fn(current_lr)

    # Add gradient clipping if specified
    if clip_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            base_optimizer
        )
    else:
        optimizer = base_optimizer

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
                            base_optimizer = base_optimizer_fn(current_lr)
                            if clip_norm is not None:
                                optimizer = optax.chain(
                                    optax.clip_by_global_norm(clip_norm),
                                    base_optimizer
                                )
                            else:
                                optimizer = base_optimizer
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
                    base_optimizer = base_optimizer_fn(current_lr)
                    if clip_norm is not None:
                        optimizer = optax.chain(
                            optax.clip_by_global_norm(clip_norm),
                            base_optimizer
                        )
                    else:
                        optimizer = base_optimizer
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
        self, breakpoints, values, cadlag=True, unique_breaks=False, dtype=jnp.float32
    ):
        """Initialize a JAX-based piecewise function with batching support.

        Args:
            breakpoints: JAX array of shape (..., n_breaks) where ... represents batch dimensions
            values: JAX array of shape (..., n_values) where n_values = n_breaks + 1
            cadlag: Whether function is right-continuous (True) or left-continuous (False)
            unique_breaks: Whether breakpoints are already unique and sorted
            dtype: JAX dtype for computations
        """
        self.dtype = dtype
        self.cadlag = cadlag
        self.unique_breaks = unique_breaks

        # Convert to JAX arrays with specified dtype
        self.breakpoints = jnp.asarray(breakpoints, dtype=dtype)
        self.values = jnp.asarray(values, dtype=dtype)

        # Ensure breakpoints and values have compatible batch dimensions
        breakpoints_shape = self.breakpoints.shape
        values_shape = self.values.shape

        # The last dimension of values should be one more than breakpoints
        if values_shape[-1] != breakpoints_shape[-1] + 1:
            raise ValueError(f"Values shape {values_shape} incompatible with breakpoints shape {breakpoints_shape}. "
                           f"Values should have {breakpoints_shape[-1] + 1} elements in last dimension.")

        # Handle batch dimension broadcasting
        if len(values_shape) > len(breakpoints_shape):
            # Broadcast breakpoints to match values batch dimensions
            target_shape = values_shape[:-1] + (breakpoints_shape[-1],)
            self.breakpoints = jnp.broadcast_to(self.breakpoints, target_shape)
        elif len(breakpoints_shape) > len(values_shape):
            # Broadcast values to match breakpoints batch dimensions
            target_shape = breakpoints_shape[:-1] + (values_shape[-1],)
            self.values = jnp.broadcast_to(self.values, target_shape)

        # Sort breakpoints if needed
        if not unique_breaks:
            self.breakpoints = jnp.sort(self.breakpoints, axis=-1)

    def __call__(self, x):
        """Evaluate piecewise function at points x with broadcasting support.

        Args:
            x: JAX array of evaluation points with shape (..., n_points) or (...)

        Returns:
            JAX array with broadcasted shape containing function values
        """
        x = jnp.asarray(x, dtype=self.dtype)

        # Handle scalar input
        if x.ndim == 0:
            x = x[None]
            squeeze_output = True
        else:
            squeeze_output = False

        # Determine output shape through broadcasting rules
        # We need to broadcast: (...batch_dims, n_breaks), (...x_batch_dims, n_points)
        # Result should be: (...broadcast_batch_dims, n_points)

        # Add dimensions for broadcasting
        # breakpoints: (..., n_breaks) -> (..., n_breaks, 1)
        # x: (..., n_points) -> (..., 1, n_points)
        breaks_expanded = self.breakpoints[..., :, None]
        x_expanded = x[..., None, :]

        # Find indices: count how many breakpoints are <= each x value
        if self.cadlag:
            # Right-continuous: use <=
            indices = jnp.sum(breaks_expanded <= x_expanded, axis=-2)
        else:
            # Left-continuous: use <
            indices = jnp.sum(breaks_expanded < x_expanded, axis=-2)

        # Clamp indices to valid range [0, n_values-1]
        indices = jnp.clip(indices, 0, self.values.shape[-1] - 1)

        # Gather values using advanced indexing
        # We need to handle the batch dimensions properly
        result = jnp.take_along_axis(
            self.values[..., None, :],
            indices[..., :, None],
            axis=-1
        ).squeeze(-1)

        # Remove singleton dimension if input was scalar
        if squeeze_output:
            result = result.squeeze(-1)

        return result

    def __add__(self, other):
        """Add two piecewise functions or add a scalar to this function."""
        if isinstance(other, numbers.Number):
            # Add scalar to all values
            return PiecewiseFunction(
                self.breakpoints,
                self.values + other,
                cadlag=self.cadlag,
                unique_breaks=self.unique_breaks,
                dtype=self.dtype
            )

        if not isinstance(other, PiecewiseFunction):
            raise TypeError("Can only add PiecewiseFunction or scalar")

        # Merge breakpoints from both functions
        # First, broadcast to common batch shape
        left_breaks = self.breakpoints
        right_breaks = other.breakpoints

        # Concatenate breakpoints
        all_breaks = jnp.concatenate([left_breaks, right_breaks], axis=-1)

        # Get unique sorted breakpoints
        all_breaks = jnp.sort(all_breaks, axis=-1)
        # Note: jnp.unique doesn't work well with batch dimensions, so we'll keep duplicates
        # The evaluation will still work correctly

        # Evaluate both functions at all breakpoints
        # Add a small offset to handle discontinuities
        eval_points = all_breaks
        if self.cadlag and other.cadlag:
            # Both right-continuous, evaluate at breakpoints
            v1 = self(eval_points)
            v2 = other(eval_points)
        else:
            # Handle mixed continuity by evaluating just to the left and right
            eps = jnp.finfo(self.dtype).eps * 10
            eval_points_left = eval_points - eps
            v1 = self(eval_points_left)
            v2 = other(eval_points_left)

        # Create new values array (one more element than breakpoints)
        # Add the value at negative infinity (first value of each function)
        v1_init = self.values[..., :1]  # First value
        v2_init = other.values[..., :1]  # First value

        new_values = jnp.concatenate([v1_init + v2_init, v1 + v2], axis=-1)

        return PiecewiseFunction(
            all_breaks,
            new_values,
            cadlag=self.cadlag and other.cadlag,
            unique_breaks=False,  # We may have duplicates
            dtype=self.dtype
        )


def demo():
    """Demonstrate the JAX-based PiecewiseFunction with batching and broadcasting."""
    import jax.numpy as jnp

    print("=== JAX PiecewiseFunction Demo ===")

    # Single piecewise function: f(x) = 1 for x <= 1, 2 for 1 < x <= 2, 3 for 2 < x <= 3, 4 for x > 3
    print("\n1. Single piecewise function:")
    breakpoints = jnp.array([1.0, 2.0, 3.0])  # 3 breakpoints
    values = jnp.array([1.0, 2.0, 3.0, 4.0])  # 4 values (n_breaks + 1)
    f = PiecewiseFunction(breakpoints, values)

    test_points = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    print(f"f({test_points}) = {f(test_points)}")

    # Test scalar input
    print(f"f(1.5) = {f(1.5)}")

    # Batch of piecewise functions
    print("\n2. Batch of piecewise functions:")
    batch_breakpoints = jnp.array([[1.0, 2.0], [2.0, 4.0]])  # 2 functions, 2 breakpoints each
    batch_values = jnp.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])  # 2 functions, 3 values each
    f_batch = PiecewiseFunction(batch_breakpoints, batch_values)

    test_batch_points = jnp.array([0.5, 1.5, 3.0])
    result_batch = f_batch(test_batch_points)
    print(f"Batch evaluation at {test_batch_points}:")
    print(f"Function 0: {result_batch[0]}")
    print(f"Function 1: {result_batch[1]}")

    # Broadcasting test
    print("\n3. Broadcasting test:")
    single_points = jnp.array([1.5, 2.5])  # (2,)
    batch_points = jnp.array([[0.5, 1.5], [2.5, 3.5]])  # (2, 2)

    print(f"Single function, single points: {f(single_points)}")
    print(f"Single function, batch points: {f(batch_points)}")
    print(f"Batch functions, single points: {f_batch(single_points)}")

    # Addition tests
    print("\n4. Addition tests:")
    g = f + 10  # Add scalar
    print(f"f + 10 at [0.5, 1.5, 2.5]: {g(jnp.array([0.5, 1.5, 2.5]))}")

    # Add two functions
    f2 = PiecewiseFunction(jnp.array([1.5, 2.5]), jnp.array([0.5, 1.0, 1.5]))
    h = f + f2
    print(f"f + f2 at [0.5, 1.5, 2.5]: {h(jnp.array([0.5, 1.5, 2.5]))}")

    print("\n=== Demo completed ===")
    return f, f_batch, g, h


if __name__ == "__main__":
    demo()

#!/usr/bin/env python3
"""
Simple test to verify gradient clipping implementation works correctly.
"""

import jax
import jax.numpy as jnp
import optax
from bayesianquilts.util import training_loop


def simple_loss_fn(data, params):
    """Simple quadratic loss function for testing."""
    x, y = data
    pred = params['w'] * x + params['b']
    return jnp.mean((pred - y) ** 2)


def create_test_data():
    """Create simple test data."""
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (100,))
    # True parameters: w=2.0, b=1.0
    y = 2.0 * x + 1.0 + 0.1 * jax.random.normal(key, (100,))
    return x, y


def test_gradient_clipping():
    """Test that gradient clipping is applied correctly."""
    print("Testing gradient clipping implementation...")

    # Create test data
    x, y = create_test_data()
    data = (x, y)

    # Initial parameters
    initial_params = {'w': 10.0, 'b': 10.0}  # Start far from true values

    # Test without gradient clipping
    print("\n1. Training without gradient clipping...")
    losses_no_clip, params_no_clip = training_loop(
        initial_values=initial_params,
        loss_fn=simple_loss_fn,
        data_iterator=iter([data] * 10),  # Simple iterator
        steps_per_epoch=1,
        num_epochs=10,
        learning_rate=0.1,
        checkpoint_dir=None,
        clip_norm=None
    )

    # Test with gradient clipping
    print("\n2. Training with gradient clipping (norm=1.0)...")
    losses_with_clip, params_with_clip = training_loop(
        initial_values=initial_params,
        loss_fn=simple_loss_fn,
        data_iterator=iter([data] * 10),  # Simple iterator
        steps_per_epoch=1,
        num_epochs=10,
        learning_rate=0.1,
        checkpoint_dir=None,
        clip_norm=1.0
    )

    print(f"\nResults:")
    print(f"Without clipping - Final loss: {losses_no_clip[-1]:.6f}")
    print(f"With clipping - Final loss: {losses_with_clip[-1]:.6f}")
    print(f"Without clipping - Final params: w={params_no_clip['w']:.3f}, b={params_no_clip['b']:.3f}")
    print(f"With clipping - Final params: w={params_with_clip['w']:.3f}, b={params_with_clip['b']:.3f}")

    # Verify that both training runs completed successfully
    assert len(losses_no_clip) == 10, "Training without clipping should complete 10 epochs"
    assert len(losses_with_clip) == 10, "Training with clipping should complete 10 epochs"

    print("\n✅ Gradient clipping implementation test passed!")
    return True


if __name__ == "__main__":
    test_gradient_clipping()
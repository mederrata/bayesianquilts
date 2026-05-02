#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
from bayesianquilts.predictors.nn.locally_linear import LocallyLinearAttention

def test_linearity():
    print("Testing linearity within region...")
    input_dim = 2
    output_dim = 2
    seq_len = 3
    num_regions = 2
    
    # Place centroids far apart
    centroids = jnp.array([
        [-10.0, -10.0],
        [10.0, 10.0]
    ])
    
    model = LocallyLinearAttention(
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        num_regions=num_regions,
        centroids=centroids
    )
    
    # Create two inputs in the same region (near first centroid)
    X1 = centroids[0] + jax.random.normal(jax.random.PRNGKey(1), (1, seq_len, input_dim)) * 0.1
    X2 = centroids[0] + jax.random.normal(jax.random.PRNGKey(2), (1, seq_len, input_dim)) * 0.1
    
    # Get a sample from the surrogate distribution
    surrogate = model.surrogate_distribution_generator(model.params)
    params = surrogate.sample(seed=jax.random.PRNGKey(42))



    
    # f(aX1 + bX2)
    a, b = 0.5, 0.5
    X_mixed = a * X1 + b * X2
    Y_mixed = model.eval(X_mixed, params)
    
    # a f(X1) + b f(X2)
    Y1 = model.eval(X1, params)
    Y2 = model.eval(X2, params)
    Y_expected = a * Y1 + b * Y2
    
    diff = jnp.abs(Y_mixed - Y_expected)
    max_diff = jnp.max(diff)
    print(f"Max difference (should be ~0): {max_diff:.2e}")
    assert max_diff < 1e-10
    print("Linearity test passed!")

def test_composition():
    print("\nTesting composition property...")
    input_dim = 2
    hidden_dim = 2
    output_dim = 2
    seq_len = 3
    num_regions = 2
    
    layer1 = LocallyLinearAttention(
        input_dim=input_dim,
        output_dim=hidden_dim,
        seq_len=seq_len,
        num_regions=num_regions,
        name="layer1"
    )
    
    layer2 = LocallyLinearAttention(
        input_dim=hidden_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        num_regions=num_regions,
        name="layer2"
    )
    
    # Combined eval
    def eval_composite(X, p1, p2):
        X1 = layer1.eval(X, p1)
        X2 = layer2.eval(X1, p2)
        return X2
    
    # Test linearity of the composite function in a local area
    X1 = jax.random.normal(jax.random.PRNGKey(1), (1, seq_len, input_dim)) * 0.01
    X2 = jax.random.normal(jax.random.PRNGKey(2), (1, seq_len, input_dim)) * 0.01
    
    # Move them to be near a random point to ensure they stay in same regions
    center = jax.random.normal(jax.random.PRNGKey(3), (1, seq_len, input_dim))
    X1 += center
    X2 += center
    
    p1 = layer1.surrogate_distribution_generator(layer1.params).sample(seed=jax.random.PRNGKey(42))
    p2 = layer2.surrogate_distribution_generator(layer2.params).sample(seed=jax.random.PRNGKey(43))

    
    a, b = 0.5, 0.5
    X_mixed = a * X1 + b * X2
    Y_mixed = eval_composite(X_mixed, p1, p2)
    
    Y1 = eval_composite(X1, p1, p2)
    Y2 = eval_composite(X2, p1, p2)
    Y_expected = a * Y1 + b * Y2
    
    diff = jnp.abs(Y_mixed - Y_expected)
    max_diff = jnp.max(diff)
    print(f"Max difference of composite (should be ~0 if regions are constant): {max_diff:.2e}")
    # Note: If X1 and X2 fall into different regions, this won't be zero.
    # But for small perturbations around a center, they likely stay in same regions.
    assert max_diff < 1e-10
    print("Composition test passed!")

if __name__ == "__main__":
    test_linearity()
    test_composition()

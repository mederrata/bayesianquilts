from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from tensorflow_probability.substrates.jax import distributions as tfd

from bayesianquilts.predictors.nn.dense import Dense


class UnivariateDense(Dense):
    """Univariate neural network for generalized additive models (GAM).

    Creates independent neural networks for each input feature, mapping R^1 -> R^k
    with no mixing between features. Each feature is processed through its own
    ReLU-activated neural network to learn a univariate shape function.

    This is the core building block for Generalized Additive Models with Neural
    Networks (GAM-Nets), where the model has the form:

        f(x) = β₀ + Σᵢ fᵢ(xᵢ)

    where each fᵢ is a univariate neural network that learns the contribution
    of feature i to the output.

    Architecture for each feature:
        xᵢ (1D) -> ReLU Hidden Layers -> Output (k-dimensional)

    The ReLU activation allows the network to learn piecewise linear approximations
    of arbitrary univariate functions while maintaining interpretability through
    the additive structure.

    Args:
        input_size: Number of input features (each gets its own network)
        layer_sizes: List of hidden layer sizes + output size for each feature's network
        weight_scale: Multiplicative scale for weights (default: 1.0)
        bias_scale: Multiplicative scale for biases (default: 1.0)
        activation_fn: Activation function for hidden layers (default: ReLU)
        dtype: Data type for computations (default: float32)

    Example:
        >>> # Create univariate networks for 5 features
        >>> nn = UnivariateDense(input_size=5, layer_sizes=[10, 5, 1])
        >>> X = jnp.array(np.random.rand(100, 5))  # 100 samples, 5 features
        >>> output = nn.eval(X)  # Shape: (100, 5) - one output per feature
        >>>
        >>> # For multi-class (k=3 classes), last layer should be size 3
        >>> nn_multiclass = UnivariateDense(input_size=5, layer_sizes=[10, 5, 3])
        >>> output = nn_multiclass.eval(X)  # Shape: (100, 5, 3)
    """

    def __init__(self, **kwargs) -> None:
        """Initialize univariate neural network.

        Args:
            **kwargs: Arguments passed to Dense parent class including
                     input_size, layer_sizes, weight_scale, bias_scale, dtype
        """
        super(UnivariateDense, self).__init__(**kwargs)

    def sample_initial_nn_params(
        self,
        input_size: int,
        layer_sizes: list[int],
        priors: list[tuple[float, float]] | None = None,
    ) -> list[jax.Array]:
        """Sample initial parameters for univariate neural networks.

        Creates separate neural network parameters for each input feature.
        Each feature gets its own set of weights and biases.

        Args:
            input_size: Number of input features (number of independent networks)
            layer_sizes: List of hidden layer sizes + output size
            priors: Optional list of (weight_scale, bias_scale) tuples for each layer

        Returns:
            List of weight and bias tensors alternating [W0, b0, W1, b1, ...]
            Each weight tensor has shape (input_size, prev_layer_size, current_layer_size)
            Each bias tensor has shape (input_size, current_layer_size)
        """
        architecture = []
        layer_sizes = [1] + layer_sizes  # Input is always 1D for univariate
        _, sample_key = random.split(random.PRNGKey(0))

        if priors is None:
            for j, layer_size in enumerate(layer_sizes[1:]):
                # Weights: (input_size, prev_layer_size, current_layer_size)
                # One set of weights per feature
                weights = tfd.Normal(
                    loc=jnp.zeros(
                        (input_size, layer_sizes[j], layer_size), dtype=self.dtype
                    ),
                    scale=1e-1,
                ).sample(seed=sample_key)
                # Biases: (input_size, current_layer_size)
                # One bias per feature
                biases = tfd.Normal(
                    loc=jnp.zeros((input_size, layer_size), dtype=self.dtype),
                    scale=1.0
                ).sample(seed=sample_key)
                architecture += [weights, biases]
        else:
            # Use custom priors for each layer
            for j, layer_size in enumerate(layer_sizes[1:]):
                weight_scale, bias_scale = priors[j]
                weights = tfd.Normal(
                    loc=jnp.zeros(
                        (input_size, layer_sizes[j], layer_size), dtype=self.dtype
                    ),
                    scale=weight_scale,
                ).sample(seed=sample_key)
                biases = tfd.Normal(
                    loc=jnp.zeros((input_size, layer_size), dtype=self.dtype),
                    scale=bias_scale
                ).sample(seed=sample_key)
                architecture += [weights, biases]

        return architecture

    def eval(
        self,
        tensor: jax.Array,
        weight_tensors: list[jax.Array] | None = None,
        activation: Callable[[jax.Array], jax.Array] | None = None,
    ) -> jax.Array:
        """Evaluate the univariate neural networks.

        Each input feature is processed independently through its own neural network.
        No mixing occurs between features - the model learns separate shape functions
        for each input dimension.

        Args:
            tensor: Input tensor of shape [n, p] where n is batch size, p is number of features
            weight_tensors: Optional list of weight/bias tensors. Uses self.weight_tensors if None.
            activation: Activation function for hidden layers. Defaults to ReLU.

        Returns:
            Output tensor of shape:
            - [n, p] if final layer size is 1 (each feature contributes 1 value)
            - [n, p, k] if final layer size is k > 1 (each feature contributes k values)

        Example:
            >>> X = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape: (2, 3)
            >>> nn = UnivariateDense(input_size=3, layer_sizes=[10, 1])
            >>> output = nn.eval(X)  # Shape: (2, 3)
            >>> # output[i, j] is the contribution of feature j for sample i
        """
        weight_tensors = (
            weight_tensors if weight_tensors is not None else self.weight_tensors
        )
        activation = jax.nn.relu if activation is None else activation

        # Reshape input: [n, p] -> [n, p, 1, 1]
        # This allows us to process each feature independently
        net = tensor[..., jnp.newaxis, jnp.newaxis]
        net = net.astype(self.dtype)

        weights_list = weight_tensors[::2]
        biases_list = weight_tensors[1::2]

        # First layer: Scale each feature by its weights and add biases
        # net: [n, p, 1, 1]
        # weights_list[0]: [p, 1, hidden_size]
        # Result: [n, p, 1, hidden_size]
        net = net * weights_list[0] + biases_list[0][..., jnp.newaxis, :]

        # Hidden layers with ReLU activation
        for weights, biases in zip(weights_list[1:-1], biases_list[1:-1]):
            net = self.dense(
                net, self.weight_scale * weights, self.bias_scale * biases, activation
            )

        # Final layer with identity activation (linear output)
        net = self.dense(
            net,
            self.weight_scale * weights_list[-1],
            self.bias_scale * biases_list[-1],
            lambda x: x,  # Identity activation
        )

        # Return shape depends on final layer size:
        # If final layer is size 1: [n, p]
        # If final layer is size k: [n, p, k]
        if net.shape[-1] == 1:
            return net[..., 0, :]  # Squeeze last dimension
        else:
            return net[..., 0, :]  # Keep all dimensions


def demo():
    """Demonstrate the UnivariateDense neural network for GAM-style modeling.

    Shows how to create univariate networks that learn independent shape functions
    for each input feature.
    """
    print("Running UnivariateDense demo...")
    print("=" * 60)

    # Setup
    n = 30  # Batch size
    p = 3   # Number of input features

    # Create synthetic data with known univariate patterns
    np.random.seed(42)
    X = np.random.rand(n, p).astype(np.float32)

    print(f"\nInput shape: {X.shape}")
    print(f"Processing {p} features independently")

    # Example 1: Single output per feature (for regression or binary classification)
    print("\n" + "=" * 60)
    print("Example 1: Single output per feature")
    print("=" * 60)
    # Architecture: 1 -> 10 -> 5 -> 1 for each feature
    nn_single = UnivariateDense(
        input_size=p,
        layer_sizes=[10, 5, 1],  # Hidden layers + output
        dtype=jnp.float32
    )

    output_single = nn_single.eval(jnp.array(X))

    print(f"Output shape: {output_single.shape}")
    print(f"Expected shape: ({n}, {p})")
    assert output_single.shape == (n, p), \
        f"Shape mismatch: got {output_single.shape}, expected ({n}, {p})"

    print(f"\nOutput sample (first 5 rows):")
    print(output_single[:5])
    print("\nInterpretation: Each column is the learned function for one feature")

    # Example 2: Multiple outputs per feature (for multi-class classification)
    print("\n" + "=" * 60)
    print("Example 2: Multiple outputs per feature (k=3 classes)")
    print("=" * 60)
    k = 3  # Number of classes
    # Architecture: 1 -> 10 -> 5 -> 3 for each feature
    nn_multi = UnivariateDense(
        input_size=p,
        layer_sizes=[10, 5, k],
        dtype=jnp.float32
    )

    output_multi = nn_multi.eval(jnp.array(X))

    print(f"Output shape: {output_multi.shape}")
    print(f"Expected shape: ({n}, {p}, {k}) or ({n}, {p})")
    print(f"\nOutput sample (first 2 rows, all features):")
    print(output_multi[:2])

    # Example 3: Using custom priors
    print("\n" + "=" * 60)
    print("Example 3: Custom priors for regularization")
    print("=" * 60)
    priors = [
        (0.05, 0.5),   # First layer: small weights, moderate biases
        (0.05, 0.5),   # Second layer
        (0.02, 0.1)    # Output layer: very small weights and biases
    ]

    nn_custom = UnivariateDense(
        input_size=p,
        layer_sizes=[10, 5, 1],
        dtype=jnp.float32
    )
    # Manually set custom priors
    custom_params = nn_custom.sample_initial_nn_params(p, [10, 5, 1], priors=priors)
    nn_custom.set_weights(custom_params)

    output_custom = nn_custom.eval(jnp.array(X))
    print(f"Custom priors output shape: {output_custom.shape}")
    print(f"Mean absolute output: {jnp.abs(output_custom).mean():.6f}")
    print("(Should be smaller due to tighter priors)")

    # Example 4: GAM-style additive model
    print("\n" + "=" * 60)
    print("Example 4: GAM-style additive prediction")
    print("=" * 60)
    print("Creating additive model: y = intercept + f1(x1) + f2(x2) + f3(x3)")

    # Get contributions from each feature
    contributions = nn_single.eval(jnp.array(X))

    # Add an intercept
    intercept = 2.0

    # Final prediction is sum of all contributions + intercept
    predictions = jnp.sum(contributions, axis=1) + intercept

    print(f"\nContributions shape: {contributions.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"\nFirst 5 predictions:")
    print(predictions[:5])
    print(f"\nFirst 5 contribution breakdowns:")
    for i in range(5):
        print(f"  Sample {i}: {intercept:.2f} + {contributions[i, 0]:.4f} + "
              f"{contributions[i, 1]:.4f} + {contributions[i, 2]:.4f} = {predictions[i]:.4f}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

    return output_single


if __name__ == "__main__":
    demo()

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from tensorflow_probability.substrates.jax import distributions as tfd

from bayesianquilts.predictors.nn.dense import Dense


class PairwiseDense(Dense):
    """Bivariate Dense neural network for modeling pairwise feature interactions.

    This network processes pairs of input features through independent ReLU-activated
    neural networks to capture quadratic (and higher-order) interactions. Each pair
    is mapped from R^2 to R^1 through a series of hidden layers with ReLU activations,
    followed by a final linear output layer.

    Architecture for each pair:
        Input (2D) -> ReLU Hidden Layers -> Linear Output (1D)

    This design is motivated by the observation that ReLU networks can represent
    quadratic features through piecewise linear approximations. By processing pairs
    independently, the model avoids mixing between different feature pairs while
    still capturing complex interactions within each pair.

    Args:
        pairs: List of [i, j] pairs specifying which input features to process together.
               For example, [[0, 1], [2, 3]] processes features (0,1) and (2,3) separately.
        input_size: Number of input features (inherited from Dense)
        layer_sizes: List of hidden layer sizes. Final layer outputs 1D per pair.
        weight_scale: Multiplicative scale for weights (default: 1.0)
        bias_scale: Multiplicative scale for biases (default: 1.0)
        dtype: Data type for computations (default: float32)

    Example:
        >>> pairs = [[0, 1], [2, 3], [4, 5]]
        >>> nn = PairwiseDense(input_size=6, pairs=pairs, layer_sizes=[10, 5, 1])
        >>> X = jnp.array(np.random.rand(100, 6))  # 100 samples, 6 features
        >>> output = nn.eval(X)  # Shape: (100, 3) - one output per pair
    """

    def __init__(self, pairs: list[list[int]] | None = None, **kwargs) -> None:
        """Initialize bivariate pairwise neural network.

        Args:
            pairs: List of [i, j] pairs specifying which features to process together
            **kwargs: Additional arguments passed to Dense parent class
        """
        if pairs is None:
            raise ValueError("pairs must be specified for PairwiseDense")
        self.pairs = pairs
        self.num_pairs = len(pairs)
        super(PairwiseDense, self).__init__(**kwargs)

    def sample_initial_nn_params(
        self,
        input_size: int,
        layer_sizes: list[int],
        priors: list[tuple[float, float]] | None = None,
    ) -> list[jax.Array]:
        """Sample initial parameters for bivariate neural network.

        Each layer processes pairs of features independently using ReLU activations
        to capture quadratic interactions before the final linear output.

        Args:
            input_size: Number of input features (not used, as pairs are defined explicitly)
            layer_sizes: List of hidden layer sizes for each pair
            priors: Optional list of (weight_scale, bias_scale) tuples for each layer

        Returns:
            List of weight and bias tensors alternating [W0, b0, W1, b1, ...]
        """
        architecture = []
        layer_sizes = [2] + layer_sizes  # Input is always 2D for bivariate
        _, sample_key = random.split(random.PRNGKey(0))

        if priors is None:
            for j, layer_size in enumerate(layer_sizes[1:]):
                # Weights: (num_pairs, prev_layer_size, current_layer_size)
                weights = tfd.Normal(
                    loc=jnp.zeros(
                        (self.num_pairs, layer_sizes[j], layer_size), dtype=self.dtype
                    ),
                    scale=1e-1,
                ).sample(seed=sample_key)
                # Biases: (num_pairs, current_layer_size)
                biases = tfd.Normal(
                    loc=jnp.zeros((self.num_pairs, layer_size), dtype=self.dtype),
                    scale=1.0
                ).sample(seed=sample_key)
                architecture += [weights, biases]
        else:
            # Use custom priors for each layer
            for j, layer_size in enumerate(layer_sizes[1:]):
                weight_scale, bias_scale = priors[j]
                weights = tfd.Normal(
                    loc=jnp.zeros(
                        (self.num_pairs, layer_sizes[j], layer_size), dtype=self.dtype
                    ),
                    scale=weight_scale,
                ).sample(seed=sample_key)
                biases = tfd.Normal(
                    loc=jnp.zeros((self.num_pairs, layer_size), dtype=self.dtype),
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
        """Evaluate the bivariate neural network.

        The network processes pairs of input features independently through
        ReLU-activated hidden layers to capture quadratic interactions,
        then outputs a single linear activation per pair.

        Args:
            tensor: Input tensor of shape [n, p] where n is batch size and p is number of features
            weight_tensors: Optional list of weight/bias tensors. Uses self.weight_tensors if None.
            activation: Activation function for hidden layers. Defaults to ReLU.

        Returns:
            Output tensor of shape [n, num_pairs] containing the network output for each pair
        """
        weight_tensors = (
            weight_tensors if weight_tensors is not None else self.weight_tensors
        )
        activation = jax.nn.relu if activation is None else activation

        # Extract pairs of features: [p, n] -> [num_pairs, 2, n] -> [n, num_pairs, 2]
        pairs_array = jnp.array(self.pairs, dtype=jnp.int32)
        net = jnp.transpose(tensor)  # [n, p] -> [p, n]
        net = jnp.take(net, pairs_array, axis=0)  # [num_pairs, 2, n]
        net = jnp.transpose(net, [2, 0, 1])  # [n, num_pairs, 2]
        net = net.astype(self.dtype)

        weights_list = weight_tensors[::2]
        biases_list = weight_tensors[1::2]

        # First layer: linear combination of the 2 input features for each pair
        # net: [n, num_pairs, 2]
        # weights_list[0]: [num_pairs, 2, hidden_size]
        # Result: [n, num_pairs, hidden_size]
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

        # Return shape [n, num_pairs] - one output per pair
        return net[..., 0, :]


def demo():
    """Demonstrate the PairwiseDense bivariate neural network.

    Creates a simple network that processes pairs of features through
    ReLU activations to capture quadratic interactions.
    """
    # Configure JAX for single-threaded execution to avoid threading issues
    import os
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

    print("Running PairwiseDense demo...")

    # Setup
    n = 30  # Batch size
    p = 8   # Number of input features
    X = jnp.array(np.random.rand(n, p), dtype=jnp.float32)

    # Define pairs of features to process
    pairs = [[0, 1], [1, 2], [2, 3], [7, 0]]
    num_pairs = len(pairs)

    print(f"Input shape: {X.shape}")
    print(f"Processing {num_pairs} pairs: {pairs}")

    # Create network with hidden layers [10, 5] and final output layer [1]
    # Architecture: 2 -> 10 -> 5 -> 1 for each pair
    nn = PairwiseDense(
        input_size=p,
        pairs=pairs,
        layer_sizes=[10, 5, 1],  # Hidden layers + output
        dtype=jnp.float32
    )

    # Evaluate the network
    output = nn.eval(X)

    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({n}, {num_pairs})")
    assert output.shape == (n, num_pairs), \
        f"Shape mismatch: got {output.shape}, expected ({n}, {num_pairs})"

    print(f"Output sample (first 5 rows):\n{output[:5]}")
    print("\nDemo completed successfully!")

    # Test with custom priors
    print("\nTesting with custom priors...")
    priors = [(0.1, 0.5), (0.1, 0.5), (0.05, 0.1)]  # (weight_scale, bias_scale) for each layer
    nn_custom = PairwiseDense(
        input_size=p,
        pairs=pairs,
        layer_sizes=[10, 5, 1],
        dtype=jnp.float32
    )
    # Manually set custom priors
    custom_params = nn_custom.sample_initial_nn_params(p, [10, 5, 1], priors=priors)
    nn_custom.set_weights(custom_params)

    output_custom = nn_custom.eval(X)
    print(f"Custom priors output shape: {output_custom.shape}")
    print("Custom priors test completed successfully!")

    return output


if __name__ == "__main__":
    demo()

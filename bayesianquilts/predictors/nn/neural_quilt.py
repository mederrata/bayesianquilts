#!/usr/bin/env python3
"""Neural Quilt - Neural network with discretely indexed parameter decomposition.

This module implements neural networks where weights and biases vary according
to discrete indices (categorical variables). Similar to how LogisticBayesianquilt
applies parameter decomposition to regression coefficients, NeuralQuilt applies
it to neural network weights.

The key idea is that different groups (defined by discrete indices) can have
different network weights, with hierarchical structure enabling sharing of
information across groups through the parameter decomposition.
"""

from collections import defaultdict
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd

from bayesianquilts.jax.parameter import Decomposed, Interactions
from bayesianquilts.model import BayesianModel
from bayesianquilts.predictors.nn.dense import Dense
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator

jax.config.update("jax_enable_x64", True)


class NeuralQuilt(BayesianModel):
    """Neural network with hierarchically decomposed weights.

    This class implements a neural network where weights and biases are
    decomposed according to discrete interaction structure. Each group
    (defined by categorical variables) has its own effective weights that
    are composed hierarchically from global, group-level, and local components.

    The decomposition for each layer's weights follows:
        W_effective[group] = W_global + W_group_level + ... + W_local[group]

    This enables:
    - Information sharing across groups through hierarchical structure
    - Automatic regularization via the decomposition
    - Interpretable group-specific adaptations

    Args:
        input_size: Dimension of input features
        layer_sizes: List of hidden layer sizes + output size
        weight_interactions: Interactions object defining weight decomposition structure
        bias_interactions: Interactions object defining bias decomposition structure
        activation_fn: Activation function for hidden layers (default: ReLU)
        weight_scale: Global scale for weights
        bias_scale: Global scale for biases
        dim_decay_factor: Decay factor for interaction order regularization
        weight_prior_scale: Scale for weight priors
        bias_prior_scale: Scale for bias priors
        dtype: Data type for computations
        outcome_label: Key for outcome variable in data dict
        initialize_distributions: Whether to create distributions on init

    Example:
        >>> # Define interaction structure for 2 categorical variables
        >>> interactions = Interactions(
        ...     dimensions=[("group", 5), ("subgroup", 3)],
        ...     exclusions=[]
        ... )
        >>> # Create neural quilt with 2 hidden layers
        >>> model = NeuralQuilt(
        ...     input_size=10,
        ...     layer_sizes=[20, 10, 1],
        ...     weight_interactions=interactions,
        ...     bias_interactions=interactions
        ... )
        >>> # Fit to data with discrete indices
        >>> data = {
        ...     'X': jnp.array(...),  # Input features
        ...     'y': jnp.array(...),  # Outputs
        ...     'group': jnp.array(...),  # Group indices
        ...     'subgroup': jnp.array(...)  # Subgroup indices
        ... }
    """

    def __init__(
        self,
        input_size: int,
        layer_sizes: list[int],
        weight_interactions: Interactions,
        bias_interactions: Interactions,
        activation_fn: Callable[[jax.Array], jax.Array] | None = None,
        weight_scale: float = 1.0,
        bias_scale: float = 1.0,
        dim_decay_factor: float = 0.9,
        weight_prior_scale: float = 0.1,
        bias_prior_scale: float = 1.0,
        dtype: jnp.dtype = jnp.float64,
        outcome_label: str = "y",
        initialize_distributions: bool = True,
        strategy: str | None = None,
    ):
        super(NeuralQuilt, self).__init__(dtype=dtype, strategy=strategy)

        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.weight_scale = weight_scale
        self.bias_scale = bias_scale
        self.dim_decay_factor = dim_decay_factor
        self.weight_prior_scale = weight_prior_scale
        self.bias_prior_scale = bias_prior_scale
        self.outcome_label = outcome_label
        self.activation_fn = activation_fn if activation_fn is not None else jax.nn.relu

        # Create base neural network for structure
        self.base_nn = Dense(
            input_size=input_size,
            layer_sizes=layer_sizes,
            weight_scale=weight_scale,
            bias_scale=bias_scale,
            activation_fn=self.activation_fn,
            dtype=dtype,
        )

        # Create decompositions for each layer's weights and biases
        self.weight_decompositions = []
        self.bias_decompositions = []
        self.weight_interactions = weight_interactions
        self.bias_interactions = bias_interactions

        # Build layer sizes including input
        full_layer_sizes = [input_size] + layer_sizes

        for layer_idx in range(len(layer_sizes)):
            # Weight decomposition for this layer
            weight_shape = [full_layer_sizes[layer_idx], full_layer_sizes[layer_idx + 1]]
            weight_decomp = Decomposed(
                interactions=weight_interactions,
                param_shape=weight_shape,
                name=f"weight_{layer_idx}",
                dtype=dtype,
            )
            self.weight_decompositions.append(weight_decomp)

            # Bias decomposition for this layer
            bias_shape = [full_layer_sizes[layer_idx + 1]]
            bias_decomp = Decomposed(
                interactions=bias_interactions,
                param_shape=bias_shape,
                name=f"bias_{layer_idx}",
                dtype=dtype,
            )
            self.bias_decompositions.append(bias_decomp)

        if initialize_distributions:
            self.create_distributions()

    def create_distributions(self):
        """Create prior and surrogate distributions for all parameters."""
        distribution_dict = {}
        self.var_lists = {}  # Track variables for each layer

        # Create distributions for each layer
        for layer_idx, (weight_decomp, bias_decomp) in enumerate(
            zip(self.weight_decompositions, self.bias_decompositions)
        ):
            # Weight distributions
            (
                weight_tensors,
                weight_vars,
                weight_shapes,
            ) = weight_decomp.generate_tensors(dtype=self.dtype)

            # Compute scales with dimensional decay
            weight_scales = {
                k: (
                    self.weight_prior_scale
                    * self.dim_decay_factor ** (len([d for d in v if d > 1]) - 1)
                )
                for k, v in weight_shapes.items()
            }

            self.var_lists[f"weight_{layer_idx}"] = list(weight_vars.keys())

            for label, tensor in weight_tensors.items():
                distribution_dict[label] = tfd.Independent(
                    tfd.Normal(
                        loc=jnp.zeros_like(jnp.asarray(tensor, self.dtype)),
                        scale=weight_scales[label]
                        * jnp.ones_like(jnp.asarray(tensor, self.dtype)),
                    ),
                    reinterpreted_batch_ndims=len(tensor.shape),
                )

            # Bias distributions
            (
                bias_tensors,
                bias_vars,
                bias_shapes,
            ) = bias_decomp.generate_tensors(dtype=self.dtype)

            bias_scales = {
                k: (
                    self.bias_prior_scale
                    * self.dim_decay_factor ** (len([d for d in v if d > 1]) - 1)
                )
                for k, v in bias_shapes.items()
            }

            self.var_lists[f"bias_{layer_idx}"] = list(bias_vars.keys())

            for label, tensor in bias_tensors.items():
                distribution_dict[label] = tfd.Independent(
                    tfd.Normal(
                        loc=jnp.zeros_like(jnp.asarray(tensor, self.dtype)),
                        scale=bias_scales[label]
                        * jnp.ones_like(jnp.asarray(tensor, self.dtype)),
                    ),
                    reinterpreted_batch_ndims=len(tensor.shape),
                )

        # Create joint prior distribution
        self.prior_distribution = tfd.JointDistributionNamed(distribution_dict)

        # Create surrogate distribution
        self.surrogate_distribution_generator, self.surrogate_parameter_initializer = (
            build_factored_surrogate_posterior_generator(
                self.prior_distribution, dtype=self.dtype
            )
        )

        self.params = self.surrogate_parameter_initializer()
        self.var_list = list(self.params.keys())

    def get_layer_weights(self, layer_idx: int, indices: jax.Array, params: dict) -> tuple:
        """Get effective weights and biases for a specific layer and indices.

        Args:
            layer_idx: Layer index
            indices: Discrete indices for lookups
            params: Parameter dictionary

        Returns:
            Tuple of (weights, biases) for the specified layer and indices
        """
        weight_decomp = self.weight_decompositions[layer_idx]
        bias_decomp = self.bias_decompositions[layer_idx]

        # Get weight variables for this layer
        weight_var_list = self.var_lists[f"weight_{layer_idx}"]
        weight_params = {k: params[k] for k in weight_var_list}

        # Get bias variables for this layer
        bias_var_list = self.var_lists[f"bias_{layer_idx}"]
        bias_params = {k: params[k] for k in bias_var_list}

        # Lookup decomposed weights and biases
        weights = weight_decomp.lookup(indices, tensors=weight_params)
        biases = bias_decomp.lookup(indices, tensors=bias_params)

        return weights, biases

    def eval(
        self,
        X: jax.Array,
        indices: jax.Array,
        params: dict,
        activation: Callable[[jax.Array], jax.Array] | None = None,
    ) -> jax.Array:
        """Evaluate neural network with group-specific weights.

        Args:
            X: Input features of shape [batch_size, input_dim]
            indices: Discrete indices of shape [batch_size, num_interaction_dims]
            params: Parameter dictionary
            activation: Activation function (defaults to self.activation_fn)

        Returns:
            Network output of shape [batch_size, output_dim]
        """
        activation = activation if activation is not None else self.activation_fn

        # Forward pass through layers
        net = X.astype(self.dtype)

        for layer_idx in range(len(self.layer_sizes)):
            # Get layer-specific weights for each data point's group
            weights, biases = self.get_layer_weights(layer_idx, indices, params)

            # Apply dense layer
            # weights: [batch_size, input_dim, output_dim]
            # net: [batch_size, input_dim]
            # Need to do batch-wise matrix multiplication
            net = jnp.einsum('bi,bio->bo', net, weights) + biases

            # Apply activation (except for last layer)
            if layer_idx < len(self.layer_sizes) - 1:
                net = activation(net)

        return net

    def predictive_distribution(self, data: dict, **params) -> dict:
        """Compute predictive distribution.

        Args:
            data: Dictionary containing:
                - 'X': Input features
                - outcome_label: Target outputs
                - Discrete index variables (as specified in interactions)
            **params: Model parameters

        Returns:
            Dictionary containing:
                - 'prediction': Predictive distribution
                - 'log_likelihood': Log likelihood values
                - 'outputs': Network outputs
        """
        X = data["X"]
        y = data[self.outcome_label]

        # Retrieve indices for decomposition lookup
        indices = self.weight_decompositions[0].retrieve_indices(data)

        # Get network outputs
        outputs = self.eval(X, indices, params)

        # Compute log likelihood (assuming Gaussian for regression)
        # For classification, this should be modified
        rv_outcome = tfd.Normal(loc=outputs, scale=1.0)
        log_likelihood = rv_outcome.log_prob(jnp.squeeze(y))

        return {
            "log_likelihood": log_likelihood,
            "prediction": rv_outcome,
            "outputs": outputs,
        }

    def log_likelihood(self, data: dict, **params) -> jax.Array:
        """Compute log likelihood.

        Args:
            data: Data dictionary
            **params: Model parameters

        Returns:
            Log likelihood values
        """
        return self.predictive_distribution(data, **params)["log_likelihood"]

    def unormalized_log_prob(
        self, data: dict | None = None, prior_weight: float = 1.0, **params
    ) -> jax.Array:
        """Compute unnormalized log probability (log likelihood + log prior).

        Args:
            data: Data dictionary
            prior_weight: Weight for prior term
            **params: Model parameters

        Returns:
            Unnormalized log probability
        """
        # Compute log likelihood
        prediction = self.predictive_distribution(data, **params)
        log_likelihood = prediction["log_likelihood"]

        # Handle infinite values
        max_val = jnp.max(log_likelihood)
        finite_portion = jnp.where(
            jnp.isfinite(log_likelihood),
            log_likelihood,
            jnp.zeros_like(log_likelihood),
        )
        min_val = jnp.min(finite_portion) - 1.0
        log_likelihood = jnp.where(
            jnp.isfinite(log_likelihood),
            log_likelihood,
            jnp.ones_like(log_likelihood) * min_val,
        )

        # Compute log prior
        prior = self.prior_distribution.log_prob(
            {k: jnp.asarray(params[k], self.dtype) for k in self.var_list}
        )

        prior_weight = jnp.asarray(prior_weight, self.dtype)

        # Combine
        energy = jnp.sum(log_likelihood, axis=-1) + prior_weight * prior

        return energy


def demo():
    """Demonstrate the NeuralQuilt implementation."""
    print("Running NeuralQuilt demo...")
    print("=" * 60)

    # Setup synthetic data
    np.random.seed(42)
    n_samples = 100
    input_dim = 5
    n_groups = 3
    n_subgroups = 2

    # Create synthetic data
    X = np.random.randn(n_samples, input_dim).astype(np.float64)
    group_ids = np.random.randint(0, n_groups, n_samples)
    subgroup_ids = np.random.randint(0, n_subgroups, n_samples)

    # Generate synthetic outputs with group structure
    true_effects = np.random.randn(n_groups, n_subgroups)
    y = true_effects[group_ids, subgroup_ids][:, np.newaxis] + 0.1 * np.random.randn(
        n_samples, 1
    )

    data = {
        "X": jnp.array(X),
        "y": jnp.array(y),
        "group": jnp.array(group_ids),
        "subgroup": jnp.array(subgroup_ids),
    }

    print(f"Data shapes:")
    print(f"  X: {data['X'].shape}")
    print(f"  y: {data['y'].shape}")
    print(f"  Groups: {n_groups}, Subgroups: {n_subgroups}")

    # Create interaction structure
    print("\nCreating interaction structure...")
    interactions = Interactions(
        dimensions=[("group", n_groups), ("subgroup", n_subgroups)], exclusions=[]
    )

    print(f"Interaction shape: {interactions.shape()}")

    # Create model
    print("\nCreating NeuralQuilt model...")
    model = NeuralQuilt(
        input_size=input_dim,
        layer_sizes=[10, 5, 1],  # 2 hidden layers + output
        weight_interactions=interactions,
        bias_interactions=interactions,
        weight_prior_scale=0.1,
        bias_prior_scale=1.0,
    )

    print(f"Number of layers: {len(model.layer_sizes)}")
    print(f"Number of parameters: {len(model.params)}")
    print(f"Parameter keys: {list(model.params.keys())[:5]}...")

    # Test forward pass
    print("\n" + "=" * 60)
    print("Testing forward pass...")
    indices = model.weight_decompositions[0].retrieve_indices(data)
    print(f"Indices shape: {indices.shape}")

    outputs = model.eval(data["X"], indices, model.params)
    print(f"Output shape: {outputs.shape}")
    print(f"Output sample:\n{outputs[:5]}")

    # Test predictive distribution
    print("\n" + "=" * 60)
    print("Testing predictive distribution...")
    pred = model.predictive_distribution(data, **model.params)
    print(f"Log likelihood shape: {pred['log_likelihood'].shape}")
    print(f"Mean log likelihood: {jnp.mean(pred['log_likelihood']):.4f}")

    # Test log prob
    print("\n" + "=" * 60)
    print("Testing unnormalized log probability...")
    log_prob = model.unormalized_log_prob(data, **model.params)
    print(f"Log prob: {log_prob:.4f}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()

#!/usr/bin/env python3
"""Logistic Generalized Additive Models with Neural Networks (GAM-Nets).

This module implements GAM-style models for classification where each feature
contributes through its own neural network. The model has the additive form:

    logit(p) = intercept + f₁(x₁) + f₂(x₂) + ... + fₚ(xₚ) + g₁₂(x₁,x₂) + ...

where:
- fᵢ are univariate neural networks (one per feature)
- gᵢⱼ are pairwise neural networks (for feature interactions)

This maintains interpretability while using neural networks to learn complex
univariate and bivariate shape functions.
"""

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd

from bayesianquilts.model import BayesianModel
from bayesianquilts.predictors.nn.bivariate import PairwiseDense
from bayesianquilts.predictors.nn.univariate import UnivariateDense
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator

jax.config.update("jax_enable_x64", True)


class GamiNetUnivariate(BayesianModel):
    """Generalized Additive Model with Univariate Neural Networks.

    Implements logistic regression where each feature contributes through
    its own neural network:

        logit(p) = intercept + Σᵢ fᵢ(xᵢ)

    Each fᵢ is a univariate neural network that learns the shape function
    for feature i. The additive structure maintains interpretability while
    allowing for complex non-linear feature effects.

    Args:
        input_size: Number of input features
        layer_sizes: List of hidden layer sizes for each univariate network
        outcome_classes: Number of outcome classes
        outcome_label: Key for outcome in data dictionary
        weight_scale: Scale for network weights
        bias_scale: Scale for network biases
        dtype: Data type for computations

    Example:
        >>> model = GamiNetUnivariate(
        ...     input_size=10,
        ...     layer_sizes=[20, 10, 1],  # Each feature gets this architecture
        ...     outcome_classes=2
        ... )
        >>> # Each of 10 features has independent network: x -> 20 -> 10 -> 1
    """

    def __init__(
        self,
        input_size: int,
        layer_sizes: list[int],
        outcome_classes: int = 2,
        outcome_label: str = "y",
        weight_scale: float = 1.0,
        bias_scale: float = 1.0,
        dtype: jnp.dtype = jnp.float64,
        **kwargs,
    ):
        super(GamiNetUnivariate, self).__init__(dtype=dtype, **kwargs)

        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.outcome_classes = outcome_classes
        self.outcome_label = outcome_label
        self.weight_scale = weight_scale
        self.bias_scale = bias_scale

        # Ensure output is correct size for classification
        if layer_sizes[-1] != outcome_classes - 1:
            layer_sizes = layer_sizes + [outcome_classes - 1]

        # Create univariate neural network
        self.univariate_nn = UnivariateDense(
            input_size=input_size,
            layer_sizes=layer_sizes,
            weight_scale=weight_scale,
            bias_scale=bias_scale,
            dtype=dtype,
        )

        # Create intercept parameter
        self.intercept_shape = [outcome_classes - 1]

        self.create_distributions()

    def create_distributions(self):
        """Create prior and surrogate distributions for parameters."""
        distribution_dict = {}
        bijectors = {}

        # Add univariate network parameters
        for j, weight in enumerate(self.univariate_nn.weight_tensors[::2]):
            # Weights for layer j
            distribution_dict[f"w_{j}"] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros_like(weight, dtype=self.dtype),
                    scale=jnp.ones_like(weight, dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=len(weight.shape),
            )
            bijectors[f"w_{j}"] = tfp.bijectors.Identity()

            # Biases for layer j
            bias = self.univariate_nn.weight_tensors[2 * j + 1]
            distribution_dict[f"b_{j}"] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros_like(bias, dtype=self.dtype),
                    scale=jnp.ones_like(bias, dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=len(bias.shape),
            )
            bijectors[f"b_{j}"] = tfp.bijectors.Identity()

        # Add intercept
        distribution_dict["intercept"] = tfd.Independent(
            tfd.Normal(
                loc=jnp.zeros(self.intercept_shape, dtype=self.dtype),
                scale=10.0 * jnp.ones(self.intercept_shape, dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=1,
        )
        bijectors["intercept"] = tfp.bijectors.Identity()

        # Create joint distribution
        self.prior_distribution = tfd.JointDistributionNamed(distribution_dict)

        # Create surrogate
        self.surrogate_distribution_generator, self.surrogate_parameter_initializer = (
            build_factored_surrogate_posterior_generator(
                self.prior_distribution, bijectors=bijectors, dtype=self.dtype
            )
        )

        self.params = self.surrogate_parameter_initializer()
        self.var_list = list(self.params.keys())

    def eval(self, X: jax.Array, params: dict) -> jax.Array:
        """Evaluate the GAM-Net.

        Args:
            X: Input features of shape [n, p]
            params: Parameter dictionary

        Returns:
            Logits of shape [n, num_classes-1]
        """
        # Extract network weights
        weight_tensors = []
        num_layers = len(self.layer_sizes)
        for j in range(num_layers):
            weight_tensors.append(params[f"w_{j}"])
            weight_tensors.append(params[f"b_{j}"])

        # Evaluate univariate networks
        # Returns shape [n, p, num_classes-1] or [n, p] if last layer is size 1
        univariate_contributions = self.univariate_nn.eval(
            jnp.asarray(X, dtype=self.dtype), weight_tensors=weight_tensors
        )

        # Sum contributions across features
        if len(univariate_contributions.shape) == 2:
            # Shape [n, p] -> [n]
            logits = jnp.sum(univariate_contributions, axis=1, keepdims=True)
        else:
            # Shape [n, p, k] -> [n, k]
            logits = jnp.sum(univariate_contributions, axis=1)

        # Add intercept
        logits = logits + params["intercept"]

        return logits

    def predictive_distribution(
        self, data: dict[str, jax.Array], **params
    ) -> dict[str, jax.Array]:
        """Compute predictive distribution.

        Args:
            data: Dictionary with 'X' (features) and outcome_label (targets)
            **params: Model parameters

        Returns:
            Dictionary with log_likelihood, prediction, and logits
        """
        X = jnp.asarray(data["X"], dtype=self.dtype)

        # Get logits from GAM-Net
        logits = self.eval(X, params)

        # Pad logits for reference class (class 0)
        logits = jnp.pad(logits, [(0, 0)] * (len(logits.shape) - 1) + [(1, 0)])

        # Create categorical distribution
        rv_outcome = tfd.Categorical(logits=logits)

        # Compute log likelihood
        y = jnp.squeeze(data[self.outcome_label])
        log_likelihood = rv_outcome.log_prob(y)

        return {
            "log_likelihood": log_likelihood,
            "prediction": rv_outcome,
            "logits": logits,
        }

    def log_likelihood(self, data: dict[str, jax.Array], **params) -> jax.Array:
        """Compute log likelihood."""
        return self.predictive_distribution(data, **params)["log_likelihood"]

    def unormalized_log_prob(
        self,
        data: dict[str, jax.Array] | None = None,
        prior_weight: float = 1.0,
        **params,
    ) -> jax.Array:
        """Compute unnormalized log probability (log likelihood + log prior)."""
        # Compute log likelihood
        log_likelihood = self.log_likelihood(data, **params)

        # Compute log prior
        prior = self.prior_distribution.log_prob(
            {k: jnp.asarray(params[k], self.dtype) for k in self.var_list}
        )

        # Combine
        return jnp.sum(log_likelihood, axis=-1) + prior_weight * prior


class GamiNetPairwise(BayesianModel):
    """Generalized Additive Model with Univariate and Pairwise Neural Networks.

    Extends GamiNetUnivariate by adding pairwise interaction terms:

        logit(p) = intercept + Σᵢ fᵢ(xᵢ) + Σ_{i<j} gᵢⱼ(xᵢ, xⱼ)

    Each fᵢ is a univariate network and each gᵢⱼ is a bivariate network
    capturing interactions between features i and j.

    Args:
        input_size: Number of input features
        univariate_layer_sizes: Layer sizes for univariate networks
        pairwise_layer_sizes: Layer sizes for pairwise networks
        pairs: List of [i, j] pairs for pairwise interactions
        outcome_classes: Number of outcome classes
        outcome_label: Key for outcome in data dictionary
        dtype: Data type for computations

    Example:
        >>> model = GamiNetPairwise(
        ...     input_size=5,
        ...     univariate_layer_sizes=[10, 5, 1],
        ...     pairwise_layer_sizes=[8, 4, 1],
        ...     pairs=[[0, 1], [2, 3]],  # Interactions between (x0,x1) and (x2,x3)
        ...     outcome_classes=2
        ... )
    """

    def __init__(
        self,
        input_size: int,
        univariate_layer_sizes: list[int],
        pairwise_layer_sizes: list[int],
        pairs: list[list[int]],
        outcome_classes: int = 2,
        outcome_label: str = "y",
        weight_scale: float = 1.0,
        bias_scale: float = 1.0,
        dtype: jnp.dtype = jnp.float64,
        **kwargs,
    ):
        super(GamiNetPairwise, self).__init__(dtype=dtype, **kwargs)

        self.input_size = input_size
        self.univariate_layer_sizes = univariate_layer_sizes
        self.pairwise_layer_sizes = pairwise_layer_sizes
        self.pairs = pairs
        self.outcome_classes = outcome_classes
        self.outcome_label = outcome_label
        self.weight_scale = weight_scale
        self.bias_scale = bias_scale

        # Ensure outputs are correct size
        if univariate_layer_sizes[-1] != outcome_classes - 1:
            univariate_layer_sizes = univariate_layer_sizes + [outcome_classes - 1]
        if pairwise_layer_sizes[-1] != outcome_classes - 1:
            pairwise_layer_sizes = pairwise_layer_sizes + [outcome_classes - 1]

        # Create univariate networks
        self.univariate_nn = UnivariateDense(
            input_size=input_size,
            layer_sizes=univariate_layer_sizes,
            weight_scale=weight_scale,
            bias_scale=bias_scale,
            dtype=dtype,
        )

        # Create pairwise networks
        self.pairwise_nn = PairwiseDense(
            input_size=input_size,
            pairs=pairs,
            layer_sizes=pairwise_layer_sizes,
            weight_scale=weight_scale,
            bias_scale=bias_scale,
            dtype=dtype,
        )

        self.intercept_shape = [outcome_classes - 1]

        self.create_distributions()

    def create_distributions(self):
        """Create prior and surrogate distributions for all parameters."""
        distribution_dict = {}
        bijectors = {}

        # Univariate network parameters
        for j, weight in enumerate(self.univariate_nn.weight_tensors[::2]):
            distribution_dict[f"uni_w_{j}"] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros_like(weight, dtype=self.dtype),
                    scale=jnp.ones_like(weight, dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=len(weight.shape),
            )
            bijectors[f"uni_w_{j}"] = tfp.bijectors.Identity()

            bias = self.univariate_nn.weight_tensors[2 * j + 1]
            distribution_dict[f"uni_b_{j}"] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros_like(bias, dtype=self.dtype),
                    scale=jnp.ones_like(bias, dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=len(bias.shape),
            )
            bijectors[f"uni_b_{j}"] = tfp.bijectors.Identity()

        # Pairwise network parameters
        for j, weight in enumerate(self.pairwise_nn.weight_tensors[::2]):
            distribution_dict[f"pair_w_{j}"] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros_like(weight, dtype=self.dtype),
                    scale=jnp.ones_like(weight, dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=len(weight.shape),
            )
            bijectors[f"pair_w_{j}"] = tfp.bijectors.Identity()

            bias = self.pairwise_nn.weight_tensors[2 * j + 1]
            distribution_dict[f"pair_b_{j}"] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros_like(bias, dtype=self.dtype),
                    scale=jnp.ones_like(bias, dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=len(bias.shape),
            )
            bijectors[f"pair_b_{j}"] = tfp.bijectors.Identity()

        # Intercept
        distribution_dict["intercept"] = tfd.Independent(
            tfd.Normal(
                loc=jnp.zeros(self.intercept_shape, dtype=self.dtype),
                scale=10.0 * jnp.ones(self.intercept_shape, dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=1,
        )
        bijectors["intercept"] = tfp.bijectors.Identity()

        # Create joint distribution
        self.prior_distribution = tfd.JointDistributionNamed(distribution_dict)

        # Create surrogate
        self.surrogate_distribution_generator, self.surrogate_parameter_initializer = (
            build_factored_surrogate_posterior_generator(
                self.prior_distribution, bijectors=bijectors, dtype=self.dtype
            )
        )

        self.params = self.surrogate_parameter_initializer()
        self.var_list = list(self.params.keys())

    def eval(self, X: jax.Array, params: dict) -> jax.Array:
        """Evaluate the GAM-Net with pairwise interactions.

        Args:
            X: Input features of shape [n, p]
            params: Parameter dictionary

        Returns:
            Logits of shape [n, num_classes-1]
        """
        X = jnp.asarray(X, dtype=self.dtype)

        # Univariate contributions
        uni_weights = []
        for j in range(len(self.univariate_layer_sizes)):
            uni_weights.append(params[f"uni_w_{j}"])
            uni_weights.append(params[f"uni_b_{j}"])

        univariate_contrib = self.univariate_nn.eval(X, weight_tensors=uni_weights)

        # Sum over features
        if len(univariate_contrib.shape) == 2:
            univariate_sum = jnp.sum(univariate_contrib, axis=1, keepdims=True)
        else:
            univariate_sum = jnp.sum(univariate_contrib, axis=1)

        # Pairwise contributions
        pair_weights = []
        for j in range(len(self.pairwise_layer_sizes)):
            pair_weights.append(params[f"pair_w_{j}"])
            pair_weights.append(params[f"pair_b_{j}"])

        pairwise_contrib = self.pairwise_nn.eval(X, weight_tensors=pair_weights)

        # Sum over pairs
        if len(pairwise_contrib.shape) == 2:
            pairwise_sum = jnp.sum(pairwise_contrib, axis=1, keepdims=True)
        else:
            pairwise_sum = jnp.sum(pairwise_contrib, axis=1)

        # Combine all contributions
        logits = univariate_sum + pairwise_sum + params["intercept"]

        return logits

    def predictive_distribution(
        self, data: dict[str, jax.Array], **params
    ) -> dict[str, jax.Array]:
        """Compute predictive distribution."""
        X = jnp.asarray(data["X"], dtype=self.dtype)

        # Get logits
        logits = self.eval(X, params)

        # Pad for reference class
        logits = jnp.pad(logits, [(0, 0)] * (len(logits.shape) - 1) + [(1, 0)])

        # Create distribution
        rv_outcome = tfd.Categorical(logits=logits)

        # Compute log likelihood
        y = jnp.squeeze(data[self.outcome_label])
        log_likelihood = rv_outcome.log_prob(y)

        return {
            "log_likelihood": log_likelihood,
            "prediction": rv_outcome,
            "logits": logits,
        }

    def log_likelihood(self, data: dict[str, jax.Array], **params) -> jax.Array:
        """Compute log likelihood."""
        return self.predictive_distribution(data, **params)["log_likelihood"]

    def unormalized_log_prob(
        self,
        data: dict[str, jax.Array] | None = None,
        prior_weight: float = 1.0,
        **params,
    ) -> jax.Array:
        """Compute unnormalized log probability."""
        log_likelihood = self.log_likelihood(data, **params)
        prior = self.prior_distribution.log_prob(
            {k: jnp.asarray(params[k], self.dtype) for k in self.var_list}
        )
        return jnp.sum(log_likelihood, axis=-1) + prior_weight * prior


def demo():
    """Demonstrate GamiNet models."""
    print("Running GamiNet demo...")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    # Create non-linear effects
    y_prob = (
        1 / (1 + jnp.exp(-(jnp.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.3 * X[:, 2])))
    )
    y = (np.random.rand(n_samples) < y_prob).astype(np.int32)

    data = {"X": jnp.array(X), "y": jnp.array(y)}

    print(f"Data shapes: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {jnp.bincount(jnp.array(y))}")

    # Test GamiNetUnivariate
    print("\n" + "=" * 60)
    print("Testing GamiNetUnivariate...")
    model_uni = GamiNetUnivariate(
        input_size=n_features, layer_sizes=[10, 5, 1], outcome_classes=2
    )

    print(f"Parameters: {list(model_uni.params.keys())}")
    pred_uni = model_uni.predictive_distribution(data, **model_uni.params)
    print(f"Log likelihood shape: {pred_uni['log_likelihood'].shape}")
    print(f"Mean log likelihood: {jnp.mean(pred_uni['log_likelihood']):.4f}")

    # Test GamiNetPairwise
    print("\n" + "=" * 60)
    print("Testing GamiNetPairwise...")
    pairs = [[0, 1], [2, 3]]
    model_pair = GamiNetPairwise(
        input_size=n_features,
        univariate_layer_sizes=[8, 4, 1],
        pairwise_layer_sizes=[6, 3, 1],
        pairs=pairs,
        outcome_classes=2,
    )

    print(f"Number of parameters: {len(model_pair.params)}")
    print(f"Pairs modeled: {pairs}")
    pred_pair = model_pair.predictive_distribution(data, **model_pair.params)
    print(f"Log likelihood shape: {pred_pair['log_likelihood'].shape}")
    print(f"Mean log likelihood: {jnp.mean(pred_pair['log_likelihood']):.4f}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    import numpy as np

    demo()

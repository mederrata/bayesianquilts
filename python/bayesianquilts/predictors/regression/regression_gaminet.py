#!/usr/bin/env python3
"""Regression Generalized Additive Models with Neural Networks (GAM-Nets).

This module implements GAM-style models for regression where each feature
contributes through its own neural network. The model has the additive form:

    y = intercept + f₁(x₁) + f₂(x₂) + ... + fₚ(xₚ) + g₁₂(x₁,x₂) + ... + ε

where:
- fᵢ are univariate neural networks (one per feature)
- gᵢⱼ are pairwise neural networks (for feature interactions)
- ε ~ N(0, σ²) is Gaussian noise

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


class RegressionGamiNetUnivariate(BayesianModel):
    """Generalized Additive Model with Univariate Neural Networks for Regression.

    Implements regression where each feature contributes through
    its own neural network:

        y = intercept + Σᵢ fᵢ(xᵢ) + ε

    Each fᵢ is a univariate neural network that learns the shape function
    for feature i. The additive structure maintains interpretability while
    allowing for complex non-linear feature effects.

    Args:
        input_size: Number of input features
        layer_sizes: List of hidden layer sizes for each univariate network
        outcome_label: Key for outcome in data dictionary
        weight_scale: Scale for network weights
        bias_scale: Scale for network biases
        noise_scale: Prior scale for observation noise
        dtype: Data type for computations

    Example:
        >>> model = RegressionGamiNetUnivariate(
        ...     input_size=10,
        ...     layer_sizes=[20, 10, 1],  # Each feature gets this architecture
        ... )
        >>> # Each of 10 features has independent network: x -> 20 -> 10 -> 1
    """

    def __init__(
        self,
        input_size: int,
        layer_sizes: list[int],
        outcome_label: str = "y",
        weight_scale: float = 1.0,
        bias_scale: float = 1.0,
        noise_scale: float = 1.0,
        dtype: jnp.dtype = jnp.float64,
        **kwargs,
    ):
        super(RegressionGamiNetUnivariate, self).__init__(dtype=dtype, **kwargs)

        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.outcome_label = outcome_label
        self.weight_scale = weight_scale
        self.bias_scale = bias_scale
        self.noise_scale = noise_scale

        # Ensure output is scalar for regression
        if layer_sizes[-1] != 1:
            layer_sizes = layer_sizes + [1]

        # Create univariate neural network
        self.univariate_nn = UnivariateDense(
            input_size=input_size,
            layer_sizes=layer_sizes,
            weight_scale=weight_scale,
            bias_scale=bias_scale,
            dtype=dtype,
        )

        # Create intercept parameter
        self.intercept_shape = [1]

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

        # Add noise scale
        distribution_dict["noise_scale"] = tfd.Independent(
            tfd.HalfNormal(scale=self.noise_scale * jnp.ones([1], dtype=self.dtype)),
            reinterpreted_batch_ndims=1,
        )
        bijectors["noise_scale"] = tfp.bijectors.Softplus()

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
            Predicted mean of shape [n]
        """
        # Extract network weights
        weight_tensors = []
        num_layers = len(self.layer_sizes)
        for j in range(num_layers):
            weight_tensors.append(params[f"w_{j}"])
            weight_tensors.append(params[f"b_{j}"])

        # Evaluate univariate networks
        # Returns shape [n, p, 1] or [n, p]
        univariate_contributions = self.univariate_nn.eval(
            jnp.asarray(X, dtype=self.dtype), weight_tensors=weight_tensors
        )

        # Sum contributions across features
        if len(univariate_contributions.shape) == 3:
            # Shape [n, p, 1] -> [n]
            mean = jnp.sum(univariate_contributions, axis=1)
            mean = jnp.squeeze(mean, axis=-1)
        else:
            # Shape [n, p] -> [n]
            mean = jnp.sum(univariate_contributions, axis=1)

        # Add intercept
        mean = mean + jnp.squeeze(params["intercept"])

        return mean

    def predictive_distribution(
        self, data: dict[str, jax.Array], **params
    ) -> dict[str, jax.Array]:
        """Compute predictive distribution.

        Args:
            data: Dictionary with 'X' (features) and outcome_label (targets)
            **params: Model parameters

        Returns:
            Dictionary with log_likelihood, prediction, and mean
        """
        X = jnp.asarray(data["X"], dtype=self.dtype)

        # Get mean from GAM-Net
        mean = self.eval(X, params)

        # Get noise scale
        noise_scale = jnp.squeeze(params["noise_scale"])

        # Create Normal distribution
        rv_outcome = tfd.Normal(loc=mean, scale=noise_scale)

        # Compute log likelihood
        y = jnp.squeeze(data[self.outcome_label])
        log_likelihood = rv_outcome.log_prob(y)

        return {
            "log_likelihood": log_likelihood,
            "prediction": rv_outcome,
            "mean": mean,
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


class RegressionGamiNetPairwise(BayesianModel):
    """Generalized Additive Model with Univariate and Pairwise Neural Networks for Regression.

    Extends RegressionGamiNetUnivariate by adding pairwise interaction terms:

        y = intercept + Σᵢ fᵢ(xᵢ) + Σ_{i<j} gᵢⱼ(xᵢ, xⱼ) + ε

    Each fᵢ is a univariate network and each gᵢⱼ is a bivariate network
    capturing interactions between features i and j.

    Args:
        input_size: Number of input features
        univariate_layer_sizes: Layer sizes for univariate networks
        pairwise_layer_sizes: Layer sizes for pairwise networks
        pairs: List of [i, j] pairs for pairwise interactions
        outcome_label: Key for outcome in data dictionary
        noise_scale: Prior scale for observation noise
        dtype: Data type for computations

    Example:
        >>> model = RegressionGamiNetPairwise(
        ...     input_size=5,
        ...     univariate_layer_sizes=[10, 5, 1],
        ...     pairwise_layer_sizes=[8, 4, 1],
        ...     pairs=[[0, 1], [2, 3]],  # Interactions between (x0,x1) and (x2,x3)
        ... )
    """

    def __init__(
        self,
        input_size: int,
        univariate_layer_sizes: list[int],
        pairwise_layer_sizes: list[int],
        pairs: list[list[int]],
        outcome_label: str = "y",
        weight_scale: float = 1.0,
        bias_scale: float = 1.0,
        noise_scale: float = 1.0,
        dtype: jnp.dtype = jnp.float64,
        **kwargs,
    ):
        super(RegressionGamiNetPairwise, self).__init__(dtype=dtype, **kwargs)

        self.input_size = input_size
        self.univariate_layer_sizes = univariate_layer_sizes
        self.pairwise_layer_sizes = pairwise_layer_sizes
        self.pairs = pairs
        self.outcome_label = outcome_label
        self.weight_scale = weight_scale
        self.bias_scale = bias_scale
        self.noise_scale = noise_scale

        # Ensure outputs are scalar for regression
        if univariate_layer_sizes[-1] != 1:
            univariate_layer_sizes = univariate_layer_sizes + [1]
        if pairwise_layer_sizes[-1] != 1:
            pairwise_layer_sizes = pairwise_layer_sizes + [1]

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

        self.intercept_shape = [1]

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

        # Noise scale
        distribution_dict["noise_scale"] = tfd.Independent(
            tfd.HalfNormal(scale=self.noise_scale * jnp.ones([1], dtype=self.dtype)),
            reinterpreted_batch_ndims=1,
        )
        bijectors["noise_scale"] = tfp.bijectors.Softplus()

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
            Predicted mean of shape [n]
        """
        X = jnp.asarray(X, dtype=self.dtype)

        # Univariate contributions
        uni_weights = []
        for j in range(len(self.univariate_layer_sizes)):
            uni_weights.append(params[f"uni_w_{j}"])
            uni_weights.append(params[f"uni_b_{j}"])

        univariate_contrib = self.univariate_nn.eval(X, weight_tensors=uni_weights)

        # Sum over features
        if len(univariate_contrib.shape) == 3:
            univariate_sum = jnp.sum(univariate_contrib, axis=1)
            univariate_sum = jnp.squeeze(univariate_sum, axis=-1)
        else:
            univariate_sum = jnp.sum(univariate_contrib, axis=1)

        # Pairwise contributions
        pair_weights = []
        for j in range(len(self.pairwise_layer_sizes)):
            pair_weights.append(params[f"pair_w_{j}"])
            pair_weights.append(params[f"pair_b_{j}"])

        pairwise_contrib = self.pairwise_nn.eval(X, weight_tensors=pair_weights)

        # Sum over pairs
        if len(pairwise_contrib.shape) == 3:
            pairwise_sum = jnp.sum(pairwise_contrib, axis=1)
            pairwise_sum = jnp.squeeze(pairwise_sum, axis=-1)
        else:
            pairwise_sum = jnp.sum(pairwise_contrib, axis=1)

        # Combine all contributions
        mean = univariate_sum + pairwise_sum + jnp.squeeze(params["intercept"])

        return mean

    def predictive_distribution(
        self, data: dict[str, jax.Array], **params
    ) -> dict[str, jax.Array]:
        """Compute predictive distribution."""
        X = jnp.asarray(data["X"], dtype=self.dtype)

        # Get mean
        mean = self.eval(X, params)

        # Get noise scale
        noise_scale = jnp.squeeze(params["noise_scale"])

        # Create distribution
        rv_outcome = tfd.Normal(loc=mean, scale=noise_scale)

        # Compute log likelihood
        y = jnp.squeeze(data[self.outcome_label])
        log_likelihood = rv_outcome.log_prob(y)

        return {
            "log_likelihood": log_likelihood,
            "prediction": rv_outcome,
            "mean": mean,
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
    """Demonstrate RegressionGamiNet models."""
    print("Running RegressionGamiNet demo...")
    print("=" * 60)

    # Generate synthetic data
    import numpy as np
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    # Create non-linear effects
    y = (
        jnp.sin(X[:, 0])
        + 0.5 * X[:, 1] ** 2
        + 0.3 * X[:, 2]
        + 0.2 * np.random.randn(n_samples)
    )

    data = {"X": jnp.array(X), "y": jnp.array(y)}

    print(f"Data shapes: X={X.shape}, y={y.shape}")
    print(f"Target stats: mean={jnp.mean(y):.4f}, std={jnp.std(y):.4f}")

    # Test RegressionGamiNetUnivariate
    print("\n" + "=" * 60)
    print("Testing RegressionGamiNetUnivariate...")
    model_uni = RegressionGamiNetUnivariate(
        input_size=n_features, layer_sizes=[10, 5, 1]
    )

    print(f"Parameters: {list(model_uni.params.keys())}")
    pred_uni = model_uni.predictive_distribution(data, **model_uni.params)
    print(f"Log likelihood shape: {pred_uni['log_likelihood'].shape}")
    print(f"Mean log likelihood: {jnp.mean(pred_uni['log_likelihood']):.4f}")

    # Test RegressionGamiNetPairwise
    print("\n" + "=" * 60)
    print("Testing RegressionGamiNetPairwise...")
    pairs = [[0, 1], [2, 3]]
    model_pair = RegressionGamiNetPairwise(
        input_size=n_features,
        univariate_layer_sizes=[8, 4, 1],
        pairwise_layer_sizes=[6, 3, 1],
        pairs=pairs,
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
    demo()

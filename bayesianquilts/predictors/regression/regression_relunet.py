#!/usr/bin/env python3
"""Regression neural networks with ReLU activations."""

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf

from bayesianquilts.nn.dense import DenseGaussian, DenseHorseshoe


class RegressionRelunet(DenseHorseshoe):
    """Regression model using deep ReLU network with horseshoe prior.

    This model uses a multi-layer neural network with ReLU activations
    for regression tasks. The network weights have a horseshoe prior
    which encourages sparsity.

    Args:
        dim_regressors: Number of input features
        layer_sizes: List of hidden layer sizes. Output layer size is
            automatically set to 1 for regression.
        regressor_scales: Optional scaling for input features
        regressor_offsets: Optional offset for input features
        dtype: Data type for computations
        outcome_label: Key for outcome in data dictionary
        **kwargs: Additional arguments passed to DenseHorseshoe

    Example:
        >>> model = RegressionRelunet(
        ...     dim_regressors=10,
        ...     layer_sizes=[20, 10],  # Final layer of size 1 added automatically
        ...     dtype=jnp.float64
        ... )
    """

    def __init__(
        self,
        dim_regressors: int,
        layer_sizes: list[int] | None = None,
        regressor_scales: list[float] | None = None,
        regressor_offsets: list[float] | None = None,
        dtype: tf.DType = jnp.float64,
        outcome_label: str = "y",
        **kwargs
    ):
        if layer_sizes is None:
            layer_sizes = [int(dim_regressors / 10), 20, 1]
        else:
            layer_sizes = layer_sizes + [1]  # Output dimension is 1 for regression

        super(RegressionRelunet, self).__init__(
            input_size=dim_regressors,
            layer_sizes=layer_sizes,
            activation_fn=tf.nn.relu,
            weight_scale=1.0,
            bias_scale=1.0,
            dtype=dtype,
            **kwargs
        )
        if regressor_scales is None:
            self.regressor_scales = 1
        else:
            self.regressor_scales = regressor_scales
        self.regressor_offsets = (
            regressor_offsets if regressor_offsets is not None else 0
        )
        self.outcome_label = outcome_label

    def predictive_distribution(self, data: dict[str, jax.typing.ArrayLike], **params):
        """Compute predictive distribution for regression.

        Args:
            data: Dictionary with 'X' (features) and outcome_label (targets)
            **params: Model parameters

        Returns:
            Dictionary with log_likelihood, prediction (Normal distribution), and mean
        """
        X = data["X"].astype(self.dtype)

        # Network output is the mean
        mean = self.eval(X, params)
        mean = jnp.squeeze(mean, axis=-1)  # Remove last dimension

        # Use unit variance for likelihood (can be extended to learned variance)
        rv_outcome = tfd.Normal(loc=mean, scale=1.0)
        log_likelihood = rv_outcome.log_prob(tf.squeeze(data[self.outcome_label]))

        return {
            "log_likelihood": log_likelihood,
            "prediction": rv_outcome,
            "mean": mean,
        }

    def log_likelihood(self, data, **params):
        """Compute log likelihood."""
        return self.predictive_distribution(data, **params)["log_likelihood"]

    def unormalized_log_prob(
        self,
        data: dict[str, jax.typing.ArrayLike] = None,
        prior_weight: jax.typing.ArrayLike | float = 1.,
        **params
    ):
        """Compute unnormalized log probability (log likelihood + log prior).

        Args:
            data: Dictionary with input data
            prior_weight: Weight for prior term (for tempering)
            **params: Model parameters

        Returns:
            Unnormalized log probability
        """
        log_likelihood = self.log_likelihood(data, **params)
        prior = self.prior_distribution.log_prob(params)
        return (
            tf.reduce_sum(log_likelihood, axis=-1)
            + tf.cast(prior_weight, prior.dtype) * prior
        )


class ShallowGaussianRegressionRelunet(DenseGaussian):
    """Shallow regression network with Gaussian priors.

    Single hidden layer regression network with Gaussian priors on weights.
    Simpler and faster than the deep horseshoe version.

    Args:
        dim_regressors: Number of input features
        hidden_size: Size of hidden layer
        regressor_scales: Optional scaling for input features
        regressor_offsets: Optional offset for input features
        dtype: Data type for computations
        outcome_label: Key for outcome in data dictionary
        **kwargs: Additional arguments passed to DenseGaussian

    Example:
        >>> model = ShallowGaussianRegressionRelunet(
        ...     dim_regressors=10,
        ...     hidden_size=20,
        ...     dtype=jnp.float64
        ... )
    """

    def __init__(
        self,
        dim_regressors: int,
        hidden_size: int,
        regressor_scales: list[float] | None = None,
        regressor_offsets: list[float] | None = None,
        dtype: tf.DType = jnp.float64,
        outcome_label: str = "y",
        **kwargs
    ):
        layer_sizes = [hidden_size, 1]  # Output dimension is 1 for regression
        self.hidden_size = hidden_size
        super(ShallowGaussianRegressionRelunet, self).__init__(
            input_size=dim_regressors,
            layer_sizes=layer_sizes,
            activation_fn=jax.nn.relu,
            weight_scale=1.0,
            bias_scale=1.0,
            dtype=dtype,
            **kwargs
        )
        if regressor_scales is None:
            self.regressor_scales = 1
        else:
            self.regressor_scales = regressor_scales
        self.regressor_offsets = (
            regressor_offsets if regressor_offsets is not None else 0
        )
        self.outcome_label = outcome_label

    def predictive_distribution(self, data: dict[str, jax.typing.ArrayLike], **params):
        """Compute predictive distribution for regression.

        Args:
            data: Dictionary with 'X' (features) and outcome_label (targets)
            **params: Model parameters

        Returns:
            Dictionary with log_likelihood, prediction (Normal distribution), and mean
        """
        X = tf.cast(data["X"], self.dtype)

        # Network output is the mean
        mean = self.eval(X, params)
        mean = jnp.squeeze(mean, axis=-1)  # Remove last dimension

        # Use unit variance for likelihood (can be extended to learned variance)
        rv_outcome = tfd.Normal(loc=mean, scale=1.0)
        log_likelihood = rv_outcome.log_prob(jnp.squeeze(data[self.outcome_label]))

        return {
            "log_likelihood": log_likelihood,
            "prediction": rv_outcome,
            "mean": mean,
        }

    def log_likelihood(self, data, **params):
        """Compute log likelihood."""
        return self.predictive_distribution(data, **params)["log_likelihood"]

    def unormalized_log_prob(
        self,
        data: dict[str, tf.Tensor] = None,
        prior_weight: tf.Tensor | float = 1.,
        **params
    ):
        """Compute unnormalized log probability (log likelihood + log prior).

        Args:
            data: Dictionary with input data
            prior_weight: Weight for prior term (for tempering)
            **params: Model parameters

        Returns:
            Unnormalized log probability
        """
        log_likelihood = self.log_likelihood(data, **params)
        finite_portion = tf.where(
            tf.math.is_finite(log_likelihood),
            log_likelihood,
            jnp.zeros_like(log_likelihood),
        )
        min_val = tf.reduce_min(finite_portion) - 1.0
        max_val = 100.0  # Reasonable max for regression log likelihood
        log_likelihood = tf.clip_by_value(log_likelihood, min_val, max_val)
        log_likelihood = tf.where(
            tf.math.is_finite(log_likelihood),
            log_likelihood,
            jnp.ones_like(log_likelihood) * min_val,
        )
        prior = self.prior_distribution.log_prob(params)
        return (
            jnp.sum(log_likelihood, axis=-1)
            + jnp.array(prior_weight).astype(prior.dtype) * prior
        )

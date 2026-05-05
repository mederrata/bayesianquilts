#!/usr/bin/env python3
"""Locally Linear Attention - Piecewise linear attention mechanism.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates.jax import distributions as tfd

from bayesianquilts.jax.parameter import Decomposed, Interactions
from bayesianquilts.model import BayesianModel
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator

jax.config.update("jax_enable_x64", True)


class LocallyLinearAttention(BayesianModel):
    """Truly Locally Linear Attention layer.

    This layer partitions the input space into discrete regions and applies
    fixed routing matrices (M) and weight matrices (W) for each region.
    This ensures that the layer is strictly locally linear.

    Args:
        input_dim: Dimension of input features
        output_dim: Dimension of output features
        seq_len: Fixed sequence length (N)
        num_regions: Number of discrete regions to partition the space into
        centroids: Optional initial centroids for region selection.
                   If None, they will be initialized randomly.
        weight_prior_scale: Scale for parameter priors
        dtype: Data type for computations
        outcome_label: Key for outcome variable in data dict
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        num_regions: int,
        centroids: Optional[jax.Array] = None,
        weight_prior_scale: float = 0.1,
        dtype: jnp.dtype = jnp.float64,
        outcome_label: str = "y",
        initialize_distributions: bool = True,
        strategy: str | None = None,
        name: str = "",
    ):
        super(LocallyLinearAttention, self).__init__(dtype=dtype, strategy=strategy)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.num_regions = num_regions
        self.weight_prior_scale = weight_prior_scale
        self.outcome_label = outcome_label
        self._name = name


        # Centroids for region selection (partitioning the space)
        if centroids is None:
            # Simple initialization: random points in the input space
            _, key = jax.random.split(jax.random.PRNGKey(0))
            self.centroids = jax.random.normal(key, (num_regions, input_dim), dtype=dtype)
        else:
            self.centroids = jnp.asarray(centroids, dtype=dtype)

        # Create interaction structure for regions
        self.region_interactions = Interactions(
            dimensions=[("region", num_regions)],
            exclusions=[]
        )

        # Routing matrix M: [seq_len, seq_len]
        self.routing_decomp = Decomposed(
            interactions=self.region_interactions,
            param_shape=[seq_len, seq_len],
            name="routing_M",
            dtype=dtype,
        )

        # Weight matrix W: [input_dim, output_dim]
        self.weight_decomp = Decomposed(
            interactions=self.region_interactions,
            param_shape=[input_dim, output_dim],
            name="weight_W",
            dtype=dtype,
        )

        if initialize_distributions:
            self.create_distributions()

    def create_distributions(self):
        """Create prior and surrogate distributions."""
        distribution_dict = {}
        self.var_lists = {}

        # Routing M distributions
        (
            m_tensors,
            m_vars,
            m_shapes,
        ) = self.routing_decomp.generate_tensors(dtype=self.dtype)
        self.var_lists["routing_M"] = list(m_vars.keys())

        for label, tensor in m_tensors.items():
            distribution_dict[label] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros_like(jnp.asarray(tensor, self.dtype)),
                    scale=self.weight_prior_scale * jnp.ones_like(jnp.asarray(tensor, self.dtype)),
                ),
                reinterpreted_batch_ndims=len(tensor.shape),
            )

        # Weight W distributions
        (
            w_tensors,
            w_vars,
            w_shapes,
        ) = self.weight_decomp.generate_tensors(dtype=self.dtype)
        self.var_lists["weight_W"] = list(w_vars.keys())

        for label, tensor in w_tensors.items():
            distribution_dict[label] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros_like(jnp.asarray(tensor, self.dtype)),
                    scale=self.weight_prior_scale * jnp.ones_like(jnp.asarray(tensor, self.dtype)),
                ),
                reinterpreted_batch_ndims=len(tensor.shape),
            )

        self.prior_distribution = tfd.JointDistributionNamed(distribution_dict)
        self.surrogate_distribution_generator, self.surrogate_parameter_initializer = (
            build_factored_surrogate_posterior_generator(
                self.prior_distribution, dtype=self.dtype
            )
        )
        self.params = self.surrogate_parameter_initializer()
        self.var_list = list(self.params.keys())

    def get_region_index(self, X: jax.Array) -> jax.Array:
        """Map input to region index.
        
        Args:
            X: Input of shape [batch, seq_len, input_dim]
        
        Returns:
            Region indices of shape [batch, 1]
        """
        # Average over sequence to get a representative vector for the whole sequence
        # We assume X has shape [..., seq_len, input_dim]
        X_avg = jnp.mean(X, axis=-2) # [..., input_dim]
        
        # L2 distance to centroids
        # X_avg: [..., 1, input_dim], centroids: [num_regions, input_dim]
        # Reshape centroids for broadcasting
        c = self.centroids.reshape((1,) * (X_avg.ndim - 1) + self.centroids.shape)
        dists = jnp.sum((X_avg[..., jnp.newaxis, :] - c)**2, axis=-1)

        
        # Hard argmin for strict local linearity
        indices = jnp.argmin(dists, axis=-1)
        return indices[..., jnp.newaxis]


    def eval(
        self,
        X: jax.Array,
        params: dict,
    ) -> jax.Array:
        """Evaluate the locally linear attention layer.
        
        Args:
            X: Input features of shape [batch, seq_len, input_dim]
            params: Parameter dictionary
            
        Returns:
            Output of shape [batch, seq_len, output_dim]
        """
        # 1. Identify region
        region_indices = self.get_region_index(X)
        
        # 2. Lookup M and W for these regions
        M = self.routing_decomp.lookup(region_indices, tensors=params) # [batch, seq_len, seq_len]
        W = self.weight_decomp.lookup(region_indices, tensors=params)  # [batch, input_dim, output_dim]


        
        # 3. Apply: Y = M X W
        # We need batch-wise matrix multiplication
        # X is [batch, seq_len, input_dim]
        # Y = M @ X @ W
        # First: M @ X -> [batch, seq_len, seq_len] @ [batch, seq_len, input_dim] -> [batch, seq_len, input_dim]
        MX = jnp.matmul(M, X)
        # Second: MX @ W -> [batch, seq_len, input_dim] @ [batch, input_dim, output_dim] -> [batch, seq_len, output_dim]
        Y = jnp.matmul(MX, W)
        
        return Y

    def predictive_distribution(self, data: dict, **params) -> dict:
        """Compute predictive distribution."""
        X = data["X"]
        y = data[self.outcome_label]

        outputs = self.eval(X, params)
        
        # Assuming regression for now
        rv_outcome = tfd.Normal(loc=outputs, scale=1.0)
        log_likelihood = rv_outcome.log_prob(y)

        return {
            "log_likelihood": log_likelihood,
            "prediction": rv_outcome,
            "outputs": outputs,
        }

    def log_likelihood(self, data: dict, **params) -> jax.Array:
        return self.predictive_distribution(data, **params)["log_likelihood"]

    def unormalized_log_prob(self, data: dict, prior_weight: float = 1.0, **params) -> jax.Array:
        pred = self.predictive_distribution(data, **params)
        log_likelihood = jnp.sum(pred["log_likelihood"])
        log_prior = self.prior_distribution.log_prob(params)
        return log_likelihood + prior_weight * log_prior

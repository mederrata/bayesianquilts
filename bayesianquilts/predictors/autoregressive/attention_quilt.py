#!/usr/bin/env python3
"""Autoregressive Attention Quilt - Autoregressive attention mechanism with discretely indexed parameter decomposition.
"""

from collections import defaultdict
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd
from flax.linen import Embed

from bayesianquilts.jax.parameter import Decomposed, Interactions
from bayesianquilts.model import BayesianModel
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator

jax.config.update("jax_enable_x64", True)


class AutoregressiveAttentionQuilt(BayesianModel):
    """Autoregressive Multi-head attention with hierarchically decomposed weights.
    """

    def __init__(
        self,
        vocab_size: int,
        input_dim: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        key_interactions: Interactions,
        query_interactions: Interactions,
        value_interactions: Interactions,
        output_interactions: Interactions,
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
        super(AutoregressiveAttentionQuilt, self).__init__(dtype=dtype, strategy=strategy)

        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        self.key_interactions = key_interactions
        self.query_interactions = query_interactions
        self.value_interactions = value_interactions
        self.output_interactions = output_interactions

        self.weight_scale = weight_scale
        self.bias_scale = bias_scale
        self.dim_decay_factor = dim_decay_factor
        self.weight_prior_scale = weight_prior_scale
        self.bias_prior_scale = bias_prior_scale
        self.outcome_label = outcome_label
        self.activation_fn = activation_fn if activation_fn is not None else jax.nn.relu

        # Embedding layer (not decomposed for now)
        self.embedding = Embed(num_embeddings=vocab_size, features=input_dim, dtype=dtype)

        # Create decompositions for each head's weights
        self.query_decompositions = []
        self.key_decompositions = []
        self.value_decompositions = []
        
        for h in range(self.num_heads):
            # Query decomposition
            query_shape = [self.input_dim, self.key_dim]
            query_decomp = Decomposed(
                interactions=query_interactions,
                param_shape=query_shape,
                name=f"query_h{h}",
                dtype=dtype,
            )
            self.query_decompositions.append(query_decomp)

            # Key decomposition
            key_shape = [self.input_dim, self.key_dim]
            key_decomp = Decomposed(
                interactions=key_interactions,
                param_shape=key_shape,
                name=f"key_h{h}",
                dtype=dtype,
            )
            self.key_decompositions.append(key_decomp)

            # Value decomposition
            value_shape = [self.input_dim, self.value_dim]
            value_decomp = Decomposed(
                interactions=value_interactions,
                param_shape=value_shape,
                name=f"value_h{h}",
                dtype=dtype,
            )
            self.value_decompositions.append(value_decomp)
        
        # Output projection
        output_shape = [self.num_heads * self.value_dim, self.vocab_size]
        self.output_decomposition = Decomposed(
            interactions=output_interactions,
            param_shape=output_shape,
            name="output_proj",
            dtype=dtype,
        )

        if initialize_distributions:
            self.create_distributions()

    def create_distributions(self):
        """Create prior and surrogate distributions for all parameters."""
        distribution_dict = {}
        self.var_lists = {}  # Track variables for each component

        # Embedding distributions
        embedding_params = self.embedding.init(jax.random.PRNGKey(0), jnp.ones((1, 1), dtype=jnp.int32))['params']
        self.var_lists["embedding"] = ["embedding"]
        distribution_dict["embedding"] = tfd.Independent(
            tfd.Normal(
                loc=jnp.zeros_like(embedding_params['embedding']),
                scale=self.weight_prior_scale * jnp.ones_like(embedding_params['embedding']),
            ),
            reinterpreted_batch_ndims=2,
        )


        # Loop over heads for Q, K, V
        for h in range(self.num_heads):
            # Query distributions
            (
                query_tensors,
                query_vars,
                query_shapes,
            ) = self.query_decompositions[h].generate_tensors(dtype=self.dtype)
            
            query_scales = {
                k: (
                    self.weight_prior_scale
                    * self.dim_decay_factor ** (len([d for d in v if d > 1]) - 1)
                )
                for k, v in query_shapes.items()
            }

            self.var_lists[f"query_h{h}"] = list(query_vars.keys())

            for label, tensor in query_tensors.items():
                distribution_dict[label] = tfd.Independent(
                    tfd.Normal(
                        loc=jnp.zeros_like(jnp.asarray(tensor, self.dtype)),
                        scale=query_scales[label]
                        * jnp.ones_like(jnp.asarray(tensor, self.dtype)),
                    ),
                    reinterpreted_batch_ndims=len(tensor.shape),
                )
            
            # Key distributions
            (
                key_tensors,
                key_vars,
                key_shapes,
            ) = self.key_decompositions[h].generate_tensors(dtype=self.dtype)

            key_scales = {
                k: (
                    self.weight_prior_scale
                    * self.dim_decay_factor ** (len([d for d in v if d > 1]) - 1)
                )
                for k, v in key_shapes.items()
            }

            self.var_lists[f"key_h{h}"] = list(key_vars.keys())

            for label, tensor in key_tensors.items():
                distribution_dict[label] = tfd.Independent(
                    tfd.Normal(
                        loc=jnp.zeros_like(jnp.asarray(tensor, self.dtype)),
                        scale=key_scales[label]
                        * jnp.ones_like(jnp.asarray(tensor, self.dtype)),
                    ),
                    reinterpreted_batch_ndims=len(tensor.shape),
                )
            
            # Value distributions
            (
                value_tensors,
                value_vars,
                value_shapes,
            ) = self.value_decompositions[h].generate_tensors(dtype=self.dtype)

            value_scales = {
                k: (
                    self.weight_prior_scale
                    * self.dim_decay_factor ** (len([d for d in v if d > 1]) - 1)
                )
                for k, v in value_shapes.items()
            }
            
            self.var_lists[f"value_h{h}"] = list(value_vars.keys())

            for label, tensor in value_tensors.items():
                distribution_dict[label] = tfd.Independent(
                    tfd.Normal(
                        loc=jnp.zeros_like(jnp.asarray(tensor, self.dtype)),
                        scale=value_scales[label]
                        * jnp.ones_like(jnp.asarray(tensor, self.dtype)),
                    ),
                    reinterpreted_batch_ndims=len(tensor.shape),
                )

        # Output projection distributions
        (
            output_tensors,
            output_vars,
            output_shapes,
        ) = self.output_decomposition.generate_tensors(dtype=self.dtype)

        output_scales = {
            k: (
                self.weight_prior_scale
                * self.dim_decay_factor ** (len([d for d in v if d > 1]) - 1)
            )
            for k, v in output_shapes.items()
        }

        self.var_lists["output_proj"] = list(output_vars.keys())
        for label, tensor in output_tensors.items():
            distribution_dict[label] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros_like(jnp.asarray(tensor, self.dtype)),
                    scale=output_scales[label]
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

    def get_head_weights(self, head_idx: int, indices: jax.Array, params: dict) -> tuple:
        """
        """
        query_decomp = self.query_decompositions[head_idx]
        key_decomp = self.key_decompositions[head_idx]
        value_decomp = self.value_decompositions[head_idx]

        # Get query variables for this head
        query_var_list = self.var_lists[f"query_h{head_idx}"]
        query_params = {k: params[k] for k in query_var_list}

        # Get key variables for this head
        key_var_list = self.var_lists[f"key_h{head_idx}"]
        key_params = {k: params[k] for k in key_var_list}

        # Get value variables for this head
        value_var_list = self.var_lists[f"value_h{head_idx}"]
        value_params = {k: params[k] for k in value_var_list}


        # Lookup decomposed weights and biases
        queries = query_decomp.lookup(indices, tensors=query_params)
        keys = key_decomp.lookup(indices, tensors=key_params)
        values = value_decomp.lookup(indices, tensors=value_params)

        return queries, keys, values

    def eval(
        self,
        X: jax.Array,
        indices: jax.Array,
        params: dict,
        activation: Callable[[jax.Array], jax.Array] | None = None,
    ) -> jax.Array:
        """Evaluate neural network with group-specific weights.

        Args:
            X: Input token ids of shape [batch_size, sequence_length]
            indices: Discrete indices of shape [batch_size, num_interaction_dims]
            params: Parameter dictionary
            activation: Activation function (defaults to self.activation_fn)

        Returns:
            Next token logits of shape [batch_size, sequence_length, vocab_size]
        """
        batch_size, sequence_length = X.shape
        
        # 1. Embed tokens
        X_embedded = self.embedding.apply({'params': {'embedding': params['embedding']}}, X)


        # 2. Calculate Q, K, V for each head
        queries, keys, values = [], [], []
        for h in range(self.num_heads):
            W_q, W_k, W_v = self.get_head_weights(h, indices, params)
            
            Q_h = jnp.einsum('bsi,bio->bso', X_embedded, W_q)
            K_h = jnp.einsum('bsi,bio->bso', X_embedded, W_k)
            V_h = jnp.einsum('bsi,biv->bsv', X_embedded, W_v)
            queries.append(Q_h)
            keys.append(K_h)
            values.append(V_h)

        Q = jnp.stack(queries, axis=1)
        K = jnp.stack(keys, axis=1)
        V = jnp.stack(values, axis=1)

        # 3. Attention scores
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(self.key_dim)
        
        # 4. Causal mask
        mask = jnp.tril(jnp.ones((sequence_length, sequence_length)))
        scores = jnp.where(mask, scores, -jnp.inf)

        # 5. Attention weights
        attention_weights = jax.nn.softmax(scores, axis=-1)

        # 6. Weighted sum of values
        attention_output = jnp.matmul(attention_weights, V)
        
        # 7. Concatenate head outputs and project
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)

        output_decomp = self.output_decomposition
        output_var_list = self.var_lists["output_proj"]
        output_params = {k: params[k] for k in output_var_list}
        W_o = output_decomp.lookup(indices, tensors=output_params)
        
        logits = jnp.einsum('bsi,bio->bso', attention_output, W_o)

        return logits

    def predictive_distribution(self, data: dict, **params) -> dict:
        """Compute predictive distribution for next token prediction.
        """
        X = data["X"]
        
        # Assume all interactions are the same for simplicity
        indices = self.query_decompositions[0].retrieve_indices(data)

        logits = self.eval(X, indices, params)
        
        rv_outcome = tfd.Categorical(logits=logits)
        
        # The target 'y' should be the shifted input sequence
        y = data[self.outcome_label]
        log_likelihood = rv_outcome.log_prob(y)

        return {
            "log_likelihood": log_likelihood,
            "prediction": rv_outcome,
            "logits": logits,
        }

    def log_likelihood(self, data: dict, **params) -> jax.Array:
        """Compute log likelihood.
        """
        return self.predictive_distribution(data, **params)["log_likelihood"]
        
if __name__ == "__main__":
    pass


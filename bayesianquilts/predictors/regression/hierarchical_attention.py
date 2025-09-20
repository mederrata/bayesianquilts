#!/usr/bin/env python3


from bayesianquilts.jax.parameter import Decomposed, Interactions
from bayesianquilts.model import QuiltedBayesianModel

"""Hierarchical Attention Models
"""

class HierarchicalAttentionRegression(QuiltedBayesianModel):
    """Hierarchical Attention Model

    This model implements a hierarchical attention mechanism for processing
    sequential data. It consists of multiple layers of attention, allowing the
    model to focus on different levels of abstraction in the input data.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden attention layers.
        output_dim (int): Dimension of the output layer.

    Example:
        >>> model = HierarchicalAttention(input_dim=128, hidden_dim=256, output_dim=10)
        >>> output = model(input_data)
    """

    def __init__(self, interactions: Interactions, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.interactions = interactions

        self.create_distributions()

    def forward(self, data):
        """Forward pass through the hierarchical attention model.

        Args:
            data (dict[str:Tensor]): Input dict of tensors of shape (batch_size, seq_length, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Implement the forward pass logic here
        # ...
        return output
    
    def create_distributions():
        self.attention_k_decompostiion = Decomposed(
            interaction,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            dropout,
        )
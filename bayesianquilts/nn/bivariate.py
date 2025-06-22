from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow_probability.python import distributions as tfd

from bayesianquilts.nn.dense import Dense


class PairwiseDense(Dense):
    """Pairwise Dense neural network, taking R^2 to R^p with no additional variable mixing

    Args:
        Dense (_type_): _description_
    """

    def __init__(self, pairs: list[int] | None = None, **kwargs) -> None:
        """Iniitialize univariate neural network"""
        self.pairs = pairs
        self.num_pairs = len(pairs)
        super(PairwiseDense, self).__init__(**kwargs)

    def sample_initial_nn_params(
        self,
        input_size: int,
        layer_sizes: list[int],
        priors: list[tuple[float, float]] | None = None,
    ) -> list[tf.Tensor]:
        """
        layer_sizes correspond to each feature
        """
        architecture = []
        layer_sizes = [2] + layer_sizes

        if priors is None:
            for j, layer_size in enumerate(layer_sizes[1:]):
                weights = tfd.Normal(
                    loc=jnp.zeros(
                        (self.num_pairs, layer_sizes[j], layer_size), dtype=self.dtype
                    ),
                    scale=1e-1,
                ).sample()
                biases = tfd.Normal(
                    loc=jnp.zeros((self.num_pairs, layer_size), dtype=self.dtype), scale=1.0
                ).sample()
                architecture += [weights, biases]
        else:
            pass

        return architecture

    def eval(
        self,
        tensor: tf.Tensor,
        weight_tensors: tf.Tensor | None = None,
        activation: Callable[[tf.Tensor], tf.Tensor] | None = None,
    ) -> tf.Tensor:
        """Evaluate the model

        Args:
            tensor (self.dtype): [n x p]
            weight_tensors (_type_, optional): _description_. Defaults to None.
            activation (_type_, optional): _description_. Defaults to None.
        """
        weight_tensors = (
            weight_tensors if weight_tensors is not None else self.weight_tensors
        )
        activation = tf.nn.relu if activation is None else activation

        net = tf.gather(tf.transpose(tensor), self.pairs)
        net = tf.transpose(net, [2, 0, 1])
        net = tf.cast(net, self.dtype)
        weights_list = weight_tensors[::2]
        biases_list = weight_tensors[1::2]
        net = net * weights_list[0] + biases_list[0][..., tf.newaxis, :]

        for weights, biases in zip(weights_list[1:-1], biases_list[1:-1]):
            net = self.dense(
                net, self.weight_scale * weights, self.bias_scale * biases, activation
            )

        net = self.dense(
            net,
            self.weight_scale * weights_list[-1],
            self.bias_scale * biases_list[-1],
            tf.identity,
        )
        return net[..., 0, :]


def demo():
    n = 30
    p = 8
    X = np.random.rand(n, p)
    nn = PairwiseDense(
        input_size=p, pairs=[[0, 1], [1, 2], [2, 3], [7, 0]], layer_sizes=[10, 5, 7]
    )

    x = nn.eval(X)
    return


if __name__ == "__main__":
    demo()

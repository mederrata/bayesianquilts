import inspect
from abc import abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayesianquilts.distributions import SqrtInverseGamma
from bayesianquilts.metastrings import (
    cauchy_code,
    horseshoe_code,
    horseshoe_lambda_code,
    igamma_code,
    sq_igamma_code,
    weight_code,
)
from bayesianquilts.model import BayesianModel
from bayesianquilts.vi.advi import (
    build_surrogate_posterior,
    build_trainable_InverseGamma_dist,
    build_trainable_normal_dist,
)

from tensorflow_probability.python import distributions as tfd
from bayesianquilts.nn.dense import Dense


class UnivariateNN(Dense):
    """Univariate neural network, taking R^n to R^n with no variable mixing

    Args:
        Dense (_type_): _description_
    """

    def __init__(self, **kwargs):
        """Iniitialize univariate neural network"""
        super(UnivariateNN, self).__init__(**kwargs)

    def sample_initial_nn_params(self, input_size, layer_sizes, priors=None):
        """
        layer_sizes correspond to each feature
        """
        architecture = []
        layer_sizes = [1] + layer_sizes

        if priors is None:
            for j, layer_size in enumerate(layer_sizes[1:]):
                weights = tfd.Normal(
                    loc=tf.zeros(
                        (input_size, layer_sizes[j], layer_size), dtype=self.dtype
                    ),
                    scale=1e-1,
                ).sample()
                biases = tfd.Normal(
                    loc=tf.zeros((input_size, layer_size), dtype=self.dtype), scale=1.0
                ).sample()
                architecture += [weights, biases]
        else:
            pass

        return architecture

    def eval(self, input, weight_tensors=None, activation=None):
        """Evaluate the model

        Args:
            input (self.dtype): [n x p]
            weight_tensors (_type_, optional): _description_. Defaults to None.
            activation (_type_, optional): _description_. Defaults to None.
        """
        weight_tensors = (
            weight_tensors if weight_tensors is not None else self.weight_tensors
        )
        activation = tf.nn.relu if activation is None else activation

        net = input[..., tf.newaxis, tf.newaxis]
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
    p = 3
    X = np.random.rand(n, p)
    nn = UnivariateNN(input_size=p, layer_sizes=[10, 5, 7])

    x = nn.eval(X)
    return


if __name__ == "__main__":
    demo()

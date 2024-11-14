import inspect
from abc import abstractmethod
from typing import Callable, Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayesianquilts.model import BayesianModel
from bayesianquilts.vi.advi import build_surrogate_posterior

from tensorflow_probability.python import distributions as tfd


class Dense(object):
    def __init__(
        self,
        input_size: int | None = None,
        layer_sizes: list[int] | None = None,
        weight_scale: float = 1.0,
        bias_scale: float = 1.0,
        activation_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
        dtype: tf.DType = tf.float32,
    ) -> None:

        self.dtype = dtype
        self.weight_scale = weight_scale
        self.bias_scale = bias_scale
        self.input_size = input_size
        self.activation_fn = tf.nn.relu if (activation_fn is None) else activation_fn
        self.weight_tensors = self.sample_initial_nn_params(input_size, layer_sizes)

    def dense(
        self,
        X: tf.Tensor,
        W: tf.Tensor,
        b: tf.Tensor,
        activation: Callable[[tf.Tensor], tf.Tensor],
    ) -> tf.Tensor:
        out = activation(
            tf.matmul(tf.cast(X, self.dtype), tf.cast(W, self.dtype))
            + tf.cast(b[..., tf.newaxis, :], self.dtype)
        )
        return out

    def set_weights(self, weight_tensors: list[tf.Tensor]) -> None:
        self.weight_tensors = weight_tensors

    def eval(
        self,
        tensor: tf.Tensor,
        weight_tensors: dict[str, tf.Tensor] | None =None,
        activation: Callable[[tf.Tensor], tf.Tensor]  | None = None,
    ) -> tf.Tensor:
        activation = self.activation_fn if (activation is None) else activation
        weight_tensors = (
            weight_tensors if weight_tensors is not None else self.weight_tensors
        )

        net = tensor
        net = tf.cast(net, self.dtype)
        weights_list = weight_tensors[::2]
        biases_list = weight_tensors[1::2]

        for weights, biases in zip(weights_list[:-1], biases_list[:-1]):
            net = self.dense(
                net, self.weight_scale * weights, self.bias_scale * biases, activation
            )
        net = self.dense(
            net,
            self.weight_scale * weights_list[-1],
            self.bias_scale * biases_list[-1],
            tf.identity,
        )
        return net

    def sample_initial_nn_params(
        self,
        input_size: int,
        layer_sizes: list[int],
        priors: list[tuple[float, float]]  | None = None,
    ) -> list[tf.Tensor]:
        """
        Priors should be either none or a list of tuples:
        [(weight prior, bias prior) for layer in layer_sizes]
        """
        architecture = []
        layer_sizes = [input_size] + layer_sizes

        if priors is None:
            for j, layer_size in enumerate(layer_sizes[1:]):
                weights = tfd.Normal(
                    loc=tf.zeros((layer_sizes[j], layer_size), dtype=self.dtype),
                    scale=1e-1,
                ).sample()
                biases = tfd.Normal(
                    loc=tf.zeros((layer_size), dtype=self.dtype), scale=1.0
                ).sample()
                architecture += [weights, biases]
        else:
            pass

        return architecture

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state.copy()
        self.saved_state = state

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        keys = self.__dict__.keys()
        for k in keys:
            # print(k)
            if isinstance(state[k], tf.Tensor) or isinstance(state[k], tf.Variable):
                state[k] = state[k].numpy()
            elif isinstance(state[k], dict) or isinstance(state[k], list):
                flat = tf.nest.flatten(state[k])
                new = []
                for t in flat:
                    if isinstance(t, tf.Tensor) or isinstance(t, tf.Variable):
                        # print(k)
                        new += [t.numpy()]
                    elif hasattr(inspect.getmodule(t), "__name__"):
                        if inspect.getmodule(t).__name__.startswith("tensorflow"):
                            if not isinstance(t, tf.dtypes.DType):
                                new += [None]
                            else:
                                new += [None]
                        else:
                            new += [t]
                    else:
                        new += [t]
                state[k] = tf.nest.pack_sequence_as(state[k], new)
            elif hasattr(inspect.getmodule(state[k]), "__name__"):
                if inspect.getmodule(state[k]).__name__.startswith("tensorflow"):
                    if not isinstance(state[k], tf.dtypes.DType):
                        del state[k]
        return state


class DenseHorseshoe(BayesianModel):
    """Dense horseshoe network of given layer sizes

    Arguments:
        DenseNetwork {[type]} -- [description]
    """

    distribution = None
    surrogate_distribution = None
    reparameterized = True

    def __init__(
        self,
        input_size: int  | None = None,
        layer_sizes: list[int]  | None = None,
        decay: float = 0.5,
        activation_fn: Callable[[tf.Tensor], tf.Tensor]  | None = None,
        weight_scale: float = 1.0,
        bias_scale: float = 1.0,
        dtype: tf.DType = tf.float64,
        **kwargs,
    ) -> None:
        super(DenseHorseshoe, self).__init__(
            **kwargs,
        )
        self.dtype = dtype
        self.layer_sizes = [input_size] + layer_sizes
        self.weight_scale = weight_scale
        self.bias_scale = bias_scale
        self.nn = Dense(
            input_size=input_size,
            layer_sizes=layer_sizes,
            weight_scale=weight_scale,
            bias_scale=bias_scale,
            activation_fn=activation_fn,
            dtype=dtype,
        )
        self.decay = decay  # dimensional decay
        self.input_size = input_size

        self.create_distributions()

    def set_weights(self, weights: list[tf.Tensor]):
        self.nn.set_weights(weights)

    def log_prob(self, x: dict[str, tf.Tensor]):
        return self.prior_distribution.log_prob(x)

    def sample_weights(self, *args, **kwargs):
        return self.prior_distribution.sample(*args, **kwargs)

    def eval(
        self,
        tensor: tf.Tensor,
        sample: dict[str, tf.Tensor],
        activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
    ):
        weight_tensors = []
        for j in range(int(len(self.nn.weight_tensors) / 2)):
            weight_tensors += [sample["w_" + str(j)]] + [sample["b_" + str(j)]]
        net = self.nn.eval(tensor, weight_tensors, activation=activation)
        return net

    def create_distributions(self):
        distribution_dict = {}
        bijectors = {}
        var_list = []
        weight_var_list = []
        initial = {}
        for j, weight in enumerate(self.nn.weight_tensors[::2]):
            var_list += [f"w_{j}"] + [f"b_{j}"]
            weight_var_list += [f"w_{j}"] + [f"b_{j}"]

            # Top level priors
            # weight

            # w ~ Horseshoe(0, w_tau)
            # w_tau ~ cauchy(0, w_tau_scale)
            # b ~ Horseshoe(0, b_tau)
            # b_tau ~ cauchy(0, b_tau_scale)

            bijectors[f"w_{j}"] = tfp.bijectors.Identity()
            distribution_dict[f"w_{j}"] = tfd.Independent(
                tfd.Horseshoe(
                    scale=tf.ones(
                        [self.layer_sizes[j], self.layer_sizes[j + 1]], dtype=self.dtype
                    )
                ),
                reinterpreted_batch_ndims=2,
            )

            initial[f"w_{j}"] = tf.convert_to_tensor(
                1e-3
                * np.random.normal(
                    np.zeros([self.layer_sizes[j], self.layer_sizes[j + 1]]),
                    np.ones([self.layer_sizes[j], self.layer_sizes[j + 1]]),
                ),
                self.dtype,
            )

            # bias
            bijectors[f"b_{j}"] = tfp.bijectors.Identity()

            distribution_dict[f"b_{j}"] = tfd.Independent(
                tfd.Horseshoe(
                    scale=tf.ones(
                        [
                            self.layer_sizes[j + 1],
                        ],
                        dtype=self.dtype,
                    )
                ),
                reinterpreted_batch_ndims=1,
            )

        self.bijectors = bijectors
        self.prior_distribution = tfd.JointDistributionNamed(distribution_dict)
        self.surrogate_distribution = build_surrogate_posterior(
            self.prior_distribution, bijectors, dtype=self.dtype, initializers=initial
        )
        self.var_list = list(self.surrogate_distribution.model.keys())
        self.surrogate_vars = self.surrogate_distribution.variables

    def sample(self, *args, **kwargs):
        return self.surrogate_distribution.sample(*args, **kwargs)

    @abstractmethod
    def predictive_distribution(self, data: dict[str, tf.Tensor], **params):
        pass

    @abstractmethod
    def log_likelihood(self, data: dict[str, tf.Tensor], **params):
        pass

class DenseGaussian(BayesianModel):
    """Dense horseshoe network of given layer sizes

    Arguments:
        DenseNetwork {[type]} -- [description]
    """

    distribution = None
    surrogate_distribution = None
    reparameterized = True

    def __init__(
        self,
        input_size: int  | None = None,
        layer_sizes: list[int]  | None = None,
        decay: float = 0.5,
        activation_fn: Callable[[tf.Tensor], tf.Tensor]  | None = None,
        weight_scale: float = 1.0,
        bias_scale: float = 1.0,
        dtype: tf.DType = tf.float64,
        **kwargs,
    ) -> None:
        super(DenseGaussian, self).__init__(
            **kwargs,
        )
        self.dtype = dtype
        self.layer_sizes = [input_size] + layer_sizes
        self.weight_scale = weight_scale
        self.bias_scale = bias_scale
        self.nn = Dense(
            input_size=input_size,
            layer_sizes=layer_sizes,
            weight_scale=weight_scale,
            bias_scale=bias_scale,
            activation_fn=activation_fn,
            dtype=dtype,
        )
        self.decay = decay  # dimensional decay
        self.input_size = input_size

        self.create_distributions()

    def set_weights(self, weights: list[tf.Tensor]):
        self.nn.set_weights(weights)

    def log_prob(self, x: dict[str, tf.Tensor]):
        return self.prior_distribution.log_prob(x)

    def sample_weights(self, *args, **kwargs):
        return self.prior_distribution.sample(*args, **kwargs)

    def eval(
        self,
        tensor: tf.Tensor,
        sample: dict[str, tf.Tensor],
        activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
    ):
        weight_tensors = []
        for j in range(int(len(self.nn.weight_tensors) / 2)):
            weight_tensors += [sample["w_" + str(j)]] + [sample["b_" + str(j)]]
        net = self.nn.eval(tensor, weight_tensors, activation=activation)
        return net

    def create_distributions(self):
        distribution_dict = {}
        bijectors = {}
        var_list = []
        weight_var_list = []
        initial = {}
        for j, weight in enumerate(self.nn.weight_tensors[::2]):
            var_list += [f"w_{j}"] + [f"b_{j}"]
            weight_var_list += [f"w_{j}"] + [f"b_{j}"]

            # Top level priors
            # weight

            # w ~ Horseshoe(0, w_tau)
            # w_tau ~ cauchy(0, w_tau_scale)
            # b ~ Horseshoe(0, b_tau)
            # b_tau ~ cauchy(0, b_tau_scale)

            bijectors[f"w_{j}"] = tfp.bijectors.Identity()
            distribution_dict[f"w_{j}"] = tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(
                        [self.layer_sizes[j], self.layer_sizes[j + 1]], dtype=self.dtype
                    ),
                    scale=tf.ones(
                        [self.layer_sizes[j], self.layer_sizes[j + 1]], dtype=self.dtype
                    )
                ),
                reinterpreted_batch_ndims=2,
            )

            initial[f"w_{j}"] = tf.convert_to_tensor(
                1e-3
                * np.random.normal(
                    np.zeros([self.layer_sizes[j], self.layer_sizes[j + 1]]),
                    np.ones([self.layer_sizes[j], self.layer_sizes[j + 1]]),
                ),
                self.dtype,
            )

            # bias
            bijectors[f"b_{j}"] = tfp.bijectors.Identity()

            distribution_dict[f"b_{j}"] = tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(
                        [
                            self.layer_sizes[j + 1],
                        ],
                        dtype=self.dtype,
                    ),
                    scale=tf.ones(
                        [
                            self.layer_sizes[j + 1],
                        ],
                        dtype=self.dtype,
                    )
                ),
                reinterpreted_batch_ndims=1,
            )

        self.bijectors = bijectors
        self.prior_distribution = tfd.JointDistributionNamed(distribution_dict)
        self.surrogate_distribution = build_surrogate_posterior(
            self.prior_distribution, bijectors, dtype=self.dtype, initializers=initial
        )
        self.var_list = list(self.surrogate_distribution.model.keys())
        self.surrogate_vars = self.surrogate_distribution.variables

    def sample(self, *args, **kwargs):
        return self.surrogate_distribution.sample(*args, **kwargs)

    @abstractmethod
    def predictive_distribution(self, data: dict[str, tf.Tensor], **params):
        pass

    @abstractmethod
    def log_likelihood(self, data: dict[str, tf.Tensor], **params):
        pass


def demo():
    nn = Dense(input_size=5, layer_sizes=[10, 5, 1])
    n = 30
    p = 5
    X = np.random.rand(n, p)
    x = nn.eval(X)

    class AutoEncoder(DenseHorseshoe):
        def log_likelihood(self, data, **params):
            return super().log_likelihood(data, **params)

        def predictive_distribution(self, data, **params):
            return super().predictive_distribution(**params)

        def unormalized_log_prob(self, data, *args, **kwargs):
            return super().unormalized_log_prob(data, *args, **kwargs)

    denseH = AutoEncoder(10, [20, 12, 2])
    sample = denseH.prior_distribution.sample(10)

    return


if __name__ == "__main__":
    demo()

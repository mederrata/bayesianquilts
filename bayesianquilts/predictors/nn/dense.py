import inspect
from abc import abstractmethod
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import random
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf

from bayesianquilts.model import BayesianModel
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator


class Dense(object):
    def __init__(
        self,
        input_size: int | None = None,
        layer_sizes: list[int] | None = None,
        weight_scale: float = 1.0,
        bias_scale: float = 1.0,
        activation_fn: Callable[[jax.typing.ArrayLike], tf.Tensor] | None = None,
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
            jnp.matmul(X.astype(self.dtype), W.astype(self.dtype))
            + b[..., jnp.newaxis, :].astype(self.dtype)
        )
        return out

    def set_weights(self, weight_tensors: list[jax.typing.ArrayLike]) -> None:
        self.weight_tensors = weight_tensors

    def eval(
        self,
        tensor: jax.typing.ArrayLike,
        weight_tensors: dict[str, jax.typing.ArrayLike] | None = None,
        activation: Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike] | None = None,
    ) -> jax.typing.ArrayLike:
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
        priors: list[tuple[float, float]] | None = None,
    ) -> list[jax.typing.ArrayLike]:
        """
        Priors should be either none or a list of tuples:
        [(weight prior, bias prior) for layer in layer_sizes]
        """
        architecture = []
        layer_sizes = [input_size] + layer_sizes
        _, sample_key = random.split(random.PRNGKey(0))

        if priors is None:
            for j, layer_size in enumerate(layer_sizes[1:]):
                weights = tfd.Normal(
                    loc=jnp.zeros((layer_sizes[j], layer_size), dtype=self.dtype),
                    scale=1e-1,
                ).sample(seed=sample_key)
                biases = tfd.Normal(
                    loc=jnp.zeros((layer_size), dtype=self.dtype), scale=1.0
                ).sample(seed=sample_key)
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
                            if not isinstance(t, jnp.dtype):
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
                    if not isinstance(state[k], jnp.dtype):
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
        input_size: int | None = None,
        layer_sizes: list[int] | None = None,
        decay: float = 0.5,
        activation_fn: Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike] | None = None,
        weight_scale: float = 1.0,
        bias_scale: float = 1.0,
        prior_scale: float = 1.0,
        extra_batch_dims=0,
        dtype: jax.typing.DTypeLike = jnp.float64,
        **kwargs,
    ) -> None:
        super(DenseHorseshoe, self).__init__(
            **kwargs,
        )
        self.dtype = dtype
        self.layer_sizes = [input_size] + layer_sizes
        self.weight_scale = weight_scale
        self.bias_scale = bias_scale
        self.prior_scale = prior_scale
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
        self.extra_batch_dims = extra_batch_dims

        self.create_distributions()

    def set_weights(self, weights: list[tf.Tensor]):
        self.nn.set_weights(weights)

    def log_prob(self, x: dict[str, tf.Tensor]):
        return self.prior_distribution.log_prob(x)

    def sample_weights(self, *args, **kwargs):
        return self.prior_distribution.sample(*args, **kwargs)

    def eval(
        self,
        tensor: jax.typing.ArrayLike,
        sample: dict[str, jax.typing.ArrayLike],
        activation: Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike] = jax.nn.relu,
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
                    scale=self.prior_scale * jnp.ones(
                        [1] * self.extra_batch_dims
                        + [self.layer_sizes[j], self.layer_sizes[j + 1]],
                        dtype=self.dtype,
                    )
                ),
                reinterpreted_batch_ndims=2 + self.extra_batch_dims,
            )

            initial[f"w_{j}"] = tf.convert_to_tensor(
                1e-3
                * np.random.normal(
                    np.zeros(
                        [1] * self.extra_batch_dims
                        + [self.layer_sizes[j], self.layer_sizes[j + 1]]
                    ),
                    np.ones(
                        [1] * self.extra_batch_dims
                        + [self.layer_sizes[j], self.layer_sizes[j + 1]]
                    ),
                ),
                self.dtype,
            )

            # bias
            bijectors[f"b_{j}"] = tfp.bijectors.Identity()

            distribution_dict[f"b_{j}"] = tfd.Independent(
                tfd.Horseshoe(
                    scale=self.prior_scale * jnp.ones(
                        [1] * self.extra_batch_dims
                        + [
                            self.layer_sizes[j + 1],
                        ],
                        dtype=self.dtype,
                    )
                ),
                reinterpreted_batch_ndims=1 + self.extra_batch_dims,
            )

        self.bijectors = bijectors
        self.prior_distribution = tfd.JointDistributionNamed(distribution_dict)
        self.surrogate_distribution_generator, self.surrogate_parameter_initializer = (
            build_factored_surrogate_posterior_generator(
                self.prior_distribution,
                bijectors=self.bijectors,
                dtype=self.dtype,
            )
        )
        self.params = self.surrogate_parameter_initializer()
        self.var_list = list(self.prior_distribution.model.keys())

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
        input_size: int | None = None,
        layer_sizes: list[int] | None = None,
        decay: float = 0.5,
        activation_fn: Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike] | None = None,
        weight_scale: float = 1.0,
        bias_scale: float = 1.0,
        prior_scale: float = 1.0,
        dtype: jnp.dtype = jnp.float64,
        **kwargs,
    ) -> None:
        super(DenseGaussian, self).__init__(
            **kwargs,
        )
        self.dtype = dtype
        self.layer_sizes = [input_size] + layer_sizes
        self.weight_scale = weight_scale
        self.bias_scale = bias_scale
        self.prior_scale = prior_scale
        self.extra_batch_dims = kwargs.get("extra_batch_dims", 0)
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
        activation: Callable[[tf.Tensor], tf.Tensor] = jax.nn.relu,
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

            bijectors[f"w_{j}"] = tfb.Identity()
            distribution_dict[f"w_{j}"] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(
                        [1] * self.extra_batch_dims
                        + [self.layer_sizes[j], self.layer_sizes[j + 1]],
                        dtype=self.dtype,
                    ),
                    scale=self.prior_scale * jnp.ones(
                        [1] * self.extra_batch_dims
                        + [self.layer_sizes[j], self.layer_sizes[j + 1]],
                        dtype=self.dtype,
                    ),
                ),
                reinterpreted_batch_ndims=2 + self.extra_batch_dims,
            )

            initial[f"w_{j}"] = tf.convert_to_tensor(
                1e-3
                * np.random.normal(
                    np.zeros(
                        [1] * self.extra_batch_dims
                        + [self.layer_sizes[j], self.layer_sizes[j + 1]]
                    ),
                    np.ones(
                        [1] * self.extra_batch_dims
                        + [self.layer_sizes[j], self.layer_sizes[j + 1]]
                    ),
                ),
                self.dtype,
            )

            # bias
            bijectors[f"b_{j}"] = tfp.bijectors.Identity()

            distribution_dict[f"b_{j}"] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(
                        [1] * self.extra_batch_dims
                        + [
                            self.layer_sizes[j + 1],
                        ],
                        dtype=self.dtype,
                    ),
                    scale=self.prior_scale * jnp.ones(
                        [1] * self.extra_batch_dims
                        + [
                            self.layer_sizes[j + 1],
                        ],
                        dtype=self.dtype,
                    ),
                ),
                reinterpreted_batch_ndims=1 + self.extra_batch_dims,
            )

        self.bijectors = bijectors
        self.prior_distribution = tfd.JointDistributionNamed(distribution_dict)
        self.surrogate_distribution_generator, self.surrogate_parameter_initializer = build_factored_surrogate_posterior_generator(
            self.prior_distribution, bijectors, dtype=self.dtype, surrogate_initializers=initial
        )
        self.params = self.surrogate_parameter_initializer()
        self.var_list = list(self.prior_distribution.model.keys())


    @abstractmethod
    def predictive_distribution(self, data: dict[str, tf.Tensor], **params):
        pass

    @abstractmethod
    def log_likelihood(self, data: dict[str, tf.Tensor], **params):
        pass

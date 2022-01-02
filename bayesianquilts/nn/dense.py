import tensorflow as tf
import tensorflow_probability as tfp

from bayesianquilts.metastrings import (
    weight_code,
    cauchy_code,
    sq_igamma_code,
    igamma_code,
)

from bayesianquilts.util import (
    clip_gradients,
    run_chain,
    build_trainable_InverseGamma_dist,
    build_trainable_normal_dist,
    build_surrogate_posterior,
)
from bayesianquilts.distributions import SqrtInverseGamma
from bayesianquilts.model import BayesianModel
from bayesianquilts.metastrings import horseshoe_code, horseshoe_lambda_code

tfd = tfp.distributions


class Dense(object):
    fn = None
    weights = None
    dtype = tf.float32

    def __init__(
        self,
        input_size=None,
        layer_sizes=None,
        weight_scale=1.0,
        bias_scale=1.0,
        activation_fn=None,
        dtype=tf.float32,
    ):

        self.dtype = dtype
        self.weight_scale = weight_scale
        self.bias_scale = bias_scale
        self.activation_fn = tf.nn.relu if (activation_fn is None) else activation_fn
        if (input_size is None) or (layer_sizes is None):
            self.fn = lambda x: x
        else:
            self.weights = self.sample_initial_nn_params(input_size, layer_sizes)
            self.fn = self.build_network(self.weights)

    def dense(self, X, W, b, activation):
        return activation(
            tf.matmul(tf.cast(X, self.dtype), tf.cast(W, self.dtype))
            + tf.cast(b[..., tf.newaxis, :], self.dtype)
        )

    def set_weights(self, weights):
        self.weights = weights
        self.fn = self.build_network(self.weights)

    def build_network(self, weight_tensors, activation=None):
        activation = self.activation_fn if (activation is None) else activation

        # @tf.function
        def model(X):
            net = X
            net = tf.cast(net, self.dtype)
            weights_list = weight_tensors[::2]
            biases_list = weight_tensors[1::2]

            for (weights, biases) in zip(weights_list, biases_list):
                net = self.dense(
                    net, self.weight_scale * weights, self.bias_scale * biases, activation
                )
            return net

        return model

    def sample_initial_nn_params(self, input_size, layer_sizes, priors=None):
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


class DenseHorseshoe(Dense, BayesianModel):
    """Dense horseshoe network of given layer sizes

    Arguments:
        DenseNetwork {[type]} -- [description]
    """

    distribution = None
    surrogate_distribution = None
    reparameterized = True

    def __init__(
        self,
        input_size=None,
        layer_sizes=None,
        decay=0.5,
        activation_fn=None,
        weight_scale=1.0,
        bias_scale=1.0,
        dtype=tf.float64,
        *args,
        **kwargs,
    ):
        super(DenseHorseshoe, self).__init__(
            input_size,
            layer_sizes,
            activation_fn=activation_fn,
            dtype=dtype,
            weight_scale=weight_scale,
            bias_scale=bias_scale,
            *args,
            **kwargs,
        )
        self.dtype = dtype
        self.layer_sizes = [input_size] + layer_sizes
        self.decay = decay  # dimensional decay
        self.weight_scale = weight_scale
        self.bias_scale = bias_scale
        self.create_distributions()

    def set_weights(self, weights):
        super(DenseHorseshoe, self).set_weights(weights)

    def log_prob(self, x):
        return self.prior_distribution.log_prob(x)

    def sample_weights(self, *args, **kwargs):
        return self.prior_distribution.sample(*args, **kwargs)

    def assemble_networks(self, sample, activation=tf.nn.relu):
        weight_tensors = []
        for j in range(int(len(self.weights) / 2)):
            weight_tensors += [sample["w_" + str(j)]] + [sample["b_" + str(j)]]
        net = self.build_network(weight_tensors, activation=activation)
        return net

    def create_distributions(self):
        distribution_dict = {}
        bijectors = {}
        var_list = []
        weight_var_list = []
        for j, weight in enumerate(self.weights[::2]):
            var_list += [f"w_{j}"] + [f"b_{j}"]
            weight_var_list += [f"w_{j}"] + [f"b_{j}"]

            # Top level priors
            ## weight

            # w ~ Horseshoe(0, w_tau)
            # w_tau ~ cauchy(0, w_tau_scale)
            # b ~ Horseshoe(0, b_tau)
            # b_tau ~ cauchy(0, b_tau_scale)

            bijectors[f"w_{j}"] = tfp.bijectors.Identity()

            distribution_dict[f"w_{j}"] = eval(
                horseshoe_lambda_code.format(
                    f"w_{j}_tau", f"w_{j}_tau", 2, "tf." + self.dtype.name
                )
            )

            ## bias
            bijectors[f"b_{j}"] = tfp.bijectors.Identity()
            distribution_dict[f"b_{j}"] = eval(
                horseshoe_lambda_code.format(
                    f"b_{j}_tau", f"b_{j}_tau", 1, "tf." + self.dtype.name
                )
            )

            # using the auxiliary inverse-gamma parameterization for cauchy vars

            # Scale
            bijectors[f"w_{j}_tau"] = tfp.bijectors.Softplus()
            distribution_dict[f"w_{j}_tau"] = eval(
                sq_igamma_code.format(
                    f"({self.layer_sizes[j]}, {self.layer_sizes[j+1]})",
                    f"w_{j}_tau_a",
                    2,
                    "tf." + self.dtype.name,
                )
            )
            bijectors[f"b_{j}_tau"] = tfp.bijectors.Softplus()
            distribution_dict[f"b_{j}_tau"] = eval(
                sq_igamma_code.format(
                    f"({self.layer_sizes[j+1]}, )",
                    f"b_{j}_tau_a",
                    1,
                    "tf." + self.dtype.name,
                )
            )

            # auxiliary scale variables
            bijectors[f"w_{j}_tau_a"] = tfp.bijectors.Softplus()
            distribution_dict[f"w_{j}_tau_a"] = eval(
                igamma_code.format(
                    f"({self.layer_sizes[j]}, {self.layer_sizes[j+1]})",
                    1.0,
                    2,
                    "tf." + self.dtype.name,
                )
            )

            bijectors[f"b_{j}_tau_a"] = tfp.bijectors.Softplus()
            distribution_dict[f"b_{j}_tau_a"] = eval(
                igamma_code.format(
                    f"({self.layer_sizes[j+1]}, )",
                    1.0,
                    1,
                    "tf." + self.dtype.name,
                )
            )

        self.bijectors = bijectors
        self.prior_distribution = tfd.JointDistributionNamed(distribution_dict)
        self.surrogate_distribution = build_surrogate_posterior(
            self.prior_distribution, bijectors, dtype=self.dtype
        )
        self.var_list = list(self.surrogate_distribution.model.keys())
        self.surrogate_vars = self.surrogate_distribution.variables

    def sample(self, *args, **kwargs):
        return self.surrogate_distribution.sample(*args, **kwargs)


def main():
    denseH = DenseHorseshoe(10, [20, 12, 2])
    sample = denseH.prior_distribution.sample(10)
    prob = denseH.log_prob(sample)
    sample2 = denseH.surrogate_distribution.sample(10)
    prob2 = denseH.log_prob(sample2)
    networks = denseH.assemble_networks(sample)
    networks(tf.ones((1, 2, 10)))
    return


if __name__ == "__main__":
    main()

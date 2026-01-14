
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf
from bayesianquilts.predictors.nn.dense import DenseGaussian

class NeuralPoissonRegression(DenseGaussian):
    def __init__(
        self,
        dim_regressors: int,
        hidden_size: int = 4,
        depth: int = 2,
        dtype: tf.DType = jnp.float32,
        **kwargs
    ):
        # Architecture: input -> [hidden]*depth -> output (1)
        layer_sizes = [hidden_size] * depth + [1]

        super(NeuralPoissonRegression, self).__init__(
            input_size=dim_regressors,
            layer_sizes=layer_sizes,
            activation_fn=jax.nn.relu,
            weight_scale=1.0,
            bias_scale=1.0,
            dtype=dtype,
            **kwargs
        )
        self.input_dim = dim_regressors
        # Fix for DenseHorseshoe not setting var_list
        # We must use latent variable names from prior, NOT variational param names from params
        if hasattr(self, 'prior_distribution') and hasattr(self.prior_distribution, 'model'):
             self.var_list = list(self.prior_distribution.model.keys())
        elif hasattr(self, 'params') and self.params is not None:
             # Fallback, though likely incorrect for ADVI
             self.var_list = list(self.params.keys())


    def predictive_distribution(self, data: dict, **params):
        X = data["X"].astype(self.dtype)

        # Output of eval is (batch, 1) usually, or (samples, batch, 1)
        # We need to handle parameter sampling dimensions
        out = self.eval(X, params) # Shape: (..., batch_size, 1)

        # Squeeze the last dimension to get rate log-scale
        log_rate = jnp.squeeze(out, axis=-1) 
        rate = jnp.exp(log_rate)

        log_lik = None
        if 'y' in data:
            rv = tfd.Poisson(rate=rate)
            log_lik = rv.log_prob(data['y'])

        return {
            "prediction": rate,
            "log_likelihood": log_lik,
             "log_rate": log_rate
        }

    def log_likelihood(self, data, **params):
        return self.predictive_distribution(data, **params)["log_likelihood"]

    def unormalized_log_prob(self, data, prior_weight=1.0, **params):
        log_lik = self.log_likelihood(data, **params)
        prior = self.prior_distribution.log_prob(params)
        # Sum log_lik over data points?
        # log_lik shape: (S, N) or (N,)
        if log_lik.ndim > 1:
            total_ll = jnp.sum(log_lik, axis=-1)
        else:
            total_ll = jnp.sum(log_lik)

        return total_ll + prior * prior_weight

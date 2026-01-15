
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
        output_scale: float = 1.0,
        prior_scale: float = 1.0,
        dtype: tf.DType = jnp.float32,
        **kwargs
    ):
        # Architecture: input -> [hidden]*depth -> output (1)
        layer_sizes = [hidden_size] * depth + [1]

        super(NeuralPoissonRegression, self).__init__(
            input_size=dim_regressors,
            layer_sizes=layer_sizes,
            activation_fn=jax.nn.relu,
            weight_scale=0.1,
            bias_scale=0.1,
            prior_scale=prior_scale,
            dtype=dtype,
            **kwargs
        )
        self.input_dim = dim_regressors
        self.dim_regressors = dim_regressors
        self.hidden_size = hidden_size
        self.depth = depth
        self.output_scale = output_scale
        pass


    def predictive_distribution(self, data: dict, **params):
        X = data["X"].astype(self.dtype)

        # Output of eval is (batch, 1) usually, or (samples, batch, 1)
        # We need to handle parameter sampling dimensions
        out = self.eval(X, params) # Shape: (..., batch_size, 1)

        # Squeeze the last dimension to get rate log-scale
        log_rate = jnp.squeeze(out, axis=-1)
        # Apply output scaling: rate = output_scale * exp(network_output)
        log_rate = log_rate + jnp.log(self.output_scale)
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

from bayesianquilts.metrics.ais import LikelihoodFunction
import jax.flatten_util

class NeuralPoissonLikelihood(LikelihoodFunction):
    def __init__(self, model):
        self.model = model
        self.dtype = model.dtype

    def log_likelihood(self, data, params):
        w0 = params['w_0']
        # Check w_0 rank to detect if we have per-datum parameters
        # Standard MCMC: (S, D_in, H) -> Rank 3
        # AIS per-datum: (S, N, D_in, H) -> Rank 4
        if w0.ndim == 4:
             # Case: (S, N, D, H). We must map over N.
             X = data['X']
             y = data['y']

             def single_data_ll(x_i, y_i, params_i):
                 # params_i: (S, D, H)
                 d = {'X': x_i[None, :], 'y': y_i} 
                 ll = self.model.log_likelihood(d, **params_i)
                 # ll should be (S, 1) or (S,)
                 return jnp.squeeze(ll)

             # Map over N (axis 0 of data, axis 1 of params)
             in_axes_params = jax.tree_util.tree_map(lambda x: 1, params)

             # Result: (N, S)
             ll_val = jax.vmap(single_data_ll, in_axes=(0, 0, in_axes_params))(X, y, params)
             
             # Return (S, N). Use swapaxes to be safe for high rank
             # If ll_val is (N, S), swaps to (S, N)
             return jnp.swapaxes(ll_val, 0, 1)
        else:
             return self.model.log_likelihood(data, **params)

    def _flatten_params(self, params):
        flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(params)
        return flat_params, unflatten_fn

    def log_likelihood_gradient(self, data, params):
        # Optimized gradient computation per datum
        one_sample_params = jax.tree_util.tree_map(lambda x: x[0], params)
        flat_proto, unflatten = jax.flatten_util.ravel_pytree(one_sample_params)
        flat_params_S = jax.vmap(lambda p: jax.flatten_util.ravel_pytree(p)[0])(params)
        X = data['X']
        y = data['y']

        def log_lik_fn(flat_theta, x, y):
            theta = unflatten(flat_theta)
            d = {'X': x[None, :], 'y': y}
            ll = self.model.log_likelihood(d, **theta)
            return jnp.squeeze(ll)

        grad_fn = jax.grad(log_lik_fn)
        grad_vmap_N = jax.vmap(grad_fn, in_axes=(None, 0, 0))
        grads = jax.vmap(lambda p: grad_vmap_N(p, X, y))(flat_params_S)
        return grads

    def log_likelihood_hessian_diag(self, data, params):
        one_sample_params = jax.tree_util.tree_map(lambda x: x[0], params)
        flat_proto, unflatten = jax.flatten_util.ravel_pytree(one_sample_params)
        flat_params_S = jax.vmap(lambda p: jax.flatten_util.ravel_pytree(p)[0])(params)
        X = data['X']
        y = data['y']

        def log_lik_fn(flat_theta, x, y):
            theta = unflatten(flat_theta)
            d = {'X': x[None, :], 'y': y}
            ll = self.model.log_likelihood(d, **theta)
            return jnp.squeeze(ll)

        def hess_diag_fn(flat_theta, x, y):
            return jnp.diag(jax.hessian(log_lik_fn)(flat_theta, x, y))

        hess_diag_vmap_N = jax.vmap(hess_diag_fn, in_axes=(None, 0, 0))
        hess_diag = jax.vmap(lambda p: hess_diag_vmap_N(p, X, y))(flat_params_S)
        return hess_diag

    def extract_parameters(self, params):
        flat_params = jax.vmap(lambda p: jax.flatten_util.ravel_pytree(p)[0])(params)
        return flat_params

    def reconstruct_parameters(self, flat_params, template):
        if isinstance(template.get('w_0'), jnp.ndarray) and template['w_0'].ndim > 2:
             template = jax.tree_util.tree_map(lambda x: x[0], template)
        dummy_flat, unflatten = jax.flatten_util.ravel_pytree(template)
        K = dummy_flat.shape[0]
        input_shape = flat_params.shape
        if input_shape[-1] != K:
             raise ValueError(f"Last dimension {input_shape} != K={K}")
        batch_dims = input_shape[:-1]
        n_batch = 1
        for d in batch_dims: n_batch *= d
        flat_reshaped = flat_params.reshape((n_batch, K))
        unflattened_flat = jax.vmap(unflatten)(flat_reshaped)
        def reshape_leaf(leaf):
             leaf_param_shape = leaf.shape[1:]
             return leaf.reshape(batch_dims + leaf_param_shape)
        return jax.tree_util.tree_map(reshape_leaf, unflattened_flat)


import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf
from bayesianquilts.predictors.nn.dense import DenseHorseshoe

class NeuralNegativeBinomialRegression(DenseHorseshoe):
    def __init__(
        self,
        dim_regressors: int,
        hidden_size: int = 4,
        depth: int = 2,
        output_scale: float = 1.0,
        zero_inflated: bool = True,
        dtype: tf.DType = jnp.float32,
        **kwargs
    ):
        """
        Neural Negative Binomial Regression with optional zero-inflation.

        Args:
            dim_regressors: Number of input features
            hidden_size: Number of neurons per hidden layer
            depth: Number of hidden layers
            output_scale: Scaling factor for mean parameter (typically y.mean())
            zero_inflated: If True, uses zero-inflated negative binomial
            dtype: Data type for computations

        Network outputs:
            If zero_inflated=True: [zero_logit, log_mean, log_concentration]
            If zero_inflated=False: [log_mean, log_concentration]

        Negative Binomial parameterization:
            mean: Expected count (mu)
            concentration: Overdispersion parameter (r or alpha)
            variance = mean + mean^2 / concentration
            As concentration -> inf, approaches Poisson
        """
        # Output: 3 values if zero-inflated, 2 otherwise
        output_size = 3 if zero_inflated else 2
        layer_sizes = [hidden_size] * depth + [output_size]

        super(NeuralNegativeBinomialRegression, self).__init__(
            input_size=dim_regressors,
            layer_sizes=layer_sizes,
            activation_fn=jax.nn.relu,
            weight_scale=0.05,
            bias_scale=1.0,
            dtype=dtype,
            **kwargs
        )
        self.input_dim = dim_regressors
        self.dim_regressors = dim_regressors
        self.hidden_size = hidden_size
        self.depth = depth
        self.output_scale = output_scale
        self.zero_inflated = zero_inflated

    def predictive_distribution(self, data: dict, **params):
        X = data["X"].astype(self.dtype)

        # Output of eval is (..., batch_size, output_size)
        out = self.eval(X, params)

        if self.zero_inflated:
            # Split outputs: [zero_logit, log_mean, log_concentration]
            zero_logit = out[..., 0]  # Logit for zero-inflation probability
            log_mean = out[..., 1]
            log_concentration = out[..., 2]

            # Zero-inflation probability
            zero_prob = jax.nn.sigmoid(zero_logit)
        else:
            # Split outputs: [log_mean, log_concentration]
            log_mean = out[..., 0]
            log_concentration = out[..., 1]
            zero_prob = None

        # Apply output scaling to mean
        log_mean = log_mean + jnp.log(self.output_scale)
        mean = jnp.exp(log_mean)

        # Concentration (overdispersion parameter)
        # Add small constant for numerical stability
        concentration = jnp.exp(log_concentration) + 1e-6

        # Convert (mean, concentration) to NegativeBinomial parameters
        # For NegativeBinomial(total_count, probs):
        #   mean = total_count * probs / (1 - probs)
        # Solving: total_count = concentration, probs = concentration / (concentration + mean)
        total_count = concentration
        probs = concentration / (concentration + mean)

        # Clip probs for numerical stability
        probs = jnp.clip(probs, 1e-6, 1 - 1e-6)

        log_lik = None
        if 'y' in data:
            y = data['y']

            if self.zero_inflated:
                # Zero-inflated negative binomial log-likelihood
                # P(Y=0) = zero_prob + (1 - zero_prob) * NB(0)
                # P(Y>0) = (1 - zero_prob) * NB(y)

                nb_dist = tfd.NegativeBinomial(total_count=total_count, probs=probs)
                nb_logprob = nb_dist.log_prob(y)

                # For y=0: log(zero_prob + (1-zero_prob)*exp(nb_logprob(0)))
                # For y>0: log(1-zero_prob) + nb_logprob(y)
                nb_logprob_zero = nb_dist.log_prob(jnp.zeros_like(y))

                # Compute log-likelihood for y=0 case
                # log(p + (1-p)*q) = log(p + (1-p)*q) using logsumexp
                log_zero_prob = jnp.log(zero_prob + 1e-10)
                log_one_minus_zero_prob = jnp.log(1 - zero_prob + 1e-10)

                ll_zero = jnp.logaddexp(
                    log_zero_prob,
                    log_one_minus_zero_prob + nb_logprob_zero
                )

                # Compute log-likelihood for y>0 case
                ll_nonzero = log_one_minus_zero_prob + nb_logprob

                # Select based on whether y is zero
                log_lik = jnp.where(y == 0, ll_zero, ll_nonzero)
            else:
                # Standard negative binomial
                rv = tfd.NegativeBinomial(total_count=total_count, probs=probs)
                log_lik = rv.log_prob(y)

        result = {
            "prediction": mean,
            "log_likelihood": log_lik,
            "mean": mean,
            "concentration": concentration,
        }

        if self.zero_inflated:
            result["zero_prob"] = zero_prob

        return result

    def log_likelihood(self, data, **params):
        return self.predictive_distribution(data, **params)["log_likelihood"]

    def unormalized_log_prob(self, data, prior_weight=1.0, **params):
        log_lik = self.log_likelihood(data, **params)
        prior = self.prior_distribution.log_prob(params)

        if log_lik.ndim > 1:
            total_ll = jnp.sum(log_lik, axis=-1)
        else:
            total_ll = jnp.sum(log_lik)

        return total_ll + prior * prior_weight

from bayesianquilts.metrics.ais import LikelihoodFunction
import jax.flatten_util

class NeuralNegativeBinomialLikelihood(LikelihoodFunction):
    def __init__(self, model):
        self.model = model
        self.dtype = model.dtype

    def log_likelihood(self, data, params):
        w0 = params['w_0']
        if w0.ndim == 4:
             # Case: (S, N, D, H). We must map over N.
             X = data['X']
             y = data['y']

             def single_data_ll(x_i, y_i, params_i):
                 d = {'X': x_i[None, :], 'y': y_i}
                 ll = self.model.log_likelihood(d, **params_i)
                 return jnp.squeeze(ll)

             in_axes_params = jax.tree_util.tree_map(lambda x: 1, params)
             ll_val = jax.vmap(single_data_ll, in_axes=(0, 0, in_axes_params))(X, y, params)
             return jnp.swapaxes(ll_val, 0, 1)
        else:
             return self.model.log_likelihood(data, **params)

    def _flatten_params(self, params):
        flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(params)
        return flat_params, unflatten_fn

    def log_likelihood_gradient(self, data, params):
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

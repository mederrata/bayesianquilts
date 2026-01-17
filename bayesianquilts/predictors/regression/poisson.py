
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
from bayesianquilts.model import BayesianModel

class PoissonRegression(BayesianModel):
    def __init__(self, input_dim, dtype=jnp.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.input_dim = input_dim
        self.var_list = ['beta', 'intercept']
        self.create_distributions()

    def create_distributions(self):
        # Surrogate: Mean-field Normal
        pass # Using generator

    def surrogate_parameter_initializer(self, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key, 2)
        
        def init_mean_scale(shape, k):
            ska, skb = jax.random.split(k)
            mean = jax.random.normal(ska, shape, dtype=self.dtype) * 0.01
            raw_scale = jnp.log(jnp.exp(0.01) - 1.0) + jax.random.normal(skb, shape, dtype=self.dtype) * 0.001
            return mean, raw_scale

        beta_loc, beta_scale = init_mean_scale((self.input_dim,), k1)
        int_loc, int_scale = init_mean_scale((1,), k2)

        return {
            'beta_loc': beta_loc, 'beta_raw_scale': beta_scale,
            'intercept_loc': int_loc, 'intercept_raw_scale': int_scale
        }

    def surrogate_distribution_generator(self, params):
        return tfd.JointDistributionNamed({
            'beta': tfd.Independent(tfd.Normal(params['beta_loc'], jax.nn.softplus(params['beta_raw_scale']) + 1e-5), reinterpreted_batch_ndims=1),
            'intercept': tfd.Independent(tfd.Normal(params['intercept_loc'], jax.nn.softplus(params['intercept_raw_scale']) + 1e-5), reinterpreted_batch_ndims=1)
        })

    def unormalized_log_prob(self, data, prior_weight=1.0, **params):
        X = data['X']
        y = data['y']
        
        beta = params['beta']
        intercept = params['intercept']
        
        has_sample_dim = (beta.ndim > 1)
        
        # Priors
        if has_sample_dim:
             p_beta = jnp.sum(tfd.Normal(0., 10.).log_prob(beta), axis=-1)
             p_int = jnp.sum(tfd.Normal(0., 10.).log_prob(intercept), axis=-1)
             log_prior = p_beta + p_int
             
             # Likelihood
             # X: (B, D), beta: (S, D) -> (S, B)
             eta = jnp.einsum('bd,sd->sb', X, beta) + intercept
        else:
            log_prior = jnp.sum(tfd.Normal(0., 10.).log_prob(beta)) + jnp.sum(tfd.Normal(0., 10.).log_prob(intercept))
            eta = jnp.dot(X, beta) + intercept
            
        rate = jnp.exp(eta)
        
        # Poisson Log Likelihood
        if has_sample_dim:
            log_lik = tfd.Poisson(rate=rate).log_prob(y)
            log_lik = jnp.sum(log_lik, axis=-1)
        else:
            log_lik = jnp.sum(tfd.Poisson(rate=rate).log_prob(y))
            
        return log_lik + log_prior * prior_weight

    def log_likelihood(self, data, **params):
        X = data['X']
        y = data['y']
        beta = params['beta']
        intercept = params['intercept']
        
        has_sample_dim = (beta.ndim > 1)
        
        if has_sample_dim:
            eta = jnp.einsum('bd,sd->sb', X, beta) + intercept
        else:
            eta = jnp.dot(X, beta) + intercept
            
        rate = jnp.exp(eta)
        # Returns (S, B) or (B,)
        return tfd.Poisson(rate=rate).log_prob(y)

    def predictive_distribution(self, data, **params):
        X = data['X']
        beta = params['beta']
        intercept = params['intercept']
        
        has_sample_dim = (beta.ndim > 1)
        if has_sample_dim:
             eta = jnp.einsum('bd,sd->sb', X, beta) + intercept
        else:
             eta = jnp.dot(X, beta) + intercept
        
        rate = jnp.exp(eta)
        return {'prediction': rate}

from typing import Dict, Any

from bayesianquilts.metrics.ais import LikelihoodFunction, AutoDiffLikelihoodMixin

class PoissonRegressionLikelihood(AutoDiffLikelihoodMixin):
    """Likelihood function for Poisson regression."""

    def __init__(self, dtype=jnp.float32):
        self.dtype = dtype

    def log_likelihood(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        """Compute log-likelihood for Poisson regression."""
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)  # counts
        y = jnp.squeeze(y)

        beta = params["beta"]
        intercept = params["intercept"]
        
        # Handle accumulated dimensions from chained transformations
        # Both beta and intercept can accumulate dimensions, so we need to handle both
        
        # Squeeze beta down to at most 3D (S, N, F) or 2D (S, F)
        while beta.ndim > 3:
            beta = jnp.squeeze(beta, axis=tuple(i for i in range(1, beta.ndim-1) if beta.shape[i] == 1))
            if beta.ndim > 3:
                # If still 4D+ and no singleton dims, take diagonal or first slice
                # For (S, N, N, F), we want (S, N, F) by taking diagonal
                if beta.ndim == 4 and beta.shape[1] == beta.shape[2]:
                    # Take diagonal: beta[s, n, n, f] -> beta[s, n, f]
                    beta = jnp.einsum('snnf->snf', beta)
                else:
                    # Fallback: just take first slice along axis 1
                    beta = beta[:, 0, ...]
        
        # Squeeze intercept to match beta's dimensionality
        # If beta is 3D (S, N, F), intercept should be (S, N) or (S,)
        # If beta is 2D (S, F), intercept should be (S,)
        while intercept.ndim > 2:
            intercept = jnp.squeeze(intercept, axis=tuple(i for i in range(1, intercept.ndim-1) if intercept.shape[i] == 1))
            if intercept.ndim > 2:
                if intercept.ndim == 3 and intercept.shape[1] == intercept.shape[2]:
                    # Take diagonal for (S, N, N) -> (S, N)
                    intercept = jnp.einsum('snn->sn', intercept)
                else:
                    intercept = intercept[:, 0, ...]
        
        # Final squeeze to remove any trailing singleton dimensions
        intercept = jnp.squeeze(intercept)
        if intercept.ndim == 0:
            intercept = jnp.atleast_1d(intercept)
        
        # Now beta should be 2D or 3D, and intercept should match appropriately
        if beta.ndim == 2:
            # Shape (S, F), intercept should be (S,)
            log_rate = jnp.einsum('df,sf->sd', X, beta) + intercept[:, jnp.newaxis]
        elif beta.ndim == 3:
            # Shape (S, N, F), intercept could be (S,) or (S, N)
            if intercept.ndim == 1:
                # Broadcast (S,) to match (S, N) output from einsum
                log_rate = jnp.einsum('df,sdf->sd', X, beta) + intercept[:, jnp.newaxis]
            else:
                # intercept is already (S, N)
                log_rate = jnp.einsum('df,sdf->sd', X, beta) + intercept
        else:
            raise ValueError(f"beta shape {beta.shape} not supported in log_likelihood after squeezing")

        rate = jnp.exp(log_rate)

        # Poisson log-likelihood: y * log(rate) - rate
        # We omit log(y!)
        log_lik = y[jnp.newaxis, :] * log_rate - rate

        return log_lik

    def extract_parameters(self, params: Dict[str, Any]) -> jnp.ndarray:
        """Extract parameters into flattened array."""
        beta = params["beta"]
        intercept = params["intercept"]
        intercept = jnp.squeeze(intercept)
        theta = jnp.concatenate([beta, intercept[..., jnp.newaxis]], axis=-1)
        return theta

    def reconstruct_parameters(self, flat_params: jnp.ndarray, template: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct parameters from flattened array."""
        n_features = template["beta"].shape[-1]
        beta = flat_params[..., :n_features]
        intercept = flat_params[..., n_features]
        return {"beta": beta, "intercept": intercept[..., jnp.newaxis]}

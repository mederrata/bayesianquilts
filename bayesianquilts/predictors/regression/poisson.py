
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

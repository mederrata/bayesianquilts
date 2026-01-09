
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb
from bayesianquilts.model import BayesianModel

class BayesianMultinomialRegression(BayesianModel):

    def __init__(
        self,
        input_dim,
        num_classes,
        global_shrinkage=1.0,
        dtype=jnp.float32,
        **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.global_shrinkage = global_shrinkage
        self.var_list = ['beta', 'intercept']
        self.create_distributions()

    def create_distributions(self):
        # Surrogate
        # Mean-field Normal for everything
        self.surrogate_distribution = tfd.JointDistributionNamedAutoBatched({
            'beta': tfd.Normal(
                loc=jax.nn.initializers.zeros(jax.random.PRNGKey(0), (self.input_dim, self.num_classes), dtype=self.dtype),
                scale=1e-2 + jax.nn.softplus(jax.nn.initializers.uniform(scale=0.01)(jax.random.PRNGKey(1), (self.input_dim, self.num_classes), dtype=self.dtype))
            ),
            'intercept': tfd.Normal(
                loc=jax.nn.initializers.zeros(jax.random.PRNGKey(2), (self.num_classes,), dtype=self.dtype),
                scale=1e-2 + jax.nn.softplus(jax.nn.initializers.uniform(scale=0.01)(jax.random.PRNGKey(3), (self.num_classes,), dtype=self.dtype))
            )
        })

    def surrogate_parameter_initializer(self, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(42)
        
        k1, k2 = jax.random.split(key, 2)
        
        def init_mean_scale(shape, k):
            ska, skb = jax.random.split(k)
            mean = jax.random.normal(ska, shape, dtype=self.dtype) * 0.01
            raw_scale = jnp.log(jnp.exp(0.01) - 1.0) + jax.random.normal(skb, shape, dtype=self.dtype) * 0.001
            return mean, raw_scale

        beta_loc, beta_scale = init_mean_scale((self.input_dim, self.num_classes), k1)
        int_loc, int_scale = init_mean_scale((self.num_classes,), k2)

        return {
            'beta_loc': beta_loc, 'beta_raw_scale': beta_scale,
            'intercept_loc': int_loc, 'intercept_raw_scale': int_scale
        }

    def surrogate_distribution_generator(self, params):
        return tfd.JointDistributionNamed({
            'beta': tfd.Independent(tfd.Normal(params['beta_loc'], jax.nn.softplus(params['beta_raw_scale']) + 1e-5), reinterpreted_batch_ndims=2),
            'intercept': tfd.Independent(tfd.Normal(params['intercept_loc'], jax.nn.softplus(params['intercept_raw_scale']) + 1e-5), reinterpreted_batch_ndims=1)
        })

    def unormalized_log_prob(self, data, prior_weight=1.0, **params):
        X = data['X']
        y = data['y']
        
        beta = params['beta']
        intercept = params['intercept']
        
        # Check for sample dimension
        has_sample_dim = (beta.ndim == 3)
        
        # --- Priors ---
        # Sum over parameter dimensions (D, K, etc), but KEEP sample dimension (S)
        sum_axes = list(range(1, beta.ndim)) if has_sample_dim else None
        
        # beta ~ Normal(0, global_shrinkage)
        p_beta = tfd.Normal(loc=0., scale=self.global_shrinkage).log_prob(beta)
        p_beta = jnp.sum(p_beta, axis=sum_axes) # (S,) or scalar
        
        p_int = tfd.Normal(0., 5.).log_prob(intercept)
        p_int = jnp.sum(p_int, axis=(1 if has_sample_dim else 0)) 
        
        log_prior = p_beta + p_int
        
        # --- Likelihood ---
        if has_sample_dim:
            # X: (B, D), beta: (S, D, K) -> (S, B, K)
            logits = jnp.einsum('bd,sdk->sbk', X, beta)
            # intercept: (S, K) -> (S, 1, K)
            logits = logits + intercept[:, None, :]
            
            # log_prob(y): y is (B,), logits (S, B, K) -> (S, B)
            log_lik = tfd.Categorical(logits=logits).log_prob(y)
            log_lik = jnp.sum(log_lik, axis=1) # Sum over Batch -> (S,)
        else:
            logits = jnp.dot(X, beta) + intercept
            log_lik = jnp.sum(tfd.Categorical(logits=logits).log_prob(y))
        
        return log_lik + log_prior * prior_weight

    def predict_probs(self, params, X):
        beta = params['beta']
        intercept = params['intercept']
        
        has_sample_dim = (beta.ndim == 3)
        
        if has_sample_dim:
            # X: (N, D), beta: (S, D, K) -> (S, N, K)
            logits = jnp.einsum('nd,sdk->snk', X, beta)
            # intercept: (S, K) -> (S, 1, K)
            logits = logits + intercept[:, None, :]
        else:
            logits = jnp.dot(X, beta) + intercept
            
        return jax.nn.softmax(logits, axis=-1)

    def predictive_distribution(self, data, **params):
        X = data['X']
        probs = self.predict_probs(params, X)
        return tfd.Categorical(probs=probs)
        
    def log_likelihood(self, data, params):
        X = data['X']
        y = data['y']
        
        beta = params['beta']
        intercept = params['intercept']
        
        has_sample_dim = (beta.ndim == 3)
        
        if has_sample_dim:
            # X: (B, D), beta: (S, D, K) -> (S, B, K)
            logits = jnp.einsum('bd,sdk->sbk', X, beta)
            # intercept: (S, K) -> (S, 1, K)
            logits = logits + intercept[:, None, :]
            
            # log_prob(y): y is (B,), logits (S, B, K) -> (S, B)
            log_lik = tfd.Categorical(logits=logits).log_prob(y)
        else:
            logits = jnp.dot(X, beta) + intercept
            log_lik = tfd.Categorical(logits=logits).log_prob(y)
            
        return log_lik


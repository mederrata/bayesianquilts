
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
        self.var_list = ['beta', 'intercept', 'tau', 'lam']
        self.create_distributions()

    def create_distributions(self):
        # Priors
        # tau ~ HalfCauchy(0, global_shrinkage)
        prior_tau = tfd.HalfCauchy(loc=0., scale=self.global_shrinkage)
        
        # lam ~ HalfCauchy(0, 1) -> shape (input_dim, num_classes)
        prior_lam = tfd.Sample(tfd.HalfCauchy(loc=0., scale=1.), sample_shape=[self.input_dim, self.num_classes])
        
        # intercept ~ Normal(0, 5)
        prior_intercept = tfd.Sample(tfd.Normal(0., 5.), sample_shape=[self.num_classes])

        # beta is conditional on tau and lam, handled in log_prob usually, 
        # but for consistency we can define a hierarchy or just leaving it implicit in log_prob.
        # However, for BQ structure, we usually define the joint if we can. 
        # But Horseshoe is often easier to handle by just defining the log_prob term.
        
        self.prior_distribution = None # Explicit joint not strictly needed if unormalized_log_prob handles it

        # Surrogate
        # Mean-field Normal for everything
        # To handle positive constraints on tau and lam, we model log_tau and log_lam in surrogate
        self.surrogate_distribution = tfd.JointDistributionNamedAutoBatched({
            'beta': tfd.Normal(
                loc=jax.nn.initializers.zeros(jax.random.PRNGKey(0), (self.input_dim, self.num_classes), dtype=self.dtype),
                scale=1e-2 + jax.nn.softplus(jax.nn.initializers.uniform(scale=0.01)(jax.random.PRNGKey(1), (self.input_dim, self.num_classes), dtype=self.dtype))
            ),
            'intercept': tfd.Normal(
                loc=jax.nn.initializers.zeros(jax.random.PRNGKey(2), (self.num_classes,), dtype=self.dtype),
                scale=1e-2 + jax.nn.softplus(jax.nn.initializers.uniform(scale=0.01)(jax.random.PRNGKey(3), (self.num_classes,), dtype=self.dtype))
            ),
            'log_tau': tfd.Normal(
                loc=jax.nn.initializers.zeros(jax.random.PRNGKey(4), (), dtype=self.dtype),
                scale=1e-2 + jax.nn.softplus(jax.nn.initializers.uniform(scale=0.01)(jax.random.PRNGKey(5), (), dtype=self.dtype))
            ),
            'log_lam': tfd.Normal(
                loc=jax.nn.initializers.zeros(jax.random.PRNGKey(6), (self.input_dim, self.num_classes), dtype=self.dtype),
                scale=1e-2 + jax.nn.softplus(jax.nn.initializers.uniform(scale=0.01)(jax.random.PRNGKey(7), (self.input_dim, self.num_classes), dtype=self.dtype))
            )
        })

    def surrogate_parameter_initializer(self, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(42)
        
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        def init_mean_scale(shape, k):
            ska, skb = jax.random.split(k)
            mean = jax.random.normal(ska, shape, dtype=self.dtype) * 0.01
            raw_scale = jnp.log(jnp.exp(0.01) - 1.0) + jax.random.normal(skb, shape, dtype=self.dtype) * 0.001
            return mean, raw_scale

        beta_loc, beta_scale = init_mean_scale((self.input_dim, self.num_classes), k1)
        int_loc, int_scale = init_mean_scale((self.num_classes,), k2)
        tau_loc, tau_scale = init_mean_scale((), k3)
        lam_loc, lam_scale = init_mean_scale((self.input_dim, self.num_classes), k4)

        return {
            'beta_loc': beta_loc, 'beta_raw_scale': beta_scale,
            'intercept_loc': int_loc, 'intercept_raw_scale': int_scale,
            'log_tau_loc': tau_loc, 'log_tau_raw_scale': tau_scale,
            'log_lam_loc': lam_loc, 'log_lam_raw_scale': lam_scale
        }

    def surrogate_distribution_generator(self, params):
        return tfd.JointDistributionNamed({
            'beta': tfd.Independent(tfd.Normal(params['beta_loc'], jax.nn.softplus(params['beta_raw_scale']) + 1e-5), reinterpreted_batch_ndims=2),
            'intercept': tfd.Independent(tfd.Normal(params['intercept_loc'], jax.nn.softplus(params['intercept_raw_scale']) + 1e-5), reinterpreted_batch_ndims=1),
            'log_tau': tfd.Normal(params['log_tau_loc'], jax.nn.softplus(params['log_tau_raw_scale']) + 1e-5),
            'log_lam': tfd.Independent(tfd.Normal(params['log_lam_loc'], jax.nn.softplus(params['log_lam_raw_scale']) + 1e-5), reinterpreted_batch_ndims=2)
        })

    def unormalized_log_prob(self, data, prior_weight=1.0, **params):
        X = data['X']
        y = data['y']
        
        beta = params['beta']
        intercept = params['intercept']
        tau = jnp.exp(params['log_tau'])
        lam = jnp.exp(params['log_lam'])
        
        # Check for sample dimension
        # If params come from VI sampling, they have shape (S, ...).
        # We assume if beta is 3D (S, D, K), we have samples.
        # If beta is 2D (D, K), we don't.
        has_sample_dim = (beta.ndim == 3)
        
        # --- Priors ---
        # Sum over parameter dimensions (D, K, etc), but KEEP sample dimension (S)
        sum_axes = list(range(1, beta.ndim)) if has_sample_dim else None
        
        # log_prob(tau) + transform correction
        # tau is scalar or (S,)
        # log_prob is (S,) or scalar
        p_tau = tfd.HalfCauchy(loc=0., scale=self.global_shrinkage).log_prob(tau) + params['log_tau']
        
        # lam: (S, D, K) or (D, K)
        p_lam = tfd.HalfCauchy(loc=0., scale=1.).log_prob(lam) + params['log_lam']
        p_lam = jnp.sum(p_lam, axis=sum_axes) # (S,) or scalar
        
        sigma_beta = tau[..., None, None] * lam if has_sample_dim else tau * lam
        p_beta = tfd.Normal(loc=0., scale=sigma_beta).log_prob(beta)
        p_beta = jnp.sum(p_beta, axis=sum_axes) # (S,) or scalar
        
        p_int = tfd.Normal(0., 5.).log_prob(intercept)
        p_int = jnp.sum(p_int, axis=(1 if has_sample_dim else 0)) 
        
        log_prior = p_tau + p_lam + p_beta + p_int
        
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
        logits = jnp.dot(X, beta) + intercept
        return jax.nn.softmax(logits, axis=-1)

    def predictive_distribution(self, data, **params):
        X = data['X']
        probs = self.predict_probs(params, X)
        return {
            "probs": probs,
            "prediction": jnp.argmax(probs, axis=-1)
        }


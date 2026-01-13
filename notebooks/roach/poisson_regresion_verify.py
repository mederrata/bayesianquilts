
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import tensorflow_probability.substrates.jax as tfp
import matplotlib.pyplot as plt
import arviz as az
import os
import pkg_resources

tfd = tfp.distributions
tfb = tfp.bijectors

from bayesianquilts.model import BayesianModel
from bayesianquilts.vi.minibatch import minibatch_fit_surrogate_posterior
from bayesianquilts.metrics.ais import AdaptiveImportanceSampler, PoissonRegressionLikelihood

# Load the dataset
try:
    data_path = pkg_resources.resource_filename('bayesianquilts.data', 'roachdata.csv')
except ImportError:
    # Fallback if package not installed
    data_path = '../../bayesianquilts/data/roachdata.csv'
    if not os.path.exists(data_path):
        data_path = 'bayesianquilts/data/roachdata.csv'

print(f'Loading data from {data_path}')
df = pd.read_csv(data_path)
print(df.head())

# Preprocessing
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Target
y_data = df['y'].values

# Features
# Drop target and any index columns
X_df = df.drop(columns=['y'])
X_data = X_df.values.astype(np.float32)
y_data = y_data.astype(np.float32)

print(f'X shape: {X_data.shape}, y shape: {y_data.shape}')

class PoissonRegression(BayesianModel):
    def __init__(self, input_dim, dtype=jnp.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.input_dim = input_dim
        self.var_list = ['beta', 'intercept']
        self.create_distributions()

    def create_distributions(self):
        # Surrogate: Mean-field Normal
        self.surrogate_distribution = tfd.JointDistributionNamedAutoBatched({
            'beta': tfd.Normal(
                loc=jax.nn.initializers.zeros(jax.random.PRNGKey(0), (self.input_dim,), dtype=self.dtype),
                scale=1e-2 + jax.nn.softplus(jax.nn.initializers.uniform(scale=0.01)(jax.random.PRNGKey(1), (self.input_dim,), dtype=self.dtype))
            ),
            'intercept': tfd.Normal(
                loc=jax.nn.initializers.zeros(jax.random.PRNGKey(2), (1,), dtype=self.dtype),
                scale=1e-2 + jax.nn.softplus(jax.nn.initializers.uniform(scale=0.01)(jax.random.PRNGKey(3), (1,), dtype=self.dtype))
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
        
        log_lik = None
        if 'y' in data:
            log_lik = tfd.Poisson(rate=rate).log_prob(data['y'])
            
        return {'prediction': rate, 'log_likelihood': log_lik}

model = PoissonRegression(input_dim=X_data.shape[1])

# Data factory for batching
def data_factory_builder(batch_size=None):
    if batch_size is None:
        batch_size = len(X_data)
    
    num_batches = int(np.ceil(len(X_data) / batch_size))
    
    def generator():
        indices = np.arange(len(X_data))
        np.random.shuffle(indices)
        for i in range(num_batches):
            idx = indices[i*batch_size : (i+1)*batch_size]
            if len(idx) > 0:
                yield {'X': X_data[idx], 'y': y_data[idx]}
            
    return generator

# Check init
print('Checking initialization...')
try:
    params = model.surrogate_distribution.sample(2)
    print('Log prob check:', model.unormalized_log_prob({'X': X_data, 'y': y_data}, **params))
except Exception as e:
    print(f'Initialization failed: {e}')
    pass

# Fit
print('Fitting model...')
batch_size = 32
losses, params = model.fit(
    batched_data_factory=data_factory_builder(batch_size),
    batch_size=batch_size,
    dataset_size=len(X_data),
    num_epochs=50,
    learning_rate=0.01,
    patience=20
)

plt.figure()
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('ADVI Loss')
plt.show()

# Evaluation with AIS
print('Computing LOO-IC using Adaptive Importance Sampling (AIS)...')

# Initialize Likelihood
likelihood_fn = PoissonRegressionLikelihood()

# Create Sampler
ais_sampler = AdaptiveImportanceSampler(likelihood_fn=likelihood_fn)

# Prepare data and params
# We need to extract params from the model and format them as expected by AIS
# AIS expects params as a dictionary of arrays with shape (n_samples, ...)
params = model.surrogate_distribution.sample(100) # Sample from posterior/surrogate

# Run AIS LOO
# We need to pass data as a dictionary {'X': ..., 'y': ...}
# Ensure data is jax arrays
data_jax = {'X': jnp.array(X_data), 'y': jnp.array(y_data)}

results = ais_sampler.adaptive_is_loo(
    data=data_jax,
    params=params,
    hbar=1.0,
    variational=True, # We used VI
    transformations=['ll', 'kl', 'var', 'identity']
)

for method, res in results.items():
    print(f'Method: {method}')
    print(f'  LOO-IC (eta): {-2 * jnp.sum(jnp.log(res["p_loo_eta"]))}')
    print(f'  LOO-IC (psis): {-2 * jnp.sum(jnp.log(res["p_loo_psis"]))}')
    print(f'  Pareto k (min/max): {res["khat"].min():.3f}/{res["khat"].max():.3f}')


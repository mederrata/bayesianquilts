
import json
import os
import nest_asyncio
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from tqdm import tqdm

nest_asyncio.apply()

import importlib

import jax.numpy as jnp

logistic_horseshoe_code = """
data {
  int <lower=0> N;                // number  of  observations
  int <lower=0> d;                // number  of  predictors
  array[N-1] int<lower=0,upper=1> y;      // outputs
  matrix[N-1,d] x;                  // inputs
  real <lower=0>  scale_icept;    // prior  std for  the  intercept
  real <lower=0>  scale_global;   // scale  for  the half -t prior  for  tau
  real <lower=1>  nu_global;      // degrees  of  freedom  for the half -t prior for tau
  real <lower=1> nu_local;        // degrees  of  freedom  for  the half -t priors for  lambdas
  real <lower=0>  slab_scale;     // slab  scale  for  the  regularized  horseshoe
  real <lower=0> slab_df;         // slab  degrees  of  freedom  for the  regularized horseshoe

  //int<lower=0> N_tilde;
  //matrix[N_tilde, d] x_tilde;
  //array[N_tilde] int<lower=0,upper=1> y_obs;
}
parameters {
  real  beta0;
  vector[d] z;
  real <lower=0> tau;             // global  shrinkage  parameter
  vector <lower =0>[d] lambda;    // local  shrinkage  parameter
  real <lower=0> caux;
}
transformed  parameters {
  vector <lower =0>[d] lambda_tilde;    // ’truncated ’ local  shrinkage  parameter
  real <lower=0> c;                     // slab  scale
  vector[d] beta;                       // regression  coefficients
  vector[N-1] f;                          // latent  function  values
  c = slab_scale * sqrt(caux);
  lambda_tilde = sqrt( c^2 * square(lambda) ./ (c^2 + tau^2* square(lambda )) );
  beta = z .*  lambda_tilde*tau;
  f = beta0 + x*beta;
}
model {
  z ~ normal(0.0, 1.0); // half -t priors  for  lambdas  and tau , and  inverse -gamma  for c^2
  lambda ~ student_t(nu_local , 0.0, 1.0);
  tau ~ student_t(nu_global , 0.0, scale_global);
  caux ~ inv_gamma (0.5* slab_df , 0.5* slab_df );
  beta0 ~ normal(0.0,  scale_icept );
  y ~ bernoulli_logit(f);
}
generated quantities {
  vector[N-1] log_lik;
  // vector[N_tilde] loo_log_lik;

  for (nn in 1:(N-1))
    log_lik[nn] = bernoulli_logit_lpmf(y[nn] | x[nn] * beta + beta0);

  //for (nn in 1:N_tilde)
  //  loo_log_lik[nn] = bernoulli_logit_lpmf(y_obs[nn] | x_tilde[nn] * beta + beta0);
}
"""

try:
  os.makedirs("/tmp/ovarian")
except:
  pass
with open("/tmp/ovarian/ovarian_model.stan", 'w') as f:
  f.writelines(logistic_horseshoe_code)


with importlib.resources.path('bayesianquilts.data',   "overianx.csv") as xpath:
  X = pd.read_csv(xpath, header=None)
with importlib.resources.path('bayesianquilts.data',   "overiany.csv") as ypath:
  y = pd.read_csv(ypath, header=None)

batch_size = 6

X_scaled = (X - X.mean())/X.std()
X_scaled = X_scaled.fillna(0)
n = X_scaled.shape[0]
p = X_scaled.shape[1]

print((n, p))

guessnumrelevcov = n / 10  # 20.
slab_scale = 2.5
scale_icept = 5.0
nu_global = 1
nu_local = 1
slab_df = 1
scale_global = guessnumrelevcov / ((p - guessnumrelevcov) * np.sqrt(n))

control = {"adapt_delta": 0.999, "max_treedepth": 14}


sm = CmdStanModel(stan_file="/tmp/ovarian/ovarian_model.stan")

for i in tqdm(range(n)):
    y_ = y.drop(i)
    X_ = X_scaled.drop(i)
    
    _ovarian_data = {
        "N": n,
        "d": p,
        "slab_df": slab_df,
        "slab_scale": slab_scale,
        "scale_icept": scale_icept,
        "nu_global": 1,
        "nu_local": 1,
        "scale_global": np.abs(scale_global),
        "y": y_.astype(int)[0].to_numpy().tolist(),
        "x": X_.to_numpy().tolist(),
    }
    
    with open(f"/tmp/ovarian/_ovarian_data_{i}.json", "w") as f:
        json.dump(_ovarian_data, f)
        
    fit = sm.sample(
        data=f"/tmp/ovarian/_ovarian_data_{i}.json",
        iter_warmup=20000,
        iter_sampling=2000,
        thin=2,
        adapt_delta=0.9995,
        max_treedepth=15,
    )
    
    params = fit.stan_variables()
    params.keys()
    params['c'] = params['c'][:, jnp.newaxis]
    params['tau'] = params['tau'][:, jnp.newaxis]
    params['caux'] = params['caux'][:, jnp.newaxis]
    params['beta0'] = params['beta0'][:, jnp.newaxis]
    
    np.save(f'/tmp/ovarian/ovarian_loo_{i}.npy', params)



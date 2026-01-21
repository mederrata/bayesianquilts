
import json
import os
import nest_asyncio
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from tqdm import tqdm
import jax.numpy as jnp
import importlib

nest_asyncio.apply()

logistic_relu_horseshoe_code = """
data {
  int<lower=0> N;               // number of observations
  int<lower=0> D_in;            // number of input features
  int<lower=0> D_hidden;        // number of hidden units
  matrix[N, D_in] X;            // input data (N x D_in matrix)
  array[N] int<lower=0, upper=1> y;   // binary target labels (0 or 1)
}

parameters {
  matrix[D_in, D_hidden] w_0;    // weights for the first layer
  vector[D_hidden] b_0;          // biases for the first layer
  
  vector[D_hidden] w_1;          // weights for the output layer
  real b_1;                      // bias for the output layer
}

transformed parameters {
  vector[N] z_output;           // pre-activation output (logits)
  vector[D_hidden] z_hidden;    // hidden layer (latent outputs)
  for (n in 1:N) {
    // Compute hidden layer activations with ReLU
    for (j in 1:D_hidden) {
      z_hidden[j] = X[n] * w_0[, j] + b_0[j];
      z_hidden[j] = fmax(0, z_hidden[j]);  // ReLU activation
    }
    
    // Compute the output logits (before applying sigmoid)
    z_output[n] = dot_product(w_1, z_hidden) + b_1;
  }
}

model {
  // Priors on weights and biases (adjust based on your problem)
  to_vector(w_0) ~ normal(0, 1);
  b_0 ~ normal(0, 1);
  w_1 ~ normal(0, 1);
  b_1 ~ normal(0, 1);

  // Likelihood (logistic sigmoid output)
  y ~ bernoulli_logit(z_output);
}
"""

try:
  os.makedirs("/tmp/ovarian")
except:
  pass
with open("/tmp/ovarian/ovarian_relu_model.stan", 'w') as f:
  f.writelines(logistic_relu_horseshoe_code)


with importlib.resources.path('bayesianquilts.data',   "overianx.csv") as xpath:
  X = pd.read_csv(xpath, header=None)
with importlib.resources.path('bayesianquilts.data',   "overiany.csv") as ypath:
  y = pd.read_csv(ypath, header=None)

X_scaled = (X - X.mean())/X.std()
X_scaled = X_scaled.fillna(0)
n = X_scaled.shape[0]
p = X_scaled.shape[1]

print((n, p))

D_hidden = 3

control = {"adapt_delta": 0.999, "max_treedepth": 14}


sm = CmdStanModel(stan_file="/tmp/ovarian/ovarian_relu_model.stan")

for i in tqdm(range(n)):
    y_ = y.drop(i)
    X_ = X_scaled.drop(i)
    
    _ovarian_data = {
        "N": n - 1,
        "D_in": p,
        "D_hidden": D_hidden,
        "y": y_.astype(int)[0].to_numpy().tolist(),
        "X": X_.to_numpy().tolist(),
    }
    
    with open(f"/tmp/ovarian/_ovarian_relu_data_{i}.json", "w") as f:
        json.dump(_ovarian_data, f)
        
    fit = sm.sample(
        data=f"/tmp/ovarian/_ovarian_relu_data_{i}.json",
        iter_warmup=20000,
        iter_sampling=2000,
        thin=2,
        adapt_delta=0.9995,
        max_treedepth=15,
        show_progress=False # Suppress progress bars for individual loo runs to avoid clutter
    )
    
    params = fit.stan_variables()
    
    # Add newaxis to match the previous script's output format if necessary 
    # The previous script did: params['c'] = params['c'][:, jnp.newaxis]
    # Here we have w_0, b_0, w_1, b_1. 
    # b_1 is real in Stan, so it comes out as (draws,). We might want (draws, 1).
    # w_1 is vector, so (draws, D_hidden).
    # b_0 is vector, so (draws, D_hidden).
    # w_0 is matrix, so (draws, D_in, D_hidden).
    
    # Let's keep them as numpy arrays. The jnp.newaxis usage in the original might be for specific compatibility.
    # I will add new axis to scalar/vector if consistent with standard bayesianquilts usage or the user's workflow which often expects shape consistency.
    # In the provided notebook: "params['b_1'] = params['b_1'][..., tf.newaxis]" and same for w_1.
    # So I will replicate that here using numpy.
    
    params['b_1'] = params['b_1'][:, np.newaxis]
    params['w_1'] = params['w_1'][:, :, np.newaxis] # Wait, in notebook: params['w_1'] = params['w_1'][..., tf.newaxis]
    # w_1 in Stan is vector[D_hidden]. In python: (draws, D_hidden). 
    # If we add newaxis at end: (draws, D_hidden, 1).
    
    # Let's check the notebook again.
    # params['w_1'] = params['w_1'][..., tf.newaxis]
    # Yes.
    
    np.save(f'/tmp/ovarian/ovarian_relu_loo_{i}.npy', params)

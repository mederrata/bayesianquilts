
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

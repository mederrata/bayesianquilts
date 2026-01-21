# bayesianquilts R Package

R port of the `bayesianquilts` library for interpretable Bayesian machine learning.

## Features

- **Adaptive Importance Sampling (AIS)** for Leave-One-Out (LOO) Cross-Validation
- **Transformation Strategies**: Identity (PSIS-LOO), Likelihood Descent (T_ll)
- **brms Support**: Helpers to use AIS with `brms` fitted models

## Installation

### From GitHub (recommended)

```R
# Install devtools if you don't have it
install.packages("devtools")

# Install bayesianquilts from GitHub
devtools::install_github("mederrata/bayesianquilts", subdir = "R")
```

### From Local Source

If you have cloned the repository locally:

```R
# Navigate to the repository root and install
devtools::install("R")

# Or using remotes
remotes::install_local("R")
```

### Dependencies

The package requires:
- `R6` - For object-oriented programming
- `loo` - For PSIS diagnostics

Optional (for brms integration):
- `brms` - Bayesian regression models

## Usage

### Basic Usage with Manual Likelihood

```R
library(bayesianquilts)

# Create sample data
N <- 100
K <- 3
X <- matrix(rnorm(N * K), nrow = N, ncol = K)
true_beta <- c(1, -0.5, 0.3)
y <- rbinom(N, 1, plogis(X %*% true_beta))

# Simulate posterior samples (in practice, use MCMC output)
S <- 500
beta_samples <- matrix(rnorm(S * K, mean = rep(true_beta, each = S), sd = 0.1), nrow = S, ncol = K)
intercept_samples <- rnorm(S, mean = 0, sd = 0.1)

# Prepare data and params
data <- list(X = X, y = y)
params <- list(beta = beta_samples, intercept = intercept_samples)

# Create likelihood function and sampler
lik <- LogisticRegressionLikelihood$new()
sampler <- AdaptiveImportanceSampler$new(lik)

# Run AIS for LOO-CV
results <- sampler$adaptive_is_loo(data, params, transformations = c("identity", "ll"))

# Inspect k-hat diagnostics (should be < 0.7 for reliable estimates)
print(summary(results$best$khat))
```

### Usage with brms

```R
library(bayesianquilts)
library(brms)

# Fit a brms model
fit <- brm(y ~ x1 + x2, data = df, family = bernoulli())

# Run AIS-LOO (automatically infers likelihood for bernoulli family)
results <- ais_brms(fit, transformations = c("identity", "ll"))

# Compare k-hats
print(mean(results$identity$khat))
print(mean(results$best$khat))
```

### Available Likelihood Functions

- `LogisticRegressionLikelihood` - For binary outcomes (bernoulli/binomial with logit link)
- `PoissonRegressionLikelihood` - For count data (poisson with log link)

### Custom Likelihood Functions

You can create custom likelihood functions by inheriting from `LikelihoodFunction`:

```R
CustomLikelihood <- R6::R6Class("CustomLikelihood",
  inherit = LikelihoodFunction,
  public = list(
    log_likelihood = function(data, params) {
      # Return S x N matrix of log-likelihoods
    },
    log_likelihood_gradient = function(data, params) {
      # Return list of gradients (S x N x K arrays)
    },
    log_likelihood_hessian_diag = function(data, params) {
      # Return list of Hessian diagonals (S x N x K arrays)
    }
  )
)
```

## Package Structure

- `R/ais.R` - Main `AdaptiveImportanceSampler` class
- `R/ais_classes.R` - Base classes (`LikelihoodFunction`, `Transformation`)
- `R/likelihoods.R` - Standard likelihood implementations
- `R/transformations.R` - Transformation strategies (LikelihoodDescent)
- `R/brms_utils.R` - Utilities for brms integration
- `R/psis.R` - PSIS wrapper around the `loo` package
- `R/utils.R` - Helper functions

## References

- Chang et al. (2024) "Gradient-flow adaptive importance sampling for Bayesian leave-one-out cross-validation" ArXiv 2402.08151v2
- Vehtari et al. (2017) "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC"

## License

MIT

# bayesianquilts R Package

R port of the `bayesianquilts` library for interpretable Bayesian machine learning.

## Features

- **Adaptive Importance Sampling (AIS)** for Leave-One-Out (LOO) Cross-Validation.
- **Transformation Strategies**: Identity (PSIS-LOO), Likelihood Descent (T_ll).
- **brms Support**: Helpers to use AIS with `brms` fitted models.

## Installation

You can install the package from source:

```R
install.packages("R", type="source", repos=NULL)
```
(Adjust path to the `R` directory)

## Usage

### Basic Usage with Manual Likelihood

```R
source("R/R/utils.R")
source("R/R/psis.R")
source("R/R/ais_classes.R")
source("R/R/transformations.R")
source("R/R/likelihoods.R")
source("R/R/ais.R")

# define data and params...
lik <- LogisticRegressionLikelihood$new()
sampler <- AdaptiveImportanceSampler$new(lik)
res <- sampler$adaptive_is_loo(data, params)
```

### Usage with brms

```R
library(brms)
source("R/R/brms_utils.R")

fit <- brm(y ~ x, data = df, family = bernoulli())

# Run AIS
# Automatically infers LogisticRegressionLikelihood for bernoulli family
res <- ais_brms(fit, transformations = c("identity", "ll"))

# Inspect k-hats
print(mean(res$identity$khat))
```

## Structure

- `R/ais.R`: Main AIS implementation
- `R/likelihoods.R`: Standard LikelihoodFunctions (Logistic, Poisson)
- `R/transformations.R`: Transformation strategies
- `R/brms_utils.R`: Utilities for brms integration


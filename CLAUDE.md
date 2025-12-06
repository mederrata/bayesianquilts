# CLAUDE.md - AI Assistant Context for Bayesianquilts

This document provides essential context for AI assistants (like Claude) working with the bayesianquilts codebase.

## Project Overview

**Bayesianquilts** is a JAX-based library for interpretable Bayesian machine learning, offering an alternative to black-box neural networks through piecewise linear models and advanced cross-validation methods.

**Key Value Proposition**: Interpretable-by-design models suitable for high-stakes domains (healthcare, science) where understanding model behavior is critical.

## Core Technologies

- **JAX 0.7.1+**: Primary computation framework with JIT compilation, auto-differentiation
- **TensorFlow Probability 0.25.0**: Bayesian modeling primitives, distributions
- **Flax 0.11.2 (NNX API)**: Neural network module composition using new NNX API
- **Optax**: Gradient-based optimization
- **Orbax**: Model checkpointing
- **ArviZ 0.22.0**: Bayesian diagnostics and visualization

## Project Architecture

### Directory Structure

```
bayesianquilts/
├── bayesianquilts/           # Main package
│   ├── model.py              # BayesianModel abstract base class
│   ├── util.py               # Training loops, NaN recovery, checkpointing
│   ├── features.py           # Feature engineering utilities
│   ├── jax/
│   │   └── parameter.py      # Core: Decomposed & Interactions classes
│   ├── tf/
│   │   └── parameter.py      # TensorFlow implementation (legacy)
│   ├── predictors/
│   │   ├── classification/   # Logistic models (Bayesianquilt, Ridge, etc.)
│   │   ├── regression/       # Regression models (RegressionQuilt)
│   │   ├── nn/               # Neural network components (Dense*, GamiNet)
│   │   └── factorization/    # Matrix factorization (Gaussian, Poisson, Bernoulli)
│   ├── metrics/
│   │   ├── ais.py           # AdaptiveImportanceSampler + LikelihoodFunction protocol
│   │   ├── psis.py          # PSIS implementation
│   │   └── nppsis.py        # NumPy/JAX PSIS variant
│   ├── vi/
│   │   ├── advi.py          # ADVI implementation
│   │   └── minibatch.py     # Minibatch variational inference
│   ├── distributions/        # Custom TFP distributions
│   │   ├── generalized_gamma.py
│   │   ├── piecewise_exponential.py
│   │   ├── transformed_horseshoe.py
│   │   └── ...
│   ├── plotting/
│   │   └── forest.py        # Visualization utilities
│   └── data/                # Data utilities
├── notebooks/               # Jupyter examples and case studies
├── test_ais_framework.py    # AIS comprehensive tests
├── test_gradient_clipping.py # Training utilities tests
├── requirements.txt
└── setup.py
```

### Key Modules

#### `jax/parameter.py` - Core Parameter Decomposition

**Decomposed Class**:
- Implements additive parameter decomposition: `θ = θ_global + θ_group1 + ... + θ_local`
- Used by all piecewise models
- Handles hierarchical structure across interaction dimensions
- Methods:
  - `__init__(interaction, component_shapes, name)`: Initialize decomposition structure
  - `tensor()`: Generate full parameter tensor from decomposed components
  - `sample_prior(rng)`: Sample from hierarchical prior
  - `log_prior()`: Compute log prior probability

**Interactions Class**:
- Defines interaction structure and dimensionality
- Creates shapes for multi-way tensor factorizations
- Methods:
  - `__init__(interaction_dims, exclude)`: Specify interaction dimensions
  - `shapes()`: Compute shapes for each interaction component
  - `validate()`: Check interaction specification validity

#### `util.py` - Training Infrastructure

**training_loop() Function**:
```python
def training_loop(
    initial_values,
    loss_fn,
    data_iterator=None,
    steps_per_epoch=1,
    num_epochs=100,
    learning_rate=0.01,
    clip_norm=None,           # Gradient clipping threshold
    patience=None,            # Early stopping patience
    lr_decay_factor=0.9,      # Learning rate decay
    checkpoint_dir=None,      # Orbax checkpoint directory
    recover_from_nan=True     # Enable NaN recovery
)
```

Features:
- **Gradient clipping**: Prevents exploding gradients
- **Early stopping**: Monitors loss with patience threshold
- **Learning rate decay**: Reduces LR when loss plateaus
- **NaN recovery**: Detects NaN/Inf and applies recovery strategies (gradient clipping, LR reduction)
- **Checkpointing**: Saves best model state using Orbax
- **Progress bars**: Uses tqdm for epoch/step tracking

**NaN Recovery Strategies**:
1. Detect NaN/Inf in loss or gradients
2. Restore previous good parameters
3. Reduce learning rate
4. Apply gradient clipping if not already enabled
5. Continue training from recovered state

#### `metrics/ais.py` - Adaptive Importance Sampling

**AdaptiveImportanceSampler Class**:

Core method:
```python
def adaptive_is_loo(
    data,                    # Data dict
    params,                  # Parameter samples (posterior)
    hbar=1.0,               # Step size for transformations
    variational=True,       # Use variational approximation
    transformations=['ll', 'kl', 'var', 'identity']
) -> dict
```

Returns dictionary with:
- `eta_weights`: Raw importance weights
- `psis_weights`: PSIS-smoothed weights
- `ll_loo_eta/psis`: LOO log-likelihood estimates
- `p_loo_eta/psis`: Effective parameter count
- `khat`: PSIS diagnostic (< 0.7 is good)
- `predictions`: Model predictions

**Transformation Strategies**:
- `T_ll`: Negative log-likelihood gradient descent
- `T_kl`: KL-weighted gradients (uses posterior weights)
- `T_var`: Variance-weighted using Hessian diagonal
- `T_I`: Identity (baseline, no transformation)

**LikelihoodFunction Protocol**:
Abstract base class for custom likelihoods. Required methods:
```python
class CustomLikelihood(LikelihoodFunction):
    def log_likelihood(self, data, params) -> Array
    def log_likelihood_gradient(self, data, params) -> PyTree
    def log_likelihood_hessian_diag(self, data, params) -> PyTree
    def extract_parameters(self, params) -> Array
    def reconstruct_parameters(self, params_flat) -> PyTree
```

Provided implementations:
- `LogisticRegressionLikelihood`
- `PoissonRegressionLikelihood`
- `LinearRegressionLikelihood`

### Model Hierarchy

```
BayesianModel (ABC in model.py)
│
├── Classification (predictors/classification/)
│   ├── LogisticBayesianquilt    # Piecewise linear with decomposition
│   ├── LogisticRegression        # Standard Bayesian logistic
│   ├── LogisticRelunet          # ReLU neural network
│   ├── LogisticGamiNet          # Generalized additive model
│   └── LogisticRidge            # Ridge regularization
│
├── Regression (predictors/regression/)
│   ├── RegressionQuilt          # Piecewise linear regression
│   └── HierarchicalAttention    # Attention mechanism
│
├── Neural Networks (predictors/nn/)
│   ├── DenseHorseshoe/Gaussian  # Dense layers with Bayesian priors
│   ├── GamiNetUnivariate        # Univariate shape functions
│   └── GamiNetPairwise          # Pairwise interactions
│
└── Factorization (predictors/factorization/)
    ├── GaussianFactorization    # Continuous data
    ├── PoissonFactorization     # Count data
    └── BernoulliFactorization   # Binary data
```

## Common Workflows

### 1. Training a Piecewise Linear Model

```python
from bayesianquilts.predictors.classification import LogisticBayesianquilt
from bayesianquilts.util import training_loop
import jax
import jax.numpy as jnp

# 1. Create model instance
model = LogisticBayesianquilt(num_features=X.shape[1], num_classes=2)

# 2. Initialize parameters
rng = jax.random.PRNGKey(0)
params = model.initialize(rng)

# 3. Define loss function
def loss_fn(p):
    return model.negative_log_likelihood(p, X_train, y_train)

# 4. Train with utilities
losses, trained_params = training_loop(
    initial_values=params,
    loss_fn=loss_fn,
    num_epochs=100,
    learning_rate=0.01,
    clip_norm=1.0,
    patience=10,
    checkpoint_dir="./checkpoints"
)
```

### 2. Running Adaptive Importance Sampling for LOO-CV

```python
from bayesianquilts.metrics.ais import (
    AdaptiveImportanceSampler,
    LogisticRegressionLikelihood
)

# 1. Define likelihood
likelihood_fn = LogisticRegressionLikelihood()

# 2. Define prior and surrogate (variational approximation)
def prior_fn(params):
    return model.log_prior(params)

def surrogate_fn(params):
    return model.variational_log_prob(params)

# 3. Create AIS sampler
sampler = AdaptiveImportanceSampler(
    likelihood_fn,
    prior_log_prob_fn=prior_fn,
    surrogate_log_prob_fn=surrogate_fn
)

# 4. Run LOO-CV
results = sampler.adaptive_is_loo(
    data={'X': X, 'y': y},
    params=posterior_samples,
    hbar=1.0,
    transformations=['ll', 'kl', 'var', 'identity']
)

# 5. Check diagnostics
print(f"PSIS k-hat: {results['khat']}")  # Should be < 0.7
print(f"LOO log-likelihood: {results['ll_loo_psis']}")
```

### 3. Creating Custom Parameter Decomposition

```python
from bayesianquilts.jax.parameter import Decomposed, Interactions

# 1. Define interaction structure
# Example: 2-way interactions between feature groups
interactions = Interactions(
    interaction_dims=[
        (10,),        # Global intercept (1 value per 10 classes)
        (5, 10),      # Group-level (5 groups, 10 classes)
        (100, 10)     # Feature-level (100 features, 10 classes)
    ]
)

# 2. Create decomposed parameter
beta = Decomposed(
    interaction=interactions,
    component_shapes=interactions.shapes(),
    name="regression_coefficients"
)

# 3. Sample from prior
rng = jax.random.PRNGKey(0)
beta_samples = beta.sample_prior(rng)

# 4. Get full parameter tensor
beta_full = beta.tensor()  # Shape: (100, 10)
```

## Development Branch

**Current branch**: `jax` (JAX implementation development)
**Main branch**: `main` (stable releases)

When creating PRs, target `main` branch.

## Testing

Key test files:
- `test_ais_framework.py`: Comprehensive AIS tests with all transformations
- `test_gradient_clipping.py`: Training utilities and NaN recovery

Run tests:
```bash
python test_ais_framework.py
python test_gradient_clipping.py
```

## Important Conventions

### Code Style
- Use JAX functional programming patterns (pure functions)
- Avoid in-place mutations
- Use `jax.jit` for performance-critical functions
- Type hints encouraged but not always present in legacy code

### Parameter Conventions
- Parameters stored as PyTrees (nested dicts/lists)
- Use `jax.tree_util` for tree operations
- Random keys passed explicitly (no global RNG state)

### Naming
- `rng` or `key`: JAX random number generator key
- `params`: Model parameters (PyTree)
- `hbar`: Step size parameter (from physics notation ℏ)
- `khat`: PSIS diagnostic threshold
- `p_loo`: Effective number of parameters in LOO-CV

### Common Gotchas
1. **Flax NNX API**: Uses new NNX API (not legacy Linen), check imports
2. **TFP Integration**: Mix of JAX backend with TFP distributions
3. **Parameter Extraction**: Many models need `extract_parameters()` to flatten PyTree
4. **Shape Handling**: Interactions create complex tensor shapes, check dimensions
5. **NaN Issues**: Can occur with extreme parameters, use training utilities' recovery

## Research Context

### Publications
The library implements methods from peer-reviewed research:

1. **Piecewise Models**: Chang et al. (2024) PLOS ONE - Medical claims interpretability
2. **AIS Method**: Chang et al. (2024) ArXiv 2402.08151v2 - Gradient-flow importance sampling

### Theoretical Foundation
- **Piecewise Linearity**: Mimics ReLU networks but with explicit interpretability
- **Hierarchical Decomposition**: Automatic regularization through multilevel structure
- **Gradient Flow**: Uses gradient information to find better importance sampling proposals

## Examples and Learning Resources

- `notebooks/decomposition.ipynb`: Best introduction to parameter decomposition
- `notebooks/ovarian/`: Real-world medical claims modeling
- `notebooks/roach/`: Standard classification examples
- `test_ais_framework.py`: Complete working AIS examples

## Common Tasks for AI Assistants

### Adding a New Model
1. Inherit from `BayesianModel` in `model.py`
2. Implement required abstract methods: `__init__`, `initialize`, `log_likelihood`, `log_prior`
3. Use `Decomposed` for parameter decomposition if needed
4. Add to appropriate predictor subdirectory
5. Create tests following existing patterns

### Implementing Custom Likelihood for AIS
1. Inherit from `LikelihoodFunction` in `metrics/ais.py`
2. Implement all required methods (log_likelihood, gradients, hessian_diag, parameter extract/reconstruct)
3. Test with `AdaptiveImportanceSampler` on synthetic data
4. Validate PSIS diagnostics (k-hat < 0.7)

### Debugging NaN Issues
1. Check `util.py` training loop with `recover_from_nan=True`
2. Enable gradient clipping: `clip_norm=1.0`
3. Reduce learning rate: `learning_rate=0.001`
4. Check prior specifications (avoid overly tight priors)
5. Inspect parameter initialization for extreme values

### Performance Optimization
1. Use `jax.jit` on loss functions and gradients
2. Batch data properly for GPU utilization
3. Profile with `jax.profiler`
4. Consider mixed precision if needed
5. Check for unnecessary PyTree copying

## API Stability

The API is evolving (version 0.1.2). Breaking changes possible until 1.0 release. Pin versions in production:
```
bayesianquilts==0.1.2
```

## Key Dependencies to Understand

- **JAX**: Functional transformations (jit, grad, vmap)
- **TFP**: Distribution API (sample, log_prob)
- **Flax NNX**: Module system (different from Linen!)
- **Optax**: Optimizer API (GradientTransformation)
- **Orbax**: Checkpointing API (CheckpointManager)

## Contact and Support

- **Organization**: Mederrata Research LLC (501(c)3 non-profit)
- **Email**: info@mederrata.com
- **Repository**: https://github.com/mederrata/bayesianquilts
- **Issues**: Use GitHub issue tracker

Mederrata Research LLC is a 501(c)3 non-profit organization. Tax-deductible monetary contributions are welcome to support the development of open-source tools for interpretable machine learning.

## Summary for AI Assistants

When working with bayesianquilts:
1. **Core concept**: Piecewise linear models via parameter decomposition (not black-box NNs)
2. **Main classes**: `Decomposed`, `Interactions`, `AdaptiveImportanceSampler`, `BayesianModel`
3. **Key files**: `jax/parameter.py`, `util.py`, `metrics/ais.py`
4. **Framework**: JAX-based with TFP, using Flax NNX API
5. **Training**: Use `training_loop()` with gradient clipping and NaN recovery
6. **Evaluation**: Use AIS for LOO-CV with PSIS diagnostics
7. **Testing**: See `test_ais_framework.py` for comprehensive examples

The library prioritizes interpretability and statistical rigor over predictive performance alone, making it suitable for scientific and high-stakes applications where understanding model behavior is essential.

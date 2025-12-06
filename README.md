# Bayesianquilts

A JAX-based library for building interpretable Bayesian models using piecewise linear regression and gradient-flow adaptive importance sampling for leave-one-out cross-validation.

## Overview

**Bayesianquilts** provides tools for building truly interpretable input-output maps based on the principle of piecewise linearity. Rather than using black-box neural networks, this library combines representation learning, clustering, and multilevel linear regression modeling to create transparent, interpretable models suitable for high-stakes applications like healthcare and scientific research.

The library includes two major research contributions:

1. **Piecewise Linear Regression Models**: An additive decomposition approach where parameter values arise as sums of contributions at different length scales
2. **Gradient-Flow Adaptive Importance Sampling (AIS)**: Advanced methods for Bayesian leave-one-out cross-validation

## Key Features

- **Interpretable by Design**: Models are constructed to be inherently interpretable, not just post-hoc explainable
- **Parameter Decomposition**: Additive decomposition of parameters across interaction dimensions
- **Flexible Model Types**: Supports classification, regression, and matrix factorization
- **Advanced Cross-Validation**: Gradient-flow adaptive importance sampling for LOO-CV
- **Bayesian Inference**: Full support for variational inference (ADVI) and importance sampling
- **JAX-Accelerated**: Built on JAX for GPU/TPU acceleration and automatic differentiation
- **Robust Training**: Includes gradient clipping, learning rate scheduling, NaN recovery, and checkpointing

## Installation

### From PyPI (once published)

```bash
pip install bayesianquilts
```

### From Source

```bash
git clone https://github.com/mederrata/bayesianquilts.git
cd bayesianquilts
pip install -r requirements.txt
pip install -e .
```

### Requirements

- Python >= 3.8
- JAX >= 0.7.1
- TensorFlow Probability >= 0.25.0
- Flax >= 0.11.2
- NumPy, Pandas, SciPy, Scikit-learn
- Optax (optimization)
- Orbax (checkpointing)
- ArviZ (Bayesian diagnostics)

See `requirements.txt` for complete dependency list.

## Quick Start

### Piecewise Linear Classification

```python
from bayesianquilts.predictors.classification import LogisticBayesianquilt
from bayesianquilts.util import training_loop
import jax.numpy as jnp

# Prepare your data
X_train = jnp.array(...)  # Features
y_train = jnp.array(...)  # Labels

# Create model
model = LogisticBayesianquilt(
    num_features=X_train.shape[1],
    num_classes=2
)

# Initialize parameters
params = model.initialize(random_key)

# Train with built-in utilities
losses, trained_params = training_loop(
    initial_values=params,
    loss_fn=lambda p: model.loss(p, X_train, y_train),
    num_epochs=100,
    learning_rate=0.01,
    clip_norm=1.0,
    patience=10
)
```

### Adaptive Importance Sampling for LOO-CV

```python
from bayesianquilts.metrics.ais import (
    AdaptiveImportanceSampler,
    LogisticRegressionLikelihood
)

# Define likelihood function
likelihood_fn = LogisticRegressionLikelihood()

# Create AIS sampler
ais_sampler = AdaptiveImportanceSampler(
    likelihood_fn,
    prior_log_prob_fn=prior_fn,
    surrogate_log_prob_fn=surrogate_fn
)

# Compute LOO-CV with multiple transformation strategies
results = ais_sampler.adaptive_is_loo(
    data={'X': X, 'y': y},
    params=trained_params,
    hbar=1.0,
    variational=False,
    transformations=['ll', 'kl', 'var', 'identity']
)

# Access results
print(f"LOO log-likelihood: {results['ll_loo_psis']}")
print(f"Effective parameters (p_loo): {results['p_loo_psis']}")
print(f"PSIS k-hat diagnostic: {results['khat']}")
```

## Core Concepts

### Parameter Decomposition

The fundamental innovation is an additive decomposition of model parameters:

```
θ_effective = θ_global + θ_group1 + θ_group2 + ... + θ_local
```

Each parameter value arises as a sum of contributions at different hierarchical levels (length scales), enabling:
- Automatic regularization through hierarchical priors
- Interpretable multi-level effects
- Interaction modeling across categorical and continuous variables

See `notebooks/decomposition.ipynb` for detailed examples.

### Adaptive Importance Sampling

The AIS framework implements gradient-flow transformations for stable LOO-CV:

- **T_ll**: Likelihood descent using negative log-likelihood gradients
- **T_kl**: KL-divergence weighted gradients using posterior weights
- **T_var**: Variance-based adaptation using Hessian curvature
- **T_I**: Identity (baseline, no transformation)

Combined with Pareto Smoothed Importance Sampling (PSIS) for robust weight estimation.

## Available Models

### Classification
- `LogisticBayesianquilt`: Piecewise linear logistic regression
- `LogisticRegression`: Standard Bayesian logistic regression with decomposition
- `LogisticRelunet`: ReLU neural network classifier
- `LogisticGamiNet`: Generalized additive model with neural networks
- `LogisticRidge`: Ridge-regularized logistic regression

### Regression
- `RegressionQuilt`: Piecewise linear regression
- `HierarchicalAttention`: Attention-based regression

### Matrix Factorization
- `GaussianFactorization`: Continuous latent factor models
- `PoissonFactorization`: Count data factorization
- `BernoulliFactorization`: Binary data factorization

### Neural Network Components
- `DenseHorseshoe`: Dense layers with horseshoe priors
- `DenseGaussian`: Dense layers with Gaussian priors
- `GamiNetUnivariate`: Univariate shape functions
- `GamiNetPairwise`: Pairwise interaction networks

## Training Utilities

The `util.py` module provides robust training infrastructure:

```python
from bayesianquilts.util import training_loop

losses, params = training_loop(
    initial_values=initial_params,
    loss_fn=loss_function,
    data_iterator=data_batches,
    steps_per_epoch=100,
    num_epochs=50,
    learning_rate=0.01,
    clip_norm=1.0,              # Gradient clipping
    patience=10,                 # Early stopping
    lr_decay_factor=0.5,        # Learning rate decay
    checkpoint_dir="./ckpts",   # Automatic checkpointing
    recover_from_nan=True       # NaN recovery strategies
)
```

Features:
- Gradient clipping for stability
- Learning rate scheduling with decay
- Early stopping with patience
- Automatic checkpointing with Orbax
- NaN/Inf detection and recovery
- Progress tracking with tqdm

## Custom Distributions

Bayesianquilts includes several custom probability distributions:

- `GeneralizedGamma`: Flexible shape for positive continuous data
- `PiecewiseExponential`: For survival/duration modeling
- `TransformedHorseshoe`: Sparsity-inducing priors
- `TransformedCauchy`: Heavy-tailed priors
- `TransformedInverseGamma`: Scale parameter priors

## Examples and Notebooks

- `notebooks/decomposition.ipynb`: Parameter decomposition methodology
- `notebooks/ovarian/`: Medical claims modeling examples
- `notebooks/roach/`: Logistic regression case studies
- `notebooks/enset/`: Model comparison demonstrations
- `test_ais_framework.py`: Complete AIS usage examples
- `test_gradient_clipping.py`: Training utilities demonstration

## Publications

This library implements methods from:

### Piecewise Linear Models

- Chang TL, Xia H, Mahajan S, Mahajan R, Maisog J, et al. (2024). Interpretable (not just posthoc-explainable) medical claims modeling for discharge placement to reduce preventable all-cause readmissions or death. *PLOS ONE* 19(5): e0302871. [https://doi.org/10.1371/journal.pone.0302871](https://doi.org/10.1371/journal.pone.0302871)

- Xia H, Chang JC, Nowak S, Mahajan S, Mahajan R, Chang TL, Chow CC (2023). Proceedings of the 8th Machine Learning for Healthcare Conference, *PMLR* 219:884-905.

### Adaptive Importance Sampling

- Chang JC, Li X, Xu S, Yao HR, Porcino J, Chow CC (2024). Gradient-flow adaptive importance sampling for Bayesian leave one out cross-validation with application to sigmoidal classification models. *ArXiv* [Preprint] 2402.08151v2. PMID: 38711425; PMCID: PMC11071546. [https://arxiv.org/abs/2402.08151](https://arxiv.org/abs/2402.08151)

## Project Structure

```
bayesianquilts/
├── bayesianquilts/
│   ├── model.py              # Base BayesianModel class
│   ├── util.py               # Training loops and utilities
│   ├── features.py           # Feature engineering
│   ├── jax/
│   │   └── parameter.py      # Parameter decomposition (Decomposed, Interactions)
│   ├── predictors/
│   │   ├── classification/   # Classification models
│   │   ├── regression/       # Regression models
│   │   ├── nn/               # Neural network components
│   │   └── factorization/    # Matrix factorization
│   ├── metrics/
│   │   ├── ais.py           # Adaptive importance sampling
│   │   ├── psis.py          # Pareto smoothed IS
│   │   └── nppsis.py        # NumPy/JAX PSIS
│   ├── vi/
│   │   ├── advi.py          # ADVI implementation
│   │   └── minibatch.py     # Minibatch VI
│   ├── distributions/        # Custom distributions
│   └── plotting/            # Visualization utilities
├── notebooks/               # Example notebooks
├── requirements.txt         # Dependencies
└── setup.py                # Package setup
```

## API Status

The API is currently evolving as we prepare manuscripts on the methodology and theory. We will stabilize the API in future releases. For production use, please pin to specific versions.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Contact and Support

- **Organization**: Mederrata Research LLC (501(c)3 non-profit)
- **Email**: info@mederrata.com
- **Repository**: [https://github.com/mederrata/bayesianquilts](https://github.com/mederrata/bayesianquilts)

### Supporting This Project

Mederrata Research LLC is a 501(c)3 non-profit organization. Tax-deductible monetary contributions are welcome and help support the development and maintenance of open-source tools for interpretable machine learning in healthcare and scientific research.

To make a contribution or learn more, please contact us at info@mederrata.com.

## Citation

If you use this library in your research, please cite:

```bibtex
@article{chang2024interpretable,
  title={Interpretable (not just posthoc-explainable) medical claims modeling for discharge placement to reduce preventable all-cause readmissions or death},
  author={Chang, Ted L and Xia, Hongjing and Mahajan, Sonya and Mahajan, Rohit and Maisog, Jose and others},
  journal={PLOS ONE},
  volume={19},
  number={5},
  pages={e0302871},
  year={2024},
  publisher={Public Library of Science}
}

@article{chang2024gradient,
  title={Gradient-flow adaptive importance sampling for Bayesian leave one out cross-validation with application to sigmoidal classification models},
  author={Chang, Joshua C and Li, Xu and Xu, Shuang and Yao, Howard R and Porcino, John and Chow, Carson C},
  journal={arXiv preprint arXiv:2402.08151},
  year={2024}
}
```

## Acknowledgments

This work was developed by the Mederrata Research team with support from the research community. Special thanks to all contributors and users who have provided feedback and helped improve the library.

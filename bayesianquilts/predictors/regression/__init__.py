#!/usr/bin/env python3
"""Regression models for Bayesian analysis.

This module contains regression models using various approaches:
- Linear models with different priors (Ridge, Bayesianquilt)
- Neural network models (ReLU networks, GAM-Nets)
- Piecewise linear models with decomposition

Available models:
    Linear Models:
        - LinearRegression: Standard Bayesian linear regression
        - LinearRidge: Ridge regression with Gaussian prior
        - LinearBayesianquilt: Piecewise linear with parameter decomposition
        - LinearRegressionReparam: Reparameterized linear regression

    Neural Network Models:
        - RegressionRelunet: Deep ReLU network with horseshoe prior
        - ShallowGaussianRegressionRelunet: Shallow network with Gaussian prior
        - RegressionGamiNetUnivariate: GAM with univariate neural networks
        - RegressionGamiNetPairwise: GAM with univariate + pairwise networks

    Piecewise Models:
        - RegressionBayesianquilt: Full quilt model for regression

    Other:
        - HierarchicalAttention: Attention-based hierarchical model
"""

from bayesianquilts.predictors.regression.hierarchical_attention import (
    HierarchicalAttention,
)
from bayesianquilts.predictors.regression.linear_bayesianquilt import (
    LinearBayesianquilt,
)
from bayesianquilts.predictors.regression.linear_regression import LinearRegression
from bayesianquilts.predictors.regression.linear_regression_reparam import (
    LinearRegressionReparam,
)
from bayesianquilts.predictors.regression.linear_ridge import LinearRidge
from bayesianquilts.predictors.regression.regression_bayesianquilt import (
    RegressionBayesianquilt,
)
from bayesianquilts.predictors.regression.regression_gaminet import (
    RegressionGamiNetPairwise,
    RegressionGamiNetUnivariate,
)
from bayesianquilts.predictors.regression.regression_relunet import (
    RegressionRelunet,
    ShallowGaussianRegressionRelunet,
)

__all__ = [
    # Linear models
    "LinearRegression",
    "LinearRidge",
    "LinearBayesianquilt",
    "LinearRegressionReparam",
    # Neural network models
    "RegressionRelunet",
    "ShallowGaussianRegressionRelunet",
    "RegressionGamiNetUnivariate",
    "RegressionGamiNetPairwise",
    # Piecewise models
    "RegressionBayesianquilt",
    # Other
    "HierarchicalAttention",
]

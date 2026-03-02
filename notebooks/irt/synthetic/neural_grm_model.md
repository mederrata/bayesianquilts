# Neural Graded Response Model

## Background: The Standard GRM

The Graded Response Model (Samejima, 1969) models ordinal responses $Y_{ji} \in \{0, 1, \ldots, K-1\}$ for person $j$ on item $i$. It defines the probability of responding at or above category $k$ as

$$
P(Y_{ji} \geq k \mid \theta_j) = \sigma\bigl(a_i(\theta_j - b_{ik})\bigr), \quad k = 1, \ldots, K-1,
$$

where $\sigma(x) = 1/(1 + e^{-x})$ is the standard logistic sigmoid, $\theta_j \in \mathbb{R}$ is the latent ability of person $j$, $a_i > 0$ is the discrimination parameter for item $i$, and $b_{i1} \leq b_{i2} \leq \cdots \leq b_{i,K-1}$ are ordered difficulty thresholds. The boundary conditions are $P(Y_{ji} \geq 0) = 1$ and $P(Y_{ji} \geq K) = 0$.

The category response probabilities are obtained by differencing:

$$
P(Y_{ji} = k) = P(Y_{ji} \geq k) - P(Y_{ji} \geq k+1).
$$

The sigmoid link guarantees that $P(Y_{ji} \geq k)$ is monotone increasing in $\theta_j$ (higher ability implies higher probability of endorsing higher categories) and monotone decreasing in $b_{ik}$ (harder thresholds are more difficult to surpass).

## Motivation for the Neural GRM

The standard GRM assumes the item response function (IRF) has a logistic shape---symmetric, with a fixed functional form up to location and scale. In practice, items may exhibit:

- **Asymmetric response curves**: The probability of endorsing higher categories may rise more steeply on one side of the threshold than the other.
- **Non-logistic curvature**: The transition between categories may be sharper or more gradual than the logistic function permits.
- **Heterogeneous shapes across the latent trait**: The rate of change may vary differently across regions of $\theta$.

The Neural GRM replaces the fixed sigmoid link with a flexible, learned monotone function $g: \mathbb{R} \to (0, 1)$, while retaining the interpretable IRT parameterization of person abilities and item discrimination/difficulty.

## Model Specification

### Response Function

The Neural GRM defines:

$$
P(Y_{ji} \geq k \mid \theta_j) = g\bigl(-a_i(b_{ik} - \theta_j)\bigr), \quad k = 1, \ldots, K-1,
$$

where $g: \mathbb{R} \to (0, 1)$ is a **shared monotone function** parameterized as a mixture of sigmoids. The argument uses the sign convention $-a_i(b_{ik} - \theta_j) = a_i(\theta_j - b_{ik})$, matching the standard GRM so that $P(Y_{ji} \geq k)$ increases with ability and decreases with threshold difficulty.

### Mixture-of-Sigmoids Architecture

The monotone response function $g$ is defined as a weighted average of $H$ logistic sigmoid components:

$$
g(z) = \sum_{h=1}^{H} \pi_h \, \sigma\!\bigl(\alpha_h \, z + \beta_h\bigr),
$$

where:

- $\alpha_h > 0$ are **component slopes**, ensuring each component is monotone increasing in $z$. We parameterize $\alpha_h = \mathrm{softplus}(\tilde{\alpha}_h)$ with unconstrained $\tilde{\alpha}_h \in \mathbb{R}$.
- $\beta_h \in \mathbb{R}$ are **component offsets**, shifting each sigmoid along the $z$-axis.
- $\pi_h > 0$ with $\sum_h \pi_h = 1$ are **mixing weights**, parameterized as $\pi = \mathrm{softmax}(\tilde{w})$ with unconstrained logits $\tilde{w} \in \mathbb{R}^H$.

**Monotonicity guarantee.** Since each $\sigma(\alpha_h z + \beta_h)$ is monotone increasing in $z$ (because $\alpha_h > 0$ and $\sigma$ is increasing), and $\pi_h \geq 0$, the weighted sum $g(z)$ is monotone increasing. Moreover, $g(z) \to 0$ as $z \to -\infty$ and $g(z) \to 1$ as $z \to +\infty$, so $g$ is a valid CDF-like link function.

**Expressiveness.** A mixture of $H$ sigmoids can approximate any continuous monotone function from $\mathbb{R}$ to $(0, 1)$ to arbitrary precision as $H \to \infty$. With $H = 32$ (default), the model can capture a wide range of asymmetric, multi-modal-derivative, and non-logistic response curves. When $H = 1$ and $\alpha_1 = 1$, $\beta_1 = 0$, the model reduces to the standard GRM.

### Category Probabilities

Given the cumulative probabilities, the response probability for category $k$ is:

$$
P(Y_{ji} = k) = P(Y_{ji} \geq k) - P(Y_{ji} \geq k + 1),
$$

with boundary conventions $P(Y_{ji} \geq 0) = 1$ and $P(Y_{ji} \geq K) = 0$.

### Multidimensional Extension

For a $D$-dimensional model with abilities $\boldsymbol{\theta}_j \in \mathbb{R}^D$ and per-dimension discriminations $a_{id}$, the dimension-specific category probabilities are computed independently for each dimension $d$ and then aggregated via a discrimination-weighted average:

$$
P(Y_{ji} = k) = \sum_{d=1}^{D} w_{id} \, P_d(Y_{ji} = k),
$$

where the weights are

$$
w_{id} = \frac{|a_{id}|^{\gamma}}{\sum_{d'=1}^{D} |a_{id'}|^{\gamma}},
$$

and $\gamma \geq 0$ is a weight exponent (default $\gamma = 1$). This ensures dimensions with larger discrimination contribute proportionally more to the predicted response probabilities.

## Prior Specification

All priors match the standard GRM implementation in `bayesianquilts`, with additional priors for the mixture-of-sigmoids parameters.

### IRT Parameters

| Parameter | Prior | Shape | Description |
|-----------|-------|-------|-------------|
| $\mu_i$ | $\mathcal{N}(0, 1)$ | $(1, D, I, 1)$ | Difficulty location prior mean |
| $b_{i1}$ (difficulties0) | $\mathcal{N}(\mu_i, 1)$ | $(1, D, I, 1)$ | First threshold |
| $\Delta b_{ik}$ (ddifficulties) | $\mathrm{HalfNormal}(1)$ | $(1, D, I, K{-}2)$ | Threshold increments (positive, ensuring ordering) |
| $\theta_j$ (abilities) | $\mathcal{N}(0, 1)$ | $(N, D, 1, 1)$ | Latent abilities |
| $a_{id}$ (discriminations) | $\mathrm{AbsHorseshoe}(\eta_i \cdot \kappa_d)$ | $(1, D, I, 1)$ | Item discriminations |
| $\eta_i$ | $\mathrm{HalfNormal}(\eta_{\mathrm{scale}})$ | $(1, 1, I, 1)$ | Local shrinkage |
| $\kappa_d$ | $\sqrt{\mathrm{InverseGamma}(0.5, 1/\kappa_a)}$ | $(1, D, 1, 1)$ | Global shrinkage |
| $\kappa_a$ | $\mathrm{InverseGamma}(0.5, 1/\kappa_{\mathrm{scale}}^2)$ | $(1, D, 1, 1)$ | Hyperprior on global shrinkage |

The default scales are $\eta_{\mathrm{scale}} = 0.1$ and $\kappa_{\mathrm{scale}} = 0.5$, which produce discriminations typically in the range $[0.3, 3.0]$.

The ordered difficulty thresholds are constructed as:

$$
b_{ik} = b_{i1} + \sum_{\ell=2}^{k} \Delta b_{i\ell}, \quad k = 1, \ldots, K-1,
$$

where $\Delta b_{ik} > 0$ ensures the ordering $b_{i1} \leq b_{i2} \leq \cdots$.

### Mixture-of-Sigmoids Parameters

| Parameter | Prior | Shape | Description |
|-----------|-------|-------|-------------|
| $\tilde{\alpha}_h$ (nn\_w0) | $\mathcal{N}(0, 1)$ | $(H, 1)$ | Unconstrained slopes |
| $\beta_h$ (nn\_b0) | $\mathcal{N}(0, 2)$ | $(H,)$ | Component offsets (wider prior for diversity) |
| $\tilde{w}_h$ (nn\_w1) | $\mathcal{N}(0, 1)$ | $(1, H)$ | Mixing logits |

The wider prior ($\sigma = 2$) on the offsets $\beta_h$ encourages the sigmoid components to spread across different regions of the $z$-axis at initialization, giving the mixture good initial coverage of the response function.

## Inference

The model is fit using **Automatic Differentiation Variational Inference** (ADVI) with a mean-field factored surrogate posterior. Each parameter is approximated by an independent transformed normal distribution, where the transformation is chosen to respect parameter constraints:

| Parameter | Bijector | Effect |
|-----------|----------|--------|
| abilities, $\mu$, difficulties0 | Identity | Unconstrained |
| discriminations, $\eta$, $\kappa$, $\kappa_a$, ddifficulties | Softplus | Constrained to $\mathbb{R}^+$ |
| NN weights and biases | Identity | Unconstrained (constraints applied inside forward pass) |

The ELBO objective for a minibatch $\mathcal{B}$ of size $B$ from a dataset of size $N$ is:

$$
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}\!\left[\frac{B}{N}\bigl(\log p(\Psi) - H_{\mathrm{disc}}(\Psi)\bigr) + \sum_{j \in \mathcal{B}} \log p(\mathbf{Y}_j \mid \Psi)\right] + H(q_\phi),
$$

where $\Psi$ denotes all model parameters, $H(q_\phi)$ is the entropy of the surrogate posterior, and $H_{\mathrm{disc}}$ is a discrimination-weighted entropy regularizer:

$$
H_{\mathrm{disc}}(\Psi) = -\sum_{i,d} \frac{w_{id} \log w_{id}}{\eta_i},
$$

which encourages diversification of discrimination loadings across dimensions, scaled by the local shrinkage $\eta_i$.

## Handling Missing Data

Missing responses ($Y_{ji}$ unobserved) are handled by one of two strategies:

1. **Without imputation model**: Missing cells contribute an entropy term $\sum_k p_k \log p_k$ to the log-likelihood, where $p_k = P(Y_{ji} = k \mid \Psi)$. This is equivalent to marginalizing over a uniform missing-data mechanism.

2. **With imputation model** (MICEBayesianLOO): Missing cells are analytically marginalized using imputation PMFs $q(k)$ provided by a pre-fitted imputation model:

$$
\log P(Y_{ji} \mid \Psi, \text{missing}) = \log \sum_{k=0}^{K-1} q(k) \cdot P(Y_{ji} = k \mid \Psi).
$$

This is exact (zero-variance) and eliminates the need for multiple imputation samples.

## Synthetic Data Generation

Given a fitted model with calibrated parameter expectations $\hat{\Psi}$, synthetic responses are generated as:

1. Compute category probabilities: $\hat{p}_{jik} = P(Y_{ji} = k \mid \hat{\Psi})$ for all persons $j$, items $i$.
2. Draw responses: $Y_{ji}^{\mathrm{syn}} \sim \mathrm{Categorical}(\hat{p}_{ji0}, \ldots, \hat{p}_{ji,K-1})$.
3. Optionally introduce MCAR missingness at a specified rate.

## Relationship to the Standard GRM

The Neural GRM is a strict generalization of the standard GRM. When the mixture-of-sigmoids reduces to a single component with unit slope and zero offset---i.e., $H = 1$, $\alpha_1 = 1$, $\beta_1 = 0$---the response function becomes $g(z) = \sigma(z)$ and the model is identical to the standard GRM.

The key structural difference is that the response function $g$ is **shared across all items**. Item-specific behavior is captured entirely by the discrimination $a_i$ and difficulty thresholds $b_{ik}$. This means the Neural GRM learns a single "response curve shape" that represents how respondents transition between categories, while item parameters control where on this curve each item sits.

## Implementation

The implementation is in `bayesianquilts.irt.neural_grm.NeuralGRModel`, which inherits from `IRTModel`. Key methods:

- `_monotone_forward(z, nn_params)`: Evaluates the mixture-of-sigmoids $g(z)$.
- `neural_grm_model_prob(abilities, discriminations, difficulties, nn_params)`: Computes $P(Y = k)$ for all persons, items, and categories.
- `predictive_distribution(data, **params)`: Full predictive distribution including missing-data handling.
- `create_distributions()`: Defines the joint prior and constructs the ADVI surrogate posterior.
- `simulate_data(abilities)`: Generates synthetic responses from calibrated parameters.

## References

- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph Supplement*, 17.
- Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). Automatic differentiation variational inference. *Journal of Machine Learning Research*, 18(1), 430--474.
- Piironen, J., & Vehtari, A. (2017). Sparsity information and regularization in the horseshoe and other shrinkage priors. *Electronic Journal of Statistics*, 11(2), 5018--5051.

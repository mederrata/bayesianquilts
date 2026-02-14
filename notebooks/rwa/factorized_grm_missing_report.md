# Factorized Graded Response Model with Rao-Blackwellized Imputation

This report documents the mathematical foundations behind the
`factorized_grm_missing` notebook, covering the Graded Response Model,
variational inference, stochastic imputation, and the Rao-Blackwellized
training objective.

> **A note on ignorability and misspecification.** Under the classical
> Rubin (1976) framework, a Missing At Random (MAR) mechanism is
> *ignorable* for likelihood-based inference --- one can maximize the
> observed-data likelihood without modelling the missingness process.
> However, this result assumes the analyst's model is correctly
> specified (the *M-closed* setting). In the *M-open* regime, where the
> true data-generating process is not contained in the model class, MAR
> is **not sufficient for ignorability**. Misspecification introduces a
> coupling between the missingness pattern and the estimation bias: the
> observed-data likelihood under a wrong model no longer integrates out
> the missing data correctly, so which values happen to be missing
> affects the parameter estimates even when the probability of being
> missing does not depend on the unobserved values. Because any
> parametric IRT model is at best an approximation, we operate in the
> M-open world and therefore **cannot simply ignore the missing data**.
> Explicit imputation --- drawing plausible completions and averaging
> over them --- is necessary to reduce the bias that would otherwise
> result from dropping or zero-filling missing responses.

---

## 1. The Graded Response Model (GRM)

### 1.1 Samejima's GRM

The Graded Response Model (Samejima, 1969) defines the probability of
person $n$ selecting response category $k \in \{0, 1, \ldots, K-1\}$ on
item $i$ as:

$$
P(Y_{ni} = k \mid \theta_n, \alpha_i, \boldsymbol{\tau}_i)
= P^*(Y_{ni} \geq k) - P^*(Y_{ni} \geq k+1)
$$

where the cumulative boundary probabilities are:

$$
P^*(Y_{ni} \geq k) = \sigma\!\left(\alpha_i(\theta_n - \tau_{ik})\right)
= \frac{1}{1 + \exp\!\left(\alpha_i(\tau_{ik} - \theta_n)\right)}
$$

with boundary conditions $P^*(Y_{ni} \geq 0) = 1$ and
$P^*(Y_{ni} \geq K) = 0$, and where:

| Symbol | Description |
|---|---|
| $\theta_n \in \mathbb{R}$ | Latent ability of person $n$ |
| $\alpha_i > 0$ | Discrimination of item $i$ |
| $\tau_{i1} < \tau_{i2} < \cdots < \tau_{i,K-1}$ | Ordered difficulty thresholds for item $i$ |

### 1.2 Difficulty parameterization

To enforce the ordering constraint $\tau_{i1} < \cdots < \tau_{i,K-1}$,
the model uses a cumulative-sum parameterization:

$$
\tau_{i1} = \delta_{i0}, \qquad
\tau_{ik} = \delta_{i0} + \sum_{j=1}^{k-1} \Delta_{ij}, \quad k \geq 2
$$

where $\delta_{i0} \in \mathbb{R}$ is a free base difficulty and
$\Delta_{ij} > 0$ are positive increments (the `ddifficulties` parameter).
These increments are given a half-normal prior:

$$
\Delta_{ij} \sim \text{HalfNormal}(1)
$$

### 1.3 Multidimensional extension

The model extends to $D$ latent dimensions. Person $n$ has an ability
vector $\boldsymbol{\theta}_n \in \mathbb{R}^D$, and item $i$ has a
discrimination vector $\boldsymbol{\alpha}_i \in \mathbb{R}_+^D$. The
GRM probability becomes a discrimination-weighted mixture:

$$
P(Y_{ni} = k \mid \boldsymbol{\theta}_n, \boldsymbol{\alpha}_i, \boldsymbol{\tau}_i)
= \sum_{d=1}^{D} w_{id} \, P_d(Y_{ni} = k)
$$

where $P_d(Y_{ni} = k)$ is the GRM probability using dimension $d$
alone, and the weights are:

$$
w_{id} = \frac{|\alpha_{id}|}{\sum_{d'=1}^{D} |\alpha_{id'}|}
$$


## 2. The Factorized GRM (FactorizedGRModel)

### 2.1 Scale-factorized structure

The `FactorizedGRModel` partitions the $I$ items into $S$ disjoint
scales $\mathcal{S}_1, \ldots, \mathcal{S}_S$ (e.g., two subscales of
the RWA questionnaire). Each scale $s$ has its own independent set of
parameters:

- **Discriminations**: $\alpha_{i}^{(s)}$ for $i \in \mathcal{S}_s$,
  with prior $\alpha_i^{(s)} \sim \text{HalfNormal}(\kappa)$
- **Base difficulties**: $\delta_{i0}^{(s)} \sim \mathcal{N}(3, 1)$
- **Difficulty increments**: $\Delta_{ij}^{(s)} \sim \text{HalfNormal}(1)$
- **Abilities**: $\theta_n^{(s)} \sim \mathcal{N}(0, 5)$ independently per person

The full parameter vector for scale $s$ is:

$$
\boldsymbol{\phi}_s = \left\{
  \alpha_i^{(s)},\;
  \delta_{i0}^{(s)},\;
  \Delta_{ij}^{(s)},\;
  \theta_n^{(s)}
  : i \in \mathcal{S}_s,\; n = 1,\ldots,N
\right\}
$$

### 2.2 Joint log-probability

The joint prior factorizes across scales:

$$
\log p(\boldsymbol{\phi})
= \sum_{s=1}^{S} \log p(\boldsymbol{\phi}_s)
$$

The log-likelihood for a batch of $N$ persons is:

$$
\ell(\boldsymbol{\phi}; \mathbf{Y})
= \sum_{n=1}^{N} \sum_{s=1}^{S} \sum_{i \in \mathcal{S}_s}
  \log P(Y_{ni} = y_{ni} \mid \boldsymbol{\phi}_s)
$$

Missing responses ($y_{ni}$ is NaN or out-of-range) contribute zero to
the log-likelihood. The unnormalized log-posterior used as the VI target is:

$$
\widetilde{\ell}(\boldsymbol{\phi}; \mathbf{Y}, w)
= w \cdot \log p(\boldsymbol{\phi})
+ \ell(\boldsymbol{\phi}; \mathbf{Y})
$$

where $w = B / N_{\text{total}}$ is the prior weight (batch size divided
by dataset size), ensuring proper scaling in minibatch training.


## 3. Variational Inference (ADVI)

### 3.1 The ELBO

The model is fitted by maximizing the Evidence Lower Bound (ELBO):

$$
\mathcal{L}(q)
= \mathbb{E}_{q(\boldsymbol{\phi})}
  \left[\widetilde{\ell}(\boldsymbol{\phi}; \mathbf{Y}, w)\right]
- w \cdot \mathrm{KL}\!\left[q(\boldsymbol{\phi}) \,\|\, p(\boldsymbol{\phi})\right]
$$

In practice, the loss function minimized is the negative ELBO, estimated
via Monte Carlo:

$$
\hat{\mathcal{L}} = \frac{1}{S_q} \sum_{s=1}^{S_q}
\Bigl[
  w \cdot \log q(\boldsymbol{\phi}^{(s)})
  - \widetilde{\ell}(\boldsymbol{\phi}^{(s)}; \mathbf{Y}, w)
\Bigr]
$$

where $\boldsymbol{\phi}^{(1)}, \ldots, \boldsymbol{\phi}^{(S_q)}$ are
samples from the surrogate $q$.

### 3.2 Surrogate distribution

The surrogate posterior is a mean-field factored distribution built
automatically from the prior structure
(`build_factored_surrogate_posterior_generator`). Each parameter gets an
independent distribution matched to the prior's support:

| Prior support | Surrogate family |
|---|---|
| $\mathbb{R}$ (Normal prior) | $\mathcal{N}(\mu, \sigma^2)$ with trainable $\mu, \sigma$ |
| $\mathbb{R}_+$ (HalfNormal prior) | Softplus-transformed Normal |
| Inverse-Gamma | Trainable concentration-scale |

All surrogate parameters are optimized jointly via Adam with gradient
clipping, learning rate decay, and early stopping.


## 4. The Imputation Model (MICEBayesianLOO)

### 4.1 The missing data problem

Let $\mathbf{Y} = (\mathbf{Y}_{\text{obs}}, \mathbf{Y}_{\text{mis}})$
denote the complete response matrix partitioned into observed and missing
entries. The ideal target log-probability marginalizes over the missing data:

$$
\log p(\mathbf{Y}_{\text{obs}} \mid \boldsymbol{\phi})
= \log \int p(\mathbf{Y}_{\text{obs}}, \mathbf{Y}_{\text{mis}} \mid \boldsymbol{\phi})\;
  p(\mathbf{Y}_{\text{mis}} \mid \mathbf{Y}_{\text{obs}})
  \;\mathrm{d}\mathbf{Y}_{\text{mis}}
$$

This integral is intractable for discrete ordinal responses, so we
approximate it via Monte Carlo using draws from an imputation model.

### 4.2 Overview of MICEBayesianLOO

`MICEBayesianLOO` provides the imputation distribution
$\hat{p}(\mathbf{Y}_{\text{mis}} \mid \mathbf{Y}_{\text{obs}})$.
It is a library of simple Bayesian regression models --- one per
ordered pair of variables --- whose predictions are combined at
prediction time via Bayesian stacking.

For $P$ variables the framework fits:

| Model class | Count | Description |
|---|---|---|
| Zero-predictor | $P$ | Intercept-only model for each variable |
| One-predictor | up to $P(P-1)$ | Predict variable $i$ from variable $j$ |

Each model is fitted independently using **Pathfinder** variational
inference (or optionally ADVI) and evaluated via **PSIS-LOO-CV**.

### 4.3 The sub-model families

The model family chosen for each target variable depends on its inferred
type. For IRT items, all targets are ordinal.

#### 4.3.1 Ordinal logistic regression (cumulative link model)

This is the primary model used for IRT item imputation. The
**cumulative (proportional-odds) model** defines:

$$
P(Y \leq k \mid \mathbf{x}) = \sigma(c_k - \eta),
\qquad k = 1, \ldots, K-1
$$

where $\sigma(\cdot)$ is the logistic sigmoid, and the **linear
predictor** is:

$$
\eta = \boldsymbol{\beta}^\top \mathbf{x}
$$

The ordered cutpoints $c_1 < c_2 < \cdots < c_{K-1}$ are enforced via
the **ascending bijector**: given unconstrained parameters
$\mathbf{r} \in \mathbb{R}^{K-1}$,

$$
c_1 = r_1, \qquad
c_k = c_{k-1} + \text{softplus}(r_k) \;\text{ for } k \geq 2
$$

This ensures strict ordering. The category probabilities are:

$$
P(Y = k \mid \mathbf{x}) = P(Y \leq k) - P(Y \leq k-1)
$$

with boundary conditions $P(Y \leq 0) = 0$ and $P(Y \leq K) = 1$.
This is implemented via TFP's `OrderedLogistic` distribution.

**Priors.** The Bayesian ordinal logistic model has the following
priors:

$$
\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0},\, \sigma_0^2 \mathbf{I}),
\qquad
\mathbf{r} \sim \mathcal{N}(\mathbf{0},\, 25 \mathbf{I})
$$

where $\sigma_0$ is the `prior_scale` hyperparameter (default 1.0).
The wide prior on $\mathbf{r}$ is weakly informative for the cutpoint
locations.

**Predictor encoding.** When the predictor $j$ is itself ordinal with
values $\{0, 1, \ldots, V\}$, it is encoded using **thermometer
(ordinal one-hot) encoding**:

$$
\mathbf{x}(v) = [\mathbb{1}(v \geq 1),\; \mathbb{1}(v \geq 2),\; \ldots,\; \mathbb{1}(v \geq V)]
\;\in\; \{0,1\}^V
$$

This preserves the ordinal structure: each additional unit of the
predictor "turns on" one more indicator. For continuous predictors,
the raw value is standardized to zero mean and unit variance.

#### 4.3.2 Binary logistic regression

For binary targets ($K = 2$), the model simplifies to standard
logistic regression:

$$
P(Y = 1 \mid \mathbf{x}) = \sigma(\boldsymbol{\beta}^\top \mathbf{x} + b)
$$

$$
\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0},\, \sigma_0^2 \mathbf{I}),
\qquad b \sim \mathcal{N}(0, \sigma_0^2)
$$

#### 4.3.3 Linear regression

For continuous targets, a Bayesian linear regression is used:

$$
Y \mid \mathbf{x} \sim \mathcal{N}(\boldsymbol{\beta}^\top \mathbf{x} + b,\; \sigma^2)
$$

$$
\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0},\, \sigma_0^2 \mathbf{I}),
\quad b \sim \mathcal{N}(0, \sigma_0^2),
\quad \log \sigma \sim \mathcal{N}(\log \sigma_{\text{noise}},\, 1)
$$

This model is not used for IRT items but may appear for auxiliary
variables in the imputation framework.

### 4.4 Posterior inference with Pathfinder

Each sub-model's posterior is approximated using **Pathfinder**
(Zhang et al., 2022), a fast variational method that:

1. Runs L-BFGS on the unnormalized log-posterior.
2. At each iterate, fits a diagonal-Gaussian approximation using
   the L-BFGS inverse-Hessian estimate.
3. Selects the best approximation by ELBO.
4. Draws samples from that Gaussian.

Pathfinder is much faster than full ADVI or MCMC for these low-
dimensional sub-models (typically 1--10 parameters each), making it
feasible to fit $O(P^2)$ models.

### 4.5 Model evaluation via PSIS-LOO-CV

Each fitted sub-model is evaluated using **Pareto-Smoothed Importance
Sampling Leave-One-Out** cross-validation (Vehtari et al., 2017).
Given $S$ posterior samples and $N$ observations, the pointwise
LOO log-predictive density is estimated as:

$$
\widehat{\text{elpd}}_{\text{LOO}}
= \sum_{n=1}^{N} \log \hat{p}(y_n \mid y_{-n})
$$

where

$$
\hat{p}(y_n \mid y_{-n})
= \frac{
  \sum_{s=1}^{S} w_n^{(s)} \, p(y_n \mid \boldsymbol{\psi}^{(s)})
}{
  \sum_{s=1}^{S} w_n^{(s)}
}
$$

and $w_n^{(s)}$ are Pareto-smoothed importance weights derived from
the leave-one-out ratios $1 / p(y_n \mid \boldsymbol{\psi}^{(s)})$.

The PSIS diagnostic $\hat{k}$ quantifies the reliability of the
importance sampling approximation:

| $\hat{k}$ | Interpretation |
|---|---|
| $< 0.5$ | Excellent; IS estimate is reliable |
| $0.5$--$0.7$ | Acceptable; moderate IS variance |
| $> 0.7$ | Unreliable; IS approximation breaks down |

Each sub-model stores:
- `elpd_loo_per_obs`: $\widehat{\text{elpd}}_{\text{LOO}} / N$ (normalized for comparability across different sample sizes)
- `khat_max`: worst-case $\hat{k}$ across observations
- `converged`: whether Pathfinder converged
- Point estimates (`beta_mean`, `intercept_mean`, `cutpoints_mean`) for prediction

### 4.6 Stacking weights at prediction time

When predicting item $i$ for person $n$, the imputation model
assembles all available sub-models (zero-predictor + one-predictor
models for each observed item $j$) and computes **stacking weights**.

Let $\mathcal{M} = \{M_0, M_1, \ldots, M_J\}$ denote the set of
available models for target $i$, where $M_0$ is the zero-predictor
and $M_j$ uses observed item $j$ as predictor. Each model $M_j$ has
an associated LOO-ELPD $E_j$ and standard error $\text{SE}_j$.

The uncertainty-penalized stacking weight for model $j$ is:

$$
\tilde{w}_j = \exp(E_j - \lambda \cdot \text{SE}_j)
$$

where $\lambda$ is the `uncertainty_penalty` parameter (default 1.0,
corresponding to a roughly one-standard-error lower confidence bound).
The normalized weights are:

$$
w_j = \frac{\tilde{w}_j}{\sum_{j'} \tilde{w}_{j'}}
$$

This is a softmax over uncertainty-adjusted ELPDs: models with higher
predictive accuracy get more weight, and models with uncertain ELPD
estimates are penalized.

### 4.7 Constructing the imputation PMF

For ordinal targets (the IRT case), each sub-model $M_j$ produces a
categorical PMF over $\{0, \ldots, K-1\}$ via the cumulative model:

$$
p_{M_j}(Y = k \mid x_j)
= \sigma(c_k^{(j)} - \eta_j) - \sigma(c_{k-1}^{(j)} - \eta_j)
$$

where $\eta_j = \bar{\beta}_j \cdot \tilde{x}_j + \bar{b}_j$ uses
the posterior mean estimates $\bar{\beta}_j$, $\bar{b}_j$, and
$\bar{\mathbf{c}}^{(j)}$ (posterior mean of the transformed cutpoints).
The predictor value $\tilde{x}_j$ is standardized using the training
mean and standard deviation stored in the sub-model result.

The stacked imputation PMF is the **finite mixture**:

$$
\hat{p}(Y_{ni} = k \mid \mathbf{Y}_{n,-i}^{\text{obs}})
= \sum_{j \in \mathcal{M}} w_j \; p_{M_j}(Y_{ni} = k \mid x_{nj})
$$

This is a proper categorical distribution (sums to 1). At each
training step, the IRT model draws imputed values
$y_{ni}^{(m)} \sim \text{Categorical}(\hat{p})$ independently for
each missing cell and each of the $M$ imputation copies.

### 4.8 Why univariate models?

A natural question is why `MICEBayesianLOO` uses only **one-predictor**
models rather than multivariate models using all observed items
simultaneously. The reasons are:

1. **Scalability**: Fitting $P^2$ tiny models (each with $\leq 10$
   parameters) via Pathfinder is fast and embarrassingly parallel.
   A single multivariate model for each target would have $O(PK)$
   parameters and require more careful regularization.

2. **Robustness to missingness patterns**: Each one-predictor model
   $(i, j)$ is trained only on rows where both $i$ and $j$ are
   observed. Different predictor models may use different subsets of
   the data, avoiding the need to handle arbitrary missingness
   patterns within a single model.

3. **Stacking provides adaptive combination**: By weighting models
   according to their LOO-ELPD, the framework automatically upweights
   the most informative predictors for each target. The stacking
   mixture can approximate multivariate predictive distributions
   without explicitly fitting them.

4. **LOO-CV diagnostics per model**: Having simple models makes PSIS
   diagnostics interpretable. A high $\hat{k}$ for model $(i,j)$
   directly indicates that item $j$ is a poor predictor of item $i$.

### 4.9 Validation checks

Before fitting the IRT model, `validate_imputation_model()` performs
five checks:

1. **Fitted**: The imputation model has been trained.
2. **Coverage**: All IRT item keys appear in the imputation model.
3. **Variable type**: Items must be ordinal or binary (not continuous).
4. **Convergence**: At least one converged sub-model per item (warning if not).
5. **PSIS $\hat{k}$**: Best model's $\hat{k} < 0.7$ (warning if not).


## 5. Rao-Blackwellized Training Objective

### 5.1 The naive approach (averaging log-likelihoods)

A previous implementation yielded $M$ imputed copies as $M$ separate
mini-batches. The optimizer then averaged gradients across these batches,
which is equivalent to optimizing:

$$
\frac{1}{M} \sum_{m=1}^{M}
\widetilde{\ell}(\boldsymbol{\phi};\, \mathbf{Y}_{\text{obs}},
\mathbf{Y}_{\text{mis}}^{(m)},\, w)
$$

Since $\widetilde{\ell}$ contains a $\log$, this averages
**log-likelihoods**. By Jensen's inequality:

$$
\frac{1}{M} \sum_{m=1}^{M}
\log p(\mathbf{Y}_{\text{obs}}, \mathbf{Y}_{\text{mis}}^{(m)}
\mid \boldsymbol{\phi})
\;\leq\;
\log \left[\frac{1}{M} \sum_{m=1}^{M}
p(\mathbf{Y}_{\text{obs}}, \mathbf{Y}_{\text{mis}}^{(m)}
\mid \boldsymbol{\phi})\right]
$$

The left side is a **lower bound** on the properly marginalized
log-likelihood, introducing downward bias in the ELBO.

### 5.2 The Rao-Blackwellized approach (averaging likelihoods)

The correct Monte Carlo estimate of the marginalized log-likelihood is:

$$
\log p(\mathbf{Y}_{\text{obs}} \mid \boldsymbol{\phi})
\approx \log \left[\frac{1}{M} \sum_{m=1}^{M}
p(\mathbf{Y}_{\text{obs}}, \mathbf{Y}_{\text{mis}}^{(m)}
\mid \boldsymbol{\phi})\right]
$$

In log-space, this is computed via the log-sum-exp trick:

$$
\log \left[\frac{1}{M} \sum_{m=1}^{M}
p(\mathbf{Y}_{\text{obs}}, \mathbf{Y}_{\text{mis}}^{(m)}
\mid \boldsymbol{\phi})\right]
= \mathrm{logsumexp}_{m}(\ell_m) - \log M
$$

where
$\ell_m = \log p(\mathbf{Y}_{\text{obs}}, \mathbf{Y}_{\text{mis}}^{(m)}
\mid \boldsymbol{\phi})$.

### 5.3 Incorporating the prior

The full unnormalized log-posterior for a batch is:

$$
\widetilde{\ell}(\boldsymbol{\phi}; \mathbf{Y}^{(m)}, w)
= w \cdot \log p(\boldsymbol{\phi})
+ \ell_m(\boldsymbol{\phi})
$$

Since the prior term $w \cdot \log p(\boldsymbol{\phi})$ is constant
across imputations (it depends only on $\boldsymbol{\phi}$, not on
$\mathbf{Y}_{\text{mis}}$), the logsumexp factors cleanly:

$$
\mathrm{logsumexp}_{m}(\widetilde{\ell}_m) - \log M
= \mathrm{logsumexp}_{m}(C + \ell_m) - \log M
= C + \mathrm{logsumexp}_{m}(\ell_m) - \log M
$$

where $C = w \cdot \log p(\boldsymbol{\phi})$. This equals:

$$
w \cdot \log p(\boldsymbol{\phi})
+ \log\left[\frac{1}{M}\sum_{m=1}^{M}
p(\mathbf{Y}_{\text{obs}}, \mathbf{Y}_{\text{mis}}^{(m)}
\mid \boldsymbol{\phi})\right]
$$

which is the correct Rao-Blackwellized target. The implementation
therefore calls `unormalized_log_prob` on each imputed copy and applies
`logsumexp` directly to those values.

### 5.4 Why "Rao-Blackwellization"?

The Rao-Blackwell theorem states that conditioning an unbiased estimator
on a sufficient statistic reduces its variance without introducing bias.
Here, averaging **likelihoods** before taking the log is the
Rao-Blackwellized estimator of the marginal log-likelihood under the
imputation distribution. It has lower variance (and no Jensen bias)
compared to averaging log-likelihoods.

### 5.5 Special cases

- **$M = 1$**: $\mathrm{logsumexp}([\ell_1]) - \log 1 = \ell_1$.
  Reduces to ordinary single-imputation; no overhead.

- **No missing data**: Item arrays have `ndim == 1` (not stacked).
  The wrapper falls through to `unormalized_log_prob` directly.

- **All copies identical** (no missing values in a batch): Since all
  $\ell_m$ are equal, $\mathrm{logsumexp}([\ell, \ldots, \ell]) - \log M
  = \ell + \log M - \log M = \ell$. Correct.


## 6. Implementation Details

### 6.1 Data layout

For a batch of $N$ persons with $M$ imputation copies:

| Key | Shape (no missing) | Shape (with missing) |
|---|---|---|
| `person` | $(N,)$ | $(N,)$ |
| each item key | $(N,)$ | $(M, N)$ |

The stacking is done in NumPy (CPU) before the batch enters the
JIT-compiled JAX computation graph.

### 6.2 Wrapped log-probability function

The `rao_blackwell_log_prob` closure is passed as
`unormalized_log_prob_fn` to `_calibrate_minibatch_advi`. It detects
stacked data via `ndim > 1` on the first item key:

```python
def rao_blackwell_log_prob(data, prior_weight, **params):
    if data[first_item].ndim > 1:
        M = data[first_item].shape[0]
        results = [
            self.unormalized_log_prob(data=data_m, ...)
            for m in range(M)
        ]
        return logsumexp(stack(results)) - log(M)
    else:
        return self.unormalized_log_prob(data=data, ...)
```

Since $M$ is a compile-time constant (the Python `int`
`n_imputation_samples`), JAX unrolls the loop during tracing. The
`logsumexp` and `log(M)` operations are standard differentiable
primitives, so gradients flow correctly through the entire computation.

### 6.3 Gradient considerations

Let $L(\boldsymbol{\phi}) = \mathrm{logsumexp}_m(\ell_m) - \log M$.
The gradient with respect to $\boldsymbol{\phi}$ is:

$$
\nabla_{\boldsymbol{\phi}} L
= \sum_{m=1}^{M} \tilde{w}_m \,\nabla_{\boldsymbol{\phi}} \ell_m
$$

where the softmax weights are:

$$
\tilde{w}_m
= \frac{\exp(\ell_m)}{\sum_{m'} \exp(\ell_{m'})}
= \frac{p(\mathbf{Y}^{(m)} \mid \boldsymbol{\phi})}
       {\sum_{m'} p(\mathbf{Y}^{(m')} \mid \boldsymbol{\phi})}
$$

This means that imputed copies with higher likelihood under the current
parameters receive larger gradient weight --- a natural importance
weighting effect that focuses optimization on the most plausible
imputations.


## 7. Notebook Workflow Summary

1. **Data**: RWA scale (22 items, 9 response categories), subsampled to 500 persons.
2. **Artificial missingness**: 15% MCAR on top of the natural ~0.3%.
3. **Imputation model**: `MICEBayesianLOO` with Bayesian stacking of ordinal logistic regressions.
4. **Baseline**: `FactorizedGRModel` with zero-fill for missing data (missing responses contribute 0 to $\ell$).
5. **Imputed**: `FactorizedGRModel` with `imputation_model` and `n_imputation_samples=3`, using Rao-Blackwellized logsumexp objective.
6. **Comparison**: Training loss curves and posterior estimates for discriminations and abilities across scales.


## References

- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph Supplement*, 34(4, Pt. 2), 1--97.
- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859--877.
- Zhang, L., Carpenter, B., Gelman, A., & Vehtari, A. (2022). Pathfinder: Parallel quasi-Newton variational inference. *Journal of Machine Learning Research*, 23(306), 1--49.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413--1432.
- Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using stacking to average Bayesian predictive distributions. *Bayesian Analysis*, 13(3), 917--1007.
- Agresti, A. (2010). *Analysis of Ordinal Categorical Data* (2nd ed.). Wiley.
- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley.

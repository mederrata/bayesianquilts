# Factorized Graded Response Model with Rao-Blackwellized Imputation

This report documents the mathematical foundations behind the
`factorized_grm_missing` notebook, covering the Graded Response Model,
variational inference, stochastic imputation, and the Rao-Blackwellized
training objective.

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
P^*(Y_{ni} \geq k) = \sigma\bigl(\alpha_i(\theta_n - \tau_{ik})\bigr)
= \frac{1}{1 + \exp\bigl(\alpha_i(\tau_{ik} - \theta_n)\bigr)}
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
\boldsymbol{\phi}_s = \bigl\{
  \alpha_i^{(s)},\;
  \delta_{i0}^{(s)},\;
  \Delta_{ij}^{(s)},\;
  \theta_n^{(s)}
  : i \in \mathcal{S}_s,\; n = 1,\ldots,N
\bigr\}
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
  \log P\bigl(Y_{ni} = y_{ni} \mid \boldsymbol{\phi}_s\bigr)
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
  \bigl[\widetilde{\ell}(\boldsymbol{\phi}; \mathbf{Y}, w)\bigr]
- w \cdot \mathrm{KL}\bigl[q(\boldsymbol{\phi}) \,\|\, p(\boldsymbol{\phi})\bigr]
$$

In practice, the loss function minimized is the negative ELBO, estimated
via Monte Carlo:

$$
\hat{\mathcal{L}} = \frac{1}{S_q} \sum_{s=1}^{S_q}
\Bigl[
  w \cdot \log q(\boldsymbol{\phi}^{(s)})
  - \widetilde{\ell}\bigl(\boldsymbol{\phi}^{(s)}; \mathbf{Y}, w\bigr)
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


## 4. Stochastic Imputation

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
approximate it via Monte Carlo.

### 4.2 Imputation model: MICEBayesianLOO

The imputation distribution
$p(\mathbf{Y}_{\text{mis}} \mid \mathbf{Y}_{\text{obs}})$ is provided
by a `MICEBayesianLOO` model. For each item $i$ with a missing response
for person $n$, the imputation model provides a categorical PMF:

$$
\hat{p}(Y_{ni} = k \mid \mathbf{Y}_{n, -i}^{\text{obs}})
\quad\text{for}\quad k = 0, \ldots, K-1
$$

This PMF is a Bayesian stacking mixture of ordinal logistic regression
models, each predicting item $i$ from a single other observed item $j$.
The stacking weights are determined by LOO-CV expected log pointwise
predictive density (ELPD).

At each training step, imputed values $y_{ni}^{(m)}$ are sampled from
these PMFs independently for $m = 1, \ldots, M$, producing $M$
completed datasets.

### 4.3 Validation checks

Before fitting, `validate_imputation_model()` performs five checks:

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
\widetilde{\ell}\bigl(\boldsymbol{\phi};\, \mathbf{Y}_{\text{obs}},
\mathbf{Y}_{\text{mis}}^{(m)},\, w\bigr)
$$

Since $\widetilde{\ell}$ contains a $\log$, this averages
**log-likelihoods**. By Jensen's inequality:

$$
\frac{1}{M} \sum_{m=1}^{M}
\log p\bigl(\mathbf{Y}_{\text{obs}}, \mathbf{Y}_{\text{mis}}^{(m)}
\mid \boldsymbol{\phi}\bigr)
\;\leq\;
\log \biggl[\frac{1}{M} \sum_{m=1}^{M}
p\bigl(\mathbf{Y}_{\text{obs}}, \mathbf{Y}_{\text{mis}}^{(m)}
\mid \boldsymbol{\phi}\bigr)\biggr]
$$

The left side is a **lower bound** on the properly marginalized
log-likelihood, introducing downward bias in the ELBO.

### 5.2 The Rao-Blackwellized approach (averaging likelihoods)

The correct Monte Carlo estimate of the marginalized log-likelihood is:

$$
\log p(\mathbf{Y}_{\text{obs}} \mid \boldsymbol{\phi})
\approx \log \biggl[\frac{1}{M} \sum_{m=1}^{M}
p\bigl(\mathbf{Y}_{\text{obs}}, \mathbf{Y}_{\text{mis}}^{(m)}
\mid \boldsymbol{\phi}\bigr)\biggr]
$$

In log-space, this is computed via the log-sum-exp trick:

$$
\log \biggl[\frac{1}{M} \sum_{m=1}^{M}
p\bigl(\mathbf{Y}_{\text{obs}}, \mathbf{Y}_{\text{mis}}^{(m)}
\mid \boldsymbol{\phi}\bigr)\biggr]
= \mathrm{logsumexp}_{m}\bigl(\ell_m\bigr) - \log M
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
\mathrm{logsumexp}_{m}\bigl(\widetilde{\ell}_m\bigr) - \log M
= \mathrm{logsumexp}_{m}\bigl(C + \ell_m\bigr) - \log M
= C + \mathrm{logsumexp}_{m}\bigl(\ell_m\bigr) - \log M
$$

where $C = w \cdot \log p(\boldsymbol{\phi})$. This equals:

$$
w \cdot \log p(\boldsymbol{\phi})
+ \log\biggl[\frac{1}{M}\sum_{m=1}^{M}
p\bigl(\mathbf{Y}_{\text{obs}}, \mathbf{Y}_{\text{mis}}^{(m)}
\mid \boldsymbol{\phi}\bigr)\biggr]
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
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413--1432.
- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley.

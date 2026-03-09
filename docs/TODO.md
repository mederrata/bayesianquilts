# ADVI & Training Enhancements — TODO

Status of the improvement program started on the `advi_enhancements` branch.
Items marked **done** are implemented and unit-tested; items marked **todo**
are planned next steps.

---

## Completed

### 1. STL gradient estimator (`minibatch.py`)
- `stl=True` (default) applies `stop_gradient` to `log q(theta)` in the ELBO,
  eliminating the high-variance score-function term (Roeder et al., 2017).

### 2. Gradient accumulation averaging (`util.py`)
- Accumulated gradients are now divided by `accumulation_steps` before the
  optimizer update. Previously the accumulator was an identity (no averaging).

### 3. `inject_hyperparams` for LR decay (`util.py`)
- Learning rate decay now uses `optax.inject_hyperparams`, so reducing the LR
  only changes the step-size hyperparameter while preserving Adam's
  first/second moment estimates. Previously the optimizer was re-created,
  destroying momentum state.

### 4. Reproducible PRNG key threading (`util.py`, `minibatch.py`)
- `training_loop` accepts `seed: int`. When provided, a JAX PRNG key is split
  per step and passed to the loss function, making sampling reproducible under
  JIT. Falls back to `np.random` when `seed=None`.

### 5. Log-scale parameterization (`advi.py`)
- `parameterization="log_scale"` stores `log(sigma)` directly and computes
  `scale = exp(log_scale)`. Avoids the softplus underflow that causes NaN
  gradients at small scales, and is more numerically stable in float16/bfloat16.

### 6. Low-rank Gaussian surrogate (`advi.py`)
- `rank=r` (r > 0) gives a surrogate with covariance
  `diag(exp(2*log_diag_scale)) + F @ F^T` per variable.
  Implemented via `tfd.MultivariateNormalTriL` with an explicit Cholesky of
  `D + FF^T` (O(d^3) traced once under JIT). Reshape bijector maps flat
  samples back to the original event shape.

### 7. Mixed precision support (`util.py`)
- `compute_dtype` parameter casts data arrays to a specified dtype
  (e.g. `jnp.float32`) before passing to the loss function, while variational
  parameters stay in their native precision.

### 8. Natural parameterization (`advi.py`)
- `parameterization="natural"` stores `(eta1, eta2)` where
  `eta1 = mu/sigma^2` and `eta2 = -softplus(raw)`.
  Standard Adam on natural parameters implicitly approximates natural gradient
  descent on `(mu, sigma^2)`.

### 9. Standard ELBO scaling (`minibatch.py`)
- ELBO now uses the textbook formulation:
  `loss = E_q[log q - (N/B) * target_log_prob]`.
  **Note:** effective gradients are `N/B` times larger than the old scaling.
  Users coming from the previous formulation should reduce their learning rate
  by a factor of `B/N` (e.g. for `N=10000, B=256`: multiply old LR by ~0.026).

### 10. Per-datum ELBO normalization (`minibatch.py`)
- The ELBO loss is now divided by `dataset_size` so the loss and its gradients
  are O(1) regardless of dataset size. Under STL the effective gradient is
  `-(1/B) * grad[log_lik(batch)] - (1/N) * grad[log_prior]`, which is
  dataset-size-independent. The `dataset_size=1, batch_size=1` path used by
  MICE-LOO is unaffected (dividing by 1 is a no-op).

### 15. Pathfinder initialization (`advi.py`, `model.py`)
- `pathfinder_initialize()` uses the Pathfinder algorithm (via blackjax) to
  find a good initial `(loc, scale)` from a few L-BFGS iterations on the log
  joint density before starting ADVI. Supports all parameterizations
  (softplus, log_scale, natural, low-rank). Wired through
  `_calibrate_minibatch_advi(pathfinder_init=True)` and `fit()`.

---

## TODO — Next Priorities

### 11. Warm-start low-rank from mean-field
Allow initializing a `rank>0` surrogate from a previously fitted mean-field
solution. Copy the mean-field `loc` and `log_scale` into the low-rank
`loc` and `log_diag_scale`, and zero-initialize the factor matrix. This
avoids re-learning the diagonal component when upgrading to low-rank.

### 12. Structured low-rank: block-diagonal covariance
The current low-rank implementation computes a full `d x d` Cholesky for each
variable, which is O(d^3). For variables with large `d` (e.g. abilities with
`d = num_people`), this is prohibitive. Options:

- **Block-diagonal**: partition the `d` dimensions into blocks of size `b`
  and fit a separate low-rank MVN per block. Reduces cost to O(d/b * b^3).
- **Kronecker-factored**: for variables with event shape `(D, K)`, model
  covariance as `Sigma_D (x) Sigma_K` (KFAC-style).
- **Woodbury-based sampling/log_prob**: avoid the full Cholesky by using
  the Woodbury identity directly for sampling (`z = loc + D^{1/2} eps1 + F eps2`)
  and log_prob (O(d*r^2 + r^3)). This was the original approach but hit TFP
  tree-structure issues; could be revisited with a non-TFP wrapper class
  that implements `sample`, `log_prob`, and `experimental_sample_and_log_prob`.

### 13. Control-variate baselines for the ELBO
The STL estimator removes the score function gradient but the
reparameterization gradient can still have high variance for complex models.
Control variates (e.g. a linear baseline fitted to recent gradient history)
can further reduce variance without additional bias.

### 14. Entropy bonus / KL annealing
Add optional KL annealing (`beta` warmup) to prevent posterior collapse in
the early stages of training, especially for models with many latent
variables (e.g. per-person abilities in IRT):

```python
loss = beta * E_q[log q - log p(theta)] - (N/B) * E_q[log p(batch | theta)]
```

where `beta` ramps from 0 to 1 over the first K epochs.

### 16. Gradient variance monitoring
Log the per-parameter gradient variance (or signal-to-noise ratio) during
training. This helps diagnose:

- Score function gradient explosion (fixed by STL, but good to verify)
- Scale parameter updates being dominated by noise
- Need for more MC samples or control variates

Could be implemented as an optional callback in `training_loop`.

### 17. Multi-sample IWAE bound
Support the importance-weighted ELBO (IWAE, Burda et al., 2016) as an
alternative to the standard single-sample ELBO. The `cost="tfp"` path
partially supports this via `importance_sample_size`, but the interaction
with STL and the new scaling hasn't been verified. The IWAE bound is tighter
than ELBO and can improve posterior approximation quality at the cost of
higher gradient variance.

### 18. Stein variational gradient descent (SVGD) option
For models where mean-field / low-rank Gaussians are too restrictive,
offer a particle-based alternative using SVGD kernels. This would maintain
the minibatch training infrastructure but replace the parametric surrogate
with a set of particles.

### 19. Automatic rank selection for low-rank surrogate
Adaptively increase the rank `r` during training based on a criterion like:

- Fraction of variance explained by the top-r eigenvalues of the
  empirical gradient covariance
- Improvement in ELBO relative to the additional parameter cost

Start with `r=0` (mean-field), then promote to `r=1, 2, ...` when the
mean-field approximation plateaus.

---

## Testing gaps

- **Integration test with IRT models**: The IRT notebooks
  (`grm_single_scale.ipynb`, `factorized_grm_missing.ipynb`) use
  `learning_rate=2e-4` which was tuned for the old ELBO scaling. Need to
  verify they still converge after the scaling change (item 9), or
  document the required LR adjustment.
- **Natural parameterization convergence**: The natural parameterization
  initializes `eta1 = mu/noise^2` which can be very large when `noise` is
  small (default `1e-2` gives `eta1 ~ 10^4 * mu`). Needs either a
  recommended `noise` value in the docstring or automatic scaling.
- **Mixed precision end-to-end**: The `compute_dtype` flag casts data but
  not parameters. Need to test that float32 parameters + float16 data
  produces stable gradients for a representative model.
- **Checkpoint compatibility**: The `inject_hyperparams` change alters the
  structure of `opt_state` (adds `.hyperparams` dict). Existing Orbax
  checkpoints from before this change cannot be restored. Document the
  breaking change or add a migration path.

# Regularized moment matching for LOO

`bayesianquilts.metrics.regularized_loo` is the correctness-first PMM API used
by the revised LOO experiments. It differs from the legacy exploratory AIS API
in four deliberate ways:

1. the PMM step is the literal interpolation `T_h = (1-h) I + h T_1`, with
   `0 <= h <= 1`;
2. PMM1, PMM2, and PMM3 use their exact affine Jacobian determinants;
3. transformed weights require a full-posterior log-density callback, so the
   prior contribution cannot be omitted;
4. map fitting, step tuning, and final estimation take separate draw arrays.

Step tuning uses a bounded loss,
`mean(min((log_weight - candidate_fit_mean)**2, clip**2))`.  Each candidate's
center is estimated only from the fitting split and then held fixed on the
tuning split.  Clipping affects selection only; returned evaluation weights
are exact and unclipped.

The callbacks receive a two-dimensional array of flattened, unconstrained
parameter draws and return one log density per row. Normalizing constants may
be omitted. For one held-out observation:

```python
import jax
import jax.numpy as jnp

from bayesianquilts.metrics.regularized_loo import (
    crossfit_regularized_pmm_loo_fold,
    self_normalized_expectation,
    three_fold_split,
)

master_key = jax.random.PRNGKey(20_260_902)
folds = three_fold_split(posterior_draws, master_key)

result = crossfit_regularized_pmm_loo_fold(
    folds,
    log_full_posterior=log_full_posterior,       # includes the prior
    log_heldout_likelihood=log_likelihood_i,
    method="pmm3",
    h_grid=jnp.linspace(0.0, 1.0, 21),
)

loo_mean = self_normalized_expectation(
    target_function(result.transformed_evaluation_draws),
    result.evaluation_log_weights,
)
```

The three-fold routine rotates `(fit, tune, evaluate)` roles as `(A,B,C)`,
`(B,C,A)`, and `(C,A,B)`. Each draw is therefore used once for final
estimation, but no final estimate is evaluated on draws that fitted its map or
selected its step.

All transformations must be performed in unconstrained coordinates. PMM3
requires a positive-definite empirical covariance; its `ridge` argument is a
numerical regularizer, not the partial-step regularizer. The selected partial
step is available as `result.selected_steps`.

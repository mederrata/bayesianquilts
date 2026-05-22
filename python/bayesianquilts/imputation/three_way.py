"""Three-way Yao stacking imputation: MICE + IRT baseline + IRT shared-disc.

For each item, solves the 3-simplex stacking optimization:

    max_{w in simplex^2} sum_i log[
        w_m * exp(lpd_mice_i)  +
        w_i * exp(lpd_irt_i)   +
        w_s * exp(lpd_shared_i)
    ]

where lpd_* are pointwise LOO log predictive densities for that item.
Each item gets its own (w_m, w_i, w_s) — items the IRT baseline already
predicts well will downweight MICE and shared-disc, items where per-item
slopes are weakly identified will give weight to shared-disc, and items
that benefit from observable conditional structure will give weight to MICE.

Exposes the same ``predict_pmf`` / ``predict`` interface IRTModel expects
from an ``imputation_model``.
"""

import sys
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


def _warn_fallback(msg, exc=None):
    detail = f" ({type(exc).__name__}: {exc})" if exc else ""
    sys.stderr.write(
        f"\033[91mWARNING: {msg}{detail}\033[0m\n"
    )
    sys.stderr.flush()


def _compute_grm_loo_per_item(grm_model, data_factory, n_quad: int = 61):
    """Per-item LOO log predictive density from a fitted single-dim GRM.

    Uses Gauss-Hermite quadrature on the N(0,1) ability prior. Works for
    both the per-item-discriminations baseline GRM and the shared-disc
    GRM because both share ``grm_model_prob_d`` and have ``discriminations``
    in their surrogate samples; the shared-disc variant just has
    discriminations broadcasting across items.

    Returns
    -------
    elpd_per_item : dict[str, float]
    loo_per_obs   : dict[str, np.ndarray]  pointwise log densities
    """
    from numpy.polynomial.hermite import hermgauss

    item_keys = grm_model.item_keys
    K = grm_model.response_cardinality

    samples = grm_model.surrogate_sample
    if hasattr(grm_model, 'transform') and 'discriminations' not in samples:
        samples = grm_model.transform(dict(samples))

    disc_mean = np.asarray(samples["discriminations"]).mean(0)
    diff0_mean = np.asarray(samples["difficulties0"]).mean(0)
    ddiff_mean = (np.asarray(samples["ddifficulties"]).mean(0)
                  if "ddifficulties" in samples else None)

    nodes, weights = hermgauss(n_quad)
    quad_points = nodes * np.sqrt(2)
    quad_weights = weights / np.sqrt(np.pi)

    D = disc_mean.shape[0] if disc_mean.ndim >= 2 else 1
    theta_q = np.zeros((n_quad, D, 1, 1), dtype=np.float64)
    theta_q[:, 0, 0, 0] = quad_points

    probs_at_quad = np.asarray(grm_model.grm_model_prob_d(
        jnp.asarray(theta_q),
        jnp.asarray(disc_mean),
        jnp.asarray(diff0_mean),
        jnp.asarray(ddiff_mean) if ddiff_mean is not None else None,
    ))  # (Q, I, K)
    probs_at_quad = np.clip(probs_at_quad, 1e-30, None)
    log_probs_at_quad = np.log(probs_at_quad)
    log_quad_w = np.log(quad_weights)

    item_loo_scores: Dict[str, List[np.ndarray]] = {k: [] for k in item_keys}

    for batch in data_factory():
        responses = np.stack(
            [np.asarray(batch[k], dtype=np.float64) for k in item_keys],
            axis=-1,
        )
        observed = ~np.isnan(responses) & (responses >= 0) & (responses < K)
        responses_int = np.where(observed, responses.astype(np.int64), 0)

        log_lik_per_item = log_probs_at_quad[:, np.newaxis, :, :]  # (Q,1,I,K)
        resp_idx = responses_int[np.newaxis, :, :, np.newaxis]
        log_lik_qi = np.take_along_axis(
            log_lik_per_item, resp_idx, axis=-1
        )[..., 0]  # (Q, N, I)
        log_lik_qi = np.where(observed[np.newaxis, :, :], log_lik_qi, 0.0)
        total_ll = np.sum(log_lik_qi, axis=-1)

        for i_idx, item_key in enumerate(item_keys):
            mask_i = observed[:, i_idx]
            if not np.any(mask_i):
                continue
            loo_ll = total_ll - log_lik_qi[:, :, i_idx]
            log_posterior = log_quad_w[:, np.newaxis] + loo_ll
            log_Z = np.logaddexp.reduce(log_posterior, axis=0)
            log_post_norm = log_posterior - log_Z[np.newaxis, :]
            log_pred = log_post_norm + log_lik_qi[:, :, i_idx]
            log_cond_pred = np.logaddexp.reduce(log_pred, axis=0)
            item_loo_scores[item_key].append(log_cond_pred[mask_i])

    elpd_per_item: Dict[str, float] = {}
    loo_per_obs: Dict[str, np.ndarray] = {}
    for item_key in item_keys:
        scores = item_loo_scores[item_key]
        if scores:
            arr = np.concatenate(scores)
            elpd_per_item[item_key] = float(np.mean(arr))
            loo_per_obs[item_key] = arr
        else:
            elpd_per_item[item_key] = -np.inf
            loo_per_obs[item_key] = np.array([])
    return elpd_per_item, loo_per_obs


def _precompute_grm_pmf_matrix(grm_model):
    """Posterior-mean PMF matrix (N, I, K) for a fitted single-dim GRM."""
    samples = grm_model.surrogate_sample
    if hasattr(grm_model, 'transform') and 'discriminations' not in samples:
        samples = grm_model.transform(dict(samples))
    disc = np.asarray(samples["discriminations"]).mean(0)
    diff0 = np.asarray(samples["difficulties0"]).mean(0)
    ddiff = (np.asarray(samples["ddifficulties"]).mean(0)
             if "ddifficulties" in samples else None)
    abil = np.asarray(samples["abilities"]).mean(0)
    probs = np.asarray(grm_model.grm_model_prob_d(
        jnp.asarray(abil[np.newaxis]),
        jnp.asarray(disc[np.newaxis]),
        jnp.asarray(diff0[np.newaxis]),
        jnp.asarray(ddiff[np.newaxis]) if ddiff is not None else None,
    ))  # (1, N, I, K)
    return probs[0]


def _solve_two_way_stacking(lpd_m, lpd_i):
    """Per-item Yao stacking on the (w_m, w_i) 1-simplex (no shared-disc)."""
    lpd_m = np.asarray(lpd_m, dtype=np.float64)
    lpd_i = np.asarray(lpd_i, dtype=np.float64)
    n = min(len(lpd_m), len(lpd_i))
    if n == 0:
        return np.array([0.5, 0.5, 0.0])
    lpd_m, lpd_i = lpd_m[:n], lpd_i[:n]

    def neg_log_score(logit):
        # w_m = sigmoid(logit), w_i = 1 - w_m. logit is a length-1 array from
        # scipy.optimize; extract the scalar before exponentiating.
        s = 1.0 / (1.0 + np.exp(-float(np.asarray(logit).ravel()[0])))
        w_m, w_i = float(s), float(1.0 - s)
        a = lpd_m + np.log(max(w_m, 1e-15))
        b = lpd_i + np.log(max(w_i, 1e-15))
        m = np.maximum(a, b)
        log_mix = m + np.log(np.exp(a - m) + np.exp(b - m))
        return -float(np.sum(log_mix))

    res = minimize(
        neg_log_score, x0=np.array([0.0]), method='BFGS',
        options={'maxiter': 100, 'gtol': 1e-7},
    )
    s = 1.0 / (1.0 + np.exp(-res.x[0]))
    return np.array([float(s), float(1.0 - s), 0.0])


def _solve_three_way_stacking(lpd_m, lpd_i, lpd_s):
    """Per-item Yao stacking on the 2-simplex.

    max_{w_m + w_i + w_s = 1, w_* >= 0}
        sum_n log [ w_m exp(lpd_m_n) + w_i exp(lpd_i_n) + w_s exp(lpd_s_n) ]

    Parametrize w via free softmax over 3 logits to avoid constraint
    handling; SLSQP with simplex constraints is also fine but softmax is
    smoother for the BFGS path.

    Returns
    -------
    np.ndarray of shape (3,): (w_m, w_i, w_s)
    """
    lpd_m = np.asarray(lpd_m, dtype=np.float64)
    lpd_i = np.asarray(lpd_i, dtype=np.float64)
    lpd_s = np.asarray(lpd_s, dtype=np.float64)
    n = min(len(lpd_m), len(lpd_i), len(lpd_s))
    if n == 0:
        return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    lpd_m, lpd_i, lpd_s = lpd_m[:n], lpd_i[:n], lpd_s[:n]

    def neg_log_score(logits):
        z = logits - logits.max()
        w = np.exp(z)
        w = w / w.sum()
        # numerically stable per-obs log-mix
        a = lpd_m + np.log(max(w[0], 1e-15))
        b = lpd_i + np.log(max(w[1], 1e-15))
        c = lpd_s + np.log(max(w[2], 1e-15))
        m = np.maximum(np.maximum(a, b), c)
        log_mix = m + np.log(
            np.exp(a - m) + np.exp(b - m) + np.exp(c - m)
        )
        return -float(np.sum(log_mix))

    # Warm-start from equal weights; BFGS over R^3 softmax logits.
    res = minimize(
        neg_log_score, x0=np.zeros(3), method='BFGS',
        options={'maxiter': 200, 'gtol': 1e-7},
    )
    z = res.x - res.x.max()
    w = np.exp(z)
    w = w / w.sum()
    return w


class ThreeWayImputationModel:
    """Per-item three-way stacking of MICE + IRT-baseline + IRT-shared-disc.

    Parameters
    ----------
    irt_model : GRModel (baseline, per-item discriminations)
        Fitted with ``surrogate_sample`` and ``calibrated_expectations``.
    shared_disc_model : SharedDiscGRModel
        Fitted with the same interface; provides per-item LOO with the
        slope constrained to be shared across items.
    mice_model : PairwiseOrdinalStackingModel (or any MICE-style model
        with ``predict_pmf`` and ``variable_names``).
    data_factory : callable
        Returns iterator over data batches (same format as IRT training).
    uncertainty_penalty : float
        Penalty factor for MICE stacking uncertainty.
    """

    def __init__(
        self,
        irt_model,
        shared_disc_model,
        mice_model,
        data_factory,
        uncertainty_penalty: float = 1.0,
    ):
        self.irt_model = irt_model
        self.shared_disc_model = shared_disc_model
        self.mice_model = mice_model
        self.uncertainty_penalty = uncertainty_penalty
        self._data_factory = data_factory

        # Mirror the attributes IRTModel.validate_imputation_model checks
        self.variable_names: List[str] = list(mice_model.variable_names)
        self.variable_types: Dict[int, str] = dict(mice_model.variable_types)
        self.marginal_results = mice_model.marginal_results
        self.univariate_results = mice_model.univariate_results

        # Per-item LOO from each component
        print("Three-way stacking: IRT baseline per-item LOO...", flush=True)
        self._irt_elpd, self._irt_loo = _compute_grm_loo_per_item(
            irt_model, data_factory)
        if shared_disc_model is not None:
            print("Three-way stacking: shared-disc per-item LOO...", flush=True)
            self._shared_elpd, self._shared_loo = _compute_grm_loo_per_item(
                shared_disc_model, data_factory)
        else:
            print("Three-way stacking: shared-disc absent, falling back to 2-way stack",
                  flush=True)
            self._shared_elpd, self._shared_loo = {}, {}
        print("Three-way stacking: MICE per-item LOO...", flush=True)
        self._mice_elpd, self._mice_loo = self._compute_mice_loo()

        # Per-item simplex weights (w_m, w_i, w_s); w_s forced to 0 when shared absent
        print("Three-way stacking: solving per-item weights...", flush=True)
        self._weights: Dict[str, np.ndarray] = self._compute_weights()

        # PMF matrices for both GRM components
        self._irt_pmf_matrix = _precompute_grm_pmf_matrix(irt_model)
        if shared_disc_model is not None:
            self._shared_pmf_matrix = _precompute_grm_pmf_matrix(shared_disc_model)
        else:
            self._shared_pmf_matrix = None
        self._item_to_idx = {k: i for i, k in enumerate(irt_model.item_keys)}

    def _compute_mice_loo(self):
        """Per-item pointwise LOO from MICE (same as mixed.py)."""
        mice = self.mice_model
        irt = self.irt_model
        item_keys = irt.item_keys
        K = irt.response_cardinality

        all_responses = {k: [] for k in item_keys}
        seen = set()
        for batch in self._data_factory():
            pids = batch.get(irt.person_key)
            for row_idx in range(len(batch[item_keys[0]])):
                pid = int(pids[row_idx]) if pids is not None else row_idx
                if pid in seen:
                    continue
                seen.add(pid)
                for k in item_keys:
                    all_responses[k].append(float(batch[k][row_idx]))

        N = len(seen)
        responses = {k: np.array(all_responses[k]) for k in item_keys}

        elpd_per_item: Dict[str, float] = {}
        loo_per_obs: Dict[str, np.ndarray] = {}

        for item_key in item_keys:
            if item_key not in mice.variable_names:
                elpd_per_item[item_key] = -np.inf
                loo_per_obs[item_key] = np.array([])
                continue
            obs_mask = (~np.isnan(responses[item_key])
                        & (responses[item_key] >= 0)
                        & (responses[item_key] < K))
            if not np.any(obs_mask):
                elpd_per_item[item_key] = -np.inf
                loo_per_obs[item_key] = np.array([])
                continue
            log_scores = np.full(N, np.nan)
            for n in range(N):
                if not obs_mask[n]:
                    continue
                y_true = int(responses[item_key][n])
                observed_items = {}
                for k in item_keys:
                    if k == item_key:
                        continue
                    val = responses[k][n]
                    if not np.isnan(val) and val >= 0 and val < K:
                        observed_items[k] = float(val)
                try:
                    pmf = mice.predict_pmf(
                        observed_items, target=item_key, n_categories=K,
                    )
                    p = float(pmf[y_true])
                    log_scores[n] = np.log(max(p, 1e-30))
                except (ValueError, KeyError, AttributeError, TypeError) as exc:
                    _warn_fallback(
                        f"MICE predict_pmf failed for item '{item_key}' "
                        f"person {n}", exc)
                    log_scores[n] = np.log(1.0 / K)
            valid = log_scores[~np.isnan(log_scores)]
            elpd_per_item[item_key] = float(np.mean(valid))
            loo_per_obs[item_key] = valid

        return elpd_per_item, loo_per_obs

    def _compute_weights(self):
        """Per-item Yao stacking. 3-way (m, i, s) when shared-disc is present,
        2-way (m, i) with w_s=0 when shared-disc is absent."""
        item_keys = self.irt_model.item_keys
        has_shared = self.shared_disc_model is not None
        weights: Dict[str, np.ndarray] = {}
        for item_key in item_keys:
            lpd_m = self._mice_loo.get(item_key)
            lpd_i = self._irt_loo.get(item_key)
            lpd_s = self._shared_loo.get(item_key) if has_shared else None

            if has_shared:
                if (lpd_m is None or lpd_i is None or lpd_s is None
                        or len(lpd_m) == 0 or len(lpd_i) == 0
                        or len(lpd_s) == 0):
                    weights[item_key] = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
                    continue
                n = min(len(lpd_m), len(lpd_i), len(lpd_s))
                w = _solve_three_way_stacking(
                    lpd_m[:n], lpd_i[:n], lpd_s[:n])
                weights[item_key] = w
            else:
                if (lpd_m is None or lpd_i is None
                        or len(lpd_m) == 0 or len(lpd_i) == 0):
                    weights[item_key] = np.array([0.5, 0.5, 0.0])
                    continue
                weights[item_key] = _solve_two_way_stacking(lpd_m, lpd_i)
        return weights

    # ------------------------------------------------------------------
    # Public interface (matches IRTModel.imputation_model expectations)
    # ------------------------------------------------------------------

    def predict_pmf(
        self,
        items: Dict[str, float],
        target: str,
        n_categories: int,
        uncertainty_penalty: Optional[float] = None,
        person_idx: Optional[int] = None,
    ) -> np.ndarray:
        if uncertainty_penalty is None:
            uncertainty_penalty = self.uncertainty_penalty
        w = self._weights.get(
            target, np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]))
        w_m, w_i, w_s = float(w[0]), float(w[1]), float(w[2])

        # MICE
        try:
            mice_pmf = self.mice_model.predict_pmf(
                items, target, n_categories,
                uncertainty_penalty=uncertainty_penalty,
            )
        except (ValueError, KeyError, AttributeError) as exc:
            _warn_fallback(
                f"Three-way: MICE predict_pmf failed for '{target}'", exc)
            mice_pmf = np.ones(n_categories) / n_categories

        # GRM-baseline
        idx = self._item_to_idx.get(target)
        if idx is not None and self._irt_pmf_matrix is not None:
            if person_idx is not None and 0 <= person_idx < self._irt_pmf_matrix.shape[0]:
                irt_pmf = self._irt_pmf_matrix[person_idx, idx, :n_categories]
            else:
                irt_pmf = self._irt_pmf_matrix[:, idx, :n_categories].mean(axis=0)
        else:
            irt_pmf = np.ones(n_categories) / n_categories
        irt_pmf = irt_pmf / max(irt_pmf.sum(), 1e-15)

        # Shared-disc
        if idx is not None and self._shared_pmf_matrix is not None:
            if person_idx is not None and 0 <= person_idx < self._shared_pmf_matrix.shape[0]:
                shared_pmf = self._shared_pmf_matrix[person_idx, idx, :n_categories]
            else:
                shared_pmf = self._shared_pmf_matrix[:, idx, :n_categories].mean(axis=0)
        else:
            shared_pmf = np.ones(n_categories) / n_categories
        shared_pmf = shared_pmf / max(shared_pmf.sum(), 1e-15)

        blended = w_m * mice_pmf + w_i * irt_pmf + w_s * shared_pmf
        total = blended.sum()
        if total > 0:
            blended = blended / total
        else:
            blended = np.ones(n_categories) / n_categories
        return blended

    def predict(
        self,
        items: Dict[str, float],
        target: str,
        return_details: bool = False,
        uncertainty_penalty: Optional[float] = None,
    ):
        if uncertainty_penalty is None:
            uncertainty_penalty = self.uncertainty_penalty
        w = self._weights.get(
            target, np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]))
        K = self.irt_model.response_cardinality
        idx = self._item_to_idx.get(target)

        mice_details = self.mice_model.predict(
            items, target,
            return_details=True,
            uncertainty_penalty=uncertainty_penalty,
        )
        mice_pred = mice_details['prediction']

        def _grm_pop_pred(pmf_matrix):
            if idx is None or pmf_matrix is None:
                return (K - 1) / 2.0
            pmf = pmf_matrix[:, idx, :K].mean(axis=0)
            pmf = pmf / max(pmf.sum(), 1e-15)
            return float(np.dot(np.arange(K), pmf))

        irt_pred = _grm_pop_pred(self._irt_pmf_matrix)
        shared_pred = _grm_pop_pred(self._shared_pmf_matrix)

        blended = (float(w[0]) * mice_pred
                   + float(w[1]) * irt_pred
                   + float(w[2]) * shared_pred)

        if return_details:
            return {
                'prediction': blended,
                'mice_prediction': mice_pred,
                'irt_baseline_prediction': irt_pred,
                'shared_disc_prediction': shared_pred,
                'weights': {'mice': float(w[0]), 'irt': float(w[1]),
                            'shared': float(w[2])},
                'mice_details': mice_details,
                'irt_elpd_per_item': self._irt_elpd.get(target),
                'shared_elpd_per_item': self._shared_elpd.get(target),
                'mice_elpd_per_item': self._mice_elpd.get(target),
            }
        return blended

    def get_item_weight(self, target: str) -> np.ndarray:
        """Return the 3-vector (w_mice, w_irt, w_shared) for one item."""
        return self._weights.get(
            target, np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]))

    @property
    def weights(self) -> Dict[str, np.ndarray]:
        return dict(self._weights)

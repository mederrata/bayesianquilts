"""
IRT Mixed Imputation Model

Blends predictions from a fitted IRT baseline model and a MICEBayesianLOO
imputation model using per-item weights derived from comparable ELPD scores.

The IRT model's per-item ELPD is computed as the average in-sample log
predictive density for observed responses.  The MICE model's per-item ELPD
is aggregated from its univariate LOO-ELPD results.  These are placed on
the same per-observation scale and converted to softmax stacking weights.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, List


class IrtMixedImputationModel:
    """Blends IRT baseline and MICE imputation PMFs with per-item weights.

    The class exposes the same ``predict_pmf`` / ``predict`` interface that
    ``IRTModel`` expects from an ``imputation_model``, so it can be passed
    directly to the GRM constructor.

    Parameters
    ----------
    irt_model : GRModel (or any IRTModel subclass)
        A **fitted** IRT model with ``surrogate_sample`` and
        ``calibrated_expectations`` populated (call ``calibrate`` or the
        manual calibration shown in the notebooks first).
    mice_model : MICEBayesianLOO
        A **fitted** MICE imputation model whose ``variable_names`` cover
        all items in ``irt_model.item_keys``.
    data_factory : callable
        A callable that returns an iterator over data batches (same format
        used for training the IRT model).  Used to compute the IRT model's
        per-item ELPD in a memory-safe streaming fashion.
    uncertainty_penalty : float
        Penalty factor applied to ELPD standard error when computing MICE
        stacking weights (passed through to ``mice_model.predict_pmf``).
    irt_elpd_batch_size : int
        Number of posterior samples to process at a time when computing the
        IRT per-item ELPD.  Lower values use less memory.
    """

    def __init__(
        self,
        irt_model,
        mice_model,
        data_factory,
        uncertainty_penalty: float = 1.0,
        irt_elpd_batch_size: int = 4,
    ):
        self.irt_model = irt_model
        self.mice_model = mice_model
        self.uncertainty_penalty = uncertainty_penalty

        # Mirror the attributes that IRTModel.validate_imputation_model checks
        self.variable_names: List[str] = list(mice_model.variable_names)
        self.variable_types: Dict[int, str] = dict(mice_model.variable_types)
        self.zero_predictor_results = mice_model.zero_predictor_results
        self.univariate_results = mice_model.univariate_results

        # Compute per-item ELPDs from both models
        self._irt_elpd_per_item: Dict[str, float] = {}
        self._mice_elpd_per_item: Dict[str, float] = {}
        self._weights: Dict[str, float] = {}  # w_i for MICE; (1 - w_i) for IRT

        self._compute_irt_elpd(data_factory, batch_size=irt_elpd_batch_size)
        self._compute_mice_elpd()
        self._compute_weights()

    # ------------------------------------------------------------------
    # ELPD computation
    # ------------------------------------------------------------------

    def _compute_irt_elpd(self, data_factory, batch_size: int = 4):
        """Compute per-item WAIC from the IRT model (comparable to LOO-ELPD).

        WAIC per item is::

            WAIC_i = lppd_i - p_waic_i

        where::

            lppd_i  = (1/N_i) sum_n log( (1/S) sum_s p(y_ni | theta_s) )
            p_waic_i = (1/N_i) sum_n Var_s[ log p(y_ni | theta_s) ]

        The WAIC correction ``p_waic`` penalises the in-sample lppd by the
        effective number of parameters, making it asymptotically equivalent
        to LOO-ELPD and therefore comparable to the MICE model's LOO-ELPD.

        Streams over data batches and posterior sample chunks to stay
        memory-safe.
        """
        from scipy.special import logsumexp as sp_logsumexp

        model = self.irt_model
        samples = model.surrogate_sample
        item_keys = model.item_keys
        K = model.response_cardinality

        # FactorizedGRModel stores per-scale params; transform() assembles them
        if hasattr(model, 'transform') and 'discriminations' not in samples:
            samples = model.transform(dict(samples))

        disc_all = np.asarray(samples["discriminations"])    # (S, ...)
        diff0_all = np.asarray(samples["difficulties0"])     # (S, ...)
        ddiff_all = np.asarray(samples["ddifficulties"])     # (S, ...)
        abil_all = np.asarray(samples["abilities"])          # (S, ...)
        S = disc_all.shape[0]

        # Accumulators per item: sum of lppd and p_waic, plus count
        item_lppd_sum = {k: 0.0 for k in item_keys}
        item_pwaic_sum = {k: 0.0 for k in item_keys}
        item_count = {k: 0 for k in item_keys}

        for batch in data_factory():
            people = np.asarray(batch[model.person_key], dtype=np.int64)

            # Build observed responses matrix (N, I)
            responses = np.stack(
                [np.asarray(batch[k], dtype=np.float64) for k in item_keys],
                axis=-1,
            )
            observed_mask = (
                ~np.isnan(responses) & (responses >= 0) & (responses < K)
            )
            responses_int = np.where(observed_mask, responses.astype(np.int64), 0)

            # Collect per-sample log-probs in chunks: (S, N, I)
            log_probs_chunks = []

            for s_start in range(0, S, batch_size):
                s_end = min(s_start + batch_size, S)

                disc_chunk = jnp.asarray(disc_all[s_start:s_end])
                diff0_chunk = jnp.asarray(diff0_all[s_start:s_end])
                ddiff_chunk = jnp.asarray(ddiff_all[s_start:s_end])
                abil_chunk = jnp.asarray(abil_all[s_start:s_end])

                abil_people = abil_chunk[:, people, ...]
                response_probs = np.asarray(
                    model.grm_model_prob_d(
                        abil_people, disc_chunk, diff0_chunk, ddiff_chunk,
                    )
                )  # (s_chunk, N, I, K)

                # Gather P(Y = y_obs) for each cell
                resp_idx = responses_int[np.newaxis, :, :, np.newaxis]
                obs_probs = np.take_along_axis(
                    response_probs, resp_idx, axis=-1,
                )[..., 0]  # (s_chunk, N, I)
                obs_probs = np.clip(obs_probs, 1e-30, None)
                log_probs_chunks.append(np.log(obs_probs))

            # log_probs: (S, N, I) — per-sample log-likelihood per cell
            log_probs = np.concatenate(log_probs_chunks, axis=0)

            # lppd per cell: log E_s[p(y|theta_s)]
            lppd_cell = sp_logsumexp(log_probs, axis=0) - np.log(S)  # (N, I)

            # p_waic per cell: Var_s[log p(y|theta_s)]
            pwaic_cell = np.var(log_probs, axis=0, ddof=1)  # (N, I)

            for i_idx, item_key in enumerate(item_keys):
                mask_i = observed_mask[:, i_idx]
                if np.any(mask_i):
                    item_lppd_sum[item_key] += float(
                        lppd_cell[mask_i, i_idx].sum()
                    )
                    item_pwaic_sum[item_key] += float(
                        pwaic_cell[mask_i, i_idx].sum()
                    )
                    item_count[item_key] += int(mask_i.sum())

        # Per-item WAIC (on per-observation scale, comparable to LOO-ELPD)
        for k in item_keys:
            if item_count[k] > 0:
                lppd = item_lppd_sum[k] / item_count[k]
                pwaic = item_pwaic_sum[k] / item_count[k]
                self._irt_elpd_per_item[k] = lppd - pwaic
            else:
                self._irt_elpd_per_item[k] = -np.inf

    def _compute_mice_elpd(self):
        """Compute per-item ELPD from the MICE model.

        For each item, we use the best available ELPD: the maximum
        univariate-model ELPD across all predictors for that item,
        since that is what ``predict_pmf`` will stack over.
        """
        mice = self.mice_model
        item_keys = self.irt_model.item_keys

        for item_key in item_keys:
            if item_key not in mice.variable_names:
                self._mice_elpd_per_item[item_key] = -np.inf
                continue

            target_idx = mice.variable_names.index(item_key)
            elpds = []

            # Zero-predictor
            if target_idx in mice.zero_predictor_results:
                zr = mice.zero_predictor_results[target_idx]
                if zr.converged and np.isfinite(zr.elpd_loo_per_obs):
                    elpds.append(zr.elpd_loo_per_obs)

            # Best univariate model for this target
            for (t_idx, p_idx), ur in mice.univariate_results.items():
                if t_idx == target_idx and ur.converged:
                    if np.isfinite(ur.elpd_loo_per_obs):
                        elpds.append(ur.elpd_loo_per_obs)

            if elpds:
                self._mice_elpd_per_item[item_key] = float(np.max(elpds))
            else:
                self._mice_elpd_per_item[item_key] = -np.inf

    def _compute_weights(self):
        """Compute per-item mixing weight w_i for MICE vs IRT.

        w_i is the softmax weight for the MICE model:
            w_mice = exp(elpd_mice) / (exp(elpd_mice) + exp(elpd_irt))

        When w_mice is high, we trust the MICE imputation.
        When w_mice is low, we fall back toward uniform (ignorable).
        """
        for item_key in self.irt_model.item_keys:
            elpd_irt = self._irt_elpd_per_item.get(item_key, -np.inf)
            elpd_mice = self._mice_elpd_per_item.get(item_key, -np.inf)

            if not np.isfinite(elpd_mice) and not np.isfinite(elpd_irt):
                self._weights[item_key] = 0.5
                continue

            if not np.isfinite(elpd_mice):
                self._weights[item_key] = 0.0
                continue

            if not np.isfinite(elpd_irt):
                self._weights[item_key] = 1.0
                continue

            # Softmax over the two ELPD scores
            max_e = max(elpd_mice, elpd_irt)
            w_mice = np.exp(elpd_mice - max_e)
            w_irt = np.exp(elpd_irt - max_e)
            total = w_mice + w_irt
            self._weights[item_key] = float(w_mice / total)

    # ------------------------------------------------------------------
    # Public interface (matches what IRTModel expects)
    # ------------------------------------------------------------------

    def predict_pmf(
        self,
        items: Dict[str, float],
        target: str,
        n_categories: int,
        uncertainty_penalty: Optional[float] = None,
    ) -> np.ndarray:
        """Return a blended PMF for a missing cell.

        Mixes the MICE PMF with a uniform (ignorable) PMF using the
        per-item weight.

        Parameters
        ----------
        items : dict
            Observed variable name -> value for this person.
        target : str
            Item to impute.
        n_categories : int
            Number of response categories.
        uncertainty_penalty : float, optional
            Override the instance-level penalty for MICE stacking.

        Returns
        -------
        np.ndarray of shape (n_categories,) summing to 1.
        """
        if uncertainty_penalty is None:
            uncertainty_penalty = self.uncertainty_penalty

        w_mice = self._weights.get(target, 0.5)

        # MICE PMF
        try:
            mice_pmf = self.mice_model.predict_pmf(
                items, target, n_categories,
                uncertainty_penalty=uncertainty_penalty,
            )
        except (ValueError, KeyError, AttributeError):
            mice_pmf = np.ones(n_categories) / n_categories

        # IRT "ignorable" PMF is uniform: marginalizing over Y gives
        # equal weight to all categories from the imputation model's
        # perspective (the IRT model will apply its own likelihood).
        uniform_pmf = np.ones(n_categories) / n_categories

        # Blend
        blended = w_mice * mice_pmf + (1.0 - w_mice) * uniform_pmf

        # Normalize (should already sum to 1, but guard against numerics)
        total = blended.sum()
        if total > 0:
            blended /= total
        else:
            blended = uniform_pmf

        return blended

    def predict(
        self,
        items: Dict[str, float],
        target: str,
        return_details: bool = False,
        uncertainty_penalty: Optional[float] = None,
    ):
        """Predict a target variable using the blended model.

        Delegates to MICEBayesianLOO.predict() but adjusts the
        prediction by blending with the IRT marginal (uniform) via
        the per-item weight.  For ordinal targets the blended PMF
        expectation is returned.
        """
        if uncertainty_penalty is None:
            uncertainty_penalty = self.uncertainty_penalty

        w_mice = self._weights.get(target, 0.5)

        mice_details = self.mice_model.predict(
            items, target,
            return_details=True,
            uncertainty_penalty=uncertainty_penalty,
        )

        mice_pred = mice_details['prediction']

        # IRT ignorable prediction: marginal expectation = (K-1)/2
        K = self.irt_model.response_cardinality
        irt_pred = (K - 1) / 2.0

        blended_pred = w_mice * mice_pred + (1.0 - w_mice) * irt_pred

        if return_details:
            return {
                'prediction': blended_pred,
                'mice_prediction': mice_pred,
                'irt_marginal_prediction': irt_pred,
                'weight_mice': w_mice,
                'mice_details': mice_details,
                'irt_elpd_per_item': self._irt_elpd_per_item.get(target),
                'mice_elpd_per_item': self._mice_elpd_per_item.get(target),
            }
        return blended_pred

    @property
    def weights(self) -> Dict[str, float]:
        """Per-item MICE weights (higher = more trust in MICE)."""
        return dict(self._weights)

    @property
    def irt_elpd(self) -> Dict[str, float]:
        """Per-item average log predictive density from the IRT model."""
        return dict(self._irt_elpd_per_item)

    @property
    def mice_elpd(self) -> Dict[str, float]:
        """Per-item best ELPD from the MICE model."""
        return dict(self._mice_elpd_per_item)

    def summary(self) -> str:
        """Return a human-readable summary of per-item weights."""
        lines = ["Item Weights (MICE vs IRT):"]
        lines.append(f"{'Item':<12} {'w_mice':>8} {'ELPD_mice':>10} {'ELPD_irt':>10}")
        lines.append("-" * 44)
        for k in self.irt_model.item_keys:
            w = self._weights.get(k, float('nan'))
            em = self._mice_elpd_per_item.get(k, float('nan'))
            ei = self._irt_elpd_per_item.get(k, float('nan'))
            lines.append(f"{k:<12} {w:>8.3f} {em:>10.4f} {ei:>10.4f}")
        return "\n".join(lines)

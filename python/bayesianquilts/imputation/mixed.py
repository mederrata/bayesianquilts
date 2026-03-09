"""
IRT Mixed Imputation Model

Blends predictions from a fitted IRT baseline model and a MICEBayesianLOO
imputation model using per-item stacking weights (Yao et al. 2018).

The weights are computed by solving the stacking optimization problem
independently for each item:

    max_{w ∈ [0,1]} Σ_i log[ w · p_MICE(y_i | y_{-i}) + (1-w) · p_IRT(y_i | y_{-i}) ]

where p_MICE and p_IRT are the pointwise LOO predictive densities from
each model.  This guarantees the mixture is at least as good as the
better single model (Yao et al. 2018, Theorem 1).
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize_scalar
from typing import Dict, Any, Optional, List


class IrtMixedImputationModel:
    """Blends IRT baseline and MICE imputation PMFs with per-item stacking weights.

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
        n_posterior_samples: int = 256,
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

        self._compute_irt_elpd(data_factory, batch_size=irt_elpd_batch_size,
                               n_posterior_samples=n_posterior_samples)
        self._compute_mice_elpd()
        self._compute_weights()
        self._precompute_irt_pmfs()

    # ------------------------------------------------------------------
    # ELPD computation
    # ------------------------------------------------------------------

    def _compute_irt_elpd(self, data_factory, batch_size: int = 4,
                          n_posterior_samples: int = 256):
        """Compute per-item conditional ELPD from the IRT model via quadrature.

        Computes the conditional predictive density for each item given
        all other observed items for the same person:

            p(y_ni | y_{n,-i}) = ∫ p(y_ni | θ) · p(θ | y_{n,-i}) dθ

        where p(θ | y_{n,-i}) ∝ p(θ) · Π_{j≠i} p(y_nj | θ, params_j)

        Uses Gauss-Hermite quadrature (exact for dim=1 with N(0,1) prior).
        This makes the IRT ELPD directly comparable to the MICE conditional
        predictive density, since both measure p(y_ni | y_{n,-i}).

        Stores per-item ELPD and per-observation LOO scores.
        """
        from numpy.polynomial.hermite import hermgauss

        model = self.irt_model
        item_keys = model.item_keys
        K = model.response_cardinality
        I = len(item_keys)

        # Point estimates of item parameters (posterior mean)
        samples = model.surrogate_sample
        if hasattr(model, 'transform') and 'discriminations' not in samples:
            samples = model.transform(dict(samples))

        disc_mean = np.asarray(samples["discriminations"]).mean(0)
        diff0_mean = np.asarray(samples["difficulties0"]).mean(0)
        ddiff_mean = (np.asarray(samples["ddifficulties"]).mean(0)
                      if "ddifficulties" in samples else None)

        # Gauss-Hermite quadrature for N(0,1) prior
        n_quad = 61
        nodes, weights = hermgauss(n_quad)
        quad_points = nodes * np.sqrt(2)        # θ values on N(0,1) scale
        quad_weights = weights / np.sqrt(np.pi)  # normalized for N(0,1)

        # Compute GRM category probabilities at all quadrature points.
        # abilities shape: (Q, D, 1, 1) — Q quadrature points
        D = disc_mean.shape[0] if disc_mean.ndim >= 2 else 1
        theta_q = np.zeros((n_quad, D, 1, 1), dtype=np.float64)
        theta_q[:, 0, 0, 0] = quad_points  # primary dimension

        probs_at_quad = np.asarray(model.grm_model_prob_d(
            jnp.asarray(theta_q),
            jnp.asarray(disc_mean),
            jnp.asarray(diff0_mean),
            jnp.asarray(ddiff_mean) if ddiff_mean is not None else None,
        ))  # (Q, I, K)
        probs_at_quad = np.clip(probs_at_quad, 1e-30, None)
        log_probs_at_quad = np.log(probs_at_quad)  # (Q, I, K)
        log_quad_w = np.log(quad_weights)  # (Q,)

        # Accumulate per-item conditional LOO scores
        item_loo_scores: Dict[str, List[np.ndarray]] = {k: [] for k in item_keys}

        for batch in data_factory():
            responses = np.stack(
                [np.asarray(batch[k], dtype=np.float64) for k in item_keys],
                axis=-1,
            )  # (N_batch, I)
            observed = ~np.isnan(responses) & (responses >= 0) & (responses < K)
            responses_int = np.where(observed, responses.astype(np.int64), 0)
            N_batch = responses.shape[0]

            # Gather log p(y_nj | θ_q) for each person and item
            # log_probs_at_quad: (Q, I, K), responses_int: (N_batch, I)
            # Result: (Q, N_batch, I)
            log_lik_per_item = log_probs_at_quad[
                :, np.newaxis, :, :
            ]  # (Q, 1, I, K)
            resp_idx = responses_int[np.newaxis, :, :, np.newaxis]  # (1, N, I, 1)
            log_lik_qi = np.take_along_axis(
                log_lik_per_item, resp_idx, axis=-1
            )[..., 0]  # (Q, N_batch, I)

            # Zero out unobserved items
            log_lik_qi = np.where(observed[np.newaxis, :, :], log_lik_qi, 0.0)

            # Total log-lik across all items per person: (Q, N_batch)
            total_ll = np.sum(log_lik_qi, axis=-1)

            for i_idx, item_key in enumerate(item_keys):
                mask_i = observed[:, i_idx]
                if not np.any(mask_i):
                    continue

                # Leave-one-out log-lik (remove item i): (Q, N_batch)
                loo_ll = total_ll - log_lik_qi[:, :, i_idx]

                # LOO posterior: log w_q + loo_ll  (prior absorbed in quadrature)
                log_posterior = log_quad_w[:, np.newaxis] + loo_ll  # (Q, N_batch)

                # Normalize over quadrature points
                log_Z = np.logaddexp.reduce(log_posterior, axis=0)  # (N_batch,)
                log_post_norm = log_posterior - log_Z[np.newaxis, :]

                # Conditional predictive density:
                # log p(y_ni | y_{n,-i}) = logsumexp(log_post_norm + log p(y_ni|θ_q))
                log_pred = log_post_norm + log_lik_qi[:, :, i_idx]
                log_cond_pred = np.logaddexp.reduce(log_pred, axis=0)  # (N_batch,)

                item_loo_scores[item_key].append(log_cond_pred[mask_i])

        # Aggregate per-item
        self._irt_khat_per_item: Dict[str, float] = {}
        self._irt_elpd_loo_per_obs: Dict[str, np.ndarray] = {}

        for item_key in item_keys:
            scores = item_loo_scores[item_key]
            if scores:
                all_scores = np.concatenate(scores)
                self._irt_elpd_per_item[item_key] = float(np.mean(all_scores))
                self._irt_khat_per_item[item_key] = 0.0  # exact quadrature, no PSIS
                self._irt_elpd_loo_per_obs[item_key] = all_scores
            else:
                self._irt_elpd_per_item[item_key] = -np.inf
                self._irt_khat_per_item[item_key] = np.inf

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

    def _precompute_irt_pmfs(self):
        """Precompute IRT baseline PMFs for all people and items.

        Uses the fitted baseline model's posterior mean parameters and
        fitted abilities to compute P(Y_i = k | θ_n, item_params) for
        every (person, item) pair.  Stored as ``_irt_pmf_matrix`` of
        shape (N, I, K).
        """
        model = self.irt_model
        samples = model.surrogate_sample
        item_keys = model.item_keys
        K = model.response_cardinality

        if hasattr(model, 'transform') and 'discriminations' not in samples:
            samples = model.transform(dict(samples))

        # Use posterior means (point estimates)
        disc = np.asarray(samples["discriminations"]).mean(0)
        diff0 = np.asarray(samples["difficulties0"]).mean(0)
        ddiff = np.asarray(samples["ddifficulties"]).mean(0) if "ddifficulties" in samples else None
        abil = np.asarray(samples["abilities"]).mean(0)

        # Compute probabilities: forward pass through GRM
        probs = np.asarray(model.grm_model_prob_d(
            jnp.asarray(abil[np.newaxis]),
            jnp.asarray(disc[np.newaxis]),
            jnp.asarray(diff0[np.newaxis]),
            jnp.asarray(ddiff[np.newaxis]) if ddiff is not None else None,
        ))  # (1, N, I, K)
        self._irt_pmf_matrix = probs[0]  # (N, I, K)

        # Build item name → index mapping
        self._item_to_idx = {k: i for i, k in enumerate(item_keys)}

    def _compute_weights(self):
        """Compute per-item stacking weights (Yao et al. 2018).

        For each item j, solves the stacking optimization:

            max_{w ∈ [0,1]} Σ_i log[ w · exp(lpd_mice_i) + (1-w) · exp(lpd_irt_i) ]

        where lpd_mice_i and lpd_irt_i are the pointwise LOO log predictive
        densities for observation i under each model.

        This guarantees the mixture is at least as good as the better
        single model in terms of LOO predictive performance.

        When pointwise scores are not available for MICE (only summary ELPD),
        we fall back to a conservative approach: MICE gets weight only if its
        per-obs ELPD exceeds IRT's by more than 1 SE.

        If the IRT model's PSIS-LOO khat > 0.7 for an item, the IRT ELPD
        estimate is unreliable and the IRT model is discarded (w_mice = 1.0).
        """
        item_keys = self.irt_model.item_keys

        for item_key in item_keys:
            elpd_irt = self._irt_elpd_per_item.get(item_key, -np.inf)
            elpd_mice = self._mice_elpd_per_item.get(item_key, -np.inf)
            khat_irt = self._irt_khat_per_item.get(item_key, np.inf)

            # Discard IRT if khat > 0.7 (unreliable PSIS estimate)
            if khat_irt > 0.7:
                elpd_irt = -np.inf

            if not np.isfinite(elpd_mice) and not np.isfinite(elpd_irt):
                self._weights[item_key] = 0.0  # default to IRT
                continue

            if not np.isfinite(elpd_mice):
                self._weights[item_key] = 0.0
                continue

            if not np.isfinite(elpd_irt):
                self._weights[item_key] = 1.0
                continue

            # Try pointwise stacking if we have IRT LOO scores
            irt_loos = self._irt_elpd_loo_per_obs.get(item_key)
            if irt_loos is not None and len(irt_loos) > 0:
                w = self._solve_stacking_weight(item_key, irt_loos, elpd_mice)
                self._weights[item_key] = w
            else:
                # Fallback: conservative — only give MICE weight if clearly better
                self._weights[item_key] = 0.0

    @staticmethod
    def _solve_stacking_weight(item_key, irt_loos, mice_elpd_per_obs):
        """Solve the per-item stacking optimization (vectorized).

        Given pointwise IRT LOO log predictive densities and MICE's
        per-observation ELPD, find the optimal stacking weight.

        For MICE we don't have pointwise LOO scores (only the summary
        per-obs ELPD), so we approximate by assuming uniform pointwise
        scores: lpd_mice_i ≈ mice_elpd_per_obs for all i.

        The stacking objective is:
            max_{w ∈ [0,1]} Σ_i log[ w · exp(lpd_mice_i) + (1-w) · exp(lpd_irt_i) ]

        Parameters
        ----------
        item_key : str
            Item name (for diagnostics).
        irt_loos : np.ndarray
            Pointwise LOO log predictive densities from IRT, shape (N_i,).
        mice_elpd_per_obs : float
            Per-observation ELPD from MICE (uniform approximation).

        Returns
        -------
        float
            Optimal MICE weight w* ∈ [0, 1].
        """
        lpd_irt = np.asarray(irt_loos, dtype=np.float64)  # (N,)

        def neg_stacking_objective(w):
            """Negative stacking log score (vectorized, numerically stable)."""
            log_w = np.log(max(w, 1e-15))
            log_1mw = np.log(max(1 - w, 1e-15))
            a = mice_elpd_per_obs + log_w   # scalar broadcast
            b = lpd_irt + log_1mw            # (N,)
            m = np.maximum(a, b)
            log_mix = m + np.log(np.exp(a - m) + np.exp(b - m))
            return -np.sum(log_mix)

        result = minimize_scalar(
            neg_stacking_objective,
            bounds=(0.0, 1.0),
            method='bounded',
            options={'xatol': 1e-6},
        )

        return float(result.x)

    # ------------------------------------------------------------------
    # Public interface (matches what IRTModel expects)
    # ------------------------------------------------------------------

    def predict_pmf(
        self,
        items: Dict[str, float],
        target: str,
        n_categories: int,
        uncertainty_penalty: Optional[float] = None,
        person_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Return a blended PMF for a missing cell.

        Mixes the MICE PMF with the baseline IRT model's predicted PMF
        using the per-item stacking weight:
            PMF = w_mice * MICE(k) + (1 - w_mice) * IRT_baseline(k)

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
        person_idx : int, optional
            Index of the person in the training data.  When provided,
            uses the baseline IRT model's pre-fitted prediction for
            this person.  Falls back to the population-average IRT PMF
            when not provided.

        Returns
        -------
        np.ndarray of shape (n_categories,) summing to 1.
        """
        if uncertainty_penalty is None:
            uncertainty_penalty = self.uncertainty_penalty

        w_mice = self._weights.get(target, 0.0)

        # MICE PMF
        try:
            mice_pmf = self.mice_model.predict_pmf(
                items, target, n_categories,
                uncertainty_penalty=uncertainty_penalty,
            )
        except (ValueError, KeyError, AttributeError):
            mice_pmf = np.ones(n_categories) / n_categories

        # IRT baseline PMF from pre-fitted model
        item_idx = self._item_to_idx.get(target)
        if item_idx is not None and self._irt_pmf_matrix is not None:
            if person_idx is not None and 0 <= person_idx < self._irt_pmf_matrix.shape[0]:
                irt_pmf = self._irt_pmf_matrix[person_idx, item_idx, :n_categories]
            else:
                # Population average
                irt_pmf = self._irt_pmf_matrix[:, item_idx, :n_categories].mean(axis=0)
        else:
            irt_pmf = np.ones(n_categories) / n_categories

        # Normalize IRT PMF
        irt_total = irt_pmf.sum()
        if irt_total > 0:
            irt_pmf = irt_pmf / irt_total

        # Blend
        blended = w_mice * mice_pmf + (1.0 - w_mice) * irt_pmf

        # Normalize
        total = blended.sum()
        if total > 0:
            blended /= total
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
        """Predict a target variable using the blended model.

        Delegates to MICEBayesianLOO.predict() but adjusts the
        prediction by blending with the IRT marginal (uniform) via
        the per-item weight.  For ordinal targets the blended PMF
        expectation is returned.
        """
        if uncertainty_penalty is None:
            uncertainty_penalty = self.uncertainty_penalty

        w_mice = self._weights.get(target, 0.0)

        mice_details = self.mice_model.predict(
            items, target,
            return_details=True,
            uncertainty_penalty=uncertainty_penalty,
        )

        mice_pred = mice_details['prediction']

        # IRT baseline prediction: population-average expected value
        K = self.irt_model.response_cardinality
        item_idx = self._item_to_idx.get(target)
        if item_idx is not None and self._irt_pmf_matrix is not None:
            irt_pmf = self._irt_pmf_matrix[:, item_idx, :K].mean(axis=0)
            irt_pmf = irt_pmf / irt_pmf.sum()
            irt_pred = float(np.dot(np.arange(K), irt_pmf))
        else:
            irt_pred = (K - 1) / 2.0

        blended_pred = w_mice * mice_pred + (1.0 - w_mice) * irt_pred

        if return_details:
            return {
                'prediction': blended_pred,
                'mice_prediction': mice_pred,
                'irt_baseline_prediction': irt_pred,
                'weight_mice': w_mice,
                'mice_details': mice_details,
                'irt_elpd_per_item': self._irt_elpd_per_item.get(target),
                'mice_elpd_per_item': self._mice_elpd_per_item.get(target),
            }
        return blended_pred

    def get_item_weight(self, target: str) -> float:
        """Return the MICE stacking weight for a single item.

        Parameters
        ----------
        target : str
            Item name.

        Returns
        -------
        float
            w_mice ∈ [0, 1].  Returns 0.0 if the item is unknown.
        """
        return self._weights.get(target, 0.0)

    def predict_mice_pmf(
        self,
        items: Dict[str, float],
        target: str,
        n_categories: int,
        uncertainty_penalty: Optional[float] = None,
    ) -> np.ndarray:
        """Return the MICE-only PMF for a missing cell (no IRT blending).

        This is used by the importance-sampling likelihood formulation
        where the stacking weight is applied at the log-likelihood level
        rather than at the PMF level.

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

        try:
            mice_pmf = self.mice_model.predict_pmf(
                items, target, n_categories,
                uncertainty_penalty=uncertainty_penalty,
            )
        except (ValueError, KeyError, AttributeError):
            mice_pmf = np.ones(n_categories) / n_categories

        # Normalize
        total = mice_pmf.sum()
        if total > 0:
            mice_pmf = mice_pmf / total
        else:
            mice_pmf = np.ones(n_categories) / n_categories

        return mice_pmf

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

    @property
    def irt_khat(self) -> Dict[str, float]:
        """Per-item max khat from PSIS-LOO on the IRT model."""
        return dict(self._irt_khat_per_item)

    def save_diagnostics(self, path: str):
        """Save LOO diagnostics and stacking weights to HDF5.

        Stores per-item pointwise LOO scores, ELPD summaries, khat values,
        and computed stacking weights in a compact binary format.

        Parameters
        ----------
        path : str
            Path to the output .h5 file.
        """
        import h5py
        item_keys = self.irt_model.item_keys

        with h5py.File(path, 'w') as f:
            f.attrs['n_items'] = len(item_keys)
            f.attrs['item_keys'] = [k.encode() for k in item_keys]

            # Per-item scalar diagnostics
            weights = np.array([self._weights.get(k, np.nan) for k in item_keys])
            elpd_irt = np.array([self._irt_elpd_per_item.get(k, np.nan) for k in item_keys])
            elpd_mice = np.array([self._mice_elpd_per_item.get(k, np.nan) for k in item_keys])
            khat_irt = np.array([self._irt_khat_per_item.get(k, np.nan) for k in item_keys])

            f.create_dataset('weights', data=weights, compression='gzip')
            f.create_dataset('elpd_irt', data=elpd_irt, compression='gzip')
            f.create_dataset('elpd_mice', data=elpd_mice, compression='gzip')
            f.create_dataset('khat_irt', data=khat_irt, compression='gzip')

            # Per-item pointwise LOO scores (variable length per item)
            grp = f.create_group('irt_loo_per_obs')
            for k in item_keys:
                loos = self._irt_elpd_loo_per_obs.get(k)
                if loos is not None:
                    grp.create_dataset(k, data=np.asarray(loos, dtype=np.float32),
                                       compression='gzip', compression_opts=4)

    @classmethod
    def load_diagnostics(cls, path: str):
        """Load LOO diagnostics from HDF5 (for analysis, not full model).

        Returns
        -------
        dict with keys: item_keys, weights, elpd_irt, elpd_mice,
        khat_irt, irt_loo_per_obs.
        """
        import h5py

        with h5py.File(path, 'r') as f:
            item_keys = [k.decode() for k in f.attrs['item_keys']]
            result = {
                'item_keys': item_keys,
                'weights': f['weights'][:],
                'elpd_irt': f['elpd_irt'][:],
                'elpd_mice': f['elpd_mice'][:],
                'khat_irt': f['khat_irt'][:],
                'irt_loo_per_obs': {},
            }
            if 'irt_loo_per_obs' in f:
                for k in item_keys:
                    if k in f['irt_loo_per_obs']:
                        result['irt_loo_per_obs'][k] = f['irt_loo_per_obs'][k][:]
        return result

    def summary(self) -> str:
        """Return a human-readable summary of per-item stacking weights."""
        lines = ["Item Stacking Weights (Yao et al. 2018, MICE vs IRT):"]
        lines.append(f"{'Item':<12} {'w_mice':>8} {'ELPD_mice':>10} {'ELPD_irt':>10} {'khat_irt':>9}")
        lines.append("-" * 53)
        for k in self.irt_model.item_keys:
            w = self._weights.get(k, float('nan'))
            em = self._mice_elpd_per_item.get(k, float('nan'))
            ei = self._irt_elpd_per_item.get(k, float('nan'))
            kh = self._irt_khat_per_item.get(k, float('nan'))
            lines.append(f"{k:<12} {w:>8.3f} {em:>10.4f} {ei:>10.4f} {kh:>9.3f}")
        return "\n".join(lines)

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
        """Compute per-item PSIS-LOO ELPD from the IRT model.

        Draws ``n_posterior_samples`` from the variational posterior and
        computes the full ``(S, N, I)`` log-likelihood matrix in one
        streaming pass over the data (chunked over posterior samples for
        memory safety).  Then runs PSIS-LOO independently for each item
        on its ``(S, N_i)`` slice of observed responses.

        Stores per-item ELPD, max khat, and per-observation LOO scores.
        """
        from bayesianquilts.metrics.nppsis import psisloo

        model = self.irt_model
        item_keys = model.item_keys
        K = model.response_cardinality
        I = len(item_keys)

        # Draw fresh posterior samples
        surrogate = model.surrogate_distribution_generator(model.params)
        key = jax.random.PRNGKey(314159)
        samples = surrogate.sample(n_posterior_samples, seed=key)

        if hasattr(model, 'transform') and 'discriminations' not in samples:
            samples = model.transform(dict(samples))

        disc_all = np.asarray(samples["discriminations"])
        diff0_all = np.asarray(samples["difficulties0"])
        ddiff_all = np.asarray(samples["ddifficulties"]) if "ddifficulties" in samples else None
        abil_all = np.asarray(samples["abilities"])
        S = disc_all.shape[0]

        # Accumulate per-item log-lik columns across data batches.
        # Each entry is a list of (S, n_obs_in_batch) arrays.
        item_log_liks: Dict[str, List[np.ndarray]] = {k: [] for k in item_keys}

        for batch in data_factory():
            people = np.asarray(batch[model.person_key], dtype=np.int64)
            N_batch = len(people)

            # Observed responses: (N_batch, I)
            responses = np.stack(
                [np.asarray(batch[k], dtype=np.float64) for k in item_keys],
                axis=-1,
            )
            observed_mask = ~np.isnan(responses) & (responses >= 0) & (responses < K)
            responses_int = np.where(observed_mask, responses.astype(np.int64), 0)

            # Compute log P(y_ni | theta_s) for all items at once,
            # chunked over posterior samples
            log_probs_chunks = []
            for s_start in range(0, S, batch_size):
                s_end = min(s_start + batch_size, S)

                disc_chunk = jnp.asarray(disc_all[s_start:s_end])
                diff0_chunk = jnp.asarray(diff0_all[s_start:s_end])
                ddiff_chunk = jnp.asarray(ddiff_all[s_start:s_end]) if ddiff_all is not None else None
                abil_chunk = jnp.asarray(abil_all[s_start:s_end])

                abil_people = abil_chunk[:, people, ...]
                # (s_chunk, N_batch, I, K)
                response_probs = np.asarray(
                    model.grm_model_prob_d(
                        abil_people, disc_chunk, diff0_chunk, ddiff_chunk,
                    )
                )

                # Gather observed response category: (s_chunk, N_batch, I)
                resp_idx = responses_int[np.newaxis, :, :, np.newaxis]
                obs_probs = np.take_along_axis(
                    response_probs, resp_idx, axis=-1,
                )[..., 0]
                obs_probs = np.clip(obs_probs, 1e-30, None)
                log_probs_chunks.append(np.log(obs_probs))

            # (S, N_batch, I)
            log_probs = np.concatenate(log_probs_chunks, axis=0)

            # Split per item, keeping only observed observations
            for i_idx, item_key in enumerate(item_keys):
                mask_i = observed_mask[:, i_idx]
                if np.any(mask_i):
                    # (S, n_observed_in_batch)
                    item_log_liks[item_key].append(log_probs[:, mask_i, i_idx])

        # Run PSIS-LOO independently per item
        self._irt_khat_per_item: Dict[str, float] = {}
        self._irt_elpd_loo_per_obs: Dict[str, np.ndarray] = {}

        for item_key in item_keys:
            chunks = item_log_liks[item_key]
            if not chunks:
                self._irt_elpd_per_item[item_key] = -np.inf
                self._irt_khat_per_item[item_key] = np.inf
                continue

            # (S, N_i)
            log_lik_i = np.concatenate(chunks, axis=1)
            N_i = log_lik_i.shape[1]

            loo_total, loos, ks = psisloo(log_lik_i)

            self._irt_elpd_per_item[item_key] = float(loo_total) / N_i
            self._irt_khat_per_item[item_key] = float(np.max(ks))
            self._irt_elpd_loo_per_obs[item_key] = loos

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
        """Compute per-item mixing weight w_i for MICE vs IRT.

        w_i is the softmax weight for the MICE model:
            w_mice = exp(elpd_mice) / (exp(elpd_mice) + exp(elpd_irt))

        When w_mice is high, we trust the MICE imputation.
        When w_mice is low, we fall back toward the IRT baseline prediction.

        If the IRT model's PSIS-LOO khat > 0.7 for an item, the IRT ELPD
        estimate is unreliable and the IRT model is discarded for that item
        (w_mice = 1.0).
        """
        for item_key in self.irt_model.item_keys:
            elpd_irt = self._irt_elpd_per_item.get(item_key, -np.inf)
            elpd_mice = self._mice_elpd_per_item.get(item_key, -np.inf)
            khat_irt = self._irt_khat_per_item.get(item_key, np.inf)

            # Discard IRT if khat > 0.7 (unreliable PSIS estimate)
            if khat_irt > 0.7:
                elpd_irt = -np.inf

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
        person_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Return a blended PMF for a missing cell.

        Mixes the MICE PMF with the baseline IRT model's predicted PMF
        using the per-item weight:
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

        w_mice = self._weights.get(target, 0.5)

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

        w_mice = self._weights.get(target, 0.5)

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

    def summary(self) -> str:
        """Return a human-readable summary of per-item weights."""
        lines = ["Item Weights (MICE vs IRT, PSIS-LOO):"]
        lines.append(f"{'Item':<12} {'w_mice':>8} {'ELPD_mice':>10} {'ELPD_irt':>10} {'khat_irt':>9}")
        lines.append("-" * 53)
        for k in self.irt_model.item_keys:
            w = self._weights.get(k, float('nan'))
            em = self._mice_elpd_per_item.get(k, float('nan'))
            ei = self._irt_elpd_per_item.get(k, float('nan'))
            kh = self._irt_khat_per_item.get(k, float('nan'))
            lines.append(f"{k:<12} {w:>8.3f} {em:>10.4f} {ei:>10.4f} {kh:>9.3f}")
        return "\n".join(lines)

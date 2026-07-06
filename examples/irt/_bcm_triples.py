"""Shared helpers for building BCM training triples from a fitted GRModel.

Both ``fit_bcm_with_imputation.py`` and ``fit_marginal_irt.py`` need to
produce ``(subset_score, item_indicator_vector, gold_score)`` triples by
repeatedly masking random item subsets to "missing" and re-scoring each
respondent under the imputation-blended likelihood. Those helpers live
here so the two pipelines stay in sync.

All scoring assumes the model is already standardised to N(0,1) abilities
(joint-ADVI: ``model.standardize_abilities()``; marginal-MCMC:
``model.standardize_marginal(data)``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np


def attach_imputation_pmfs(model, data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute imputation PMFs for the (possibly masked) data and attach
    them under ``_imputation_pmfs`` / ``_imputation_weights`` so
    ``compute_eap_abilities`` integrates over them for missing items."""
    out = dict(data)
    pmfs, weights = model._compute_batch_pmfs(out)
    if pmfs is not None:
        out['_imputation_pmfs'] = pmfs
        if weights is not None:
            out['_imputation_weights'] = weights
    return out


def score_subset(
    model,
    base_data: Dict[str, Any],
    item_keys: Sequence[str],
    subset: Set[str],
    item_params: Dict[str, Any],
) -> np.ndarray:
    """Return per-respondent EAP scores for an item subset.

    Items not in ``subset`` are masked to -1 (missing) so that the
    imputation model attached to ``model`` fills them in via Rao-Blackwellised
    PMFs. ``item_params`` (e.g. from ``extract_item_params``) is passed
    explicitly to avoid relying on ``compute_eap_abilities``'s fallback
    that requires ``mcmc_samples``.
    """
    masked: Dict[str, Any] = {}
    for k in item_keys:
        arr = np.asarray(base_data[k], dtype=np.float32).copy()
        if k not in subset:
            arr[:] = -1.0
        masked[k] = arr
    if 'person' in base_data:
        masked['person'] = base_data['person']
    masked = attach_imputation_pmfs(model, masked)
    eap = model.compute_eap_abilities(masked, item_params=item_params)
    return np.asarray(eap['eap'])


def extract_item_params(model) -> Dict[str, Any]:
    """Pull mean item parameters from the surrogate (joint-ADVI path).

    ``_item_var_list()`` excludes ``abilities`` and ``mu`` (the latter is
    a location-shift that ``compute_eap_abilities`` absorbs via the data,
    not a feature of the GRM kernel itself).
    """
    keys = model._item_var_list()
    return {
        k: model.calibrated_expectations[k]
        for k in keys
        if k in model.calibrated_expectations
    }


def extract_item_params_from_mcmc(model) -> Dict[str, Any]:
    """Pull mean item parameters from MCMC samples (marginal-MCMC path).

    Averages over chains and samples. Matches the convention
    ``compute_eap_abilities`` uses internally when no ``item_params`` is
    passed and ``mcmc_samples`` is available -- so this is mostly useful
    when we want the same item params used for several subset evaluations
    without re-averaging each call.
    """
    import jax.numpy as jnp
    item_keys = set(model._item_var_list())
    out: Dict[str, Any] = {}
    for k, v in model.mcmc_samples.items():
        if k not in item_keys:
            continue
        arr = np.asarray(v)
        out[k] = jnp.asarray(arr.reshape(-1, *arr.shape[2:]).mean(axis=0))
    return out


def stratify_respondents(
    base_data: Dict[str, Any],
    item_keys: Sequence[str],
    max_respondents: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sum-score-stratified subsample of respondent indices."""
    n = int(base_data['person'].shape[0])
    if n <= max_respondents:
        return np.arange(n)
    sum_score = np.zeros(n, dtype=np.float64)
    for k in item_keys:
        arr = np.asarray(base_data[k], dtype=np.float64)
        sum_score += np.where(arr >= 0, arr, 0.0)
    sum_score += rng.normal(0, 1e-9, sum_score.shape)
    sort_order = np.argsort(sum_score)
    pick = np.linspace(0, n - 1, max_respondents).astype(int)
    return sort_order[pick]


def subsample_data(
    base_data: Dict[str, Any],
    item_keys: Sequence[str],
    indices: np.ndarray,
) -> Dict[str, Any]:
    """Return a new data dict with only the selected respondents."""
    sub: Dict[str, Any] = {}
    for k in item_keys:
        sub[k] = np.asarray(base_data[k])[indices]
    sub['person'] = np.arange(len(indices), dtype=np.float32)
    if 'sample_weights' in base_data:
        sub['sample_weights'] = np.asarray(base_data['sample_weights'])[indices]
    return sub


def build_bcm_triples(
    model,
    base_data: Dict[str, Any],
    item_keys: Sequence[str],
    subset_sizes: Sequence[int],
    n_subsets_per_size: int,
    max_respondents: int,
    rng: np.random.Generator,
    item_params: Dict[str, Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Produce (subset_scores, indicators, gold_scores, chosen_indices).

    ``subset_scores`` and ``gold_scores`` are 1D arrays of length
    ``n_triples = sum_J min(n_subsets_per_size, C(I,J)) * n_chosen``.
    ``indicators`` is shape (n_triples, I) with 0/1 entries marking which
    items each row's subset includes.

    If ``item_params`` is None, extracts from the model (joint-ADVI path).
    """
    import gc
    from math import comb

    if item_params is None:
        item_params = extract_item_params(model)

    chosen = stratify_respondents(base_data, item_keys, max_respondents, rng)
    sub_data = subsample_data(base_data, item_keys, chosen)
    n_sub = len(chosen)
    print(f"  stratified respondents: {n_sub}")

    print("  computing gold scores (full bank + imputation)...")
    gold = score_subset(model, sub_data, item_keys, set(item_keys), item_params)
    print(f"    gold range: [{gold.min():.3f}, {gold.max():.3f}]")

    I = len(item_keys)
    # Always anchor the upper boundary of test length so the correction learns
    # to vanish at (near-)full administration. In addition to the caller's
    # requested sizes, include the full battery (J=I) and a few near-full sizes:
    # without a fully-administered row in training, the all-indicators-on region
    # is pure extrapolation, so neither the point correction nor the
    # sqrt(err^2 + bias^2) uncertainty term (which folds in the estimated bias)
    # goes to zero as the instrument is completed. comb(I, I) == 1, so the full
    # battery contributes a single anchoring draw (subset_score == gold,
    # indicators all 1); out-of-range sizes are dropped.
    anchored = {int(s) for s in subset_sizes if 1 <= int(s) <= I}
    anchored.update(s for s in (I, I - 1, I - 2, I - 3) if s >= 1)
    subset_sizes = sorted(anchored)

    key_to_idx = {k: i for i, k in enumerate(item_keys)}
    subset_scores: List[float] = []
    indicators: List[np.ndarray] = []
    golds: List[float] = []

    for size in subset_sizes:
        n_draws = min(n_subsets_per_size, comb(I, size))
        print(f"    J={size}: {n_draws} draws")
        for _ in range(n_draws):
            subset = rng.choice(item_keys, size=size, replace=False)
            subset_set = set(subset)
            scores = score_subset(
                model, sub_data, item_keys, subset_set, item_params)
            indicator = np.zeros(I, dtype=np.float32)
            for k in subset:
                indicator[key_to_idx[k]] = 1.0
            for pi in range(n_sub):
                subset_scores.append(scores[pi])
                indicators.append(indicator)
                golds.append(gold[pi])
        gc.collect()

    return (
        np.asarray(subset_scores, dtype=np.float64),
        np.stack(indicators, axis=0).astype(np.float64),
        np.asarray(golds, dtype=np.float64),
        chosen,
    )

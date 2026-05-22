"""Export the fitted pipeline artifacts as a gofluttercat-loadable bundle.

This module is shared by ``fit_bcm_with_imputation.py`` and
``fit_marginal_irt.py``. Both scripts call :func:`export_bundle` after
fitting (and after standardizing abilities to N(0,1)) to produce::

    <bundle_root>/
        items/<item_key>.json          # per-item GRM params + metadata
        imputation/config.yaml.gz      # v2.0 pairwise-stacking artifact
        bcm_<scale>.json               # per-J isotonic BCMSet
        manifest.yaml                  # provenance

The per-item JSON shape matches what gofluttercat already ships under
``backend-golang/<scale>/factorized/`` and what
``libfabulouscatpy.irt.converged.load_artifact`` reads. The imputation
YAML matches what ``gofluttercat/python/convert_pairwise.py`` writes via
its v1.1 -> v2.0 conversion; we inline that conversion so the example is
self-contained. The BCMSet is fit with ``libfabulouscatpy.biascorrection.
fit_bcm_set`` (per-J isotonic regression), which is the only BCM format
the Go runtime currently consumes.

Standardisation to N(0,1) abilities must happen BEFORE calling
:func:`export_bundle`; this module assumes ``model.calibrated_expectations``
(for joint-ADVI) or ``model.mcmc_samples`` (for marginal-MCMC) already
contain the standardised item parameters.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# GRM parameter extraction (joint-ADVI or marginal-MCMC)
# ---------------------------------------------------------------------------


def extract_grm_params(model) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(discriminations, cumulative_cutpoints)`` from a fitted GRModel.

    Prefers ``model.mcmc_samples`` when populated (marginal-MCMC path);
    otherwise falls back to ``model.calibrated_expectations`` (joint-ADVI
    path). Either source is assumed to be already standardised so the
    implied abilities are N(0,1).

    Returns:
        discriminations: shape (I,) of softplus-applied positive slopes.
        cutpoints: shape (I, K-1) of cumulative ordinal thresholds. K is
            inferred from ``ddifficulties``.
    """
    src = None
    if getattr(model, 'mcmc_samples', None):
        src = {k: np.array(v) for k, v in model.mcmc_samples.items()}
        # MCMC samples: (chains, samples, ...). Mean over chains+samples.
        for k, v in src.items():
            if v.ndim >= 2:
                src[k] = v.reshape(-1, *v.shape[2:]).mean(axis=0)
    elif getattr(model, 'calibrated_expectations', None) is not None:
        src = {k: np.array(v) for k, v in model.calibrated_expectations.items()}
    else:
        raise ValueError(
            "Model has neither mcmc_samples nor calibrated_expectations; "
            "fit the model and call calibrate_model or standardize_marginal first."
        )

    disc = np.squeeze(src['discriminations'])
    diff0 = np.squeeze(src['difficulties0'])
    ddiff = np.squeeze(src.get('ddifficulties', np.zeros((disc.shape[0], 0))))

    # disc shape: (I,)  --  diff0 shape: (I,)  --  ddiff shape: (I, K-2)
    if disc.ndim != 1:
        disc = disc.reshape(-1)
    if diff0.ndim != 1:
        diff0 = diff0.reshape(-1)
    if ddiff.ndim == 1:
        ddiff = ddiff[:, None] if disc.shape[0] > 1 else ddiff[None, :]
    if ddiff.ndim == 0 or ddiff.size == 0:
        ddiff = np.zeros((disc.shape[0], 0))

    # Cumulative cutpoints: [diff0, diff0+ddiff[:,0], diff0+ddiff[:,0]+ddiff[:,1], ...]
    cutpoints = np.zeros((disc.shape[0], 1 + ddiff.shape[1]), dtype=float)
    cutpoints[:, 0] = diff0
    for k in range(ddiff.shape[1]):
        cutpoints[:, k + 1] = cutpoints[:, k] + ddiff[:, k]

    return disc.astype(float), cutpoints


# ---------------------------------------------------------------------------
# Per-item JSON (gofluttercat factorized shape)
# ---------------------------------------------------------------------------


def _default_responses(n_categories: int) -> Dict[str, Dict[str, Any]]:
    """Stringified integer labels when no question/response metadata is given."""
    return {
        str(i): {"text": str(i), "value": i} for i in range(n_categories)
    }


def write_per_item_json(
    bundle_root: Path,
    item_keys: Sequence[str],
    discriminations: np.ndarray,
    cutpoints: np.ndarray,
    scale_name: str,
    item_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> int:
    """Write one JSON file per item under ``bundle_root/items/``.

    Each file follows the gofluttercat per-item shape::

        {"item": ..., "question": ..., "responses": {...},
         "scales": {<scale>: {"discrimination": ..., "difficulties": [...]}}}

    ``item_metadata[key]`` may supply ``question`` and ``responses``;
    otherwise placeholder values keyed by integer string are used.
    """
    items_dir = bundle_root / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    n_categories = cutpoints.shape[1] + 1
    written = 0
    for i, key in enumerate(item_keys):
        meta = (item_metadata or {}).get(key, {})
        payload = {
            "item": key,
            "question": meta.get("question", key),
            "responses": meta.get("responses", _default_responses(n_categories)),
            "scales": {
                scale_name: {
                    "discrimination": float(discriminations[i]),
                    "difficulties": [float(x) for x in cutpoints[i]],
                }
            },
        }
        (items_dir / f"{key}.json").write_text(json.dumps(payload, indent=2))
        written += 1
    return written


# ---------------------------------------------------------------------------
# Imputation v2.0 YAML (mirrors gofluttercat/python/convert_pairwise.py)
# ---------------------------------------------------------------------------


def _wrap_intercept(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def _convert_univariate(entry: Mapping[str, Any]) -> Dict[str, Any]:
    result = entry.get("result", {})
    out: Dict[str, Any] = {
        "target_idx": entry.get("target_idx", result.get("target_idx")),
        "predictor_idx": entry.get("predictor_idx", result.get("predictor_idx")),
        "n_obs": result.get("n_obs", 0),
        "elpd_loo": result.get("elpd_loo", 0.0),
        "elpd_loo_per_obs": result.get("elpd_loo_per_obs", 0.0),
        "elpd_loo_per_obs_se": result.get("elpd_loo_per_obs_se", 0.0),
        "khat_max": result.get("khat_max", 0.0),
        "khat_mean": result.get("khat_mean", 0.0),
        "converged": result.get("converged", False),
        "predictor_mean": result.get("predictor_mean", 0.0) or 0.0,
        "predictor_std": result.get("predictor_std", 1.0) or 1.0,
    }
    if result.get("beta_mean") is not None:
        out["beta_mean"] = result["beta_mean"]
    out["intercept_mean"] = _wrap_intercept(result.get("intercept_mean"))
    if result.get("cutpoints_mean") is not None:
        out["cutpoints_mean"] = result["cutpoints_mean"]
    return out


def _convert_marginal(entry: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "target_idx": entry.get("target_idx", 0),
        "n_obs": entry.get("n_obs", 0),
        "elpd_loo": entry.get("elpd_loo", 0.0),
        "elpd_loo_per_obs": entry.get("elpd_loo_per_obs", 0.0),
        "elpd_loo_per_obs_se": entry.get("elpd_loo_per_obs_se", 0.0),
        "khat_max": entry.get("khat_max", 0.0),
        "khat_mean": entry.get("khat_mean", 0.0),
        "converged": entry.get("converged", False),
    }
    if entry.get("beta_mean") is not None:
        out["beta_mean"] = entry["beta_mean"]
    out["intercept_mean"] = _wrap_intercept(entry.get("intercept_mean"))
    if entry.get("cutpoints_mean") is not None:
        out["cutpoints_mean"] = entry["cutpoints_mean"]
    return out


def write_imputation_v2_yaml(
    bundle_root: Path,
    stacking_yaml_path: Path,
    mixed_weights: Optional[Mapping[str, float]] = None,
) -> Path:
    """Convert a v1.0 PairwiseOrdinalStackingModel YAML to gofluttercat's
    v2.0 gzipped format and write it under ``bundle_root/imputation/``."""
    import yaml
    with open(stacking_yaml_path) as f:
        # unsafe_load is required because PairwiseOrdinalStackingModel.save
        # emits numpy-tagged YAML; this file is locally produced and trusted.
        src = yaml.unsafe_load(f)

    out: Dict[str, Any] = {"version": "2.0"}
    out["config"] = src.get("config", {})
    out["data"] = src.get("data", {})
    out["prediction_graph"] = src.get("prediction_graph", {})

    marg = src.get("marginal_results", {})
    out["marginal_meta"] = {str(k): _convert_marginal(v) for k, v in marg.items()}
    out["univariate_meta"] = [
        _convert_univariate(e) for e in src.get("univariate_results", [])
    ]
    if mixed_weights:
        out["mixed_weights"] = {str(k): float(v) for k, v in mixed_weights.items()}

    imp_dir = bundle_root / "imputation"
    imp_dir.mkdir(parents=True, exist_ok=True)
    dest = imp_dir / "config.yaml.gz"
    raw = yaml.dump(out, default_flow_style=False, allow_unicode=True).encode("utf-8")
    with gzip.open(dest, "wb") as f:
        f.write(raw)
    return dest


# ---------------------------------------------------------------------------
# BCMSet (per-J isotonic) for the Go runtime
# ---------------------------------------------------------------------------


def fit_and_save_bcm_set(
    bundle_root: Path,
    subset_scores: np.ndarray,
    indicators: np.ndarray,
    gold_scores: np.ndarray,
    scale_name: str,
) -> Path:
    """Group ``(subset_score, gold_score)`` rows by ``J = sum(indicator)`` and
    fit one isotonic BCM per J via ``libfabulouscatpy.biascorrection.fit_bcm_set``.
    Saves under ``bundle_root/bcm_<scale>.json``.
    """
    from libfabulouscatpy.biascorrection import fit_bcm_set

    js = indicators.sum(axis=1).astype(int)
    cells: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for j in np.unique(js):
        if j < 1:
            continue
        mask = js == j
        if mask.sum() < 2:
            continue
        cells[int(j)] = (subset_scores[mask], gold_scores[mask])
    bcm_set = fit_bcm_set(cells, scale=scale_name)
    dest = bundle_root / f"bcm_{scale_name}.json"
    bcm_set.save(str(dest))
    return dest


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def write_manifest(bundle_root: Path, **fields: Any) -> Path:
    import yaml
    manifest = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "bayesianquilts_pipeline": "examples/irt",
    }
    manifest.update({k: v for k, v in fields.items() if v is not None})
    dest = bundle_root / "manifest.yaml"
    with open(dest, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    return dest


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------


def export_bundle(
    bundle_root,
    *,
    model,
    item_keys: Sequence[str],
    scale_name: str,
    stacking_yaml_path: Optional[Path] = None,
    subset_scores: Optional[np.ndarray] = None,
    indicators: Optional[np.ndarray] = None,
    gold_scores: Optional[np.ndarray] = None,
    mixed_weights: Optional[Mapping[str, float]] = None,
    item_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    manifest_fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    """End-to-end bundle writer. Returns a dict of paths actually written.

    The IRT model must already be standardised (joint-ADVI: call
    ``model.standardize_abilities()``; marginal-MCMC: call
    ``model.standardize_marginal(data)``) before invoking this function.
    Optional inputs gate optional outputs: omit ``stacking_yaml_path`` to
    skip the imputation export; omit the BCM triples to skip the BCM
    export.
    """
    bundle_root = Path(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    disc, cutpoints = extract_grm_params(model)
    n = write_per_item_json(bundle_root, item_keys, disc, cutpoints,
                            scale_name, item_metadata=item_metadata)
    written["items"] = bundle_root / "items"
    print(f"  gofluttercat: wrote {n} item JSONs to {written['items']}")

    if stacking_yaml_path is not None and Path(stacking_yaml_path).exists():
        written["imputation"] = write_imputation_v2_yaml(
            bundle_root, Path(stacking_yaml_path), mixed_weights=mixed_weights)
        print(f"  gofluttercat: wrote imputation v2.0 YAML to {written['imputation']}")

    if (subset_scores is not None and indicators is not None
            and gold_scores is not None):
        written["bcm"] = fit_and_save_bcm_set(
            bundle_root,
            np.asarray(subset_scores, dtype=float),
            np.asarray(indicators, dtype=float),
            np.asarray(gold_scores, dtype=float),
            scale_name,
        )
        print(f"  gofluttercat: wrote per-J BCMSet to {written['bcm']}")

    written["manifest"] = write_manifest(
        bundle_root, scale=scale_name, **(manifest_fields or {}))
    print(f"  gofluttercat: wrote {written['manifest']}")
    return written

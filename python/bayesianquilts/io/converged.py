"""Write a fitted GRModel + imputation model to the artifact layout that
both libfabulouscatpy and the gofluttercat Go runtime consume.

Layout
------
``<out_dir>/``
    ``items/<item_key>.json``  -- one per item, with the union of keys
        expected by libfab's ``ItemDatabase`` and gofluttercat's
        ``irtcat.LoadItem``: ``item``, ``question``, ``responses``,
        ``scored_vales`` (sic), ``scales``. Each scale entry stores
        ``discrimination`` (scalar) and ``difficulties`` (list of K-1
        cutpoints on the libfab parameterization: cumulative thresholds,
        not gaps).
    ``scales.json``  -- libfab ``ScaleDatabase`` payload: one entry per
        scale with metadata.
    ``imputation/`` -- the imputation model's ``save_to_disk`` output
        (gofluttercat's ``imputation.LoadFromDisk`` reads this directly).
        Skipped when ``imputation_model`` is ``None``.
    ``manifest.yaml`` -- provenance: source script, timestamp, fit
        method (IS/marginal/weighted/factorized), ``share_discriminations``,
        list of item keys, scale names, response cardinality.

Parameterization mapping (bayesianquilts -> libfab/gofluttercat)
----------------------------------------------------------------
bayesianquilts stores discriminations and (difficulties0, ddifficulties)
where the K-1 cutpoints are ``cumsum(concat([d0, ddiff]))``.  libfab's
``GradedResponseModel`` and gofluttercat's ``irtcat.Calibration`` both
take cumulative cutpoints directly as ``difficulties``.  This module
performs the cumsum so consumers do not need to.

The shared-discriminations case is handled transparently: when the disc
tensor has a singleton item axis we broadcast to ``num_items``.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

try:
    import yaml  # PyYAML
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "bayesianquilts.io.converged needs PyYAML; "
        "install with `pip install pyyaml`."
    ) from exc


def _jsonify(value):
    """Recursively coerce numpy scalars/arrays to native Python types so
    that ``yaml.safe_dump`` / ``json.dumps`` can serialise the result."""
    if isinstance(value, dict):
        return {str(k) if not isinstance(k, str) else k: _jsonify(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.str_, bytes)):
        return str(value)
    return value


PathLike = Union[str, os.PathLike]


def grm_to_items(
    irt_model,
    scale_names: Optional[Sequence[str]] = None,
    item_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> tuple[list[dict], dict]:
    """Convert a fitted unidimensional or factorized GRModel to the
    libfab/gofluttercat item-JSON shape.

    Returns ``(items, scales)`` where ``items`` is a list of per-item
    dicts (one entry per item key) and ``scales`` is the
    ScaleDatabase-style mapping ``scale_name -> {description}``.

    Parameters
    ----------
    irt_model
        Fitted ``GRModel`` or ``FactorizedGRModel``. Must have
        ``calibrated_expectations`` populated (call
        ``set_calibration_expectations()`` after fitting).
    scale_names
        Names for each dimension, in order.  Defaults to
        ``["dim_<d>"]``.  For ``FactorizedGRModel`` you can pass the
        scale labels you used to build ``scale_indices``.
    item_metadata
        Optional mapping ``item_key -> {question, responses, scored_vales,
        ...}`` -- extra fields copied verbatim into each item dict.
        Anything not provided is filled in with sensible defaults.
    """
    if irt_model.calibrated_expectations is None:
        raise ValueError(
            "irt_model.calibrated_expectations is None — "
            "call set_calibration_expectations() after fitting first"
        )

    ce = irt_model.calibrated_expectations
    if "discriminations" not in ce:
        raise ValueError(
            "calibrated_expectations missing 'discriminations'; "
            "is this really a fitted GRM-style model?"
        )

    # discriminations: (1, D, I_or_1, 1)
    disc = np.asarray(ce["discriminations"])
    diff0 = np.asarray(ce["difficulties0"])  # (1, D, I, 1)
    ddiff = ce.get("ddifficulties")
    if ddiff is not None and np.asarray(ddiff).size > 0:
        ddiff = np.asarray(ddiff)  # (1, D, I, K-2)
    else:
        ddiff = None

    # Squeeze leading sample axis if it survived from upstream code.
    while disc.ndim > 4 and disc.shape[0] == 1:
        disc = disc[0]
        diff0 = diff0[0]
        if ddiff is not None:
            ddiff = ddiff[0]

    if disc.ndim != 4 or diff0.ndim != 4:
        raise ValueError(
            f"unexpected calibrated_expectations shapes: "
            f"disc={disc.shape}, diff0={diff0.shape}"
        )

    _, n_dims, disc_items, _ = disc.shape
    _, _, num_items, _ = diff0.shape
    if disc_items == 1 and num_items > 1:
        # Shared-discrimination model: broadcast α scalar across items.
        disc = np.broadcast_to(disc, (1, n_dims, num_items, 1)).copy()

    item_keys = list(irt_model.item_keys)
    if num_items != len(item_keys):
        raise ValueError(
            f"calibrated_expectations has {num_items} items but "
            f"irt_model.item_keys has {len(item_keys)}"
        )

    if scale_names is None:
        scale_names = [f"dim_{d}" for d in range(n_dims)]
    scale_names = list(scale_names)
    if len(scale_names) != n_dims:
        raise ValueError(
            f"got {len(scale_names)} scale_names but model has {n_dims} dims"
        )

    # Build cumulative cutpoints τ = cumsum([d0, ddiff]) per (dim, item).
    if ddiff is None:
        cuts = diff0[..., 0:1]
    else:
        cuts = np.concatenate([diff0, ddiff], axis=-1)
    cuts = np.cumsum(cuts, axis=-1)  # (1, D, I, K-1)

    # FactorizedGRModel uses scale_indices to slot items into dims.
    scale_indices = getattr(irt_model, "scale_indices", None)

    # For each item, build a "scales" sub-dict containing only the dims
    # it actually loads on. With FactorizedGRModel scale_indices restricts
    # the slot; without it, every item loads on every dim.
    items_out: list[dict] = []
    response_cardinality = int(getattr(irt_model, "response_cardinality", 0))
    meta_lookup = dict(item_metadata or {})
    for i, key in enumerate(item_keys):
        scales_payload: dict[str, dict[str, Any]] = {}
        for d, scale in enumerate(scale_names):
            if scale_indices is not None and i not in scale_indices[d]:
                continue
            alpha = float(disc[0, d, i, 0])
            tau = cuts[0, d, i, :].tolist()
            scales_payload[scale] = {
                "discrimination": alpha,
                "difficulties": [float(t) for t in tau],
            }
        item_dict: dict[str, Any] = {
            "item": key,
            "scales": scales_payload,
        }
        extra = meta_lookup.get(key, {})
        # gofluttercat reads "responses", "question", "scored_vales".
        # Fill in defaults that produce a valid (if minimal) item file.
        if "question" in extra:
            item_dict["question"] = extra["question"]
        else:
            item_dict["question"] = ""
        if "responses" in extra:
            item_dict["responses"] = extra["responses"]
        elif response_cardinality:
            item_dict["responses"] = {
                str(v): {"text": str(v), "value": v}
                for v in range(response_cardinality)
            }
        if "scored_vales" in extra:
            item_dict["scored_vales"] = extra["scored_vales"]
        elif "scored_values" in extra:
            item_dict["scored_vales"] = extra["scored_values"]
        elif response_cardinality:
            item_dict["scored_vales"] = list(range(response_cardinality))
        for k, v in extra.items():
            if k not in item_dict and k not in {"scored_values"}:
                item_dict[k] = v
        items_out.append(item_dict)

    scales_out = {
        scale: {"description": scale, "response_cardinality": response_cardinality}
        for scale in scale_names
    }
    return items_out, scales_out


def export_multi_scale_artifact(
    irt_models: Mapping[str, Any],
    imputation_model,
    out_dir: PathLike,
    *,
    item_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    fit_method: str = "unknown",
    source_script: Optional[str] = None,
    extra_manifest: Optional[Mapping[str, Any]] = None,
    imputation_backend: str = "hdf5",
) -> Path:
    """Combine several unidimensional GRMs (one per scale) into a single
    artifact bundle.

    Items can appear under multiple scales -- duplicate item entries are
    merged so the final ``items/<key>.json`` lists every scale that the
    item loads on. Use this for factorized examples where each scale was
    fit independently as a 1D GRModel.

    The single shared ``imputation/`` directory is written once.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items").mkdir(exist_ok=True)

    merged_items: dict[str, dict] = {}
    merged_scales: dict[str, dict] = {}

    for scale_name, irt_model in irt_models.items():
        items, scales = grm_to_items(
            irt_model,
            scale_names=[scale_name],
            item_metadata=item_metadata,
        )
        merged_scales.update(scales)
        for item in items:
            existing = merged_items.get(item["item"])
            if existing is None:
                merged_items[item["item"]] = item
            else:
                existing["scales"].update(item["scales"])
                # Prefer richer metadata where the new entry has more.
                for key in ("question", "responses", "scored_vales"):
                    if key in item and key not in existing:
                        existing[key] = item[key]

    for item in merged_items.values():
        path = out / "items" / f"{item['item']}.json"
        path.write_text(json.dumps(item, indent=2, sort_keys=False))

    (out / "scales.json").write_text(
        json.dumps(merged_scales, indent=2, sort_keys=False)
    )

    if imputation_model is not None:
        imp_dir = out / "imputation"
        if hasattr(imputation_model, "save_to_disk"):
            try:
                imputation_model.save_to_disk(str(imp_dir), backend=imputation_backend)
            except TypeError:
                imputation_model.save_to_disk(str(imp_dir))
        elif hasattr(imputation_model, "save"):
            imp_dir.mkdir(parents=True, exist_ok=True)
            imputation_model.save(str(imp_dir / "model.h5"))
        else:
            raise TypeError(
                f"imputation_model of type {type(imputation_model).__name__} "
                "has neither save_to_disk nor save"
            )

    sample_model = next(iter(irt_models.values()))
    manifest = {
        "format_version": "1.0",
        "format": "gofluttercat-libfab-converged-grm",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fit_method": fit_method,
        "source_script": source_script or os.path.basename(sys.argv[0] or ""),
        "python_version": platform.python_version(),
        "share_discriminations": bool(
            getattr(sample_model, "share_discriminations", False)
        ),
        "n_items": len(merged_items),
        "scale_names": list(merged_scales.keys()),
        "item_keys": list(merged_items.keys()),
        "response_cardinality": int(
            getattr(sample_model, "response_cardinality", 0)
        ),
        "dimensions": len(merged_scales),
        "imputation_model_type": (
            type(imputation_model).__name__ if imputation_model is not None else None
        ),
    }
    if extra_manifest:
        manifest.update(dict(extra_manifest))
    (out / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    return out


def export_artifact(
    irt_model,
    imputation_model,
    out_dir: PathLike,
    *,
    scale_names: Optional[Sequence[str]] = None,
    item_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    fit_method: str = "unknown",
    source_script: Optional[str] = None,
    extra_manifest: Optional[Mapping[str, Any]] = None,
    imputation_backend: str = "hdf5",
) -> Path:
    """Write the converged artifact bundle.

    Parameters
    ----------
    irt_model
        Fitted ``GRModel`` (with ``calibrated_expectations`` set).
    imputation_model
        The imputation model used during the IRT fit (e.g. a
        ``PairwiseOrdinalStackingModel`` or ``IrtMixedImputationModel``).
        Pass ``None`` to skip the imputation sub-directory.
    out_dir
        Output directory.  Created if it does not exist.
    scale_names
        See ``grm_to_items``.
    item_metadata
        See ``grm_to_items``.
    fit_method
        Free-form label written to ``manifest.yaml`` -- e.g.
        ``"weighted_advi"``, ``"marginal_mcmc"``, ``"is_reweight"``.
    source_script
        Path or name of the calling script (recorded in manifest).
    extra_manifest
        Extra key/value pairs merged into ``manifest.yaml``.
    imputation_backend
        ``"hdf5"`` (default) or ``"safetensors"``.  Forwarded to the
        imputation model's ``save_to_disk``.

    Returns the resolved ``Path`` to ``out_dir``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items").mkdir(exist_ok=True)

    items, scales = grm_to_items(
        irt_model, scale_names=scale_names, item_metadata=item_metadata
    )

    for item_dict in items:
        # Filenames must round-trip on case-insensitive FS; use the item key
        # verbatim and trust upstream uniqueness.
        path = out / "items" / f"{item_dict['item']}.json"
        path.write_text(json.dumps(item_dict, indent=2, sort_keys=False))

    (out / "scales.json").write_text(json.dumps(scales, indent=2, sort_keys=False))

    if imputation_model is not None:
        imp_dir = out / "imputation"
        # PairwiseOrdinalStackingModel and IrtMixedImputationModel both
        # expose save_to_disk(dir, backend=...).  MICEBayesianLOO does too.
        saved = False
        if hasattr(imputation_model, "save_to_disk"):
            try:
                imputation_model.save_to_disk(str(imp_dir), backend=imputation_backend)
                saved = True
            except TypeError:
                imputation_model.save_to_disk(str(imp_dir))
                saved = True
        elif hasattr(imputation_model, "save"):
            imp_dir.mkdir(parents=True, exist_ok=True)
            imputation_model.save(str(imp_dir / "model.h5"))
            saved = True
        if not saved:
            raise TypeError(
                f"imputation_model of type {type(imputation_model).__name__} "
                "has neither save_to_disk nor save"
            )

    manifest = {
        "format_version": "1.0",
        "format": "gofluttercat-libfab-converged-grm",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fit_method": fit_method,
        "source_script": source_script or os.path.basename(sys.argv[0] or ""),
        "python_version": platform.python_version(),
        "share_discriminations": bool(
            getattr(irt_model, "share_discriminations", False)
        ),
        "n_items": len(items),
        "scale_names": [str(k) for k in scales.keys()],
        "item_keys": [str(it["item"]) for it in items],
        "response_cardinality": int(
            getattr(irt_model, "response_cardinality", 0)
        ),
        "dimensions": int(getattr(irt_model, "dimensions", 1)),
        "imputation_model_type": (
            type(imputation_model).__name__ if imputation_model is not None else None
        ),
    }
    if extra_manifest:
        manifest.update(_jsonify(dict(extra_manifest)))
    (out / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    return out

"""PROMIS 1 Wave 1 calibration dataset loader.

Single-population calibration sample (US general + multiple disease cohorts,
N≈21,000) for 14 PROMIS health domains. Data file is the public release at
Harvard Dataverse (hdl:1902.1/21134), guestbook-protected so the .tab file
must be downloaded manually.

Each domain is a 56-item polytomous (5-category) bank. Every respondent
saw a subset of items, so item-level missingness is structural by design
(planned-missing-by-form, not item nonresponse) — useful for the M-open
scoring bias study.

5 response categories (0-4, shifted from original 1-5).

Reference: PROMIS 1 Wave 1 readme codebook v4 (HealthMeasures/Northwestern).
"""

import os
import re
from pathlib import Path

import grain
import numpy as np
import polars as pl

# Populated on first call to ``get_data()`` based on ``domain``.
item_keys: list[str] = []

response_cardinality = 5

# 14 health domains, each ~56 calibration items. Prefix → semantic name.
DOMAINS = {
    "alcohol_use":            "EDALC",
    "anger":                  "EDANG",
    "anxiety":                "EDANX",
    "depression":             "EDDEP",
    "fatigue_experience":     "FATEXP",
    "fatigue_impact":         "FATIMP",
    "pain_behavior":          "PAINBE",
    "pain_interference":      "PAININ",
    "pain_quality":           "PAINQU",
    "physical_function_a":    "PFA",
    "physical_function_b":    "PFB",
    "physical_function_c":    "PFC",
    "social_personal":        "SRPPER",
    "social_satisfaction":    "SRPSAT",
}

_DEFAULT_DOMAIN = "depression"


class _ArrayDataSource(grain.sources.RandomAccessDataSource):
    def __init__(self, df: pl.DataFrame):
        self._data = {col: df[col].to_list() for col in df.columns}
        self.n = len(df)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self._data.items()}

    def __len__(self):
        return self.n


def _detect_domain_columns(all_columns, prefix):
    """Find columns matching ``<prefix><digits>``."""
    pat = re.compile(rf"^{re.escape(prefix)}\d+$")
    return sorted([c for c in all_columns if pat.match(c)])


def _candidate_data_paths(cache_dir):
    """Where to look for the manually-downloaded Wave 1 .tab file."""
    return [
        cache_dir / "promis_wave1.tab",
        cache_dir / "promis_wave1" / "promis_wave1.tab",
        Path.cwd() / "promis_wave1" / "promis_wave1.tab",
        Path(__file__).parent.parent.parent.parent / "notebooks/irt/promis_wave1/promis_wave1.tab",
    ]


def _resolve_data_path(cache_dir):
    for p in _candidate_data_paths(cache_dir):
        if p.exists():
            return p
    raise FileNotFoundError(
        "Wave 1 data file not found. Download from "
        "https://dataverse.harvard.edu/dataset.xhtml?persistentId=hdl:1902.1/21134 "
        "(accept guestbook), unzip, and place 'PROMIS 1 Wave 1.tab' "
        "(renamed to 'promis_wave1.tab') in one of: "
        + ", ".join(str(p) for p in _candidate_data_paths(cache_dir))
    )


def _prepare(df, selected_keys):
    data = df.select(selected_keys)
    data = data.with_columns([
        (pl.col(k).cast(pl.Float64) - 1).alias(k) for k in selected_keys
    ])
    num_people = len(data)
    data = data.with_columns([
        pl.when((pl.col(k) < 0) | (pl.col(k) >= response_cardinality) | pl.col(k).is_null())
        .then(-1)
        .otherwise(pl.col(k))
        .alias(k)
        for k in selected_keys
    ])
    data = data.with_row_index("person")
    return data, num_people


def get_data(reorient=False, polars_out=False, cache_dir=None, domain=None):
    """Load a single PROMIS Wave 1 health domain.

    Args:
        reorient: No-op (PROMIS items are pre-aligned by published scoring).
        polars_out: Return a Polars DataFrame instead of a grain Dataset.
        cache_dir: Directory containing the downloaded ``promis_wave1.tab``.
        domain: One of the keys in ``DOMAINS`` (default: 'depression').

    Returns:
        Tuple of ``(dataset, num_people)``.
    """
    global item_keys

    if domain is None:
        domain = _DEFAULT_DOMAIN
    if domain not in DOMAINS:
        raise ValueError(
            f"Unknown domain {domain!r}. Choose from: {list(DOMAINS)}"
        )

    if cache_dir is None:
        cache_dir = Path.cwd()
    else:
        cache_dir = Path(cache_dir)

    tab_path = _resolve_data_path(cache_dir)

    df = pl.read_csv(str(tab_path), separator="\t", infer_schema_length=10000,
                     null_values=["", "NA", "."])

    detected = _detect_domain_columns(df.columns, DOMAINS[domain])
    if not detected:
        raise ValueError(
            f"No columns with prefix {DOMAINS[domain]!r} for domain {domain!r}."
        )

    item_keys.clear()
    item_keys.extend(detected)
    print(f"Wave 1 domain '{domain}': {len(item_keys)} items (prefix '{DOMAINS[domain]}')")

    # Drop respondents with all missing in this domain (planned-missing design)
    data, num_people = _prepare(df, item_keys)
    arr = data.select(item_keys).to_numpy()
    keep_mask = (arr >= 0).any(axis=1)
    n_dropped = int((~keep_mask).sum())
    if n_dropped > 0:
        keep_idx = np.where(keep_mask)[0]
        data = data.filter(pl.col("person").is_in(keep_idx))
        # re-index "person" 0..n-1
        data = data.drop("person").with_row_index("person")
        num_people = len(data)
        print(f"  Dropped {n_dropped} respondents with no responses in this domain; "
              f"remaining N={num_people}")

    _ = reorient  # PROMIS items are pre-aligned, no-op

    if polars_out:
        return data, num_people
    dataset = grain.MapDataset.source(_ArrayDataSource(data))
    return dataset, num_people

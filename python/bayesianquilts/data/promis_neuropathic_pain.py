"""PROMIS Neuropathic Pain data loader.

Downloads and preprocesses the PROMIS Neuropathic Pain adult dataset
from Harvard Dataverse (doi:10.7910/DVN/TJ9MNM).

The dataset contains 337 items across multiple pain-related PROMIS banks:
  - Pain Interference (PAININ*): ~34 items — well-established unidimensional
  - Pain Behavior (PAINBE*): ~40 items
  - Pain Quality: Neuropathic Pain: ~7 items
  - Pain Quality: Nociceptive Pain: ~6 items
  - Pain Quality: General Pain: ~57 items
  - Pain Intensity: ~6 items
  - PROMIS-29 items
  - PROMIS Global Health items

By default, loads the Pain Interference bank (unidimensional, ~34 items).
Use the `domain` parameter to select a different bank.

The data file is SAS binary format (.sas7bdat); requires pandas for reading.

5 response categories (0-4, shifted from original 1-5).

Reference: Askew et al. (2016). "Development of a crosswalk for PRO
measures of pain interference and pain behavior." Rehabilitation
Psychology, 61(1), 88-97.
"""

import os
import urllib.request
from pathlib import Path

import grain
import numpy as np
import polars as pl

# Default domain: Pain Interference (well-established unidimensional bank)
# Set on first load based on columns matching the domain prefix.
item_keys: list[str] = []

response_cardinality = 5

_DATA_URL = "https://dataverse.harvard.edu/api/access/datafile/3194523?format=original"

# Known PROMIS item prefixes in this dataset
DOMAINS = {
    "pain_interference": "PAININ",
    "pain_behavior": "PAINBE",
    "global_health": "Global",
    "depression": "EDDEP",
    "anxiety": "EDANX",
    "fatigue": "FATEXP",
    "physical_function": "PF",
    "sleep": "Sleep",
}

_DEFAULT_DOMAIN = "pain_interference"


class _ArrayDataSource(grain.sources.RandomAccessDataSource):
    def __init__(self, df: pl.DataFrame):
        self._data = {col: df[col].to_list() for col in df.columns}
        self.n = len(df)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self._data.items()}

    def __len__(self):
        return self.n


def _detect_domain_columns(all_columns: list[str], prefix: str) -> list[str]:
    """Find columns matching a PROMIS item prefix."""
    return sorted([c for c in all_columns if c.startswith(prefix)])


def _load_sas(cache_dir):
    """Download and return the SAS file as a pandas DataFrame."""
    sas_path = cache_dir / "neuropath_adults_07192018.sas7bdat"
    if not sas_path.exists():
        os.makedirs(cache_dir, exist_ok=True)
        req = urllib.request.Request(_DATA_URL, headers={"User-Agent": "bayesianquilts"})
        with urllib.request.urlopen(req) as resp, open(str(sas_path), "wb") as f:
            f.write(resp.read())
    import pandas as pd
    return pd.read_sas(str(sas_path), format="sas7bdat", encoding="latin1")


def _prepare_data(df_pandas, selected_keys):
    """Select columns, shift 1-5 → 0-4, mark invalid as -1."""
    data = pl.from_pandas(df_pandas[selected_keys])
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


def get_data(polars_out=False, cache_dir=None, domain=None):
    """Load a single PROMIS domain from the Neuropathic Pain dataset.

    Args:
        polars_out: If True, return Polars DataFrame instead of grain Dataset.
        cache_dir: Directory to cache downloaded data. Defaults to cwd.
        domain: Which PROMIS domain to load (default: 'pain_interference').

    Returns:
        Tuple of (dataset, num_people).
    """
    global item_keys

    if domain is None:
        domain = _DEFAULT_DOMAIN
    if domain not in DOMAINS:
        raise ValueError(
            f"Unknown domain {domain!r}. Choose from: {list(DOMAINS.keys())}"
        )

    if cache_dir is None:
        cache_dir = Path.cwd()
    else:
        cache_dir = Path(cache_dir)

    df_pandas = _load_sas(cache_dir)
    detected = _detect_domain_columns(list(df_pandas.columns), DOMAINS[domain])
    if not detected:
        raise ValueError(
            f"No columns with prefix {DOMAINS[domain]!r}. "
            f"Available: {sorted(set(c[:5] for c in df_pandas.columns if len(c) >= 5))}"
        )

    item_keys.clear()
    item_keys.extend(detected)
    print(f"Domain '{domain}': {len(item_keys)} items (prefix '{DOMAINS[domain]}')")

    data, num_people = _prepare_data(df_pandas, item_keys)

    if polars_out:
        return data, num_people
    dataset = grain.MapDataset.source(_ArrayDataSource(data))
    return dataset, num_people


def get_multidomain_data(polars_out=False, cache_dir=None, domains=None,
                         min_items=10):
    """Load multiple PROMIS domains for FactorizedGRModel.

    Args:
        polars_out: If True, return Polars DataFrame instead of grain Dataset.
        cache_dir: Directory to cache downloaded data. Defaults to cwd.
        domains: List of domain names to load. If None, loads all domains
            with at least ``min_items`` items.
        min_items: Minimum number of items for a domain to be included.

    Returns:
        Tuple of (dataset, num_people, scale_indices) where
        scale_indices maps domain name → list of integer indices into
        the flat item_keys list.
    """
    global item_keys

    if cache_dir is None:
        cache_dir = Path.cwd()
    else:
        cache_dir = Path(cache_dir)

    df_pandas = _load_sas(cache_dir)
    all_columns = list(df_pandas.columns)

    if domains is None:
        domains = []
        for name, prefix in DOMAINS.items():
            cols = _detect_domain_columns(all_columns, prefix)
            if len(cols) >= min_items:
                domains.append(name)
        print(f"Auto-selected {len(domains)} domains with ≥{min_items} items: {domains}")

    all_keys = []
    scale_indices = {}
    for domain in domains:
        if domain not in DOMAINS:
            raise ValueError(f"Unknown domain {domain!r}")
        cols = _detect_domain_columns(all_columns, DOMAINS[domain])
        if not cols:
            print(f"  Warning: no items for '{domain}', skipping")
            continue
        start = len(all_keys)
        all_keys.extend(cols)
        scale_indices[domain] = list(range(start, start + len(cols)))
        print(f"  {domain}: {len(cols)} items (indices {start}-{start + len(cols) - 1})")

    item_keys.clear()
    item_keys.extend(all_keys)
    print(f"Total: {len(item_keys)} items across {len(scale_indices)} domains")

    data, num_people = _prepare_data(df_pandas, item_keys)

    if polars_out:
        return data, num_people, scale_indices
    dataset = grain.MapDataset.source(_ArrayDataSource(data))
    return dataset, num_people, scale_indices

"""PROMIS COPD cohort data loader.

Downloads and preprocesses the PROMIS Wave 2 COPD dataset from
Harvard Dataverse (doi:10.7910/DVN/UOQNJF).

This longitudinal dataset contains item-level responses for multiple
PROMIS banks administered to COPD patients. Each bank is independently
unidimensional. Available domains:

  - Depression (EDDEP*): ~28 items
  - Anxiety (EDANX*): ~29 items
  - Anger (EDANG*): ~29 items
  - Fatigue Experience (FATEXP*): variable
  - Fatigue Impact (FATIMP*): variable
  - Pain Interference (PAININ*): ~34 items
  - Pain Behavior (PAINBE*): ~40 items
  - Physical Function (PFA*/PFB*/PFC*): variable
  - Social Satisfaction — Discretionary (SRPSAT*): ~12 items

By default, loads the Depression bank (EDDEP*, well-characterized
unidimensional bank with ~28 items).

5 response categories (0-4, shifted from original 1-5).

Reference: DeWalt et al. PROMIS 1 Wave 2 COPD Study.
University of North Carolina at Chapel Hill.
"""

import os
import urllib.request
from pathlib import Path

import grain
import polars as pl

# Populated on first load based on selected domain.
item_keys: list[str] = []

response_cardinality = 5

_DATA_URL = "https://dataverse.harvard.edu/api/access/datafile/2911730"

# Known PROMIS item prefixes in this dataset
DOMAINS = {
    "depression": ["EDDEP"],
    "anxiety": ["EDANX"],
    "anger": ["EDANG"],
    "fatigue_experience": ["FATEXP"],
    "fatigue_impact": ["FATIMP"],
    "pain_interference": ["PAININ"],
    "pain_behavior": ["PAINBE"],
    "physical_function": ["PFA", "PFB", "PFC"],
    "social_satisfaction": ["SRPSAT"],
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


def _detect_domain_columns(all_columns: list[str], prefixes: list[str]) -> list[str]:
    """Find _Bank columns matching any of the given PROMIS item prefixes.

    Only selects columns ending in '_Bank' (full item bank responses),
    excluding '_SF' (short form), theta/SE scores, and recoded variants.
    """
    matched = []
    for col in all_columns:
        if not col.endswith("_Bank"):
            continue
        if any(col.startswith(p) for p in prefixes):
            matched.append(col)
    return sorted(matched)


def list_domains(cache_dir=None):
    """Download data and report available domains with item counts."""
    if cache_dir is None:
        cache_dir = Path.cwd()
    else:
        cache_dir = Path(cache_dir)

    tab_path = cache_dir / "promis_copd.tab"
    if not tab_path.exists():
        os.makedirs(cache_dir, exist_ok=True)
        req = urllib.request.Request(_DATA_URL, headers={"User-Agent": "bayesianquilts"})
        with urllib.request.urlopen(req) as resp, open(str(tab_path), "wb") as f:
            f.write(resp.read())

    data = pl.read_csv(str(tab_path), separator="\t", n_rows=1, infer_schema_length=10000)
    all_cols = data.columns
    for name, prefixes in DOMAINS.items():
        cols = _detect_domain_columns(all_cols, prefixes)
        print(f"  {name}: {len(cols)} items (prefixes: {prefixes})")


def _download_raw(cache_dir):
    """Download and return the raw tab file as a Polars DataFrame."""
    tab_path = cache_dir / "promis_copd.tab"
    if not tab_path.exists():
        os.makedirs(cache_dir, exist_ok=True)
        req = urllib.request.Request(_DATA_URL, headers={"User-Agent": "bayesianquilts"})
        with urllib.request.urlopen(req) as resp, open(str(tab_path), "wb") as f:
            f.write(resp.read())
    return pl.read_csv(str(tab_path), separator="\t", infer_schema_length=5000)


def _prepare_data(raw, selected_keys):
    """Select columns, shift 1-5 → 0-4, mark invalid as -1."""
    data = raw.select(selected_keys)
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
    """Load a single PROMIS domain from the COPD dataset.

    Args:
        reorient: If True, reverse-score items with negative item-total
            correlations so all items load in the same direction.
        polars_out: If True, return Polars DataFrame instead of grain Dataset.
        cache_dir: Directory to cache downloaded data. Defaults to cwd.
        domain: Which PROMIS domain to load. One of:
            'depression' (default), 'anxiety', 'anger',
            'fatigue_experience', 'fatigue_impact',
            'pain_interference', 'pain_behavior',
            'physical_function', 'social_satisfaction'.

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

    raw = _download_raw(cache_dir)
    detected = _detect_domain_columns(raw.columns, DOMAINS[domain])
    if not detected:
        raise ValueError(
            f"No columns found for domain {domain!r}. Run list_domains()."
        )

    item_keys.clear()
    item_keys.extend(detected)
    print(f"Domain '{domain}': {len(item_keys)} items")

    data, num_people = _prepare_data(raw, item_keys)

    if reorient:
        import numpy as np
        arr = data.select(item_keys).to_numpy().astype(float)
        arr[arr < 0] = np.nan
        total = np.nanmean(arr, axis=1)
        to_reverse = []
        for i, k in enumerate(item_keys):
            valid = ~np.isnan(arr[:, i])
            if valid.sum() > 10:
                r = np.corrcoef(arr[valid, i], total[valid])[0, 1]
                if r < -0.05:
                    to_reverse.append(k)
        if to_reverse:
            K = response_cardinality
            data = data.with_columns([
                (K - 1 - pl.col(k)).alias(k) for k in to_reverse
            ])
            data = data.with_columns([
                pl.when((pl.col(k) < 0) | (pl.col(k) >= K))
                .then(-1).otherwise(pl.col(k)).alias(k)
                for k in to_reverse
            ])
            print(f"  Reoriented {len(to_reverse)} items: {to_reverse}")

    if polars_out:
        return data, num_people
    dataset = grain.MapDataset.source(_ArrayDataSource(data))
    return dataset, num_people


def get_multidomain_data(reorient=False, polars_out=False, cache_dir=None,
                         domains=None, min_items=10):
    """Load multiple PROMIS domains for FactorizedGRModel.

    Args:
        reorient: If True, reverse-score items with negative item-total
            correlations within each domain.
        polars_out: If True, return Polars DataFrame instead of grain Dataset.
        cache_dir: Directory to cache downloaded data. Defaults to cwd.
        domains: List of domain names to load. If None, loads all domains
            with at least ``min_items`` items.
        min_items: Minimum number of items for a domain to be included
            (only used when ``domains`` is None).

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

    raw = _download_raw(cache_dir)
    all_columns = raw.columns

    if domains is None:
        domains = []
        for name, prefixes in DOMAINS.items():
            cols = _detect_domain_columns(all_columns, prefixes)
            if len(cols) >= min_items:
                domains.append(name)
        print(f"Auto-selected {len(domains)} domains with ≥{min_items} items: {domains}")

    # Build flat item_keys and scale_indices
    all_keys = []
    scale_indices = {}
    for domain in domains:
        if domain not in DOMAINS:
            raise ValueError(f"Unknown domain {domain!r}")
        cols = _detect_domain_columns(all_columns, DOMAINS[domain])
        if not cols:
            print(f"  Warning: no items found for '{domain}', skipping")
            continue
        start = len(all_keys)
        all_keys.extend(cols)
        scale_indices[domain] = list(range(start, start + len(cols)))
        print(f"  {domain}: {len(cols)} items (indices {start}-{start + len(cols) - 1})")

    item_keys.clear()
    item_keys.extend(all_keys)
    print(f"Total: {len(item_keys)} items across {len(scale_indices)} domains")

    data, num_people = _prepare_data(raw, item_keys)

    if reorient:
        import numpy as np
        # Reorient per domain (item-total correlation within each domain)
        for domain, indices in scale_indices.items():
            domain_keys = [item_keys[i] for i in indices]
            arr = data.select(domain_keys).to_numpy().astype(float)
            arr[arr < 0] = np.nan
            total = np.nanmean(arr, axis=1)
            to_reverse = []
            for j, k in enumerate(domain_keys):
                valid = ~np.isnan(arr[:, j])
                if valid.sum() > 10:
                    r = np.corrcoef(arr[valid, j], total[valid])[0, 1]
                    if r < -0.05:
                        to_reverse.append(k)
            if to_reverse:
                K = response_cardinality
                data = data.with_columns([
                    (K - 1 - pl.col(k)).alias(k) for k in to_reverse
                ])
                data = data.with_columns([
                    pl.when((pl.col(k) < 0) | (pl.col(k) >= K))
                    .then(-1).otherwise(pl.col(k)).alias(k)
                    for k in to_reverse
                ])
                print(f"  Reoriented {len(to_reverse)} items in {domain}: {to_reverse}")

    if polars_out:
        return data, num_people, scale_indices
    dataset = grain.MapDataset.source(_ArrayDataSource(data))
    return dataset, num_people, scale_indices

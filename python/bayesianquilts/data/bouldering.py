"""IFSC bouldering competition data loader.

Loads and pivots IFSC bouldering competition results into person × item
format suitable for IRT modeling. Each "item" is a unique
(competition, round, boulder) combination. Responses are ordinal:
0 = no zone, 1 = zone, 2 = top, 3 = flash.

Data is embedded in the package as bouldering_men.csv / bouldering_women.csv.
"""

import importlib.resources
import re
import unicodedata
from difflib import SequenceMatcher

import grain
import numpy as np
import polars as pl

# These are set dynamically by get_data() after pivoting
item_keys = []
response_cardinality = 4

# Mapping from item keys (B_0000, B_0001, ...) to human-readable labels
item_labels = {}

# Mapping from consolidated climber name to row index
climber_names = []


class _ArrayDataSource(grain.sources.RandomAccessDataSource):
    def __init__(self, df: pl.DataFrame):
        self._data = {col: df[col].to_list() for col in df.columns}
        self.n = len(df)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self._data.items()}

    def __len__(self):
        return self.n


def _normalize_name(name):
    """Lowercase, remove accents, clean whitespace."""
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r'[^\w\s]', ' ', name)
    return ' '.join(name.split())


def _consolidate_names(df, threshold=0.9):
    """Merge near-duplicate climber names via fuzzy matching.

    Normalizes names (lowercasing, accent removal, whitespace cleanup)
    and uses SequenceMatcher to merge variations above the similarity
    threshold. The first-seen spelling is kept as canonical.
    """
    unique_names = df['Name'].unique().to_list()
    name_mapping = {}
    canonical = {}

    for name in unique_names:
        normed = _normalize_name(name)
        matched = False
        for canon_norm, canon_name in canonical.items():
            if SequenceMatcher(None, normed, canon_norm).ratio() >= threshold:
                name_mapping[name] = canon_name
                matched = True
                break
        if not matched:
            name_mapping[name] = name
            canonical[normed] = name

    n_merged = sum(1 for k, v in name_mapping.items() if k != v)
    if n_merged:
        print(f"  Consolidated {n_merged} name variations")

    return df.with_columns(
        pl.col('Name').replace(name_mapping).alias('Name')
    )


def get_data(polars_out=False, cache_dir=None, gender='men'):
    """Load IFSC bouldering data as a person × item matrix.

    Each unique (competition, level, boulder) triple becomes one item.
    Responses are 0-3; unobserved items are -1.

    Args:
        polars_out: If True return (DataFrame, num_people), else grain dataset.
        cache_dir: Unused (data is packaged).
        gender: 'men' or 'women'.

    Returns:
        (dataset_or_df, num_people)
    """
    global item_keys, item_labels, climber_names

    # Load gzipped CSV from package data via importlib.resources
    import gzip

    gz_filename = f"bouldering_{gender}.csv.gz"
    data_pkg = importlib.resources.files("bayesianquilts.data")
    gz_bytes = (data_pkg / gz_filename).read_bytes()
    csv_bytes = gzip.decompress(gz_bytes)
    raw = pl.read_csv(csv_bytes)

    # Drop the unnamed index column
    if '' in raw.columns:
        raw = raw.drop('')

    # Consolidate name variations (accent normalization + fuzzy matching)
    raw = _consolidate_names(raw)

    boulder_cols = [c for c in raw.columns if c.startswith('Boulder')]

    # Melt to long format: one row per (climber, competition, level, boulder)
    long = raw.unpivot(
        on=boulder_cols,
        index=['Name', 'Competition', 'Level'],
        variable_name='boulder_col',
        value_name='outcome',
    )

    # Create unique item identifier
    long = long.with_columns(
        (pl.col('Competition') + '|' + pl.col('Level') + '|' + pl.col('boulder_col'))
        .alias('item_id')
    )

    # Parse outcome: "NA" or null → -1, otherwise int
    long = long.with_columns(
        pl.when(pl.col('outcome') == 'NA')
        .then(pl.lit(-1))
        .otherwise(pl.col('outcome').cast(pl.Int64, strict=False))
        .fill_null(-1)
        .alias('outcome')
    )

    # Build item key mapping: sorted unique item_ids → B_0000, B_0001, ...
    unique_items = sorted(long['item_id'].unique().to_list())
    n_items = len(unique_items)
    key_map = {item_id: f"B_{i:04d}" for i, item_id in enumerate(unique_items)}
    item_keys_local = [f"B_{i:04d}" for i in range(n_items)]

    # Store mapping for interpretability
    item_labels.clear()
    item_labels.update({v: k for k, v in key_map.items()})

    long = long.with_columns(
        pl.col('item_id').replace(key_map).alias('item_key')
    )

    # Pivot to wide: rows = climbers, columns = item keys
    wide = long.pivot(
        on='item_key',
        index='Name',
        values='outcome',
        aggregate_function='first',
    )

    # Ensure all item columns exist and fill nulls with -1
    for k in item_keys_local:
        if k not in wide.columns:
            wide = wide.with_columns(pl.lit(-1).alias(k))
        else:
            wide = wide.with_columns(pl.col(k).fill_null(-1))

    # Reorder columns
    wide = wide.select(['Name'] + item_keys_local)

    num_people = len(wide)

    # Mark out-of-range values as -1
    wide = wide.with_columns([
        pl.when((pl.col(k) < 0) | (pl.col(k) >= response_cardinality))
        .then(-1)
        .otherwise(pl.col(k))
        .alias(k)
        for k in item_keys_local
    ])

    # Store climber names for later reference (sanity-checking ranked abilities)
    climber_names.clear()
    climber_names.extend(wide['Name'].to_list())

    # Drop Name column, add person index
    data = wide.drop('Name').with_row_index('person')

    # Set module-level item_keys
    item_keys.clear()
    item_keys.extend(item_keys_local)

    n_observed = sum(
        int((data[k].to_numpy() >= 0).sum()) for k in item_keys_local
    )
    n_total = num_people * n_items
    print(f"  Climbers: {num_people}, Items: {n_items}, "
          f"Observed: {n_observed}/{n_total} ({100*n_observed/n_total:.1f}%)")

    if polars_out:
        return data, num_people

    dataset = grain.MapDataset.source(_ArrayDataSource(data))
    return dataset, num_people

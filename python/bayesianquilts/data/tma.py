"""TMA (Taylor Manifest Anxiety Scale) data loader.

Downloads and preprocesses the TMA dataset from OpenPsychometrics.org.
50 items (Q1-Q50), binary true/false (0-1).
"""

import os
import urllib.request
import zipfile
from pathlib import Path

import grain
import polars as pl

item_keys = [f"Q{i}" for i in range(1, 51)]

response_cardinality = 2

# 0-indexed item positions to reverse-score (negative item-total correlations)
# These items are keyed opposite to the dominant factor direction.
to_reverse = [0, 2, 3, 8, 11, 17, 19, 28, 31, 37, 49]  # Q1,Q3,Q4,Q9,Q12,Q18,Q20,Q29,Q32,Q38,Q50

_DATA_URL = "https://openpsychometrics.org/_rawdata/TMA.zip"


class _ArrayDataSource(grain.sources.RandomAccessDataSource):
    def __init__(self, df: pl.DataFrame):
        self._data = {col: df[col].to_list() for col in df.columns}
        self.n = len(df)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self._data.items()}

    def __len__(self):
        return self.n


def get_data(reorient=False, polars_out=False, cache_dir=None):
    """Load the TMA dataset.

    Responses are shifted from 1-2 to 0-1. Values of 0 (not answered)
    become -1 after the shift and are treated as missing.

    Returns:
        Tuple of (dataset, num_people).
    """
    if cache_dir is None:
        cache_dir = Path.cwd()
    else:
        cache_dir = Path(cache_dir)

    csv_path = cache_dir / "TMA" / "data.csv"

    if not csv_path.exists():
        zip_path = cache_dir / "TMA.zip"
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(_DATA_URL, str(zip_path))
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(cache_dir))

    data = pl.read_csv(str(csv_path))
    data = data.select(item_keys)

    # Shift from 1-2 to 0-1 (0=not answered becomes -1)
    data = data.with_columns([
        (pl.col(k) - 1).alias(k) for k in item_keys
    ])

    num_people = len(data)

    if reorient:
        data = data.with_columns([
            (1 - pl.col(item_keys[i])).alias(item_keys[i])
            for i in to_reverse
        ])

    # Mark invalid values as -1
    data = data.with_columns([
        pl.when((pl.col(k) < 0) | (pl.col(k) >= response_cardinality))
        .then(-1)
        .otherwise(pl.col(k))
        .alias(k)
        for k in item_keys
    ])

    data = data.with_row_index("person")

    if polars_out:
        return data, num_people

    dataset = grain.MapDataset.source(_ArrayDataSource(data))
    return dataset, num_people

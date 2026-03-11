"""EQSQ (Empathizing-Systemizing Quotient) data loader.

Downloads and preprocesses the EQSQ dataset from OpenPsychometrics.org.
120 items (E1-E60 + S1-S60), 4 response categories (0-3), 1 scale.
"""

import os
import urllib.request
import zipfile
from pathlib import Path

import grain
import polars as pl

item_keys = [f"E{i}" for i in range(1, 61)] + [f"S{i}" for i in range(1, 61)]

response_cardinality = 4

# 0-indexed item positions to reverse-score (negative item-total correlations)
# E11,E23,E26,E32,E51,E57 and their S counterparts
_e_reverse = [10, 22, 25, 31, 50, 56]  # 0-indexed within E1-E60
_s_reverse = [10, 22, 25, 31, 50, 56]  # 0-indexed within S1-S60
to_reverse = _e_reverse + [i + 60 for i in _s_reverse]

_DATA_URL = "https://openpsychometrics.org/_rawdata/EQSQ.zip"


class _ArrayDataSource(grain.sources.RandomAccessDataSource):
    def __init__(self, df: pl.DataFrame):
        self._data = {col: df[col].to_list() for col in df.columns}
        self.n = len(df)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self._data.items()}

    def __len__(self):
        return self.n


def get_data(reorient=False, polars_out=False, cache_dir=None):
    """Load the EQSQ dataset.

    Responses are shifted from 1-4 to 0-3. Values of 0 (not answered)
    become -1 after the shift and are treated as missing.

    Returns:
        Tuple of (dataset, num_people).
    """
    if cache_dir is None:
        cache_dir = Path.cwd()
    else:
        cache_dir = Path(cache_dir)

    csv_path = cache_dir / "EQSQ" / "data.csv"

    if not csv_path.exists():
        zip_path = cache_dir / "EQSQ.zip"
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(_DATA_URL, str(zip_path))
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(cache_dir))

    data = pl.read_csv(str(csv_path), separator="\t")
    data = data.select(item_keys)

    # Shift from 1-4 to 0-3 (0=missing becomes -1)
    data = data.with_columns([
        (pl.col(k) - 1).alias(k) for k in item_keys
    ])

    num_people = len(data)

    if reorient:
        data = data.with_columns([
            (3 - pl.col(item_keys[i])).alias(item_keys[i])
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

"""PROMIS Substance Use item bank data loader.

Downloads and preprocesses the PROMIS Substance Use calibration dataset
from Harvard Dataverse (doi:10.7910/DVN/VLCJUE).

The dataset contains 263 substance use items with PROMIS item IDs
(columns named sd*R1 or sd*R2). Values 1-5 are valid responses;
8 and 9 are missing codes (refused/not applicable).

The two calibrated banks are:
  - Severity of Substance Use (37 items)
  - Positive Appeal of Substance Use (18 items)
But the full calibration pool of 263 items is loaded by default.

5 response categories (0-4, shifted from original 1-5).

Reference: Pilkonis et al. (2013). "Item banks for substance use
from the Patient-Reported Outcomes Measurement Information System
(PROMIS)." Drug and Alcohol Dependence, 130(1-3), 107-114.
"""

import os
import re
import urllib.request
from pathlib import Path

import grain
import polars as pl

# Populated on first call to get_data().
# PROMIS SUDS items use IDs like sd13523R1, sd4182R1, etc.
item_keys: list[str] = []

response_cardinality = 5

_DATA_URL = "https://dataverse.harvard.edu/api/access/datafile/3138368"


class _ArrayDataSource(grain.sources.RandomAccessDataSource):
    def __init__(self, df: pl.DataFrame):
        self._data = {col: df[col].to_list() for col in df.columns}
        self.n = len(df)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self._data.items()}

    def __len__(self):
        return self.n


def _detect_item_columns(all_columns: list[str]) -> list[str]:
    """Identify PROMIS substance use item columns.

    Items use PROMIS numeric IDs: sd{number}R{wave} (e.g., sd13523R1).
    """
    pattern = re.compile(r"^sd\d+R\d+d?$")
    return sorted([c for c in all_columns if pattern.match(c)])


def get_data(reorient=False, polars_out=False, cache_dir=None):
    """Load the PROMIS Substance Use dataset.

    On first call, downloads data and detects sd*R* item columns.
    Responses are shifted from 1-5 to 0-4. Values 8/9 (missing codes)
    and out-of-range values become -1.

    Args:
        reorient: If True, reverse-score items with negative item-total
            correlations (typically items with 'R' in the name).
        polars_out: If True, return polars DataFrame.
        cache_dir: Directory for caching downloaded data.

    Returns:
        Tuple of (dataset, num_people).
    """
    global item_keys

    if cache_dir is None:
        cache_dir = Path.cwd()
    else:
        cache_dir = Path(cache_dir)

    tab_path = cache_dir / "promis_substance_use.tab"

    if not tab_path.exists():
        os.makedirs(cache_dir, exist_ok=True)
        req = urllib.request.Request(_DATA_URL, headers={"User-Agent": "bayesianquilts"})
        with urllib.request.urlopen(req) as resp, open(str(tab_path), "wb") as f:
            f.write(resp.read())

    data = pl.read_csv(str(tab_path), separator="\t", infer_schema_length=5000)

    if not item_keys:
        detected = _detect_item_columns(data.columns)
        item_keys.clear()
        item_keys.extend(detected)
        print(f"Detected {len(item_keys)} substance use items (sd*R* pattern)")

    data = data.select(item_keys)

    # Shift from 1-5 to 0-4 (values 8, 9 become 7, 8 → caught as invalid)
    data = data.with_columns([
        (pl.col(k).cast(pl.Float64) - 1).alias(k) for k in item_keys
    ])

    num_people = len(data)

    # Mark invalid values as -1 (includes shifted 8→7 and 9→8)
    data = data.with_columns([
        pl.when((pl.col(k) < 0) | (pl.col(k) >= response_cardinality) | pl.col(k).is_null())
        .then(-1)
        .otherwise(pl.col(k))
        .alias(k)
        for k in item_keys
    ])

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
                .then(-1)
                .otherwise(pl.col(k))
                .alias(k)
                for k in to_reverse
            ])
            print(f"  Reoriented {len(to_reverse)} items: {to_reverse}")

    data = data.with_row_index("person")

    if polars_out:
        return data, num_people

    dataset = grain.MapDataset.source(_ArrayDataSource(data))
    return dataset, num_people

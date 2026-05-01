"""PROMIS Sleep-Wake Function data loader.

Downloads and preprocesses the PROMIS II Sleep-Wake calibration dataset
from Harvard Dataverse (doi:10.7910/DVN/XESLRZ).

The full calibration pool contains 120 items (Sleep1-Sleep128, skipping
Sleep49 and Sleep51). These span both the Sleep Disturbance (27-item)
and Sleep-Related Impairment (16-item) PROMIS banks, plus additional
candidate items from the calibration study.

5 response categories (0-4, shifted from original 1-5).

Reference: Buysse et al. (2010). "Development and validation of
patient-reported outcome measures for sleep disturbance and
sleep-related impairments." Sleep, 33(6), 781-792.
"""

import os
import urllib.request
from pathlib import Path

import grain
import polars as pl

item_keys = [f"Sleep{i}" for i in range(1, 129) if i not in (49, 51)]

response_cardinality = 5

_DATA_URL = "https://dataverse.harvard.edu/api/access/datafile/2671557"


class _ArrayDataSource(grain.sources.RandomAccessDataSource):
    def __init__(self, df: pl.DataFrame):
        self._data = {col: df[col].to_list() for col in df.columns}
        self.n = len(df)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self._data.items()}

    def __len__(self):
        return self.n


def get_data(reorient=False, polars_out=False, cache_dir=None):
    """Load the PROMIS Sleep-Wake dataset.

    Responses are shifted from 1-5 to 0-4. Missing or out-of-range
    values become -1.

    Args:
        reorient: If True, reverse-score items with negative item-total
            correlations so all items load in the same direction.
        polars_out: If True, return polars DataFrame.
        cache_dir: Directory for caching downloaded data.

    Returns:
        Tuple of (dataset, num_people).
    """
    if cache_dir is None:
        cache_dir = Path.cwd()
    else:
        cache_dir = Path(cache_dir)

    tab_path = cache_dir / "promis_sleep_wake.tab"

    if not tab_path.exists():
        os.makedirs(cache_dir, exist_ok=True)
        req = urllib.request.Request(_DATA_URL, headers={"User-Agent": "bayesianquilts"})
        with urllib.request.urlopen(req) as resp, open(str(tab_path), "wb") as f:
            f.write(resp.read())

    data = pl.read_csv(str(tab_path), separator="\t")
    data = data.select(item_keys)

    # Data is already 0-indexed (0-4). Value 8 = missing/skip code.
    data = data.with_columns([
        pl.col(k).cast(pl.Float64).alias(k) for k in item_keys
    ])

    num_people = len(data)

    # Mark invalid values (8=skip, out-of-range, null) as -1
    data = data.with_columns([
        pl.when((pl.col(k) < 0) | (pl.col(k) >= response_cardinality) | pl.col(k).is_null())
        .then(-1)
        .otherwise(pl.col(k))
        .alias(k)
        for k in item_keys
    ])

    if reorient:
        # Reverse-code items with negative item-total correlations.
        # These are positively-worded items (e.g., "My sleep was refreshing")
        # where higher = better sleep, opposite to the rest of the bank.
        import numpy as np
        arr = data.select(item_keys).to_numpy().astype(float)
        arr[arr < 0] = np.nan
        total = np.nanmean(arr, axis=1)
        to_reverse = []
        for i, k in enumerate(item_keys):
            valid = ~np.isnan(arr[:, i])
            if valid.sum() > 10:
                r = np.corrcoef(arr[valid, i], total[valid])[0, 1]
                if r < -0.05:  # negative item-total correlation
                    to_reverse.append(k)
        if to_reverse:
            K = response_cardinality
            data = data.with_columns([
                (K - 1 - pl.col(k)).alias(k) for k in to_reverse
            ])
            # Re-mark invalid values after flipping
            data = data.with_columns([
                pl.when((pl.col(k) < 0) | (pl.col(k) >= K))
                .then(-1)
                .otherwise(pl.col(k))
                .alias(k)
                for k in to_reverse
            ])
            print(f"  Reoriented {len(to_reverse)} items: {to_reverse[:5]}...")

    data = data.with_row_index("person")

    if polars_out:
        return data, num_people

    dataset = grain.MapDataset.source(_ArrayDataSource(data))
    return dataset, num_people

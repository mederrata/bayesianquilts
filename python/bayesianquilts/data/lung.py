"""Lung cancer survival data loader.

Downloads the R lung cancer dataset (228 patients, survival times with censoring)
from the Rdatasets repository, caches locally, and returns a Polars DataFrame.
"""

import os
import urllib.request
from pathlib import Path

import polars as pl

_DATA_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/survival/cancer.csv"


def get_data(cache_dir=None):
    """Load the R lung cancer dataset.

    Processing:
    - status: R uses 1=censored, 2=dead -> recode to 0/1 (event = status - 1)
    - sex_idx: recode from 1/2 to 0/1
    - age_group: 3 levels -- <=60 (0), 61-70 (1), >70 (2)
    - ecog_group: 2 levels -- ph.ecog 0-1 (0), ph.ecog 2+ (1)
    - Drop rows with null time, status, sex, age, or ph.ecog

    Returns:
        Tuple of (DataFrame, n_obs) where df has columns:
        time, event, sex_idx, age_group, ecog_group (each as int).
    """
    if cache_dir is None:
        cache_dir = Path.cwd()
    else:
        cache_dir = Path(cache_dir)

    csv_path = cache_dir / "cancer.csv"

    if not csv_path.exists():
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(_DATA_URL, str(csv_path))

    data = pl.read_csv(str(csv_path))

    # Drop rows with nulls in required columns
    data = data.drop_nulls(subset=["time", "status", "sex", "age", "ph.ecog"])

    # Recode status: R uses 1=censored, 2=dead -> 0/1
    data = data.with_columns(
        (pl.col("status") - 1).cast(pl.Int32).alias("event"),
    )

    # Recode sex: 1/2 -> 0/1
    data = data.with_columns(
        (pl.col("sex") - 1).cast(pl.Int32).alias("sex_idx"),
    )

    # Age group: <=60 (0), 61-70 (1), >70 (2)
    data = data.with_columns(
        pl.when(pl.col("age") <= 60)
        .then(0)
        .when(pl.col("age") <= 70)
        .then(1)
        .otherwise(2)
        .cast(pl.Int32)
        .alias("age_group"),
    )

    # ECOG group: 0-1 (0), 2+ (1)
    data = data.with_columns(
        pl.when(pl.col("ph.ecog") <= 1)
        .then(0)
        .otherwise(1)
        .cast(pl.Int32)
        .alias("ecog_group"),
    )

    # Cast time to int
    data = data.with_columns(
        pl.col("time").cast(pl.Int32),
    )

    # Select final columns
    data = data.select(["time", "event", "sex_idx", "age_group", "ecog_group"])

    n_obs = len(data)
    return data, n_obs

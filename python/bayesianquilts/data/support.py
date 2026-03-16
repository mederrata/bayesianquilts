"""SUPPORT2 survival data loader.

Downloads the SUPPORT2 dataset (9,105 ICU patients across 9 disease categories)
from GitHub, caches locally, and returns standardized features for neural
network survival models.

Reference: Knaus et al. (1995) The SUPPORT Prognostic Model.
"""

import os
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

_DATA_URL = (
    "https://raw.githubusercontent.com/MGensheimer/nnet-survival"
    "/master/data/support2.csv"
)

# Continuous clinical features (lab values, vitals, scores)
# Excludes surv2m/surv6m model estimates per user request
CONTINUOUS_FEATURES = [
    "age",
    "meanbp",
    "hrt",
    "resp",
    "temp",
    "crea",
    "sod",
    "ph",
    "wblc",
    "pafi",
    "alb",
    "bili",
    "bun",
    "glucose",
    "sps",
    "aps",
    "scoma",
    "num.co",
]


def _fix_shifted_csv(csv_path):
    """Fix R CSV with unnamed row index column.

    The CSV has N+1 fields per data row (leading row index) but only N headers.
    We prepend 'row_idx,' to the header to align columns.
    """
    with open(csv_path, "r") as f:
        lines = f.readlines()

    # Check if fix is needed: first data row has more commas than header
    if lines[0].count(",") < lines[1].count(","):
        lines[0] = "row_idx," + lines[0]
        with open(csv_path, "w") as f:
            f.writelines(lines)


def get_data(cache_dir=None, impute_median=True):
    """Load the SUPPORT2 dataset.

    Args:
        cache_dir: Directory to cache the CSV file.
        impute_median: If True, impute missing continuous features with
            column medians. If False, drop rows with any missing values.

    Returns:
        Dict with keys: X, time, event, feature_names, n_obs, X_mean, X_std
    """
    if cache_dir is None:
        cache_dir = Path.cwd()
    else:
        cache_dir = Path(cache_dir)

    csv_path = cache_dir / "support2.csv"

    if not csv_path.exists():
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(_DATA_URL, str(csv_path))

    _fix_shifted_csv(csv_path)

    data = pl.read_csv(str(csv_path), truncate_ragged_lines=True)

    # Drop the row index column
    if "row_idx" in data.columns:
        data = data.drop("row_idx")

    # Strip whitespace from all string columns and cast numeric-looking ones
    for col_name in data.columns:
        if data[col_name].dtype == pl.String:
            stripped = data[col_name].str.strip_chars()
            # Try casting to float; if it fails, keep as string
            try:
                data = data.with_columns(stripped.cast(pl.Float64).alias(col_name))
            except Exception:
                data = data.with_columns(stripped.alias(col_name))

    # Require time and event columns
    data = data.drop_nulls(subset=["d.time", "death"])

    # Encode categoricals as numeric
    # sex: already numeric after cast (0.0/1.0) or string "male"/"female"
    if data["sex"].dtype == pl.String:
        data = data.with_columns(
            pl.when(pl.col("sex") == "male")
            .then(1.0)
            .otherwise(0.0)
            .alias("sex_num")
        )
    else:
        data = data.with_columns(
            pl.col("sex").cast(pl.Float64).alias("sex_num")
        )

    # dzclass: encode as integers
    if data["dzclass"].dtype == pl.String:
        dzclass_vals = sorted(data["dzclass"].drop_nulls().unique().to_list())
        dzclass_map = {v: float(i) for i, v in enumerate(dzclass_vals)}
        data = data.with_columns(
            pl.col("dzclass").replace_strict(dzclass_map).alias("dzclass_num")
        )
    else:
        data = data.with_columns(
            pl.col("dzclass").cast(pl.Float64).alias("dzclass_num")
        )

    # race: encode as integers
    if data["race"].dtype == pl.String:
        race_vals = sorted(data["race"].drop_nulls().unique().to_list())
        race_map = {v: float(i) for i, v in enumerate(race_vals)}
        data = data.with_columns(
            pl.col("race").replace_strict(race_map, default=0.0).alias("race_num")
        )
    else:
        data = data.with_columns(
            pl.col("race").cast(pl.Float64).fill_null(0.0).alias("race_num")
        )

    # Build feature list
    feature_cols = []
    feature_names = []

    for c in CONTINUOUS_FEATURES:
        feature_cols.append(c)
        feature_names.append(c)

    feature_cols.extend(["sex_num", "dzclass_num", "race_num"])
    feature_names.extend(["sex", "dzclass", "race"])

    # Clinically-informed fill-in values for missing baseline physiologic data
    # (from SUPPORT study documentation)
    clinical_fill = {
        "alb": 3.5,
        "pafi": 333.3,
        "bili": 1.01,
        "crea": 1.01,
        "bun": 6.51,
        "wblc": 9.0,
    }

    if impute_median:
        # First apply clinical fill-in values where available
        for c, fill_val in clinical_fill.items():
            if c in CONTINUOUS_FEATURES:
                data = data.with_columns(
                    pl.col(c).fill_null(fill_val)
                )
        # Then impute remaining missing values with median
        for c in CONTINUOUS_FEATURES:
            if c not in clinical_fill:
                median_val = data[c].median()
                if median_val is None:
                    median_val = 0.0
                data = data.with_columns(
                    pl.col(c).fill_null(median_val)
                )
    else:
        data = data.drop_nulls(subset=CONTINUOUS_FEATURES)

    time = data["d.time"].to_numpy().astype(np.float64)
    event = data["death"].to_numpy().astype(np.float64)

    # Build feature matrix
    X = np.column_stack([
        data[c].to_numpy().astype(np.float64) for c in feature_cols
    ])

    # Standardize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    X = (X - X_mean) / X_std

    n_obs = len(time)

    return {
        "X": X,
        "time": time,
        "event": event,
        "feature_names": feature_names,
        "n_obs": n_obs,
        "X_mean": X_mean,
        "X_std": X_std,
    }

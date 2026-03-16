"""GBSG2 breast cancer survival data loader.

Loads the German Breast Cancer Study Group 2 dataset (686 patients) from
scikit-survival. Encodes categorical features and standardizes continuous
features for use with neural network and quilted survival models.

Reference: Schumacher et al. (1994) Randomized 2 × 2 trial evaluating
hormonal treatment and the duration of chemotherapy in node-positive
breast cancer patients.
"""

import numpy as np
from sksurv.datasets import load_gbsg2
from sklearn.preprocessing import OrdinalEncoder


# Continuous clinical features
CONTINUOUS_FEATURES = ["age", "tsize", "pnodes", "progrec", "estrec"]

# Categorical features (will be ordinally encoded)
CATEGORICAL_FEATURES = ["horTh", "menostat", "tgrade"]


def get_data():
    """Load the GBSG2 dataset.

    Returns
    -------
    dict with keys:
        X : ndarray, shape (N, D) — standardized feature matrix
        time : ndarray, shape (N,) — observed times (days)
        event : ndarray, shape (N,) — event indicator (1 = recurrence, 0 = censored)
        feature_names : list of str
        n_obs : int
        X_mean : ndarray, shape (D,)
        X_std : ndarray, shape (D,)
        cat_indices : dict mapping categorical name to (column_index, n_levels)
    """
    X_df, y = load_gbsg2()

    events = np.array([e for e, t in y]).astype(np.float64)
    times = np.array([t for e, t in y]).astype(np.float64)

    # Encode categoricals
    cat_encoded = {}
    cat_indices = {}
    col_idx = len(CONTINUOUS_FEATURES)
    for cat_col in CATEGORICAL_FEATURES:
        enc = OrdinalEncoder()
        vals = enc.fit_transform(X_df[[cat_col]]).ravel().astype(np.float64)
        n_levels = len(enc.categories_[0])
        cat_encoded[cat_col] = vals
        cat_indices[cat_col] = (col_idx, n_levels)
        col_idx += 1

    # Build feature matrix
    feature_names = list(CONTINUOUS_FEATURES) + list(CATEGORICAL_FEATURES)
    arrays = [X_df[c].values.astype(np.float64) for c in CONTINUOUS_FEATURES]
    arrays += [cat_encoded[c] for c in CATEGORICAL_FEATURES]
    X_raw = np.column_stack(arrays)

    # Standardize
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    X = (X_raw - X_mean) / X_std

    n_obs = len(times)

    return {
        "X": X,
        "time": times,
        "event": events,
        "feature_names": feature_names,
        "n_obs": n_obs,
        "X_mean": X_mean,
        "X_std": X_std,
        "cat_indices": cat_indices,
    }

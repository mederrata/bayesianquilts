"""Train and save models for interpretability analysis."""
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
import jax.numpy as jnp

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension
from uci_benchmarks import fit_logistic_model, evaluate_model

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def train_german_credit():
    """Train and save German Credit model."""
    print("Training German Credit model...")

    data = fetch_openml(data_id=31, as_frame=True, parser="auto")
    df = data.frame

    cat_cols = ["checking_status", "credit_history", "purpose", "savings_status",
                "employment", "personal_status", "other_parties", "property_magnitude",
                "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"]
    num_cols = ["duration", "credit_amount", "installment_commitment", "residence_since",
                "age", "existing_credits", "num_dependents"]

    X_num = df[num_cols].values.astype(np.float32)
    y = (df["class"].astype(str) == "good").astype(int)

    # Encode categoricals
    cat_encoders = {}
    cat_indices = {}
    for col in cat_cols:
        le = LabelEncoder()
        cat_indices[col] = le.fit_transform(df[col].astype(str))
        cat_encoders[col] = le

    # Train/test split
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_test = X_num[train_idx], X_num[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Select key categoricals for lattice
    selected_cats = ["checking_status", "credit_history", "savings_status", "employment"]

    # Build dimensions and indices
    dimensions = []
    train_indices = {}
    test_indices = {}

    for col in selected_cats:
        n_levels = len(cat_encoders[col].classes_)
        dimensions.append(Dimension(col, n_levels))
        train_indices[col] = cat_indices[col][train_idx]
        test_indices[col] = cat_indices[col][test_idx]

    # Add binned numeric features
    for col, n_bins in [("duration", 8), ("credit_amount", 8), ("age", 6)]:
        col_idx = num_cols.index(col)
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(X_train_s[:, col_idx], percentiles)
        train_indices[f"bin_{col}"] = np.clip(
            np.digitize(X_train_s[:, col_idx], edges[1:-1]), 0, n_bins - 1
        )
        test_indices[f"bin_{col}"] = np.clip(
            np.digitize(X_test_s[:, col_idx], edges[1:-1]), 0, n_bins - 1
        )
        dimensions.append(Dimension(f"bin_{col}", n_bins))

    # Build decomposition
    interactions = Interactions(dimensions=dimensions)
    decomp = Decomposed(
        interactions=interactions,
        param_shape=[len(num_cols)],
        name="beta"
    )

    # Prepare data
    train_data = {"X": X_train_s, "y": y_train, **train_indices}
    test_data = {"X": X_test_s, "y": y_test, **test_indices}

    # Get prior scales
    prior_scales = decomp.generalization_preserving_scales(
        noise_scale=1.0, total_n=len(train_idx), c=0.3, per_component=True
    )

    # Fit model
    params = fit_logistic_model(
        train_data, decomp, max_order=2,
        prior_scales=prior_scales, sparse=True, l1_weight=0.01, n_steps=5000
    )

    # Evaluate
    metrics = evaluate_model(test_data, decomp, params)
    print(f"  Test AUC: {metrics['auc']:.4f}")

    # Save everything
    model_data = {
        "params": {k: np.array(v) for k, v in params.items()},
        "dimensions": [(d.name, d.cardinality) for d in dimensions],
        "selected_cats": selected_cats,
        "cat_encoders": {col: list(le.classes_) for col, le in cat_encoders.items()},
        "num_cols": num_cols,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "prior_scales": prior_scales,
        "metrics": metrics,
        "decomp_info": {
            "tensor_parts": list(decomp._tensor_parts.keys()),
            "tensor_part_shapes": {k: v for k, v in decomp._tensor_part_shapes.items()},
        }
    }

    save_path = RESULTS_DIR / "german_credit_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  Saved to {save_path}")

    return model_data


def train_madelon():
    """Train and save Madelon model (high-dim example)."""
    print("Training Madelon model...")

    data = fetch_openml(data_id=1485, as_frame=True, parser="auto")
    df = data.frame

    X = df.drop(columns=["Class"]).values.astype(np.float32)
    y = (df["Class"].astype(str) == "2").astype(int)

    # Train/test split
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # LR for feature selection
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_train_s, y_train)
    coefs = np.abs(lr.coef_[0])
    top_idx = np.argsort(coefs)[::-1]

    n_reg = 15  # Regression features
    n_lat = 10  # Lattice features
    n_bins = 5

    reg_features = top_idx[:n_reg]
    lattice_features = top_idx[:n_lat]

    X_train_sub = X_train_s[:, reg_features]
    X_test_sub = X_test_s[:, reg_features]

    # Build lattice
    dimensions = []
    train_indices = {}
    test_indices = {}

    for i, feat in enumerate(lattice_features):
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(X_train_s[:, feat], percentiles)
        train_indices[f"f{i}"] = np.clip(
            np.digitize(X_train_s[:, feat], edges[1:-1]), 0, n_bins - 1
        )
        test_indices[f"f{i}"] = np.clip(
            np.digitize(X_test_s[:, feat], edges[1:-1]), 0, n_bins - 1
        )
        dimensions.append(Dimension(f"f{i}", n_bins))

    interactions = Interactions(dimensions=dimensions)
    decomp = Decomposed(
        interactions=interactions,
        param_shape=[n_reg],
        name="beta"
    )

    train_data = {"X": X_train_sub, "y": y_train, **train_indices}
    test_data = {"X": X_test_sub, "y": y_test, **test_indices}

    prior_scales = decomp.generalization_preserving_scales(
        noise_scale=1.0, total_n=len(train_idx), c=0.3, per_component=True
    )

    params = fit_logistic_model(
        train_data, decomp, max_order=2,
        prior_scales=prior_scales, sparse=True, l1_weight=0.01, n_steps=5000
    )

    metrics = evaluate_model(test_data, decomp, params)
    print(f"  Test AUC: {metrics['auc']:.4f}")

    model_data = {
        "params": {k: np.array(v) for k, v in params.items()},
        "dimensions": [(d.name, d.cardinality) for d in dimensions],
        "reg_features": list(reg_features),
        "lattice_features": list(lattice_features),
        "lr_coefs": coefs,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "prior_scales": prior_scales,
        "metrics": metrics,
        "decomp_info": {
            "tensor_parts": list(decomp._tensor_parts.keys()),
            "tensor_part_shapes": {k: v for k, v in decomp._tensor_part_shapes.items()},
        }
    }

    save_path = RESULTS_DIR / "madelon_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  Saved to {save_path}")

    return model_data


if __name__ == "__main__":
    train_german_credit()
    train_madelon()
    print("\nDone! Models saved to:", RESULTS_DIR)

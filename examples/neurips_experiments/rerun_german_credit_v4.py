"""Rerun German Credit - single lattice dim, proper regularization."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
import jax.numpy as jnp

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def fit_logistic_model(data, decomp, max_order=1, prior_scales=None, n_steps=2000):
    """Fit logistic regression with decomposed parameters using gen-preserving priors."""
    import jax
    import optax

    active_components = [
        name for name in decomp._tensor_parts.keys()
        if decomp.component_order(name) <= max_order
    ]

    params = {
        name: jnp.zeros(decomp._tensor_part_shapes[name])
        for name in active_components
    }
    params["_intercept"] = jnp.zeros(1)

    dim_names = [d.name for d in decomp._interactions._dimensions]
    interaction_indices = jnp.stack(
        [jnp.array(data[name]) for name in dim_names],
        axis=-1
    )

    X = jnp.array(data["X"])
    y = jnp.array(data["y"])
    N = len(y)
    p = X.shape[1]

    def loss_fn(params):
        intercept = params.get("_intercept", jnp.zeros(1))
        model_params = {k: v for k, v in params.items() if k != "_intercept"}

        beta = decomp.lookup_flat(interaction_indices, model_params)
        logits = jnp.sum(X * beta, axis=-1) + intercept[0]

        bce = jnp.mean(jnp.logaddexp(0, logits) - y * logits)

        # L2 regularization with gen-preserving scales
        l2_reg = 0.0
        for name, param in model_params.items():
            if prior_scales and name in prior_scales:
                scale = prior_scales[name]
            else:
                scale = 1.0
            l2_reg += 0.5 * jnp.sum(param ** 2) / (scale ** 2 + 1e-6)

        return bce + l2_reg / N

    opt = optax.adam(0.01)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for i in range(n_steps):
        params, opt_state, loss = step(params, opt_state)
        if i % 500 == 0:
            print(f"    Step {i}: loss = {loss:.4f}")

    return params


def evaluate_model(data, decomp, params):
    """Evaluate model on test data."""
    dim_names = [d.name for d in decomp._interactions._dimensions]
    interaction_indices = jnp.stack(
        [jnp.array(data[name]) for name in dim_names],
        axis=-1
    )

    X = jnp.array(data["X"])
    y = np.array(data["y"])

    intercept = params.get("_intercept", jnp.zeros(1))
    model_params = {k: v for k, v in params.items() if k != "_intercept"}

    beta = decomp.lookup_flat(interaction_indices, model_params)
    logits = jnp.sum(X * beta, axis=-1) + intercept[0]
    probs = 1 / (1 + jnp.exp(-logits))

    auc = roc_auc_score(y, np.array(probs))
    return {"auc": auc}


def run_german_credit():
    """German Credit with 1-dim lattice only."""
    print("\n" + "="*60)
    print("GERMAN CREDIT - V4 (1-dim lattice: checking_status)")
    print("="*60)

    data = fetch_openml(data_id=31, as_frame=True, parser="auto")
    df = data.frame

    # Only the most important categorical for lattice
    lattice_cat = "checking_status"

    # All other categoricals as one-hot features
    other_cats = ["credit_history", "purpose", "savings_status", "employment",
                  "personal_status", "other_parties", "property_magnitude",
                  "housing", "job"]

    # Numeric features
    num_cols = ["duration", "credit_amount", "installment_commitment", "residence_since",
                "age", "existing_credits", "num_dependents"]

    y = (df["class"].astype(str) == "good").astype(int).values

    # Encode lattice categorical
    le = LabelEncoder()
    cat_idx = le.fit_transform(df[lattice_cat].astype(str))
    n_levels = len(le.classes_)
    print(f"  Lattice: {lattice_cat} = {n_levels} levels")
    print(f"  Classes: {list(le.classes_)}")

    # One-hot encode other categoricals
    ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    X_other_cats = ohe.fit_transform(df[other_cats].astype(str))
    print(f"  Other categoricals one-hot: {X_other_cats.shape[1]} features")

    # Numeric features
    X_num = df[num_cols].values.astype(np.float32)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_num, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        # Combine numeric + one-hot categorical features
        X_train = np.hstack([X_num[train_idx], X_other_cats[train_idx]])
        X_test = np.hstack([X_num[test_idx], X_other_cats[test_idx]])
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale all features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        n_features = X_train_s.shape[1]
        print(f"    Features: {n_features}")

        # Build 1-dim lattice
        dimensions = [Dimension(lattice_cat, n_levels)]
        train_indices = {lattice_cat: cat_idx[train_idx]}
        test_indices = {lattice_cat: cat_idx[test_idx]}

        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(interactions=interactions, param_shape=[n_features], name="beta")

        # Get generalization-preserving prior scales
        prior_scales = decomp.generalization_preserving_scales(
            noise_scale=0.5,  # Logistic approx
            total_n=len(train_idx),
            c=0.3,
            per_component=True
        )
        print(f"    Prior scales: {[(k, f'{v:.3f}') for k, v in prior_scales.items()]}")

        train_data = {"X": X_train_s, "y": y_train, **train_indices}
        test_data = {"X": X_test_s, "y": y_test, **test_indices}

        # Fit with order-1 (global + checking_status main effect)
        params = fit_logistic_model(
            train_data, decomp, max_order=1,
            prior_scales=prior_scales, n_steps=2000
        )

        metrics = evaluate_model(test_data, decomp, params)
        aucs.append(metrics["auc"])
        print(f"    AUC: {metrics['auc']:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (v4): {mean_auc:.4f} +/- {std_auc:.4f}")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_german_credit()

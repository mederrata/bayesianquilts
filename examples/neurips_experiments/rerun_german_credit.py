"""Rerun German Credit with better lattice structure using native categoricals."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
import jax.numpy as jnp

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def fit_logistic_model(data, decomp, max_order=2, prior_scales=None, l1_weight=0.005, n_steps=3000):
    """Fit logistic regression with decomposed parameters."""
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

    def loss_fn(params):
        intercept = params.get("_intercept", jnp.zeros(1))
        model_params = {k: v for k, v in params.items() if k != "_intercept"}

        beta = decomp.lookup_flat(interaction_indices, model_params)
        logits = jnp.sum(X * beta, axis=-1) + intercept[0]

        bce = jnp.mean(jnp.logaddexp(0, logits) - y * logits)

        l2_reg = 0.0
        for name, param in model_params.items():
            scale = prior_scales.get(name, 1.0) if prior_scales else 1.0
            l2_reg += 0.5 * jnp.sum(param ** 2) / (scale ** 2)

        l1_reg = 0.0
        for name, param in model_params.items():
            order = decomp.component_order(name)
            if order > 0:
                l1_reg += l1_weight * order * jnp.sum(jnp.abs(param))

        return bce + l2_reg / N + l1_reg / N

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
        if i % 1000 == 0:
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
    """German Credit with native categorical lattice structure."""
    print("\n" + "="*60)
    print("GERMAN CREDIT - IMPROVED (N=1000)")
    print("="*60)

    data = fetch_openml(data_id=31, as_frame=True, parser="auto")
    df = data.frame

    # Key categorical features for lattice (most predictive based on domain knowledge)
    lattice_cats = ["checking_status", "credit_history", "savings_status", "employment"]

    # All numeric features
    num_cols = ["duration", "credit_amount", "installment_commitment", "residence_since",
                "age", "existing_credits", "num_dependents"]

    y = (df["class"].astype(str) == "good").astype(int).values
    X_num = df[num_cols].values.astype(np.float32)

    # Encode lattice categoricals
    cat_encoders = {}
    cat_indices = {}
    for col in lattice_cats:
        le = LabelEncoder()
        cat_indices[col] = le.fit_transform(df[col].astype(str))
        cat_encoders[col] = le
        print(f"  {col}: {len(le.classes_)} levels")

    print(f"\nN = {len(y)}, p = {len(num_cols)}")
    print(f"Lattice dimensions: {[len(cat_encoders[c].classes_) for c in lattice_cats]}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_num, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")
        X_train, X_test = X_num[train_idx], X_num[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale numeric features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Build lattice dimensions from categorical features
        dimensions = []
        train_indices = {}
        test_indices = {}

        for col in lattice_cats:
            n_levels = len(cat_encoders[col].classes_)
            dimensions.append(Dimension(col, n_levels))
            train_indices[col] = cat_indices[col][train_idx]
            test_indices[col] = cat_indices[col][test_idx]

        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(interactions=interactions, param_shape=[len(num_cols)], name="beta")

        # Get generalization-preserving prior scales
        prior_scales = decomp.generalization_preserving_scales(
            noise_scale=1.0, total_n=len(train_idx), c=0.3, per_component=True
        )

        train_data = {"X": X_train_s, "y": y_train, **train_indices}
        test_data = {"X": X_test_s, "y": y_test, **test_indices}

        # Fit with order-2 interactions
        params = fit_logistic_model(
            train_data, decomp, max_order=2,
            prior_scales=prior_scales, l1_weight=0.005, n_steps=3000
        )

        metrics = evaluate_model(test_data, decomp, params)
        aucs.append(metrics["auc"])
        print(f"    AUC: {metrics['auc']:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (improved): {mean_auc:.4f} +/- {std_auc:.4f}")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_german_credit()

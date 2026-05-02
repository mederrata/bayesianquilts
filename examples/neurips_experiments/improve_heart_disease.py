"""Heart Disease: theory-based piecewise model using natural categorical structure."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def run_heart_disease():
    """Heart Disease using natural categorical structure for lattice."""
    print("\n" + "="*60)
    print("HEART DISEASE - Natural Categorical Lattice")
    print("="*60)

    data = fetch_openml(data_id=53, as_frame=True, parser="auto")
    df = data.frame

    # Use natural categoricals for lattice: chest pain type (4) × thal (3) = 12 cells
    # These are clinically meaningful groupings
    lattice_cats = ["chest", "thal"]

    # Continuous features for global regression
    cont_cols = ["age", "resting_blood_pressure", "serum_cholestoral",
                 "maximum_heart_rate_achieved", "oldpeak"]

    # Binary features
    binary_cols = ["sex", "fasting_blood_sugar", "exercise_induced_angina"]

    # Other categoricals (as dummies)
    other_cats = ["resting_electrocardiographic_results", "slope", "number_of_major_vessels"]

    y = (df["class"].astype(str) == "present").astype(int).values
    N = len(y)

    class_balance = y.mean()
    avg_fisher_weight = class_balance * (1 - class_balance)
    sigma_eff = 1 / np.sqrt(avg_fisher_weight)
    print(f"  N = {N}")
    print(f"  Class balance: {class_balance:.3f}")
    print(f"  Effective σ: {sigma_eff:.3f}")

    # Encode lattice categoricals
    cat_encoders = {}
    cat_indices = {}
    for col in lattice_cats:
        le = LabelEncoder()
        cat_indices[col] = le.fit_transform(df[col].astype(str))
        cat_encoders[col] = le
        print(f"  Lattice dim: {col} = {len(le.classes_)} levels")

    n_chest = len(cat_encoders["chest"].classes_)
    n_thal = len(cat_encoders["thal"].classes_)
    total_cells = n_chest * n_thal
    print(f"\n  Total lattice cells: {total_cells}")
    print(f"  Bin constraint check: cells={total_cells} < N={N} ✓")

    # Build feature matrix: continuous + binary + one-hot other cats
    X_cont = df[cont_cols].values.astype(np.float32)
    X_binary = df[binary_cols].values.astype(np.float32)

    # One-hot encode other categoricals
    X_other = []
    for col in other_cats:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col].astype(str))
        n_levels = len(le.classes_)
        one_hot = np.eye(n_levels)[encoded][:, 1:]  # drop first for identifiability
        X_other.append(one_hot)
    X_other = np.hstack(X_other) if X_other else np.zeros((N, 0))

    X_all = np.hstack([X_cont, X_binary, X_other]).astype(np.float32)
    print(f"  Regression features: {X_all.shape[1]}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_all, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        N_train = len(y_train)
        n_features = X_train_s.shape[1]

        # Build lattice using natural categoricals
        dimensions = []
        train_lat_indices = {}
        test_lat_indices = {}

        for col in lattice_cats:
            n_levels = len(cat_encoders[col].classes_)
            dimensions.append(Dimension(col, n_levels))
            train_lat_indices[col] = cat_indices[col][train_idx]
            test_lat_indices[col] = cat_indices[col][test_idx]

        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(interactions=interactions, param_shape=[1], name="intercept")

        # Theory-based priors: τ ≤ σ/√(p·N^(α)) with bound factor
        c = 0.5
        bound_factor = np.sqrt(c / (1 - c))

        # Count observations per cell and per level
        cell_counts = np.zeros((n_chest, n_thal))
        chest_counts = np.bincount(cat_indices["chest"][train_idx], minlength=n_chest)
        thal_counts = np.bincount(cat_indices["thal"][train_idx], minlength=n_thal)

        for i in range(len(train_idx)):
            ci = cat_indices["chest"][train_idx[i]]
            ti = cat_indices["thal"][train_idx[i]]
            cell_counts[ci, ti] += 1

        tau_global = bound_factor * sigma_eff / np.sqrt(N_train)
        tau_chest = [bound_factor * sigma_eff / np.sqrt(max(n, 1)) for n in chest_counts]
        tau_thal = [bound_factor * sigma_eff / np.sqrt(max(n, 1)) for n in thal_counts]

        print(f"    τ_global: {tau_global:.4f}")

        # Build interaction indices
        dim_names = [d.name for d in decomp._interactions._dimensions]
        train_int_idx = jnp.stack([jnp.array(train_lat_indices[name]) for name in dim_names], axis=-1)
        test_int_idx = jnp.stack([jnp.array(test_lat_indices[name]) for name in dim_names], axis=-1)

        X_train_j = jnp.array(X_train_s)
        X_test_j = jnp.array(X_test_s)
        y_train_j = jnp.array(y_train)

        # Initialize parameters - use order 2 (full interactions) since small lattice
        active_components = [name for name in decomp._tensor_parts.keys() if decomp.component_order(name) <= 2]
        intercept_params = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active_components}
        beta = jnp.zeros(n_features)

        params = {"intercept": intercept_params, "beta": beta}

        # Prior variances per component
        prior_vars = {}
        for name in active_components:
            order = decomp.component_order(name)
            if order == 0:
                prior_vars[name] = tau_global ** 2
            elif order == 1:
                # Use average tau for the relevant dimension
                if "chest" in name:
                    prior_vars[name] = np.mean([t**2 for t in tau_chest])
                else:
                    prior_vars[name] = np.mean([t**2 for t in tau_thal])
            else:
                # Order 2: use minimum cell count
                min_cell = max(cell_counts.min(), 1)
                tau_int = bound_factor * sigma_eff / np.sqrt(max(min_cell, 1))
                prior_vars[name] = tau_int ** 2

        tau_beta = bound_factor * sigma_eff / np.sqrt(N_train)
        print(f"    τ_beta: {tau_beta:.4f}")

        def loss_fn(params):
            int_vals = decomp.lookup_flat(train_int_idx, params["intercept"])
            intercept = int_vals[:, 0]
            logits = jnp.sum(X_train_j * params["beta"], axis=-1) + intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_int = 0.0
            for name, param in params["intercept"].items():
                var = prior_vars.get(name, 1.0)
                l2_int += 0.5 * jnp.sum(param ** 2) / (var + 1e-8)

            l2_beta = 0.5 * jnp.sum(params["beta"] ** 2) / (tau_beta ** 2 + 1e-8)

            return bce + (l2_int + l2_beta) / N_train

        opt = optax.adam(0.01)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for i in range(3000):
            params, opt_state, loss = step(params, opt_state)
            if i % 1000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}")

        # Evaluate
        int_vals = decomp.lookup_flat(test_int_idx, params["intercept"])
        intercept = int_vals[:, 0]
        logits = jnp.sum(X_test_j * params["beta"], axis=-1) + intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (theory-based): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Previous: 0.788, EBM: 0.892")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_heart_disease()

"""Heart: use categorical variables as lattice dimensions."""
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


def run_heart_categorical():
    """Heart with categorical variables as lattice dimensions."""
    print("\n" + "="*60)
    print("HEART - Categorical Variables as Lattice Dims")
    print("="*60)

    data = fetch_openml(data_id=53, as_frame=True, parser="auto")
    df = data.frame.copy()

    # Target encoding
    y = (df["class"].astype(str) == "2").astype(int).values

    # Categorical variables for lattice
    categorical_cols = ["sex", "chest", "fasting_blood_sugar",
                        "resting_electrocardiographic_results",
                        "exercise_induced_angina", "slope",
                        "number_of_major_vessels", "thal"]

    # Numeric variables for regression
    numeric_cols = ["age", "resting_blood_pressure", "serum_cholestoral",
                   "maximum_heart_rate_achieved", "oldpeak"]

    # Filter to columns that exist
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    print(f"  Categorical (lattice): {categorical_cols}")
    print(f"  Numeric (regression): {numeric_cols}")

    N = len(df)
    print(f"  N = {N}")

    class_balance = y.mean()
    avg_fisher_weight = class_balance * (1 - class_balance)
    sigma_eff = 1 / np.sqrt(avg_fisher_weight)
    print(f"  Class balance: {class_balance:.3f}")
    print(f"  Effective σ: {sigma_eff:.3f}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(df, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        y_train, y_test = y[train_idx], y[test_idx]
        N_train = len(y_train)

        # Build lattice from categorical variables
        dimensions = []
        train_indices = {}
        test_indices = {}

        # Use top 2-3 categorical vars to keep lattice manageable
        # Select based on number of unique values (fewer is better for small N)
        cat_cardinalities = {}
        for col in categorical_cols:
            vals = df[col].values
            unique_vals = np.unique(vals[~np.isnan(vals.astype(float))] if np.issubdtype(vals.dtype, np.floating) else vals)
            cat_cardinalities[col] = len(unique_vals)

        # Sort by cardinality (use smaller cardinality first)
        sorted_cats = sorted(cat_cardinalities.items(), key=lambda x: x[1])

        # Compute max lattice dims based on critical condition
        # N_train / (prod of cardinalities) should be >> 1
        # For small N, be very conservative
        total_cells = 1
        selected_cats = []
        for col, card in sorted_cats:
            if total_cells * card > N_train / 20:  # Need at least 20 samples per cell
                break
            total_cells *= card
            selected_cats.append(col)
            if len(selected_cats) >= 2:  # Limit to 2 dims for small N
                break

        print(f"    Selected lattice dims: {selected_cats} ({total_cells} cells)")

        for col in selected_cats:
            le = LabelEncoder()
            # Handle missing values
            col_vals = df[col].fillna(-999).values
            le.fit(col_vals)

            train_vals = le.transform(df.iloc[train_idx][col].fillna(-999).values)
            test_vals = le.transform(df.iloc[test_idx][col].fillna(-999).values)

            cardinality = len(le.classes_)
            train_indices[col] = train_vals
            test_indices[col] = test_vals
            dimensions.append(Dimension(col, cardinality))

        # Numeric features - standardize and one-hot encode bins for shape functions
        if numeric_cols:
            X_numeric = df[numeric_cols].values.astype(np.float32)
            X_numeric = np.nan_to_num(X_numeric, nan=0.0)

            scaler = StandardScaler()
            X_train_num = scaler.fit_transform(X_numeric[train_idx])
            X_test_num = scaler.transform(X_numeric[test_idx])
        else:
            X_train_num = np.zeros((N_train, 0))
            X_test_num = np.zeros((len(test_idx), 0))

        n_num_features = X_train_num.shape[1]

        # Create decomposition
        if dimensions:
            interactions = Interactions(dimensions=dimensions)
            decomp = Decomposed(interactions=interactions, param_shape=[1], name="intercept")

            # Build interaction indices
            dim_names = [d.name for d in decomp._interactions._dimensions]
            train_int_idx = jnp.stack([jnp.array(train_indices[name]) for name in dim_names], axis=-1)
            test_int_idx = jnp.stack([jnp.array(test_indices[name]) for name in dim_names], axis=-1)
        else:
            decomp = None
            train_int_idx = None
            test_int_idx = None

        X_train_j = jnp.array(X_train_num)
        X_test_j = jnp.array(X_test_num)
        y_train_j = jnp.array(y_train)

        # Theory-based priors - use tighter bound for small N
        c = 0.3  # Tighter regularization
        bound_factor = np.sqrt(c / (1 - c))

        # Initialize parameters
        if decomp is not None:
            # Include up to order 2 interactions
            active_components = [name for name in decomp._tensor_parts.keys()
                               if decomp.component_order(name) <= 2]
            intercept_params = {name: jnp.zeros(decomp._tensor_part_shapes[name])
                              for name in active_components}

            # Prior variances - use small fixed variance for small N
            prior_vars = {}
            base_tau = 0.5  # Small fixed prior std
            for name in active_components:
                prior_vars[name] = base_tau ** 2
        else:
            active_components = []
            intercept_params = {}
            prior_vars = {}

        # Linear coefficients - strong regularization
        beta = jnp.zeros(n_num_features) if n_num_features > 0 else jnp.array([])
        tau_beta = 0.5  # Small prior std

        params = {"intercept": intercept_params, "beta": beta}

        print(f"    Active components: {len(active_components)}, numeric features: {n_num_features}")

        def loss_fn(params):
            if decomp is not None and train_int_idx is not None:
                int_vals = decomp.lookup_flat(train_int_idx, params["intercept"])
                logits = int_vals[:, 0]
            else:
                logits = jnp.zeros(N_train)

            if n_num_features > 0:
                logits = logits + jnp.sum(X_train_j * params["beta"], axis=-1)

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # Regularization
            l2_int = 0.0
            for name, param in params["intercept"].items():
                tau_sq = prior_vars.get(name, 1.0)
                l2_int += 0.5 * jnp.sum(param ** 2) / (tau_sq + 1e-8)

            l2_beta = 0.5 * jnp.sum(params["beta"] ** 2) / (tau_beta ** 2 + 1e-8) if n_num_features > 0 else 0.0

            return bce + (l2_int + l2_beta) / N_train

        opt = optax.adam(0.02)
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
        if decomp is not None and test_int_idx is not None:
            int_vals = decomp.lookup_flat(test_int_idx, params["intercept"])
            logits = int_vals[:, 0]
        else:
            logits = jnp.zeros(len(y_test))

        if n_num_features > 0:
            logits = logits + jnp.sum(X_test_j * params["beta"], axis=-1)

        probs = 1 / (1 + jnp.exp(-logits))
        probs_np = np.array(probs)

        print(f"    Probs min/max: {probs_np.min():.4f}/{probs_np.max():.4f}")
        print(f"    Test y: {y_test.sum()}/{len(y_test)}")

        try:
            auc = roc_auc_score(y_test, probs_np)
        except ValueError as e:
            print(f"    AUC error: {e}")
            auc = 0.5
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (categorical lattice): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Table: 0.907, EBM: 0.892")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_heart_categorical()

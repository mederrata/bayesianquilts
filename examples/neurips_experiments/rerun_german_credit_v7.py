"""German Credit: proper generalization-preserving regularization from the paper."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def run_german_credit():
    """German Credit with theory-based regularization."""
    print("\n" + "="*60)
    print("GERMAN CREDIT - V7")
    print("Theory: τ ≤ σ/√(p·N^(α)) with σ_eff = 2 for balanced logistic")
    print("="*60)

    data = fetch_openml(data_id=31, as_frame=True, parser="auto")
    df = data.frame

    # Key categoricals for lattice
    lattice_cats = ["checking_status", "credit_history"]

    # ALL categoricals for one-hot regression features
    all_cats = ["checking_status", "credit_history", "purpose", "savings_status",
                "employment", "personal_status", "other_parties", "property_magnitude",
                "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"]

    # Continuous features
    num_cols = ["duration", "credit_amount", "installment_commitment", "residence_since",
                "age", "existing_credits", "num_dependents"]

    y = (df["class"].astype(str) == "good").astype(int).values
    class_balance = y.mean()
    print(f"  Class balance (good): {class_balance:.3f}")

    # Fisher weight for logistic: w = μ(1-μ), avg ≈ class_balance * (1 - class_balance)
    avg_fisher_weight = class_balance * (1 - class_balance)
    sigma_eff = 1 / np.sqrt(avg_fisher_weight)
    print(f"  Avg Fisher weight: {avg_fisher_weight:.3f}")
    print(f"  Effective σ: {sigma_eff:.3f}")

    # Encode lattice categoricals
    cat_encoders = {}
    cat_indices = {}
    for col in lattice_cats:
        le = LabelEncoder()
        cat_indices[col] = le.fit_transform(df[col].astype(str))
        cat_encoders[col] = le
        print(f"  Lattice dim: {col} = {len(le.classes_)} levels")

    # One-hot encode ALL categoricals
    ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    X_cat_ohe = ohe.fit_transform(df[all_cats].astype(str))

    # Continuous features
    X_num = df[num_cols].values.astype(np.float32)

    n_checking = len(cat_encoders["checking_status"].classes_)
    n_credit = len(cat_encoders["credit_history"].classes_)
    total_cells = n_checking * n_credit
    print(f"\n  Total lattice cells: {total_cells}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_num, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        # Combine one-hot + continuous
        X_train = np.hstack([X_cat_ohe[train_idx], X_num[train_idx]])
        X_test = np.hstack([X_cat_ohe[test_idx], X_num[test_idx]])
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        n_features = X_train_s.shape[1]
        N_train = len(y_train)

        # Build lattice for intercept only
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

        # Compute N^(α) for each component and derive τ^(α)
        # For intercept, p = 1
        p_int = 1
        c = 0.5  # bound df_eff ≤ c

        # Count observations per level for main effects
        check_counts = np.bincount(cat_indices["checking_status"][train_idx], minlength=n_checking)
        credit_counts = np.bincount(cat_indices["credit_history"][train_idx], minlength=n_credit)

        # Compute interaction counts
        interaction_counts = np.zeros((n_checking, n_credit))
        for i in range(len(train_idx)):
            ci = cat_indices["checking_status"][train_idx[i]]
            cr = cat_indices["credit_history"][train_idx[i]]
            interaction_counts[ci, cr] += 1

        # Compute τ for each component using τ ≤ σ/√(p·N^(α)) with bound factor
        # For df_eff ≤ c, we need τ² ≤ c/(1-c) · σ²/(N^(α))
        bound_factor = np.sqrt(c / (1 - c))

        tau_global = bound_factor * sigma_eff / np.sqrt(p_int * N_train)
        tau_checking = [bound_factor * sigma_eff / np.sqrt(p_int * max(n, 1)) for n in check_counts]
        tau_credit = [bound_factor * sigma_eff / np.sqrt(p_int * max(n, 1)) for n in credit_counts]

        print(f"    τ_global: {tau_global:.4f}")
        print(f"    τ_checking: {[f'{t:.3f}' for t in tau_checking]}")
        print(f"    τ_credit: {[f'{t:.3f}' for t in tau_credit]}")

        # Build interaction indices
        dim_names = [d.name for d in decomp._interactions._dimensions]
        train_int_idx = jnp.stack([jnp.array(train_lat_indices[name]) for name in dim_names], axis=-1)
        test_int_idx = jnp.stack([jnp.array(test_lat_indices[name]) for name in dim_names], axis=-1)

        X_train_j = jnp.array(X_train_s)
        X_test_j = jnp.array(X_test_s)
        y_train_j = jnp.array(y_train)

        # Initialize parameters - order 1 only (main effects)
        active_components = [name for name in decomp._tensor_parts.keys() if decomp.component_order(name) <= 1]
        intercept_params = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active_components}
        beta = jnp.zeros(n_features)

        params = {"intercept": intercept_params, "beta": beta}

        # Precompute prior variances (τ²) for each component
        prior_vars = {}
        for name in active_components:
            order = decomp.component_order(name)
            if order == 0:
                prior_vars[name] = tau_global ** 2
            elif order == 1:
                # Average over levels
                if "checking" in name:
                    prior_vars[name] = np.mean([t**2 for t in tau_checking])
                else:
                    prior_vars[name] = np.mean([t**2 for t in tau_credit])
            else:
                # Order 2: use minimum cell count
                min_cell = max(interaction_counts.min(), 1)
                tau_int = bound_factor * sigma_eff / np.sqrt(p_int * min_cell)
                prior_vars[name] = tau_int ** 2

        # For beta: global coefficients, use τ ≤ σ/√N per parameter
        # (not per-component since these are shared globally)
        tau_beta = bound_factor * sigma_eff / np.sqrt(N_train)
        print(f"    τ_beta (global coefficients): {tau_beta:.4f}")

        def loss_fn(params):
            # Cell-varying intercept
            int_vals = decomp.lookup_flat(train_int_idx, params["intercept"])
            intercept = int_vals[:, 0]

            # Global coefficients
            logits = jnp.sum(X_train_j * params["beta"], axis=-1) + intercept

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # L2 on intercept components with theory-based priors
            l2_int = 0.0
            for name, param in params["intercept"].items():
                var = prior_vars.get(name, 1.0)
                l2_int += 0.5 * jnp.sum(param ** 2) / (var + 1e-8)

            # L2 on global beta
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
    print(f"\n  OURS (v7 - theory): {mean_auc:.4f} +/- {std_auc:.4f}")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_german_credit()

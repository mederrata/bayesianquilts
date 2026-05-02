"""German Credit Ours #2 with Boosting + Theory-based regularization."""
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


def run_german_boosting():
    """German Credit Ours #2 with boosting + theory."""
    print("\n" + "="*60)
    print("GERMAN CREDIT - OURS #2 WITH BOOSTING")
    print("Theory-based regularization + residual fitting")
    print("="*60)

    data = fetch_openml(data_id=31, as_frame=True, parser="auto")
    df = data.frame

    cat_cols = ["checking_status", "credit_history", "purpose", "savings_status",
                "employment", "personal_status", "other_parties", "property_magnitude",
                "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"]
    num_cols = ["duration", "credit_amount", "installment_commitment", "residence_since",
                "age", "existing_credits", "num_dependents"]

    y = (df["class"].astype(str) == "good").astype(int).values
    N = len(y)
    print(f"  N = {N}, pos_rate = {y.mean():.3f}")

    # Keep same lattice dims as Ours #1: checking_status, credit_history
    lattice_cats = ["checking_status", "credit_history"]

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(df, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        y_train, y_test = y[train_idx], y[test_idx]
        N_train = len(y_train)

        # One-hot encode categoricals
        ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        X_cat = ohe.fit_transform(df[cat_cols].astype(str).iloc[train_idx])
        X_cat_test = ohe.transform(df[cat_cols].astype(str).iloc[test_idx])

        # Standardize numerics
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df[num_cols].iloc[train_idx].values.astype(np.float32))
        X_num_test = scaler.transform(df[num_cols].iloc[test_idx].values.astype(np.float32))

        X_train = np.hstack([X_cat, X_num])
        X_test = np.hstack([X_cat_test, X_num_test])
        n_features = X_train.shape[1]

        # Encode lattice indices
        cat_n_levels = {}
        cat_indices_train = {}
        cat_indices_test = {}
        for col in lattice_cats:
            le = LabelEncoder()
            cat_indices_train[col] = le.fit_transform(df[col].astype(str).iloc[train_idx])
            cat_indices_test[col] = le.transform(df[col].astype(str).iloc[test_idx])
            cat_n_levels[col] = len(le.classes_)

        # Build intercept lattice
        dims_int = [Dimension(col, cat_n_levels[col]) for col in lattice_cats]
        decomp_int = Decomposed(interactions=Interactions(dimensions=dims_int), param_shape=[1], name="intercept")

        # Build beta lattice (same dims for simplicity)
        decomp_beta = Decomposed(interactions=Interactions(dimensions=dims_int), param_shape=[n_features], name="beta")

        # Index arrays
        train_idx_arr = np.stack([cat_indices_train[col] for col in lattice_cats], axis=-1)
        test_idx_arr = np.stack([cat_indices_test[col] for col in lattice_cats], axis=-1)
        train_idx_j = jnp.array(train_idx_arr)
        test_idx_j = jnp.array(test_idx_arr)

        # Theory-based regularization scales
        prior_scales_int = decomp_int.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)
        prior_scales_beta = decomp_beta.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)

        # Limit to order 2
        active_int = [n for n in decomp_int._tensor_parts.keys() if decomp_int.component_order(n) <= 2]
        active_beta = [n for n in decomp_beta._tensor_parts.keys() if decomp_beta.component_order(n) <= 2]

        # Initialize
        params_int = {n: jnp.zeros(decomp_int._tensor_part_shapes[n]) for n in active_int}
        params_beta = {n: jnp.zeros(decomp_beta._tensor_part_shapes[n]) for n in active_beta}

        X_train_j = jnp.array(X_train)
        X_test_j = jnp.array(X_test)
        y_train_j = jnp.array(y_train)

        scale_mult = 50.0
        learning_rate_boost = 0.1
        n_boost_rounds = 10

        # BOOSTING
        accumulated_logits = jnp.zeros(N_train)

        for boost_round in range(n_boost_rounds):
            current_probs = 1 / (1 + jnp.exp(-accumulated_logits))
            residuals = y_train_j - current_probs

            # Fit intercept to residuals
            params_int_round = {n: jnp.zeros(decomp_int._tensor_part_shapes[n]) for n in active_int}

            def loss_int(p):
                vals = decomp_int.lookup_flat(train_idx_j, p)[:, 0]
                mse = jnp.mean((residuals - vals)**2)
                l2 = sum(0.5 * jnp.sum(p[n]**2) / ((prior_scales_int.get(n, 1.0) * scale_mult)**2 + 1e-8)
                        for n in active_int)
                return mse + l2 / N_train

            opt_i = optax.adam(0.02)
            opt_state_i = opt_i.init(params_int_round)

            @jax.jit
            def step_i(p, opt_state):
                loss, grads = jax.value_and_grad(loss_int)(p)
                updates, opt_state = opt_i.update(grads, opt_state, p)
                return optax.apply_updates(p, updates), opt_state, loss

            for _ in range(100):
                params_int_round, opt_state_i, _ = step_i(params_int_round, opt_state_i)

            for n in active_int:
                params_int[n] = params_int[n] + learning_rate_boost * params_int_round[n]

            # Update accumulated
            int_vals = decomp_int.lookup_flat(train_idx_j, params_int)[:, 0]
            accumulated_logits = int_vals

            # Fit beta to residuals
            current_probs = 1 / (1 + jnp.exp(-accumulated_logits))
            residuals = y_train_j - current_probs

            params_beta_round = {n: jnp.zeros(decomp_beta._tensor_part_shapes[n]) for n in active_beta}

            def loss_beta(p):
                beta_vals = decomp_beta.lookup_flat(train_idx_j, p)
                pred = jnp.sum(X_train_j * beta_vals, axis=-1)
                mse = jnp.mean((residuals - pred)**2)
                l2 = sum(0.5 * jnp.sum(p[n]**2) / ((prior_scales_beta.get(n, 1.0) * scale_mult)**2 + 1e-8)
                        for n in active_beta)
                return mse + l2 / N_train

            opt_b = optax.adam(0.01)
            opt_state_b = opt_b.init(params_beta_round)

            @jax.jit
            def step_b(p, opt_state):
                loss, grads = jax.value_and_grad(loss_beta)(p)
                updates, opt_state = opt_b.update(grads, opt_state, p)
                return optax.apply_updates(p, updates), opt_state, loss

            for _ in range(100):
                params_beta_round, opt_state_b, _ = step_b(params_beta_round, opt_state_b)

            for n in active_beta:
                params_beta[n] = params_beta[n] + learning_rate_boost * params_beta_round[n]

            beta_vals = decomp_beta.lookup_flat(train_idx_j, params_beta)
            accumulated_logits = int_vals + jnp.sum(X_train_j * beta_vals, axis=-1)

        print(f"    Completed {n_boost_rounds} boosting rounds")

        # Final joint refinement
        print("    Final joint refinement...")
        params = {"int": params_int, "beta": params_beta}

        def loss_joint(params):
            int_vals = decomp_int.lookup_flat(train_idx_j, params["int"])[:, 0]
            beta_vals = decomp_beta.lookup_flat(train_idx_j, params["beta"])
            logits = jnp.sum(X_train_j * beta_vals, axis=-1) + int_vals

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_int = sum(0.5 * jnp.sum(params["int"][n]**2) / ((prior_scales_int.get(n, 1.0) * scale_mult)**2 + 1e-8)
                        for n in active_int)
            l2_beta = sum(0.5 * jnp.sum(params["beta"][n]**2) / ((prior_scales_beta.get(n, 1.0) * scale_mult)**2 + 1e-8)
                         for n in active_beta)

            return bce + (l2_int + l2_beta) / N_train

        opt_joint = optax.adam(0.005)
        opt_state_joint = opt_joint.init(params)

        @jax.jit
        def step_joint(params, opt_state):
            loss, grads = jax.value_and_grad(loss_joint)(params)
            updates, opt_state = opt_joint.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for i in range(2000):
            params, opt_state_joint, loss = step_joint(params, opt_state_joint)

        # Evaluate
        int_vals = decomp_int.lookup_flat(test_idx_j, params["int"])[:, 0]
        beta_vals = decomp_beta.lookup_flat(test_idx_j, params["beta"])
        logits = jnp.sum(X_test_j * beta_vals, axis=-1) + int_vals
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS #2 (boosting): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  OURS #1 (v7): 0.7883 +/- 0.0211")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_german_boosting()

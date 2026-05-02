"""German Credit Ours #2: ReLU hinges + cyclic training on top of v7 architecture."""
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


def piecewise_linear_relu(x, splits, slopes):
    """f(x) = sum_k slopes[k] * relu(x - splits[k])"""
    hinges = jax.nn.relu(x[:, None] - splits[None, :])
    return jnp.sum(hinges * slopes[None, :], axis=-1)


def run_german_ours2():
    """German Credit Ours #2 with ReLU hinges and cyclic training."""
    print("\n" + "="*60)
    print("GERMAN CREDIT - OURS #2")
    print("ReLU hinges on continuous + cyclic training + v7 lattice")
    print("="*60)

    data = fetch_openml(data_id=31, as_frame=True, parser="auto")
    df = data.frame

    # Key categoricals for lattice (same as v7)
    lattice_cats = ["checking_status", "credit_history"]

    # ALL categoricals for one-hot regression features
    all_cats = ["checking_status", "credit_history", "purpose", "savings_status",
                "employment", "personal_status", "other_parties", "property_magnitude",
                "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"]

    # Continuous features
    num_cols = ["duration", "credit_amount", "installment_commitment", "residence_since",
                "age", "existing_credits", "num_dependents"]

    # Only apply ReLU hinges to most important continuous features (fewer params for small N)
    relu_cols = ["duration", "credit_amount", "age"]

    y = (df["class"].astype(str) == "good").astype(int).values
    class_balance = y.mean()
    print(f"  Class balance (good): {class_balance:.3f}")

    avg_fisher_weight = class_balance * (1 - class_balance)
    sigma_eff = 1 / np.sqrt(avg_fisher_weight)

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

    # Continuous features (raw, will be scaled per fold)
    X_num_raw = df[num_cols].values.astype(np.float32)

    n_checking = len(cat_encoders["checking_status"].classes_)
    n_credit = len(cat_encoders["credit_history"].classes_)
    n_splits = 3  # Fewer splits for small dataset

    print(f"\n  Total lattice cells: {n_checking * n_credit}")
    print(f"  Continuous features: {len(num_cols)} with {n_splits} ReLU splits each")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_num_raw, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        # Scale continuous features
        scaler_num = StandardScaler()
        X_num_train = scaler_num.fit_transform(X_num_raw[train_idx])
        X_num_test = scaler_num.transform(X_num_raw[test_idx])

        # Combine one-hot + continuous for linear part
        X_train = np.hstack([X_cat_ohe[train_idx], X_num_train])
        X_test = np.hstack([X_cat_ohe[test_idx], X_num_test])
        y_train, y_test = y[train_idx], y[test_idx]

        n_features = X_train.shape[1]
        N_train = len(y_train)

        # Build lattice for intercept (same as v7)
        dimensions = [Dimension(col, len(cat_encoders[col].classes_)) for col in lattice_cats]
        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(interactions=interactions, param_shape=[1], name="intercept")

        # Build lattice indices
        train_lat_indices = {col: cat_indices[col][train_idx] for col in lattice_cats}
        test_lat_indices = {col: cat_indices[col][test_idx] for col in lattice_cats}

        dim_names = [d.name for d in decomp._interactions._dimensions]
        train_int_idx = jnp.stack([jnp.array(train_lat_indices[name]) for name in dim_names], axis=-1)
        test_int_idx = jnp.stack([jnp.array(test_lat_indices[name]) for name in dim_names], axis=-1)

        # Theory-based regularization (from v7)
        c = 0.5
        bound_factor = np.sqrt(c / (1 - c))
        tau_global = bound_factor * sigma_eff / np.sqrt(N_train)

        check_counts = np.bincount(cat_indices["checking_status"][train_idx], minlength=n_checking)
        credit_counts = np.bincount(cat_indices["credit_history"][train_idx], minlength=n_credit)
        tau_checking = [bound_factor * sigma_eff / np.sqrt(max(n, 1)) for n in check_counts]
        tau_credit = [bound_factor * sigma_eff / np.sqrt(max(n, 1)) for n in credit_counts]

        # Initialize parameters
        active_components = [name for name in decomp._tensor_parts.keys() if decomp.component_order(name) <= 1]
        intercept_params = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active_components}

        # Global beta for linear part
        beta = jnp.zeros(n_features)

        # ReLU hinge parameters for each continuous feature
        splits_init = {}
        for i, col in enumerate(num_cols):
            percs = np.linspace(100/(n_splits+1), 100 - 100/(n_splits+1), n_splits)
            splits_init[col] = np.percentile(X_num_train[:, i], percs)

        relu_params = {}
        for col in relu_cols:
            relu_params[f"splits_{col}"] = jnp.array(splits_init[col])
            relu_params[f"slopes_{col}"] = jnp.zeros(n_splits)
            relu_params[f"base_{col}"] = jnp.array(0.0)

        params = {
            "intercept": intercept_params,
            "beta": beta,
            "relu": relu_params,
        }

        # Prior variances for intercept
        prior_vars = {}
        for name in active_components:
            order = decomp.component_order(name)
            if order == 0:
                prior_vars[name] = tau_global ** 2
            elif "checking" in name:
                prior_vars[name] = np.mean([t**2 for t in tau_checking])
            else:
                prior_vars[name] = np.mean([t**2 for t in tau_credit])

        tau_beta = bound_factor * sigma_eff / np.sqrt(N_train)

        # Theory-based τ for ReLU slopes: each slope has effective N = samples above split
        # Use median split, so roughly N/2 samples per slope
        tau_relu_slope = bound_factor * sigma_eff / np.sqrt(N_train / 2)
        tau_relu_base = tau_beta  # Same as global beta

        X_train_j = jnp.array(X_train)
        X_test_j = jnp.array(X_test)
        X_num_train_j = jnp.array(X_num_train)
        X_num_test_j = jnp.array(X_num_test)
        y_train_j = jnp.array(y_train)

        def loss_fn(params):
            # Cell-varying intercept from lattice
            int_vals = decomp.lookup_flat(train_int_idx, params["intercept"])
            intercept = int_vals[:, 0]

            # Linear part: X * beta
            linear = jnp.sum(X_train_j * params["beta"], axis=-1)

            # ReLU hinge contributions for selected continuous features
            relu_contrib = jnp.zeros(N_train)
            for col in relu_cols:
                col_idx = num_cols.index(col)
                x = X_num_train_j[:, col_idx]
                base = params["relu"][f"base_{col}"] * x
                hinges = piecewise_linear_relu(x, params["relu"][f"splits_{col}"], params["relu"][f"slopes_{col}"])
                relu_contrib = relu_contrib + base + hinges

            logits = intercept + linear + relu_contrib

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # L2 on intercept components
            l2_int = 0.0
            for name, param in params["intercept"].items():
                var = prior_vars.get(name, 1.0)
                l2_int += 0.5 * jnp.sum(param ** 2) / (var + 1e-8)

            # L2 on beta
            l2_beta = 0.5 * jnp.sum(params["beta"] ** 2) / (tau_beta ** 2 + 1e-8)

            # L2 on ReLU params with theory-based τ
            l2_relu = 0.0
            for col in relu_cols:
                l2_relu += 0.5 * jnp.sum(params["relu"][f"slopes_{col}"]**2) / (tau_relu_slope**2 + 1e-8)
                l2_relu += 0.5 * params["relu"][f"base_{col}"]**2 / (tau_relu_base**2 + 1e-8)

            # Split ordering penalty
            split_penalty = 0.0
            for col in relu_cols:
                splits = params["relu"][f"splits_{col}"]
                split_penalty += jnp.sum(jax.nn.relu(splits[:-1] - splits[1:] + 0.05))

            return bce + (l2_int + l2_beta + l2_relu) / N_train + 0.1 * split_penalty

        # Define parameter groups for cyclic training
        int_keys = frozenset(["intercept"])
        beta_keys = frozenset(["beta"])
        relu_keys = frozenset(["relu"])

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.001, peak_value=0.02, warmup_steps=200,
            decay_steps=800, end_value=0.005
        )
        opt = optax.adam(learning_rate=schedule)
        opt_state = opt.init(params)

        def make_masked_step(active_keys):
            @jax.jit
            def step_fn(params, opt_state):
                loss, grads = jax.value_and_grad(loss_fn)(params)
                masked_grads = {}
                for k, v in grads.items():
                    if k in active_keys:
                        masked_grads[k] = v
                    elif isinstance(v, dict):
                        masked_grads[k] = {kk: jnp.zeros_like(vv) for kk, vv in v.items()}
                    else:
                        masked_grads[k] = jnp.zeros_like(v)
                updates, new_opt_state = opt.update(masked_grads, opt_state, params)
                return optax.apply_updates(params, updates), new_opt_state, loss
            return step_fn

        step_int = make_masked_step(int_keys)
        step_beta = make_masked_step(beta_keys)
        step_relu = make_masked_step(relu_keys)

        # Cyclic training
        n_cycles = 5
        steps_per_group = 200

        for cycle in range(n_cycles):
            for step_fn in [step_int, step_beta, step_relu]:
                for _ in range(steps_per_group):
                    params, opt_state, loss = step_fn(params, opt_state)

        # Final joint optimization
        @jax.jit
        def step_all(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for i in range(1000):
            params, opt_state, loss = step_all(params, opt_state)
            if i % 500 == 0:
                print(f"    Joint step {i}: loss = {loss:.4f}")

        # Evaluate
        int_vals = decomp.lookup_flat(test_int_idx, params["intercept"])
        intercept = int_vals[:, 0]
        linear = jnp.sum(X_test_j * params["beta"], axis=-1)

        relu_contrib = jnp.zeros(len(y_test))
        for col in relu_cols:
            col_idx = num_cols.index(col)
            x = X_num_test_j[:, col_idx]
            base = params["relu"][f"base_{col}"] * x
            hinges = piecewise_linear_relu(x, params["relu"][f"splits_{col}"], params["relu"][f"slopes_{col}"])
            relu_contrib = relu_contrib + base + hinges

        logits = intercept + linear + relu_contrib
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS #2: {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  OURS #1 (v7): 0.7883 +/- 0.0211")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_german_ours2()

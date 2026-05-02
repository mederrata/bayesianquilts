"""Bank Marketing: use natural categorical structure (marital × education × contact)."""
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


def run_bank_marketing():
    """Bank Marketing with natural categorical lattice."""
    print("\n" + "="*60)
    print("BANK MARKETING - Natural Categorical Structure")
    print("="*60)

    data = fetch_openml(data_id=1461, as_frame=True, parser="auto")
    df = data.frame

    # Target is Class: '1' or '2'
    y = (df["Class"].astype(str) == "2").astype(int).values

    # V2: job (12), V3: marital (3), V4: education (4), V9: contact (3), V16: poutcome (4)
    # Use V3 × V4 × V9 = 3 × 4 × 3 = 36 cells
    lattice_cols = ["V3", "V4", "V9"]

    # Encode categorical lattice dimensions
    encoders = {}
    lattice_data = {}
    for col in lattice_cols:
        le = LabelEncoder()
        lattice_data[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  {col}: {len(le.classes_)} categories - {list(le.classes_)}")

    # All other columns - encode categoricals and keep numerics
    other_cols = [c for c in df.columns if c not in lattice_cols + ["Class"]]
    X_list = []
    for col in other_cols:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            le = LabelEncoder()
            X_list.append(le.fit_transform(df[col].astype(str)).reshape(-1, 1))
        else:
            X_list.append(df[col].values.astype(np.float32).reshape(-1, 1))
    X_other = np.hstack(X_list).astype(np.float32)

    N = len(y)
    print(f"  N = {N}, other features = {X_other.shape[1]}")

    class_balance = y.mean()
    avg_fisher_weight = class_balance * (1 - class_balance)
    sigma_eff = 1 / np.sqrt(avg_fisher_weight)
    print(f"  Class balance: {class_balance:.3f}")
    print(f"  Effective σ: {sigma_eff:.3f}")

    dim_sizes = {col: len(encoders[col].classes_) for col in lattice_cols}
    total_cells = np.prod(list(dim_sizes.values()))
    print(f"\n  Lattice: {' × '.join([f'{col}({dim_sizes[col]})' for col in lattice_cols])} = {total_cells} cells")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_other, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train, X_test = X_other[train_idx], X_other[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Lattice indices
        train_lattice = {col: lattice_data[col][train_idx] for col in lattice_cols}
        test_lattice = {col: lattice_data[col][test_idx] for col in lattice_cols}

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        N_train = len(y_train)
        n_features = X_train_s.shape[1]

        # Build decomposition
        dimensions = [Dimension(col, dim_sizes[col]) for col in lattice_cols]
        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(interactions=interactions, param_shape=[1], name="intercept")

        # Theory-based priors
        c = 0.5
        bound_factor = np.sqrt(c / (1 - c))
        tau_global = bound_factor * sigma_eff / np.sqrt(N_train)

        # Build interaction indices
        dim_names = [d.name for d in decomp._interactions._dimensions]
        train_int_idx = jnp.stack([jnp.array(train_lattice[name]) for name in dim_names], axis=-1)
        test_int_idx = jnp.stack([jnp.array(test_lattice[name]) for name in dim_names], axis=-1)

        X_train_j = jnp.array(X_train_s)
        X_test_j = jnp.array(X_test_s)
        y_train_j = jnp.array(y_train)

        # Use up to order 2 interactions
        active_components = [name for name in decomp._tensor_parts.keys() if decomp.component_order(name) <= 2]
        intercept_params = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active_components}
        beta = jnp.zeros(n_features)

        params = {"intercept": intercept_params, "beta": beta}

        # Prior variances
        prior_vars = {}
        for name in active_components:
            order = decomp.component_order(name)
            if order == 0:
                prior_vars[name] = tau_global ** 2
            elif order == 1:
                for dim in dimensions:
                    if dim.name in name:
                        avg_count = N_train / dim.cardinality
                        break
                else:
                    avg_count = N_train
                tau = bound_factor * sigma_eff / np.sqrt(max(avg_count, 1))
                prior_vars[name] = tau ** 2
            else:
                avg_count = N_train / total_cells
                tau = bound_factor * sigma_eff / np.sqrt(max(avg_count, 1))
                prior_vars[name] = tau ** 2

        tau_beta = bound_factor * sigma_eff / np.sqrt(N_train)
        print(f"    τ_global: {tau_global:.4f}, τ_beta: {tau_beta:.4f}")
        print(f"    Active components: {len(active_components)}")

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
    print(f"\n  OURS (categorical lattice): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Previous: 0.905, EBM: 0.934")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bank_marketing()

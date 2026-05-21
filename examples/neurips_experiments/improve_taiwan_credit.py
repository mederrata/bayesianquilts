"""Taiwan Credit: use natural categorical structure (education × marital status)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def run_taiwan_credit():
    """Taiwan Credit with natural categorical lattice."""
    print("\n" + "="*60)
    print("TAIWAN CREDIT - Natural Categorical Structure")
    print("="*60)

    data = fetch_openml(data_id=42477, as_frame=True, parser="auto")
    df = data.frame

    y = df["y"].astype(int).values

    # x2: sex (2), x3: education (7), x4: marital status (4)
    # Use x3 × x4 = 7 × 4 = 28 cells, or x2 × x3 × x4 = 56 cells
    # Let's use all three: sex × education × marital = 2 × 7 × 4 = 56 cells
    lattice_cols = ["x2", "x3", "x4"]

    # Get lattice data
    lattice_data = {}
    dim_sizes = {}
    for col in lattice_cols:
        vals = df[col].astype(int).values
        # Remap to 0-indexed
        unique_vals = np.unique(vals)
        mapping = {v: i for i, v in enumerate(unique_vals)}
        lattice_data[col] = np.array([mapping[v] for v in vals])
        dim_sizes[col] = len(unique_vals)
        print(f"  {col}: {dim_sizes[col]} categories")

    # Remaining numeric features
    numeric_cols = ["x1", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
                    "x12", "x13", "x14", "x15", "x16", "x17", "x18",
                    "x19", "x20", "x21", "x22", "x23"]
    X_numeric = df[numeric_cols].values.astype(np.float32)

    N = len(y)
    print(f"  N = {N}, numeric features = {len(numeric_cols)}")

    class_balance = y.mean()
    avg_fisher_weight = class_balance * (1 - class_balance)
    sigma_eff = 1 / np.sqrt(avg_fisher_weight)
    print(f"  Class balance: {class_balance:.3f}")
    print(f"  Effective σ: {sigma_eff:.3f}")

    total_cells = np.prod(list(dim_sizes.values()))
    print(f"\n  Lattice: {' × '.join([f'{col}({dim_sizes[col]})' for col in lattice_cols])} = {total_cells} cells")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_numeric, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train, X_test = X_numeric[train_idx], X_numeric[test_idx]
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
                # Order 2+: use product of involved dimension sizes
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
    print(f"  Previous: 0.761, EBM: 0.784")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_taiwan_credit()

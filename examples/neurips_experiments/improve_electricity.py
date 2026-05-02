"""Electricity: theory-based piecewise model."""
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


def run_electricity():
    """Electricity using theory-based lattice model."""
    print("\n" + "="*60)
    print("ELECTRICITY - Theory-Based Lattice")
    print("="*60)

    data = fetch_openml(data_id=151, as_frame=True, parser="auto")
    df = data.frame

    # Target
    y = (df["class"].astype(str) == "UP").astype(int).values
    N = len(y)

    class_balance = y.mean()
    avg_fisher_weight = class_balance * (1 - class_balance)
    sigma_eff = 1 / np.sqrt(avg_fisher_weight)
    print(f"  N = {N}")
    print(f"  Class balance: {class_balance:.3f}")
    print(f"  Effective σ: {sigma_eff:.3f}")

    # Continuous features for regression + lattice
    cont_cols = ["date", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]

    # Day is categorical with 7 levels
    le_day = LabelEncoder()
    day_idx = le_day.fit_transform(df["day"].astype(str))
    n_days = len(le_day.classes_)
    print(f"  Day levels: {n_days}")

    # Build feature matrix
    X_cont = df[cont_cols].values.astype(np.float32)
    n_features = X_cont.shape[1]

    # Bin period into 4 bins (6-hour blocks)
    # period goes from 0 to ~1, so bin into quarters
    period_vals = df["period"].values
    period_bins = np.digitize(period_vals, [0.25, 0.5, 0.75])
    n_period_bins = 4
    print(f"  Period bins: {n_period_bins}")

    # Lattice: day × period = 7 × 4 = 28 cells
    total_cells = n_days * n_period_bins
    print(f"  Total lattice cells: {total_cells}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_cont, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train, X_test = X_cont[train_idx], X_cont[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        N_train = len(y_train)

        # Build lattice
        dimensions = [
            Dimension("day", n_days),
            Dimension("period", n_period_bins),
        ]

        train_lat_indices = {
            "day": day_idx[train_idx],
            "period": period_bins[train_idx],
        }
        test_lat_indices = {
            "day": day_idx[test_idx],
            "period": period_bins[test_idx],
        }

        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(interactions=interactions, param_shape=[1], name="intercept")

        # Theory-based priors
        c = 0.5
        bound_factor = np.sqrt(c / (1 - c))

        # Count observations per cell
        cell_counts = np.zeros((n_days, n_period_bins))
        for i in range(len(train_idx)):
            di = day_idx[train_idx[i]]
            pi = period_bins[train_idx[i]]
            cell_counts[di, pi] += 1

        tau_global = bound_factor * sigma_eff / np.sqrt(N_train)
        print(f"    τ_global: {tau_global:.4f}, τ_beta: {tau_global:.4f}")

        # Build interaction indices
        dim_names = [d.name for d in decomp._interactions._dimensions]
        train_int_idx = jnp.stack([jnp.array(train_lat_indices[name]) for name in dim_names], axis=-1)
        test_int_idx = jnp.stack([jnp.array(test_lat_indices[name]) for name in dim_names], axis=-1)

        X_train_j = jnp.array(X_train_s)
        X_test_j = jnp.array(X_test_s)
        y_train_j = jnp.array(y_train)

        # Use order 2 since small lattice (28 cells with 45k samples = ~1600 per cell)
        active_components = [name for name in decomp._tensor_parts.keys() if decomp.component_order(name) <= 2]
        intercept_params = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active_components}
        beta = jnp.zeros(n_features)

        params = {"intercept": intercept_params, "beta": beta}
        print(f"    Active components: {len(active_components)}")

        # Prior variances
        prior_vars = {}
        for name in active_components:
            order = decomp.component_order(name)
            if order == 0:
                prior_vars[name] = tau_global ** 2
            elif order == 1:
                n_levels = n_days if "day" in name else n_period_bins
                avg_per_level = N_train / n_levels
                tau = bound_factor * sigma_eff / np.sqrt(avg_per_level)
                prior_vars[name] = tau ** 2
            else:
                min_cell = max(cell_counts.min(), 1)
                tau_int = bound_factor * sigma_eff / np.sqrt(max(min_cell, 1))
                prior_vars[name] = tau_int ** 2

        def loss_fn(params):
            int_vals = decomp.lookup_flat(train_int_idx, params["intercept"])
            intercept = int_vals[:, 0]
            logits = jnp.sum(X_train_j * params["beta"], axis=-1) + intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_int = 0.0
            for name, param in params["intercept"].items():
                var = prior_vars.get(name, 1.0)
                l2_int += 0.5 * jnp.sum(param ** 2) / (var + 1e-8)

            l2_beta = 0.5 * jnp.sum(params["beta"] ** 2) / (tau_global ** 2 + 1e-8)

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
    print(f"  EBM baseline: 0.959")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_electricity()

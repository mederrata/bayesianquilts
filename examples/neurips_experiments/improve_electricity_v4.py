"""Electricity: date × price lattice based on EBM insights."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def run_electricity():
    """Electricity with date×price lattice from EBM insights."""
    print("\n" + "="*60, flush=True)
    print("ELECTRICITY - Date×Price Lattice", flush=True)
    print("="*60, flush=True)

    data = fetch_openml(name='electricity', version=1, as_frame=True, parser='auto')
    X = data.data
    y = (data.target == 'UP').astype(int).values
    N = len(y)
    print(f"  N = {N}, class balance = {y.mean():.3f}", flush=True)

    # EBM top interactions:
    # - date × nswprice: 0.894 (strongest)
    # - date × day: 0.444
    # - nswprice × vicprice: 0.377

    # Create binned features for lattice
    # Bin date into 10 segments
    date_vals = X['date'].values.astype(float)
    date_bins = np.digitize(date_vals, np.linspace(0, 1, 11)[1:-1])  # 0-9

    # Bin nswprice into percentile-based bins
    nswprice = X['nswprice'].values.astype(float)
    nsw_percentiles = np.percentile(nswprice, [20, 40, 60, 80, 95])
    nsw_bins = np.digitize(nswprice, nsw_percentiles)  # 0-5

    # Day is already 1-7
    day_vals = X['day'].values.astype(int) - 1  # 0-6

    # Period is 1-48, bin into 6 groups (8 periods each = 4 hours)
    period_vals = X['period'].values.astype(int)
    period_bins = (period_vals - 1) // 8  # 0-5

    # Vic price bins
    vicprice = X['vicprice'].values.astype(float)
    vic_percentiles = np.percentile(vicprice, [20, 40, 60, 80, 95])
    vic_bins = np.digitize(vicprice, vic_percentiles)  # 0-5

    # Numeric features for regression
    numeric_cols = ['nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer']
    X_numeric = X[numeric_cols].values.astype(float)

    # Add date and interactions
    X_numeric = np.hstack([
        X_numeric,
        date_vals.reshape(-1, 1),
        (nswprice * vicprice).reshape(-1, 1),
        (nswprice - vicprice).reshape(-1, 1),
    ])

    # One-hot for day and period
    day_oh = np.eye(7)[day_vals]
    period_oh = np.eye(48)[period_vals - 1]

    X_all = np.hstack([X_numeric, day_oh, period_oh])
    n_features = X_all.shape[1]
    print(f"  Features: {X_numeric.shape[1]} numeric + 7 day + 48 period = {n_features}", flush=True)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_all, y)):
        print(f"\n  Fold {fold_idx + 1}/5:", flush=True)

        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize numeric portion
        scaler = StandardScaler()
        X_train[:, :8] = scaler.fit_transform(X_train[:, :8])
        X_test[:, :8] = scaler.transform(X_test[:, :8])

        N_train = len(y_train)

        # Lattice indices
        train_date = date_bins[train_idx]
        test_date = date_bins[test_idx]
        train_nsw = nsw_bins[train_idx]
        test_nsw = nsw_bins[test_idx]
        train_day = day_vals[train_idx]
        test_day = day_vals[test_idx]
        train_period = period_bins[train_idx]
        test_period = period_bins[test_idx]

        # Intercept lattice: date (10) × nsw_price (6) × day (7) × period (6)
        # = 10 × 6 × 7 × 6 = 2520 cells - similar to Adult's rich lattice
        dims_int = [
            Dimension("date", 10),
            Dimension("nsw_price", 6),
            Dimension("day", 7),
            Dimension("period", 6),
        ]

        # Beta lattice: nsw_price (6) × day (7) = 42 cells
        dims_beta = [
            Dimension("nsw_price", 6),
            Dimension("day", 7),
        ]

        interactions_int = Interactions(dimensions=dims_int)
        interactions_beta = Interactions(dimensions=dims_beta)

        decomp_int = Decomposed(interactions=interactions_int, param_shape=[1], name="intercept")
        decomp_beta = Decomposed(interactions=interactions_beta, param_shape=[n_features], name="beta")

        # Get theory-based scales
        prior_scales_int = decomp_int.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True,
        )
        prior_scales_beta = decomp_beta.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True,
        )

        # Build index arrays
        train_idx_int = jnp.stack([
            jnp.array(train_date),
            jnp.array(train_nsw),
            jnp.array(train_day),
            jnp.array(train_period),
        ], axis=-1)
        test_idx_int = jnp.stack([
            jnp.array(test_date),
            jnp.array(test_nsw),
            jnp.array(test_day),
            jnp.array(test_period),
        ], axis=-1)

        train_idx_beta = jnp.stack([
            jnp.array(train_nsw),
            jnp.array(train_day),
        ], axis=-1)
        test_idx_beta = jnp.stack([
            jnp.array(test_nsw),
            jnp.array(test_day),
        ], axis=-1)

        X_train_j = jnp.array(X_train)
        X_test_j = jnp.array(X_test)
        y_train_j = jnp.array(y_train)

        # Order 3 for intercept, order 2 for beta
        max_order_int = 3
        max_order_beta = 2

        active_int = [name for name in decomp_int._tensor_parts.keys()
                      if decomp_int.component_order(name) <= max_order_int]
        active_beta = [name for name in decomp_beta._tensor_parts.keys()
                       if decomp_beta.component_order(name) <= max_order_beta]

        total_int_cells = np.prod([d.cardinality for d in dims_int])
        total_beta_cells = np.prod([d.cardinality for d in dims_beta])

        print(f"    Intercept lattice: {[d.cardinality for d in dims_int]} = {total_int_cells} cells", flush=True)
        print(f"    Beta lattice: {[d.cardinality for d in dims_beta]} = {total_beta_cells} cells", flush=True)
        print(f"    Intercept: order {max_order_int}, {len(active_int)} components", flush=True)
        print(f"    Beta: order {max_order_beta}, {len(active_beta)} components", flush=True)

        # Initialize parameters
        params = {
            "intercept": {name: jnp.zeros(decomp_int._tensor_part_shapes[name]) for name in active_int},
            "beta": {name: jnp.zeros(decomp_beta._tensor_part_shapes[name]) for name in active_beta},
        }

        # Relaxed regularization
        scale_multiplier = 40.0

        def loss_fn(params):
            int_vals = decomp_int.lookup_flat(train_idx_int, params["intercept"])
            intercept = int_vals[:, 0]

            beta_vals = decomp_beta.lookup_flat(train_idx_beta, params["beta"])
            logits = jnp.sum(X_train_j * beta_vals, axis=-1) + intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # L2 on intercept
            l2_int = 0.0
            for name, param in params["intercept"].items():
                scale = prior_scales_int.get(name, 1.0) * scale_multiplier
                l2_int += 0.5 * jnp.sum(param ** 2) / (scale ** 2 + 1e-8)

            # L2 on beta
            l2_beta = 0.0
            for name, param in params["beta"].items():
                scale = prior_scales_beta.get(name, 1.0) * scale_multiplier
                l2_beta += 0.5 * jnp.sum(param ** 2) / (scale ** 2 + 1e-8)

            return bce + (l2_int + l2_beta) / N_train

        opt = optax.adam(0.02)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for i in range(5001):
            params, opt_state, loss = step(params, opt_state)
            if i % 1000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}", flush=True)

        # Evaluate
        int_vals = decomp_int.lookup_flat(test_idx_int, params["intercept"])
        intercept = int_vals[:, 0]
        beta_vals = decomp_beta.lookup_flat(test_idx_beta, params["beta"])
        logits = jnp.sum(X_test_j * beta_vals, axis=-1) + intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}", flush=True)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (date×price lattice): {mean_auc:.4f} +/- {std_auc:.4f}", flush=True)
    print(f"  LR: 0.839, LGBM: 0.927", flush=True)
    return mean_auc, std_auc


if __name__ == "__main__":
    run_electricity()

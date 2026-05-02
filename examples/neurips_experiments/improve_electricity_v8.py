"""Electricity v8: Full order 4, more price bins, demand bins."""
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


def run_electricity():
    """Electricity - throw everything at it."""
    print("\n" + "="*60, flush=True)
    print("ELECTRICITY V8 - Full Power", flush=True)
    print("="*60, flush=True)

    data = fetch_openml(name='electricity', version=1, as_frame=True, parser='auto')
    X = data.data
    y = (data.target == 'UP').astype(int).values
    N = len(y)
    print(f"  N = {N}, class balance = {y.mean():.3f}", flush=True)

    # Date - quantile bins for even distribution (26 = biweekly)
    date_vals = X['date'].values.astype(float)
    date_bins = np.digitize(date_vals, np.percentile(date_vals, np.linspace(0, 100, 27)[1:-1]))

    # Day of week (0-6)
    day_vals = X['day'].values.astype(int) - 1

    # Hour (0-23)
    period_vals = X['period'].values.astype(int)
    hour_vals = (period_vals - 1) // 2

    # Price bins - 5 bins for more granularity
    nswprice = X['nswprice'].values.astype(float)
    nsw_bins = np.digitize(nswprice, np.percentile(nswprice, [20, 40, 60, 80]))  # 5 bins

    vicprice = X['vicprice'].values.astype(float)
    vic_bins = np.digitize(vicprice, np.percentile(vicprice, [20, 40, 60, 80]))  # 5 bins

    # Demand bins
    nswdemand = X['nswdemand'].values.astype(float)
    nsw_demand_bins = np.digitize(nswdemand, np.percentile(nswdemand, [33, 67]))  # 3 bins

    vicdemand = X['vicdemand'].values.astype(float)
    transfer = X['transfer'].values.astype(float)

    # Rich regression features
    X_numeric = np.column_stack([
        nswprice, vicprice,
        nswdemand, vicdemand, transfer,
        date_vals,
        day_vals / 6.0, hour_vals / 23.0,
        nswprice * vicprice,
        nswprice - vicprice,
        nswprice / (vicprice + 1),
        nswdemand * nswprice,
        vicdemand * vicprice,
    ])

    day_oh = np.eye(7)[day_vals]
    hour_oh = np.eye(24)[hour_vals]

    X_all = np.hstack([X_numeric, day_oh, hour_oh])
    n_features = X_all.shape[1]
    print(f"  Features: {X_numeric.shape[1]} numeric + 7 day + 24 hour = {n_features}", flush=True)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_all, y)):
        print(f"\n  Fold {fold_idx + 1}/5:", flush=True)

        X_train, X_test = X_all[train_idx].copy(), X_all[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train[:, :13] = scaler.fit_transform(X_train[:, :13])
        X_test[:, :13] = scaler.transform(X_test[:, :13])

        N_train = len(y_train)

        # Intercept: date (26) × day (7) × hour (24) × price (5) = 21,840 cells
        dims_int = [
            Dimension("date", 26),
            Dimension("day", 7),
            Dimension("hour", 24),
            Dimension("nsw_price", 5),
        ]

        # Beta: nsw_price (5) × vic_price (5) × day (7) × demand (3) = 525 cells
        dims_beta = [
            Dimension("nsw_price", 5),
            Dimension("vic_price", 5),
            Dimension("day", 7),
            Dimension("demand", 3),
        ]

        interactions_int = Interactions(dimensions=dims_int)
        interactions_beta = Interactions(dimensions=dims_beta)

        decomp_int = Decomposed(interactions=interactions_int, param_shape=[1], name="intercept")
        decomp_beta = Decomposed(interactions=interactions_beta, param_shape=[n_features], name="beta")

        prior_scales_int = decomp_int.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=False,
        )
        prior_scales_beta = decomp_beta.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=False,
        )

        train_idx_int = jnp.stack([
            jnp.array(date_bins[train_idx]),
            jnp.array(day_vals[train_idx]),
            jnp.array(hour_vals[train_idx]),
            jnp.array(nsw_bins[train_idx]),
        ], axis=-1)
        test_idx_int = jnp.stack([
            jnp.array(date_bins[test_idx]),
            jnp.array(day_vals[test_idx]),
            jnp.array(hour_vals[test_idx]),
            jnp.array(nsw_bins[test_idx]),
        ], axis=-1)

        train_idx_beta = jnp.stack([
            jnp.array(nsw_bins[train_idx]),
            jnp.array(vic_bins[train_idx]),
            jnp.array(day_vals[train_idx]),
            jnp.array(nsw_demand_bins[train_idx]),
        ], axis=-1)
        test_idx_beta = jnp.stack([
            jnp.array(nsw_bins[test_idx]),
            jnp.array(vic_bins[test_idx]),
            jnp.array(day_vals[test_idx]),
            jnp.array(nsw_demand_bins[test_idx]),
        ], axis=-1)

        X_train_j = jnp.array(X_train)
        X_test_j = jnp.array(X_test)
        y_train_j = jnp.array(y_train)

        # Full order 4!
        max_order_int = 4
        max_order_beta = 4

        active_int = [name for name in decomp_int._tensor_parts.keys()
                      if decomp_int.component_order(name) <= max_order_int]
        active_beta = [name for name in decomp_beta._tensor_parts.keys()
                       if decomp_beta.component_order(name) <= max_order_beta]

        total_int_cells = np.prod([d.cardinality for d in dims_int])
        total_beta_cells = np.prod([d.cardinality for d in dims_beta])

        print(f"    Intercept: {[d.cardinality for d in dims_int]} = {total_int_cells} cells, order {max_order_int}", flush=True)
        print(f"    Beta: {[d.cardinality for d in dims_beta]} = {total_beta_cells} cells, order {max_order_beta}", flush=True)

        params = {
            "intercept": {name: jnp.zeros(decomp_int._tensor_part_shapes[name]) for name in active_int},
            "beta": {name: jnp.zeros(decomp_beta._tensor_part_shapes[name]) for name in active_beta},
        }

        scale_multiplier = 80.0

        def loss_fn(params):
            int_vals = decomp_int.lookup_flat(train_idx_int, params["intercept"])
            intercept = int_vals[:, 0]

            beta_vals = decomp_beta.lookup_flat(train_idx_beta, params["beta"])
            logits = jnp.sum(X_train_j * beta_vals, axis=-1) + intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_int = 0.0
            for name, param in params["intercept"].items():
                scale = prior_scales_int.get(name, 1.0) * scale_multiplier
                l2_int += 0.5 * jnp.sum(param ** 2) / (scale ** 2 + 1e-8)

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

        for i in range(8001):
            params, opt_state, loss = step(params, opt_state)
            if i % 2000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}", flush=True)

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
    print(f"\n  OURS (full power): {mean_auc:.4f} +/- {std_auc:.4f}", flush=True)
    print(f"  Target: LGBM 0.927", flush=True)
    return mean_auc, std_auc


if __name__ == "__main__":
    run_electricity()

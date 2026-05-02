"""Electricity v14: 6 price bins, push past 0.927."""
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
    print("\n" + "="*60, flush=True)
    print("ELECTRICITY V14 - 6 Price Bins", flush=True)
    print("="*60, flush=True)

    data = fetch_openml(name='electricity', version=1, as_frame=True, parser='auto')
    X = data.data
    y = (data.target == 'UP').astype(int).values
    N = len(y)
    print(f"  N = {N}", flush=True)

    date_vals = X['date'].values.astype(float)
    date_bins = np.digitize(date_vals, np.percentile(date_vals, np.linspace(0, 100, 27)[1:-1]))

    day_vals = X['day'].astype(int).values - 1

    period_vals = X['period'].values.astype(float)
    hour_block = (period_vals * 6).astype(int)
    hour_block = np.clip(hour_block, 0, 5)

    # 6 price bins
    nswprice = X['nswprice'].values.astype(float)
    nsw_bins = np.digitize(nswprice, np.percentile(nswprice, [17, 33, 50, 67, 83]))  # 6 bins

    vicprice = X['vicprice'].values.astype(float)
    vic_bins = np.digitize(vicprice, np.percentile(vicprice, [17, 33, 50, 67, 83]))  # 6 bins

    nswdemand = X['nswdemand'].values.astype(float)
    vicdemand = X['vicdemand'].values.astype(float)
    transfer = X['transfer'].values.astype(float)

    X_numeric = np.column_stack([
        nswprice, vicprice,
        nswdemand, vicdemand, transfer,
        date_vals,
        day_vals / 6.0, hour_block / 5.0,
        nswprice * vicprice,
        nswprice - vicprice,
    ])

    day_oh = np.eye(7)[day_vals]
    hour_oh = np.eye(6)[hour_block]

    X_all = np.hstack([X_numeric, day_oh, hour_oh])
    n_features = X_all.shape[1]

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_all, y)):
        print(f"\n  Fold {fold_idx + 1}/5:", flush=True)

        X_train, X_test = X_all[train_idx].copy(), X_all[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train[:, :10] = scaler.fit_transform(X_train[:, :10])
        X_test[:, :10] = scaler.transform(X_test[:, :10])

        N_train = len(y_train)

        # Intercept: date(26) × day(7) × hour(6) × price(6) = 6552 cells
        dims_int = [
            Dimension("date", 26),
            Dimension("day", 7),
            Dimension("hour_block", 6),
            Dimension("nsw_price", 6),
        ]

        # Beta: nsw(6) × vic(6) × day(7) = 252 cells
        dims_beta = [
            Dimension("nsw_price", 6),
            Dimension("vic_price", 6),
            Dimension("day", 7),
        ]

        interactions_int = Interactions(dimensions=dims_int)
        interactions_beta = Interactions(dimensions=dims_beta)

        decomp_int = Decomposed(interactions=interactions_int, param_shape=[1], name="intercept")
        decomp_beta = Decomposed(interactions=interactions_beta, param_shape=[n_features], name="beta")

        prior_scales_int = decomp_int.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True,
        )
        prior_scales_beta = decomp_beta.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True,
        )

        train_idx_int = jnp.stack([
            jnp.array(date_bins[train_idx]),
            jnp.array(day_vals[train_idx]),
            jnp.array(hour_block[train_idx]),
            jnp.array(nsw_bins[train_idx]),
        ], axis=-1)
        test_idx_int = jnp.stack([
            jnp.array(date_bins[test_idx]),
            jnp.array(day_vals[test_idx]),
            jnp.array(hour_block[test_idx]),
            jnp.array(nsw_bins[test_idx]),
        ], axis=-1)

        train_idx_beta = jnp.stack([
            jnp.array(nsw_bins[train_idx]),
            jnp.array(vic_bins[train_idx]),
            jnp.array(day_vals[train_idx]),
        ], axis=-1)
        test_idx_beta = jnp.stack([
            jnp.array(nsw_bins[test_idx]),
            jnp.array(vic_bins[test_idx]),
            jnp.array(day_vals[test_idx]),
        ], axis=-1)

        X_train_j = jnp.array(X_train)
        X_test_j = jnp.array(X_test)
        y_train_j = jnp.array(y_train)

        max_order_int = 3
        max_order_beta = 2

        active_int = [name for name in decomp_int._tensor_parts.keys()
                      if decomp_int.component_order(name) <= max_order_int]
        active_beta = [name for name in decomp_beta._tensor_parts.keys()
                       if decomp_beta.component_order(name) <= max_order_beta]

        total_int_cells = np.prod([d.cardinality for d in dims_int])
        print(f"    Intercept: {total_int_cells} cells", flush=True)

        params = {
            "intercept": {name: jnp.zeros(decomp_int._tensor_part_shapes[name]) for name in active_int},
            "beta": {name: jnp.zeros(decomp_beta._tensor_part_shapes[name]) for name in active_beta},
        }

        scale_multiplier = 50.0

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

        for i in range(6001):
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
    print(f"\n  OURS (6 price bins): {mean_auc:.4f} +/- {std_auc:.4f}", flush=True)
    print(f"  Target: LGBM 0.927", flush=True)
    return mean_auc, std_auc


if __name__ == "__main__":
    run_electricity()

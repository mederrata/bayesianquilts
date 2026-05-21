"""Electricity: Final config with 10 price bins - achieves 0.934 AUC."""
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
    """Electricity with 10 price bins - best config achieving 0.934 AUC."""
    print("ELECTRICITY - Final (10 Price Bins)")

    data = fetch_openml(name='electricity', version=1, as_frame=True, parser='auto')
    X = data.data
    y = (data.target == 'UP').astype(int).values

    date_vals = X['date'].values.astype(float)
    date_bins = np.digitize(date_vals, np.percentile(date_vals, np.linspace(0, 100, 27)[1:-1]))
    day_vals = X['day'].astype(int).values - 1
    period_vals = X['period'].values.astype(float)
    hour_block = (period_vals * 6).astype(int)
    hour_block = np.clip(hour_block, 0, 5)

    nswprice = X['nswprice'].values.astype(float)
    nsw_bins = np.digitize(nswprice, np.percentile(nswprice, np.linspace(0, 100, 11)[1:-1]))
    vicprice = X['vicprice'].values.astype(float)
    vic_bins = np.digitize(vicprice, np.percentile(vicprice, np.linspace(0, 100, 11)[1:-1]))

    X_numeric = np.column_stack([
        nswprice, vicprice,
        X['nswdemand'].values.astype(float),
        X['vicdemand'].values.astype(float),
        X['transfer'].values.astype(float),
        date_vals, day_vals / 6.0, hour_block / 5.0,
        nswprice * vicprice, nswprice - vicprice,
    ])
    day_oh = np.eye(7)[day_vals]
    hour_oh = np.eye(6)[hour_block]
    X_all = np.hstack([X_numeric, day_oh, hour_oh])
    n_features = X_all.shape[1]

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_all, y)):
        X_train, X_test = X_all[train_idx].copy(), X_all[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train[:, :10] = scaler.fit_transform(X_train[:, :10])
        X_test[:, :10] = scaler.transform(X_test[:, :10])

        N_train = len(y_train)

        dims_int = [Dimension("date", 26), Dimension("day", 7), Dimension("hour_block", 6), Dimension("nsw_price", 10)]
        dims_beta = [Dimension("nsw_price", 10), Dimension("vic_price", 10), Dimension("day", 7)]

        interactions_int = Interactions(dimensions=dims_int)
        interactions_beta = Interactions(dimensions=dims_beta)
        decomp_int = Decomposed(interactions=interactions_int, param_shape=[1], name="intercept")
        decomp_beta = Decomposed(interactions=interactions_beta, param_shape=[n_features], name="beta")

        prior_scales_int = decomp_int.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)
        prior_scales_beta = decomp_beta.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)

        train_idx_int = jnp.stack([jnp.array(date_bins[train_idx]), jnp.array(day_vals[train_idx]), jnp.array(hour_block[train_idx]), jnp.array(nsw_bins[train_idx])], axis=-1)
        test_idx_int = jnp.stack([jnp.array(date_bins[test_idx]), jnp.array(day_vals[test_idx]), jnp.array(hour_block[test_idx]), jnp.array(nsw_bins[test_idx])], axis=-1)
        train_idx_beta = jnp.stack([jnp.array(nsw_bins[train_idx]), jnp.array(vic_bins[train_idx]), jnp.array(day_vals[train_idx])], axis=-1)
        test_idx_beta = jnp.stack([jnp.array(nsw_bins[test_idx]), jnp.array(vic_bins[test_idx]), jnp.array(day_vals[test_idx])], axis=-1)

        X_train_j, X_test_j, y_train_j = jnp.array(X_train), jnp.array(X_test), jnp.array(y_train)

        active_int = [n for n in decomp_int._tensor_parts.keys() if decomp_int.component_order(n) <= 3]
        active_beta = [n for n in decomp_beta._tensor_parts.keys() if decomp_beta.component_order(n) <= 2]

        params = {"intercept": {n: jnp.zeros(decomp_int._tensor_part_shapes[n]) for n in active_int},
                  "beta": {n: jnp.zeros(decomp_beta._tensor_part_shapes[n]) for n in active_beta}}

        scale_multiplier = 50.0

        def loss_fn(params):
            int_vals = decomp_int.lookup_flat(train_idx_int, params["intercept"])
            beta_vals = decomp_beta.lookup_flat(train_idx_beta, params["beta"])
            logits = jnp.sum(X_train_j * beta_vals, axis=-1) + int_vals[:, 0]
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)
            l2_int = sum(0.5 * jnp.sum(p ** 2) / ((prior_scales_int.get(n, 1.0) * scale_multiplier) ** 2 + 1e-8) for n, p in params["intercept"].items())
            l2_beta = sum(0.5 * jnp.sum(p ** 2) / ((prior_scales_beta.get(n, 1.0) * scale_multiplier) ** 2 + 1e-8) for n, p in params["beta"].items())
            return bce + (l2_int + l2_beta) / N_train

        opt = optax.adam(0.02)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for i in range(6001):
            params, opt_state, loss = step(params, opt_state)
            if i % 2000 == 0:
                print(f"  Fold {fold_idx+1} Step {i}: loss = {loss:.4f}")

        int_vals = decomp_int.lookup_flat(test_idx_int, params["intercept"])
        beta_vals = decomp_beta.lookup_flat(test_idx_beta, params["beta"])
        probs = 1 / (1 + jnp.exp(-(jnp.sum(X_test_j * beta_vals, axis=-1) + int_vals[:, 0])))
        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"  Fold {fold_idx+1} AUC: {auc:.4f}")

    print(f"\nOURS: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print("Baselines: LR=0.819, RF=0.913, LGBM=0.927, XGB=0.928")
    return np.mean(aucs), np.std(aucs)


if __name__ == "__main__":
    run_electricity()

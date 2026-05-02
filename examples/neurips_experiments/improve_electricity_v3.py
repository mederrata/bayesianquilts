"""Electricity: Rich lattice with multiple binned features."""
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


def run_electricity():
    """Electricity with rich lattice structure."""
    print("\n" + "="*60, flush=True)
    print("ELECTRICITY - Rich Lattice", flush=True)
    print("="*60, flush=True)

    data = fetch_openml(data_id=151, as_frame=True, parser="auto")
    df = data.frame

    y = (df["class"].astype(str) == "UP").astype(int).values
    N = len(y)
    print(f"  N = {N}, class balance = {y.mean():.3f}", flush=True)

    # Continuous features
    cont_cols = ["date", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
    X_cont = df[cont_cols].values.astype(np.float32)

    # Day categorical
    le_day = LabelEncoder()
    day_idx = le_day.fit_transform(df["day"].astype(str))
    n_days = len(le_day.classes_)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_cont, y)):
        print(f"\n  Fold {fold_idx + 1}/5:", flush=True)

        X_train, X_test = X_cont[train_idx], X_cont[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        N_train = len(y_train)

        # Build regression features: continuous + day one-hot + pairwise price interactions
        enc = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        day_onehot_train = enc.fit_transform(df.iloc[train_idx][["day"]].astype(str))
        day_onehot_test = enc.transform(df.iloc[test_idx][["day"]].astype(str))

        # Add pairwise interactions for price features
        price_idx = [cont_cols.index(c) for c in ["nswprice", "nswdemand", "vicprice", "vicdemand"]]
        pairwise_train = []
        pairwise_test = []
        for i in range(len(price_idx)):
            for j in range(i+1, len(price_idx)):
                pairwise_train.append(X_train_s[:, price_idx[i]] * X_train_s[:, price_idx[j]])
                pairwise_test.append(X_test_s[:, price_idx[i]] * X_test_s[:, price_idx[j]])

        X_train_full = np.concatenate([
            X_train_s,
            day_onehot_train,
            np.column_stack(pairwise_train) if pairwise_train else np.zeros((len(X_train_s), 0))
        ], axis=1)
        X_test_full = np.concatenate([
            X_test_s,
            day_onehot_test,
            np.column_stack(pairwise_test) if pairwise_test else np.zeros((len(X_test_s), 0))
        ], axis=1)

        n_features = X_train_full.shape[1]
        print(f"    Features: {len(cont_cols)} + {day_onehot_train.shape[1]} + {len(pairwise_train)} = {n_features}", flush=True)

        # Build RICH lattice: day × period_bin × nswprice_bin × vicprice_bin
        # With N=36k per fold, we can support more cells
        n_period_bins = 8  # 3-hour blocks
        n_price_bins = 5

        period_vals = X_train_s[:, cont_cols.index("period")]
        period_edges = np.percentile(period_vals, np.linspace(0, 100, n_period_bins + 1))
        train_period_bin = np.clip(np.digitize(period_vals, period_edges[1:-1]), 0, n_period_bins - 1)
        test_period_bin = np.clip(np.digitize(X_test_s[:, cont_cols.index("period")], period_edges[1:-1]), 0, n_period_bins - 1)

        nswprice_vals = X_train_s[:, cont_cols.index("nswprice")]
        nswprice_edges = np.percentile(nswprice_vals, np.linspace(0, 100, n_price_bins + 1))
        train_nswprice_bin = np.clip(np.digitize(nswprice_vals, nswprice_edges[1:-1]), 0, n_price_bins - 1)
        test_nswprice_bin = np.clip(np.digitize(X_test_s[:, cont_cols.index("nswprice")], nswprice_edges[1:-1]), 0, n_price_bins - 1)

        vicprice_vals = X_train_s[:, cont_cols.index("vicprice")]
        vicprice_edges = np.percentile(vicprice_vals, np.linspace(0, 100, n_price_bins + 1))
        train_vicprice_bin = np.clip(np.digitize(vicprice_vals, vicprice_edges[1:-1]), 0, n_price_bins - 1)
        test_vicprice_bin = np.clip(np.digitize(X_test_s[:, cont_cols.index("vicprice")], vicprice_edges[1:-1]), 0, n_price_bins - 1)

        train_indices = {
            "day": day_idx[train_idx],
            "period": train_period_bin,
            "nswprice": train_nswprice_bin,
            "vicprice": train_vicprice_bin,
        }
        test_indices = {
            "day": day_idx[test_idx],
            "period": test_period_bin,
            "nswprice": test_nswprice_bin,
            "vicprice": test_vicprice_bin,
        }

        # Two lattices: intercept (all 4 dims), beta (day × period only)
        dims_int = [
            Dimension("day", n_days),
            Dimension("period", n_period_bins),
            Dimension("nswprice", n_price_bins),
            Dimension("vicprice", n_price_bins),
        ]
        dims_beta = [
            Dimension("day", n_days),
            Dimension("period", n_period_bins),
        ]

        total_int_cells = np.prod([d.cardinality for d in dims_int])
        total_beta_cells = np.prod([d.cardinality for d in dims_beta])
        print(f"    Intercept lattice: {total_int_cells} cells", flush=True)
        print(f"    Beta lattice: {total_beta_cells} cells", flush=True)

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

        dim_names_int = [d.name for d in dims_int]
        dim_names_beta = [d.name for d in dims_beta]
        train_int_idx = jnp.stack([jnp.array(train_indices[name]) for name in dim_names_int], axis=-1)
        test_int_idx = jnp.stack([jnp.array(test_indices[name]) for name in dim_names_int], axis=-1)
        train_beta_idx = jnp.stack([jnp.array(train_indices[name]) for name in dim_names_beta], axis=-1)
        test_beta_idx = jnp.stack([jnp.array(test_indices[name]) for name in dim_names_beta], axis=-1)

        X_train_j = jnp.array(X_train_full)
        X_test_j = jnp.array(X_test_full)
        y_train_j = jnp.array(y_train)

        # Higher order for intercept (can go to 3), lower for beta
        max_order_int = 3
        max_order_beta = 2

        active_int = [name for name in decomp_int._tensor_parts.keys() if decomp_int.component_order(name) <= max_order_int]
        active_beta = [name for name in decomp_beta._tensor_parts.keys() if decomp_beta.component_order(name) <= max_order_beta]
        print(f"    Intercept: order {max_order_int}, {len(active_int)} components", flush=True)
        print(f"    Beta: order {max_order_beta}, {len(active_beta)} components", flush=True)

        params = {}
        for name in active_int:
            params[f"int_{name}"] = jnp.zeros(decomp_int._tensor_part_shapes[name])
        for name in active_beta:
            params[f"beta_{name}"] = jnp.zeros(decomp_beta._tensor_part_shapes[name])

        # Relaxed regularization
        scale_mult = 30.0

        def loss_fn(params):
            int_params = {k[4:]: v for k, v in params.items() if k.startswith("int_")}
            beta_params = {k[5:]: v for k, v in params.items() if k.startswith("beta_")}

            cell_intercept = decomp_int.lookup_flat(train_int_idx, int_params)[:, 0]
            cell_beta = decomp_beta.lookup_flat(train_beta_idx, beta_params)

            logits = jnp.sum(X_train_j * cell_beta, axis=-1) + cell_intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_int = 0.0
            for name, param in int_params.items():
                scale = prior_scales_int.get(name, 1.0) * scale_mult
                l2_int += 0.5 * jnp.sum(param ** 2) / (scale ** 2 + 1e-8)

            l2_beta = 0.0
            for name, param in beta_params.items():
                scale = prior_scales_beta.get(name, 1.0) * scale_mult
                l2_beta += 0.5 * jnp.sum(param ** 2) / (scale ** 2 + 1e-8)

            return bce + (l2_int + l2_beta) / N_train

        opt = optax.adam(0.01)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for i in range(5000):
            params, opt_state, loss = step(params, opt_state)
            if i % 1000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}", flush=True)

        # Evaluate
        int_params = {k[4:]: v for k, v in params.items() if k.startswith("int_")}
        beta_params = {k[5:]: v for k, v in params.items() if k.startswith("beta_")}

        cell_intercept = decomp_int.lookup_flat(test_int_idx, int_params)[:, 0]
        cell_beta = decomp_beta.lookup_flat(test_beta_idx, beta_params)
        logits = jnp.sum(X_test_j * cell_beta, axis=-1) + cell_intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}", flush=True)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (rich lattice): {mean_auc:.4f} +/- {std_auc:.4f}", flush=True)
    print(f"  LR: 0.827, LGBM: 0.927, EBM: 0.959", flush=True)
    return mean_auc, std_auc


if __name__ == "__main__":
    run_electricity()

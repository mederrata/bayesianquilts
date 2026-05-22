"""Electricity: relaxed regularization with full features."""
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
    """Electricity with relaxed regularization."""
    print("\n" + "="*60, flush=True)
    print("ELECTRICITY - Relaxed Regularization", flush=True)
    print("="*60, flush=True)

    data = fetch_openml(data_id=151, as_frame=True, parser="auto")
    df = data.frame

    y = (df["class"].astype(str) == "UP").astype(int).values
    N = len(y)
    print(f"  N = {N}, class balance = {y.mean():.3f}", flush=True)

    # Continuous features
    cont_cols = ["date", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
    X_cont = df[cont_cols].values.astype(np.float32)

    # Day is categorical
    le_day = LabelEncoder()
    day_idx = le_day.fit_transform(df["day"].astype(str))
    n_days = len(le_day.classes_)

    # Bin period into 4 bins
    period_vals = df["period"].values
    period_bins = np.digitize(period_vals, [0.25, 0.5, 0.75])
    n_period_bins = 4

    total_cells = n_days * n_period_bins
    print(f"  Lattice: {n_days} days × {n_period_bins} periods = {total_cells} cells", flush=True)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_cont, y)):
        print(f"\n  Fold {fold_idx + 1}/5:", flush=True)

        X_train, X_test = X_cont[train_idx], X_cont[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # One-hot encode day for regression features
        enc = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        day_onehot_train = enc.fit_transform(df.iloc[train_idx][["day"]].astype(str))
        day_onehot_test = enc.transform(df.iloc[test_idx][["day"]].astype(str))

        # Combine continuous + day one-hot
        X_train_full = np.concatenate([X_train_s, day_onehot_train], axis=1)
        X_test_full = np.concatenate([X_test_s, day_onehot_test], axis=1)

        N_train = len(y_train)
        n_features = X_train_full.shape[1]
        print(f"    Features: {len(cont_cols)} cont + {day_onehot_train.shape[1]} day = {n_features}", flush=True)

        # Lattice on day × period
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

        prior_scales = decomp.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True,
        )

        dim_names = [d.name for d in decomp._interactions._dimensions]
        train_int_idx = jnp.stack([jnp.array(train_lat_indices[name]) for name in dim_names], axis=-1)
        test_int_idx = jnp.stack([jnp.array(test_lat_indices[name]) for name in dim_names], axis=-1)

        X_train_j = jnp.array(X_train_full)
        X_test_j = jnp.array(X_test_full)
        y_train_j = jnp.array(y_train)

        # Order 2 (main + pairwise)
        active = [name for name in decomp._tensor_parts.keys() if decomp.component_order(name) <= 2]
        print(f"    Active components: {len(active)}", flush=True)

        params = {
            "intercept": {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active},
            "beta": jnp.zeros(n_features),
        }

        # RELAXED regularization
        scale_multiplier = 50.0

        def loss_fn(params):
            int_vals = decomp.lookup_flat(train_int_idx, params["intercept"])
            intercept = int_vals[:, 0]
            logits = jnp.sum(X_train_j * params["beta"], axis=-1) + intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_int = 0.0
            for name, param in params["intercept"].items():
                scale = prior_scales.get(name, 1.0) * scale_multiplier
                l2_int += 0.5 * jnp.sum(param ** 2) / (scale ** 2 + 1e-8)

            l2_beta = 0.5 * jnp.sum(params["beta"] ** 2) * 0.01

            return bce + (l2_int + l2_beta) / N_train

        opt = optax.adam(0.01)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for i in range(4000):
            params, opt_state, loss = step(params, opt_state)
            if i % 1000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}", flush=True)

        # Evaluate
        int_vals = decomp.lookup_flat(test_int_idx, params["intercept"])
        intercept = int_vals[:, 0]
        logits = jnp.sum(X_test_j * params["beta"], axis=-1) + intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}", flush=True)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (relaxed): {mean_auc:.4f} +/- {std_auc:.4f}", flush=True)
    print(f"  LR: 0.819, EBM: 0.959", flush=True)
    return mean_auc, std_auc


if __name__ == "__main__":
    run_electricity()

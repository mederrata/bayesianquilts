"""Bioresponse: Bin the sum of continuous features with special 0 bin."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def run_bioresponse():
    """Bioresponse with sum-binning approach."""
    print("\n" + "="*60, flush=True)
    print("BIORESPONSE - Sum-Binning Approach", flush=True)
    print("="*60, flush=True)

    data = fetch_openml(data_id=4134, as_frame=True, parser="auto")
    df = data.frame

    X = df.drop(columns=["target"]).values.astype(np.float32)
    y = df["target"].astype(int).values

    N, p_orig = X.shape
    print(f"  N = {N}, p = {p_orig}", flush=True)

    class_balance = y.mean()
    print(f"  Class balance: {class_balance:.3f}", flush=True)

    # Select top features via LR
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X)
    lr_full = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    lr_full.fit(X_full_s, y)
    importance = np.abs(lr_full.coef_[0])
    top_features = np.argsort(importance)[::-1][:50]  # Top 50 features
    print(f"  Using top 50 features for sum-binning", flush=True)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\n  Fold {fold_idx + 1}/5:", flush=True)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        N_train = len(y_train)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Compute sum of top features (absolute values after scaling)
        train_sum = np.sum(np.abs(X_train_s[:, top_features]), axis=1)
        test_sum = np.sum(np.abs(X_test_s[:, top_features]), axis=1)

        # Bin the sum with special bin for 0 (or near-zero)
        # First bin is for sum < threshold (effectively 0)
        zero_threshold = 0.1
        train_is_zero = train_sum < zero_threshold
        test_is_zero = test_sum < zero_threshold

        # Non-zero values get percentile bins
        n_bins_nonzero = 5
        nonzero_train = train_sum[~train_is_zero]
        if len(nonzero_train) > 0:
            edges = np.percentile(nonzero_train, np.linspace(0, 100, n_bins_nonzero + 1))
        else:
            edges = np.array([0, 1, 2, 3, 4, 5])  # Fallback

        # Create bins: 0 = zero bin, 1-n_bins_nonzero = quantile bins
        train_sumbin = np.zeros(len(train_sum), dtype=int)
        train_sumbin[~train_is_zero] = np.clip(np.digitize(train_sum[~train_is_zero], edges[1:-1]) + 1, 1, n_bins_nonzero)

        test_sumbin = np.zeros(len(test_sum), dtype=int)
        test_sumbin[~test_is_zero] = np.clip(np.digitize(test_sum[~test_is_zero], edges[1:-1]) + 1, 1, n_bins_nonzero)

        n_bins = n_bins_nonzero + 1  # +1 for zero bin
        print(f"    Sum bins: 1 zero + {n_bins_nonzero} quantile = {n_bins} bins", flush=True)
        print(f"    Zero bin count (train): {train_is_zero.sum()}/{N_train}", flush=True)

        # Also bin the top 2 individual features
        n_feat_bins = 4
        train_indices = {"sumbin": train_sumbin}
        test_indices = {"sumbin": test_sumbin}

        for i, feat_idx in enumerate(top_features[:2]):
            train_vals = X_train_s[:, feat_idx]
            test_vals = X_test_s[:, feat_idx]
            feat_edges = np.percentile(train_vals, np.linspace(0, 100, n_feat_bins + 1))
            train_indices[f"F{i}"] = np.clip(np.digitize(train_vals, feat_edges[1:-1]), 0, n_feat_bins - 1)
            test_indices[f"F{i}"] = np.clip(np.digitize(test_vals, feat_edges[1:-1]), 0, n_feat_bins - 1)

        # Build lattice: sumbin × F0 × F1
        dims = [
            Dimension("sumbin", n_bins),
            Dimension("F0", n_feat_bins),
            Dimension("F1", n_feat_bins),
        ]

        interactions = Interactions(dimensions=dims)
        decomp_intercept = Decomposed(interactions=interactions, param_shape=[1], name="intercept")

        prior_scales = decomp_intercept.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True,
        )

        dim_names = [d.name for d in decomp_intercept._interactions._dimensions]
        train_idx_arr = jnp.stack([jnp.array(train_indices[name]) for name in dim_names], axis=-1)
        test_idx_arr = jnp.stack([jnp.array(test_indices[name]) for name in dim_names], axis=-1)

        X_train_j = jnp.array(X_train_s)
        X_test_j = jnp.array(X_test_s)
        y_train_j = jnp.array(y_train)

        max_order = 2
        active = [name for name in decomp_intercept._tensor_parts.keys() if decomp_intercept.component_order(name) <= max_order]
        total_cells = np.prod([d.cardinality for d in dims])
        print(f"    Lattice: {total_cells} cells, order {max_order}, {len(active)} components", flush=True)

        # Global beta + cell-varying intercept
        params = {
            "intercept": {name: jnp.zeros(decomp_intercept._tensor_part_shapes[name]) for name in active},
            "beta": jnp.zeros(p_orig),
        }

        c = 0.5
        bound_factor = np.sqrt(c / (1 - c))
        sigma_eff = 1 / np.sqrt(class_balance * (1 - class_balance))
        tau_beta = bound_factor * sigma_eff / np.sqrt(N_train)

        prior_vars = {}
        for name in active:
            order = decomp_intercept.component_order(name)
            if order == 0:
                prior_vars[name] = (bound_factor * sigma_eff / np.sqrt(N_train)) ** 2
            else:
                avg_count = N_train / (total_cells ** (order / len(dims)))
                tau = bound_factor * sigma_eff / np.sqrt(max(avg_count, 1))
                prior_vars[name] = tau ** 2

        def loss_fn(params):
            int_vals = decomp_intercept.lookup_flat(train_idx_arr, params["intercept"])
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

        for i in range(5000):
            params, opt_state, loss = step(params, opt_state)
            if i % 1000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}", flush=True)

        # Evaluate
        int_vals = decomp_intercept.lookup_flat(test_idx_arr, params["intercept"])
        intercept = int_vals[:, 0]
        logits = jnp.sum(X_test_j * params["beta"], axis=-1) + intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}", flush=True)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (sum-binning): {mean_auc:.4f} +/- {std_auc:.4f}", flush=True)
    print(f"  Table: 0.844, EBM: 0.866, LGBM: 0.872", flush=True)
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bioresponse()

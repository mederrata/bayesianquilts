"""Bank Marketing Ours #2 with Boosting + Theory-based regularization."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from itertools import combinations
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def load_bank_data():
    import zipfile
    import urllib.request
    from io import BytesIO

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
    cache_path = Path("data/bank/bank-full.csv")
    if cache_path.exists():
        df = pd.read_csv(cache_path, sep=";")
        if "y" in df.columns:
            return df
        cache_path.unlink()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        with zipfile.ZipFile(BytesIO(response.read())) as z:
            with z.open("bank-full.csv") as f:
                df = pd.read_csv(f, sep=";")
                df.to_csv(cache_path, sep=";", index=False)
                return df


def run_bank_boosting():
    """Bank Marketing Ours #2 with boosting + theory."""
    print("\n" + "="*60)
    print("BANK MARKETING - OURS #2 WITH BOOSTING")
    print("Theory-based regularization + residual fitting")
    print("="*60)

    df = load_bank_data()
    print(f"  Loaded {len(df)} rows")

    y = (df["y"] == "yes").astype(int).values
    N = len(y)
    print(f"  N = {N}, pos_rate = {y.mean():.3f}")

    # Same setup as improve_bank.py
    numeric_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    X_numeric = df[numeric_cols].values.astype(np.float32)
    X_numeric = np.nan_to_num(X_numeric, nan=0.0)

    lattice_cats_int = ["poutcome", "month", "contact", "housing"]
    lattice_cats_beta = ["poutcome", "contact"]

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_numeric, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train_num, X_test_num = X_numeric[train_idx], X_numeric[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_num_s = scaler.fit_transform(X_train_num)
        X_test_num_s = scaler.transform(X_test_num)

        # Bin duration for intercept lattice
        duration_edges = np.percentile(X_train_num_s[:, 2], [12.5, 25, 37.5, 50, 62.5, 75, 87.5])
        duration_bins_train = np.digitize(X_train_num_s[:, 2], duration_edges)
        duration_bins_test = np.digitize(X_test_num_s[:, 2], duration_edges)

        # Add pairwise interactions
        n_top = 4
        pairwise_train = []
        pairwise_test = []
        for i, j in combinations(range(n_top), 2):
            pairwise_train.append(X_train_num_s[:, i] * X_train_num_s[:, j])
            pairwise_test.append(X_test_num_s[:, i] * X_test_num_s[:, j])

        if pairwise_train:
            X_train = np.concatenate([X_train_num_s, np.stack(pairwise_train, axis=1)], axis=1)
            X_test = np.concatenate([X_test_num_s, np.stack(pairwise_test, axis=1)], axis=1)
        else:
            X_train, X_test = X_train_num_s, X_test_num_s

        N_train = len(y_train)
        n_features = X_train.shape[1]

        # Encode categoricals
        cat_indices_train = {}
        cat_indices_test = {}
        cat_n_levels = {}

        for col in set(lattice_cats_int + lattice_cats_beta):
            le = LabelEncoder()
            cat_indices_train[col] = le.fit_transform(df.iloc[train_idx][col].astype(str))
            cat_indices_test[col] = le.transform(df.iloc[test_idx][col].astype(str))
            cat_n_levels[col] = len(le.classes_)

        # Build intercept lattice (5 dims, order 3)
        dims_int = [
            Dimension("poutcome", cat_n_levels["poutcome"]),
            Dimension("month", cat_n_levels["month"]),
            Dimension("contact", cat_n_levels["contact"]),
            Dimension("housing", cat_n_levels["housing"]),
            Dimension("duration", 8),
        ]
        decomp_int = Decomposed(interactions=Interactions(dimensions=dims_int), param_shape=[1], name="intercept")

        # Build beta lattice (2 dims, order 2)
        dims_beta = [
            Dimension("poutcome", cat_n_levels["poutcome"]),
            Dimension("contact", cat_n_levels["contact"]),
        ]
        decomp_beta = Decomposed(interactions=Interactions(dimensions=dims_beta), param_shape=[n_features], name="beta")

        # Index arrays
        train_idx_int = np.stack([
            cat_indices_train["poutcome"], cat_indices_train["month"],
            cat_indices_train["contact"], cat_indices_train["housing"],
            duration_bins_train,
        ], axis=-1)
        test_idx_int = np.stack([
            cat_indices_test["poutcome"], cat_indices_test["month"],
            cat_indices_test["contact"], cat_indices_test["housing"],
            duration_bins_test,
        ], axis=-1)

        train_idx_beta = np.stack([cat_indices_train["poutcome"], cat_indices_train["contact"]], axis=-1)
        test_idx_beta = np.stack([cat_indices_test["poutcome"], cat_indices_test["contact"]], axis=-1)

        train_idx_int_j = jnp.array(train_idx_int)
        test_idx_int_j = jnp.array(test_idx_int)
        train_idx_beta_j = jnp.array(train_idx_beta)
        test_idx_beta_j = jnp.array(test_idx_beta)

        # Theory-based regularization scales
        prior_scales_int = decomp_int.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)
        prior_scales_beta = decomp_beta.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)

        # Limit to order 3 for intercept, order 2 for beta
        active_int = [n for n in decomp_int._tensor_parts.keys() if decomp_int.component_order(n) <= 3]
        active_beta = [n for n in decomp_beta._tensor_parts.keys() if decomp_beta.component_order(n) <= 2]

        print(f"    Intercept: {len(active_int)} components, Beta: {len(active_beta)} components")

        # Initialize
        params_int = {n: jnp.zeros(decomp_int._tensor_part_shapes[n]) for n in active_int}
        params_beta = {n: jnp.zeros(decomp_beta._tensor_part_shapes[n]) for n in active_beta}

        X_train_j = jnp.array(X_train)
        X_test_j = jnp.array(X_test)
        y_train_j = jnp.array(y_train)

        scale_mult = 50.0
        learning_rate_boost = 0.1
        n_boost_rounds = 10

        # BOOSTING
        accumulated_logits = jnp.zeros(N_train)

        for boost_round in range(n_boost_rounds):
            current_probs = 1 / (1 + jnp.exp(-accumulated_logits))
            residuals = y_train_j - current_probs

            # Fit intercept to residuals
            params_int_round = {n: jnp.zeros(decomp_int._tensor_part_shapes[n]) for n in active_int}

            def loss_int(p):
                vals = decomp_int.lookup_flat(train_idx_int_j, p)[:, 0]
                mse = jnp.mean((residuals - vals)**2)
                l2 = sum(0.5 * jnp.sum(p[n]**2) / ((prior_scales_int.get(n, 1.0) * scale_mult)**2 + 1e-8)
                        for n in active_int)
                return mse + l2 / N_train

            opt_i = optax.adam(0.02)
            opt_state_i = opt_i.init(params_int_round)

            @jax.jit
            def step_i(p, opt_state):
                loss, grads = jax.value_and_grad(loss_int)(p)
                updates, opt_state = opt_i.update(grads, opt_state, p)
                return optax.apply_updates(p, updates), opt_state, loss

            for _ in range(100):
                params_int_round, opt_state_i, _ = step_i(params_int_round, opt_state_i)

            for n in active_int:
                params_int[n] = params_int[n] + learning_rate_boost * params_int_round[n]

            # Update accumulated
            int_vals = decomp_int.lookup_flat(train_idx_int_j, params_int)[:, 0]
            accumulated_logits = int_vals

            # Fit beta to residuals
            current_probs = 1 / (1 + jnp.exp(-accumulated_logits))
            residuals = y_train_j - current_probs

            params_beta_round = {n: jnp.zeros(decomp_beta._tensor_part_shapes[n]) for n in active_beta}

            def loss_beta(p):
                beta_vals = decomp_beta.lookup_flat(train_idx_beta_j, p)
                pred = jnp.sum(X_train_j * beta_vals, axis=-1)
                mse = jnp.mean((residuals - pred)**2)
                l2 = sum(0.5 * jnp.sum(p[n]**2) / ((prior_scales_beta.get(n, 1.0) * scale_mult)**2 + 1e-8)
                        for n in active_beta)
                return mse + l2 / N_train

            opt_b = optax.adam(0.01)
            opt_state_b = opt_b.init(params_beta_round)

            @jax.jit
            def step_b(p, opt_state):
                loss, grads = jax.value_and_grad(loss_beta)(p)
                updates, opt_state = opt_b.update(grads, opt_state, p)
                return optax.apply_updates(p, updates), opt_state, loss

            for _ in range(100):
                params_beta_round, opt_state_b, _ = step_b(params_beta_round, opt_state_b)

            for n in active_beta:
                params_beta[n] = params_beta[n] + learning_rate_boost * params_beta_round[n]

            beta_vals = decomp_beta.lookup_flat(train_idx_beta_j, params_beta)
            accumulated_logits = int_vals + jnp.sum(X_train_j * beta_vals, axis=-1)

        print(f"    Completed {n_boost_rounds} boosting rounds")

        # Final joint refinement
        print("    Final joint refinement...")

        params = {"int": params_int, "beta": params_beta}

        def loss_joint(params):
            int_vals = decomp_int.lookup_flat(train_idx_int_j, params["int"])[:, 0]
            beta_vals = decomp_beta.lookup_flat(train_idx_beta_j, params["beta"])
            logits = jnp.sum(X_train_j * beta_vals, axis=-1) + int_vals

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_int = sum(0.5 * jnp.sum(params["int"][n]**2) / ((prior_scales_int.get(n, 1.0) * scale_mult)**2 + 1e-8)
                        for n in active_int)
            l2_beta = sum(0.5 * jnp.sum(params["beta"][n]**2) / ((prior_scales_beta.get(n, 1.0) * scale_mult)**2 + 1e-8)
                         for n in active_beta)

            return bce + (l2_int + l2_beta) / N_train

        opt_joint = optax.adam(0.005)
        opt_state_joint = opt_joint.init(params)

        @jax.jit
        def step_joint(params, opt_state):
            loss, grads = jax.value_and_grad(loss_joint)(params)
            updates, opt_state = opt_joint.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for i in range(2000):
            params, opt_state_joint, loss = step_joint(params, opt_state_joint)

        # Evaluate
        int_vals = decomp_int.lookup_flat(test_idx_int_j, params["int"])[:, 0]
        beta_vals = decomp_beta.lookup_flat(test_idx_beta_j, params["beta"])
        logits = jnp.sum(X_test_j * beta_vals, axis=-1) + int_vals
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS #2 (boosting): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  OURS #1 (improve_bank): 0.9175 +/- 0.0040")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bank_boosting()

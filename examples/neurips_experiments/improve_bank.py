"""Bank Marketing: improve by optimizing lattice structure."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
from itertools import combinations
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def load_bank_data():
    """Load bank marketing dataset."""
    import zipfile
    import urllib.request
    from io import BytesIO

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"

    cache_path = Path("data/bank/bank-full.csv")
    if cache_path.exists():
        df = pd.read_csv(cache_path, sep=";")
        if "y" in df.columns:
            return df
        # Cache is corrupted, remove it
        cache_path.unlink()

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response:
        with zipfile.ZipFile(BytesIO(response.read())) as z:
            with z.open("bank-full.csv") as f:
                df = pd.read_csv(f, sep=";")
                df.to_csv(cache_path, sep=";", index=False)
                return df


def run_bank():
    """Bank Marketing with improved config."""
    print("\n" + "="*60)
    print("BANK MARKETING - Improved Config")
    print("="*60)

    df = load_bank_data()
    print(f"  Loaded {len(df)} rows")

    # Target
    y = (df["y"] == "yes").astype(int).values
    N = len(y)
    print(f"  N = {N}")
    print(f"  Class balance: {y.mean():.3f}")

    # Numeric features
    numeric_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    X_numeric = df[numeric_cols].values.astype(np.float32)
    X_numeric = np.nan_to_num(X_numeric, nan=0.0)

    # Categorical features - based on LR coefficient analysis:
    # Most important: poutcome, month, contact, housing
    # For intercept (p=1): use all important cats
    # For beta (p=n_features): use fewer dims
    lattice_cats_int = ["poutcome", "month", "contact", "housing"]  # For intercept
    lattice_cats_beta = ["poutcome", "contact"]  # Fewer dims for beta

    # Numeric features only for regression (no one-hot categoricals)
    print(f"  Numeric features: {len(numeric_cols)}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_numeric, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train_num, X_test_num = X_numeric[train_idx], X_numeric[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize numeric features
        scaler = StandardScaler()
        X_train_num_s = scaler.fit_transform(X_train_num)
        X_test_num_s = scaler.transform(X_test_num)

        # PCA on numeric features for additional lattice dimension
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pca_train = pca.fit_transform(X_train_num_s)[:, 0]
        pca_test = pca.transform(X_test_num_s)[:, 0]

        # Add pairwise interactions for top numeric features
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
        print(f"    Features: {X_train_num_s.shape[1]} numeric + {n_features - X_train_num_s.shape[1]} pairwise = {n_features}")

        # Build TWO lattices: richer for intercept, simpler for beta
        train_indices = {}
        test_indices = {}

        # Encode all categoricals we might use
        for col in set(lattice_cats_int + lattice_cats_beta):
            le = LabelEncoder()
            col_vals = df[col].fillna("unknown").astype(str).values
            le.fit(col_vals)
            train_indices[col] = le.transform(df.iloc[train_idx][col].fillna("unknown").astype(str).values)
            test_indices[col] = le.transform(df.iloc[test_idx][col].fillna("unknown").astype(str).values)

        # Bin duration (top numeric predictor per LR)
        n_bins = 6
        dur_idx = numeric_cols.index("duration")
        dur_edges = np.percentile(X_train_num_s[:, dur_idx], np.linspace(0, 100, n_bins + 1))
        dur_edges = np.unique(dur_edges)
        dur_edges[0] = -np.inf
        dur_edges[-1] = np.inf
        train_indices["duration_bin"] = np.digitize(X_train_num_s[:, dur_idx], dur_edges[1:-1])
        test_indices["duration_bin"] = np.digitize(X_test_num_s[:, dur_idx], dur_edges[1:-1])

        # Intercept lattice: poutcome, month, contact, housing, duration_bin
        dims_int = []
        for col in lattice_cats_int:
            cardinality = len(np.unique(train_indices[col]))
            dims_int.append(Dimension(col, cardinality))
        dims_int.append(Dimension("duration_bin", len(dur_edges) - 1))

        # Beta lattice: poutcome, contact (simpler)
        dims_beta = []
        for col in lattice_cats_beta:
            cardinality = len(np.unique(train_indices[col]))
            dims_beta.append(Dimension(col, cardinality))

        if not dims_int:
            print("    No valid dimensions, skipping")
            continue

        # Separate decompositions with DIFFERENT lattices
        interactions_int = Interactions(dimensions=dims_int)
        interactions_beta = Interactions(dimensions=dims_beta)

        decomp_intercept = Decomposed(interactions=interactions_int, param_shape=[1], name="intercept")
        decomp_beta = Decomposed(interactions=interactions_beta, param_shape=[n_features], name="beta")

        # Get prior scales for both
        prior_scales_int = decomp_intercept.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True,
        )
        prior_scales_beta = decomp_beta.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True,
        )

        # Different index arrays for each lattice
        dim_names_int = [d.name for d in decomp_intercept._interactions._dimensions]
        dim_names_beta = [d.name for d in decomp_beta._interactions._dimensions]
        train_int_idx = jnp.stack([jnp.array(train_indices[name]) for name in dim_names_int], axis=-1)
        test_int_idx = jnp.stack([jnp.array(test_indices[name]) for name in dim_names_int], axis=-1)
        train_beta_idx = jnp.stack([jnp.array(train_indices[name]) for name in dim_names_beta], axis=-1)
        test_beta_idx = jnp.stack([jnp.array(test_indices[name]) for name in dim_names_beta], axis=-1)

        X_train_j = jnp.array(X_train)
        X_test_j = jnp.array(X_test)
        y_train_j = jnp.array(y_train)

        # Intercept (p=1): can go higher order on richer lattice
        # Beta (p=n_features): lower order on simpler lattice
        max_order_int = 3  # Intercept order 3 is optimal
        max_order_beta = 2  # Beta up to order 2

        active_int = [
            name for name in decomp_intercept._tensor_parts.keys()
            if decomp_intercept.component_order(name) <= max_order_int
        ]
        active_beta = [
            name for name in decomp_beta._tensor_parts.keys()
            if decomp_beta.component_order(name) <= max_order_beta
        ]
        print(f"    Intercept lattice: {len(dims_int)} dims, order {max_order_int}, {len(active_int)} components")
        print(f"    Beta lattice: {len(dims_beta)} dims, order {max_order_beta}, {len(active_beta)} components")

        params = {}
        for name in active_int:
            params[f"int_{name}"] = jnp.zeros(decomp_intercept._tensor_part_shapes[name])
        for name in active_beta:
            params[f"beta_{name}"] = jnp.zeros(decomp_beta._tensor_part_shapes[name])

        def loss_fn(params):
            # Extract intercept and beta params
            int_params = {k[4:]: v for k, v in params.items() if k.startswith("int_")}
            beta_params = {k[5:]: v for k, v in params.items() if k.startswith("beta_")}

            # Lookup decomposed values (different lattices!)
            cell_intercept = decomp_intercept.lookup_flat(train_int_idx, int_params)[:, 0]
            cell_beta = decomp_beta.lookup_flat(train_beta_idx, beta_params)

            logits = jnp.sum(X_train_j * cell_beta, axis=-1) + cell_intercept

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # L2 regularization on intercept
            l2_int = 0.0
            for name, param in int_params.items():
                scale = prior_scales_int.get(name, 1.0)
                l2_int += 0.5 * jnp.sum(param ** 2) / ((scale * 20) ** 2)

            # L2 regularization on beta
            l2_beta = 0.0
            for name, param in beta_params.items():
                scale = prior_scales_beta.get(name, 1.0)
                l2_beta += 0.5 * jnp.sum(param ** 2) / ((scale * 20) ** 2)

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
                print(f"    Step {i}: loss = {loss:.4f}")

        # Evaluate
        int_params = {k[4:]: v for k, v in params.items() if k.startswith("int_")}
        beta_params = {k[5:]: v for k, v in params.items() if k.startswith("beta_")}

        cell_intercept = decomp_intercept.lookup_flat(test_int_idx, int_params)[:, 0]
        cell_beta = decomp_beta.lookup_flat(test_beta_idx, beta_params)
        logits = jnp.sum(X_test_j * cell_beta, axis=-1) + cell_intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (improved): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Table: 0.905, EBM: 0.934")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bank()

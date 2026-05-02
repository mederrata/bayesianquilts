"""Bank Marketing v2: Original best config + theory scales + more features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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


def run_bank():
    print("BANK V2 - Original best config + theory scales + more features")
    print("="*70)

    df = load_bank_data()
    y = (df["y"] == "yes").astype(int).values
    N = len(y)
    print(f"N = {N}, pos_rate = {y.mean():.3f}")

    # Numeric features
    numeric_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    X_numeric = df[numeric_cols].values.astype(np.float32)
    X_numeric = np.nan_to_num(X_numeric, nan=0.0)

    # Encode categoricals for lattice
    le_poutcome = LabelEncoder().fit(df["poutcome"].astype(str))
    poutcome_bins = le_poutcome.transform(df["poutcome"].astype(str))
    n_poutcome = len(le_poutcome.classes_)

    le_month = LabelEncoder().fit(df["month"].astype(str))
    month_bins = le_month.transform(df["month"].astype(str))
    n_month = len(le_month.classes_)

    le_contact = LabelEncoder().fit(df["contact"].astype(str))
    contact_bins = le_contact.transform(df["contact"].astype(str))
    n_contact = len(le_contact.classes_)

    le_housing = LabelEncoder().fit(df["housing"].astype(str))
    housing_bins = le_housing.transform(df["housing"].astype(str))
    n_housing = len(le_housing.classes_)

    print(f"\nOriginal best config: 5 dim intercept (order 3), 2 dim beta (order 2)")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(N), y)):
        print(f"\nFold {fold_idx + 1}/5:")

        X_train_num = X_numeric[train_idx]
        X_test_num = X_numeric[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_num)
        X_test_s = scaler.transform(X_test_num)

        # Bin duration
        dur_idx = numeric_cols.index("duration")
        dur_edges = np.percentile(X_train_s[:, dur_idx], [12.5, 25, 37.5, 50, 62.5, 75, 87.5])
        dur_train = np.digitize(X_train_s[:, dur_idx], dur_edges)
        dur_test = np.digitize(X_test_s[:, dur_idx], dur_edges)

        # Add pairwise interactions
        n_top = 4
        pairwise_train = [X_train_s[:, i] * X_train_s[:, j] for i, j in combinations(range(n_top), 2)]
        pairwise_test = [X_test_s[:, i] * X_test_s[:, j] for i, j in combinations(range(n_top), 2)]

        X_train = np.concatenate([X_train_s, np.stack(pairwise_train, axis=1)], axis=1)
        X_test = np.concatenate([X_test_s, np.stack(pairwise_test, axis=1)], axis=1)

        y_train, y_test = y[train_idx], y[test_idx]
        N_train = len(y_train)
        n_features = X_train.shape[1]

        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        lr.fit(X_train, y_train)
        lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
        print(f"  n_features={n_features}, LR baseline AUC: {lr_auc:.4f}")

        # Intercept lattice: poutcome, month, contact, housing, duration (order 3)
        dims_int = [
            Dimension("poutcome", n_poutcome),
            Dimension("month", n_month),
            Dimension("contact", n_contact),
            Dimension("housing", n_housing),
            Dimension("duration", 8),
        ]
        decomp_int = Decomposed(interactions=Interactions(dimensions=dims_int), param_shape=[1], name="intercept")

        # Beta lattice: poutcome, contact (order 2)
        dims_beta = [Dimension("poutcome", n_poutcome), Dimension("contact", n_contact)]
        decomp_beta = Decomposed(interactions=Interactions(dimensions=dims_beta), param_shape=[n_features], name="beta")

        train_idx_int = np.stack([
            poutcome_bins[train_idx], month_bins[train_idx],
            contact_bins[train_idx], housing_bins[train_idx], dur_train,
        ], axis=-1)
        test_idx_int = np.stack([
            poutcome_bins[test_idx], month_bins[test_idx],
            contact_bins[test_idx], housing_bins[test_idx], dur_test,
        ], axis=-1)

        train_idx_beta = np.stack([poutcome_bins[train_idx], contact_bins[train_idx]], axis=-1)
        test_idx_beta = np.stack([poutcome_bins[test_idx], contact_bins[test_idx]], axis=-1)

        prior_scales_int = decomp_int.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)
        prior_scales_beta = decomp_beta.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)

        max_order_int = 3
        max_order_beta = 2

        active_int = [n for n in decomp_int._tensor_parts.keys() if decomp_int.component_order(n) <= max_order_int]
        active_beta = [n for n in decomp_beta._tensor_parts.keys() if decomp_beta.component_order(n) <= max_order_beta]

        params = {}
        for name in active_int:
            params[f"int_{name}"] = jnp.zeros(decomp_int._tensor_part_shapes[name])
        for name in active_beta:
            params[f"beta_{name}"] = jnp.zeros(decomp_beta._tensor_part_shapes[name])

        n_int_params = sum(np.prod(decomp_int._tensor_part_shapes[n]) for n in active_int)
        n_beta_params = sum(np.prod(decomp_beta._tensor_part_shapes[n]) for n in active_beta)
        print(f"  Intercept: {len(dims_int)} dims, order {max_order_int}, {n_int_params} params")
        print(f"  Beta: {len(dims_beta)} dims, order {max_order_beta}, {n_beta_params} params")

        train_indices_int = jnp.array(train_idx_int)
        train_indices_beta = jnp.array(train_idx_beta)
        X_train_j = jnp.array(X_train)
        y_train_j = jnp.array(y_train)

        scale_mult = 50.0

        def loss_fn(params):
            int_params = {k[4:]: v for k, v in params.items() if k.startswith("int_")}
            beta_params = {k[5:]: v for k, v in params.items() if k.startswith("beta_")}

            cell_int = decomp_int.lookup_flat(train_indices_int, int_params)[:, 0]
            cell_beta = decomp_beta.lookup_flat(train_indices_beta, beta_params)

            logits = jnp.sum(X_train_j * cell_beta, axis=-1) + cell_int
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_reg = 0.0
            for name, p in int_params.items():
                scale = prior_scales_int.get(name, 1.0)
                l2_reg += 0.5 * jnp.sum(p**2) / ((scale * scale_mult)**2 + 1e-8)
            for name, p in beta_params.items():
                scale = prior_scales_beta.get(name, 1.0)
                l2_reg += 0.5 * jnp.sum(p**2) / ((scale * scale_mult)**2 + 1e-8)

            return bce + l2_reg / N_train

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.001, peak_value=0.02, warmup_steps=500,
            decay_steps=5500, end_value=0.001
        )
        opt = optax.adam(learning_rate=schedule)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for i in range(6001):
            params, opt_state, loss = step(params, opt_state)
            if i % 1000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}")

        # Evaluate
        test_indices_int = jnp.array(test_idx_int)
        test_indices_beta = jnp.array(test_idx_beta)
        X_test_j = jnp.array(X_test)

        int_params = {k[4:]: v for k, v in params.items() if k.startswith("int_")}
        beta_params = {k[5:]: v for k, v in params.items() if k.startswith("beta_")}

        cell_int = decomp_int.lookup_flat(test_indices_int, int_params)[:, 0]
        cell_beta = decomp_beta.lookup_flat(test_indices_beta, beta_params)
        logits = jnp.sum(X_test_j * cell_beta, axis=-1) + cell_int
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"  AUC = {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\nOURS: {mean_auc:.4f} +/- {std_auc:.4f}")
    print("Baselines: LR=0.907, LGBM=0.935, EBM=0.934")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bank()

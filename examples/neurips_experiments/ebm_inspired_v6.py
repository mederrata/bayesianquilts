"""EBM-inspired v6: Faster version with hard binning, fewer iterations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import jax
import jax.numpy as jnp
import optax


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


def run_bank_ebm_v6():
    """Bank with EBM-inspired approach - faster version."""
    print("BANK - EBM-Inspired v6: Faster (hard bins, fewer iters)")
    print("="*70)

    df = load_bank_data()
    y = (df["y"] == "yes").astype(int).values
    N = len(y)
    print(f"N = {N}, pos_rate = {y.mean():.3f}")

    numeric_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    X_numeric = df[numeric_cols].values.astype(np.float32)
    X_numeric = np.nan_to_num(X_numeric, nan=0.0)

    cat_cols = ["poutcome", "contact", "housing", "month"]
    cat_encoders = {}
    cat_bins = {}
    for col in cat_cols:
        le = LabelEncoder().fit(df[col].astype(str))
        cat_encoders[col] = le
        cat_bins[col] = le.transform(df[col].astype(str))

    n_cats = {col: len(le.classes_) for col, le in cat_encoders.items()}

    print(f"\nHard binning (non-differentiable but fast)")
    print(f"5k iterations instead of 20k")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    n_bins = 8  # Fewer bins for speed

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(N), y)):
        print(f"\nFold {fold_idx + 1}/5:")

        X_train_num = X_numeric[train_idx]
        X_test_num = X_numeric[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_num)
        X_test_s = scaler.transform(X_test_num)

        y_train, y_test = y[train_idx], y[test_idx]
        N_train = len(y_train)

        # Hard bin continuous features using quantiles
        cont_bins_train = {}
        cont_bins_test = {}
        for i, col in enumerate(numeric_cols):
            edges = np.percentile(X_train_s[:, i], np.linspace(0, 100, n_bins + 1)[1:-1])
            cont_bins_train[col] = np.digitize(X_train_s[:, i], edges)
            cont_bins_test[col] = np.digitize(X_test_s[:, i], edges)

        params = {"global_intercept": jnp.array(0.0)}

        # Categorical main effects
        for col in cat_cols:
            params[f"cat_{col}"] = jnp.zeros(n_cats[col])

        # Continuous main effects (per-bin values)
        for col in numeric_cols:
            params[f"cont_{col}"] = jnp.zeros(n_bins)

        # Key interactions
        params["pout_dur"] = jnp.zeros((n_cats["poutcome"], n_bins))
        params["cont_dur"] = jnp.zeros((n_cats["contact"], n_bins))
        params["pout_cont"] = jnp.zeros((n_cats["poutcome"], n_cats["contact"]))

        total_params = sum(p.size for p in params.values())
        print(f"  Total params: {total_params}")

        y_train_j = jnp.array(y_train)

        cat_train_j = {col: jnp.array(cat_bins[col][train_idx]) for col in cat_cols}
        cont_train_j = {col: jnp.array(cont_bins_train[col]) for col in numeric_cols}

        def loss_fn(params):
            logits = params["global_intercept"] * jnp.ones(N_train)

            # Categorical main effects
            for col in cat_cols:
                logits = logits + params[f"cat_{col}"][cat_train_j[col]]

            # Continuous main effects
            for col in numeric_cols:
                logits = logits + params[f"cont_{col}"][cont_train_j[col]]

            # Interactions
            pout_idx = cat_train_j["poutcome"]
            cont_idx = cat_train_j["contact"]
            dur_idx = cont_train_j["duration"]

            logits = logits + params["pout_dur"][pout_idx, dur_idx]
            logits = logits + params["cont_dur"][cont_idx, dur_idx]
            logits = logits + params["pout_cont"][pout_idx, cont_idx]

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # L2 regularization
            l2_reg = 0.0
            for key, val in params.items():
                l2_reg += 0.01 * jnp.sum(val**2)

            return bce + l2_reg / N_train

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.001, peak_value=0.02, warmup_steps=500,
            decay_steps=4500, end_value=0.001
        )
        opt = optax.adam(learning_rate=schedule)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for i in range(5001):
            params, opt_state, loss = step(params, opt_state)
            if i % 1000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}")

        # Evaluate on test
        cat_test_j = {col: jnp.array(cat_bins[col][test_idx]) for col in cat_cols}
        cont_test_j = {col: jnp.array(cont_bins_test[col]) for col in numeric_cols}

        logits_test = params["global_intercept"] * jnp.ones(len(y_test))

        for col in cat_cols:
            logits_test = logits_test + params[f"cat_{col}"][cat_test_j[col]]

        for col in numeric_cols:
            logits_test = logits_test + params[f"cont_{col}"][cont_test_j[col]]

        pout_idx_test = cat_test_j["poutcome"]
        cont_idx_test = cat_test_j["contact"]
        dur_idx_test = cont_test_j["duration"]

        logits_test = logits_test + params["pout_dur"][pout_idx_test, dur_idx_test]
        logits_test = logits_test + params["cont_dur"][cont_idx_test, dur_idx_test]
        logits_test = logits_test + params["pout_cont"][pout_idx_test, cont_idx_test]

        probs = 1 / (1 + jnp.exp(-logits_test))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"  AUC = {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\nOURS (EBM-inspired v6): {mean_auc:.4f} +/- {std_auc:.4f}")
    print("Baselines: LR=0.907, LGBM=0.935, EBM=0.934")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bank_ebm_v6()

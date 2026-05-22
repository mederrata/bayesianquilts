"""EBM-inspired v7: ReLU hinges for piecewise linear with learnable splits."""
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


def piecewise_linear_relu(x, splits, slopes):
    """Piecewise linear function using ReLU hinges.

    f(x) = sum_k slopes[k] * relu(x - splits[k])

    This creates a piecewise linear function with learned breakpoints.
    Much faster than soft binning - no intermediate arrays needed.
    """
    # x: (N,), splits: (n_splits,), slopes: (n_splits,)
    # relu(x - splits): (N, n_splits)
    hinges = jax.nn.relu(x[:, None] - splits[None, :])  # (N, n_splits)
    return jnp.sum(hinges * slopes[None, :], axis=-1)  # (N,)


def run_bank_ebm_v7():
    """Bank with ReLU-based piecewise linear functions."""
    print("BANK - EBM-Inspired v7: ReLU hinges (fast + learnable splits)")
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

    print(f"\nReLU hinges: f(x) = sum_k slope_k * relu(x - split_k)")
    print(f"Piecewise linear with learnable splits, no bin arrays")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    n_splits = 15  # 16 pieces per feature

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(N), y)):
        print(f"\nFold {fold_idx + 1}/5:")

        X_train_num = X_numeric[train_idx]
        X_test_num = X_numeric[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_num)
        X_test_s = scaler.transform(X_test_num)

        y_train, y_test = y[train_idx], y[test_idx]
        N_train = len(y_train)

        # Initialize splits at percentiles
        splits_init = {}
        for i, col in enumerate(numeric_cols):
            percs = np.linspace(100/(n_splits+1), 100 - 100/(n_splits+1), n_splits)
            splits_init[col] = np.percentile(X_train_s[:, i], percs)

        params = {"global_intercept": jnp.array(0.0)}

        # Categorical main effects
        for col in cat_cols:
            params[f"cat_{col}"] = jnp.zeros(n_cats[col])

        # Continuous main effects: base slope + ReLU hinges
        for col in numeric_cols:
            params[f"base_{col}"] = jnp.array(0.0)  # base slope
            params[f"splits_{col}"] = jnp.array(splits_init[col])
            params[f"slopes_{col}"] = jnp.zeros(n_splits)  # slope change at each hinge

        # Interactions: categorical × piecewise linear
        # poutcome × duration
        params["pout_dur_base"] = jnp.zeros(n_cats["poutcome"])
        params["pout_dur_slopes"] = jnp.zeros((n_cats["poutcome"], n_splits))

        # contact × duration
        params["cont_dur_base"] = jnp.zeros(n_cats["contact"])
        params["cont_dur_slopes"] = jnp.zeros((n_cats["contact"], n_splits))

        # Cat × Cat
        params["pout_cont"] = jnp.zeros((n_cats["poutcome"], n_cats["contact"]))

        total_params = sum(p.size for p in params.values())
        print(f"  Total params: {total_params}")

        X_train_j = jnp.array(X_train_s)
        y_train_j = jnp.array(y_train)

        cat_train_j = {col: jnp.array(cat_bins[col][train_idx]) for col in cat_cols}

        def loss_fn(params):
            logits = params["global_intercept"] * jnp.ones(N_train)

            # Categorical main effects
            for col in cat_cols:
                logits = logits + params[f"cat_{col}"][cat_train_j[col]]

            # Continuous main effects: piecewise linear
            for i, col in enumerate(numeric_cols):
                x = X_train_j[:, i]
                # base linear + ReLU hinges
                contrib = params[f"base_{col}"] * x
                contrib = contrib + piecewise_linear_relu(x, params[f"splits_{col}"], params[f"slopes_{col}"])
                logits = logits + contrib

            # Interactions
            dur_idx = numeric_cols.index("duration")
            dur_x = X_train_j[:, dur_idx]
            dur_splits = params["splits_duration"]

            pout_idx = cat_train_j["poutcome"]
            cont_idx = cat_train_j["contact"]

            # poutcome × duration: per-category piecewise linear
            pout_base = params["pout_dur_base"][pout_idx]
            pout_slopes = params["pout_dur_slopes"][pout_idx, :]  # (N, n_splits)
            hinges = jax.nn.relu(dur_x[:, None] - dur_splits[None, :])
            pout_dur_effect = pout_base * dur_x + jnp.sum(hinges * pout_slopes, axis=-1)
            logits = logits + pout_dur_effect

            # contact × duration
            cont_base = params["cont_dur_base"][cont_idx]
            cont_slopes = params["cont_dur_slopes"][cont_idx, :]
            cont_dur_effect = cont_base * dur_x + jnp.sum(hinges * cont_slopes, axis=-1)
            logits = logits + cont_dur_effect

            # Cat × Cat
            logits = logits + params["pout_cont"][pout_idx, cont_idx]

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # L2 regularization
            l2_reg = 0.0
            for key, val in params.items():
                if "splits" not in key:
                    l2_reg += 0.01 * jnp.sum(val**2)

            # Split ordering penalty (encourage sorted splits)
            split_penalty = 0.0
            for col in numeric_cols:
                splits = params[f"splits_{col}"]
                split_penalty += jnp.sum(jax.nn.relu(splits[:-1] - splits[1:] + 0.05))

            return bce + l2_reg / N_train + 0.1 * split_penalty

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.001, peak_value=0.01, warmup_steps=500,
            decay_steps=9500, end_value=0.001
        )
        opt = optax.adam(learning_rate=schedule)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for i in range(10001):
            params, opt_state, loss = step(params, opt_state)
            if i % 2000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}")

        # Evaluate on test
        X_test_j = jnp.array(X_test_s)
        cat_test_j = {col: jnp.array(cat_bins[col][test_idx]) for col in cat_cols}

        logits_test = params["global_intercept"] * jnp.ones(len(y_test))

        for col in cat_cols:
            logits_test = logits_test + params[f"cat_{col}"][cat_test_j[col]]

        for i, col in enumerate(numeric_cols):
            x = X_test_j[:, i]
            contrib = params[f"base_{col}"] * x
            contrib = contrib + piecewise_linear_relu(x, params[f"splits_{col}"], params[f"slopes_{col}"])
            logits_test = logits_test + contrib

        dur_x_test = X_test_j[:, numeric_cols.index("duration")]
        dur_splits = params["splits_duration"]

        pout_idx_test = cat_test_j["poutcome"]
        cont_idx_test = cat_test_j["contact"]

        pout_base = params["pout_dur_base"][pout_idx_test]
        pout_slopes = params["pout_dur_slopes"][pout_idx_test, :]
        hinges_test = jax.nn.relu(dur_x_test[:, None] - dur_splits[None, :])
        logits_test = logits_test + pout_base * dur_x_test + jnp.sum(hinges_test * pout_slopes, axis=-1)

        cont_base = params["cont_dur_base"][cont_idx_test]
        cont_slopes = params["cont_dur_slopes"][cont_idx_test, :]
        logits_test = logits_test + cont_base * dur_x_test + jnp.sum(hinges_test * cont_slopes, axis=-1)

        logits_test = logits_test + params["pout_cont"][pout_idx_test, cont_idx_test]

        probs = 1 / (1 + jnp.exp(-logits_test))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"  AUC = {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\nOURS (EBM-inspired v7): {mean_auc:.4f} +/- {std_auc:.4f}")
    print("Baselines: LR=0.907, LGBM=0.935, EBM=0.934")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bank_ebm_v7()

"""EBM-inspired v8: ReLU hinges at both lattice and regressor levels."""
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


def run_bank_ebm_v8():
    """Bank with ReLU hinges at lattice AND regressor levels."""
    print("BANK - EBM-Inspired v8: ReLU hinges at lattice + regressor levels")
    print("="*70)

    df = load_bank_data()
    y = (df["y"] == "yes").astype(int).values
    N = len(y)
    print(f"N = {N}, pos_rate = {y.mean():.3f}")

    numeric_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    X_numeric = df[numeric_cols].values.astype(np.float32)
    X_numeric = np.nan_to_num(X_numeric, nan=0.0)
    n_features = len(numeric_cols)

    # Key categoricals for lattice
    cat_cols = ["poutcome", "contact", "housing"]
    cat_encoders = {}
    cat_bins = {}
    for col in cat_cols:
        le = LabelEncoder().fit(df[col].astype(str))
        cat_encoders[col] = le
        cat_bins[col] = le.transform(df[col].astype(str))

    n_cats = {col: len(le.classes_) for col, le in cat_encoders.items()}

    print(f"\nTwo-level ReLU hinges:")
    print(f"  Lattice: intercept[poutcome, contact] + hinges on duration")
    print(f"  Regressor: beta[poutcome, contact] + hinges on each feature")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    n_splits_lattice = 15  # hinges for duration in lattice
    n_splits_reg = 10  # hinges per feature for regressors

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
        dur_idx = numeric_cols.index("duration")
        dur_splits_init = np.percentile(
            X_train_s[:, dur_idx],
            np.linspace(100/(n_splits_lattice+1), 100 - 100/(n_splits_lattice+1), n_splits_lattice)
        )

        reg_splits_init = {}
        for i, col in enumerate(numeric_cols):
            percs = np.linspace(100/(n_splits_reg+1), 100 - 100/(n_splits_reg+1), n_splits_reg)
            reg_splits_init[col] = np.percentile(X_train_s[:, i], percs)

        params = {}

        # === LATTICE LEVEL (intercept) ===
        # Categorical main effects
        for col in cat_cols:
            params[f"cat_{col}"] = jnp.zeros(n_cats[col])

        # Cat × Cat interaction
        params["pout_cont"] = jnp.zeros((n_cats["poutcome"], n_cats["contact"]))

        # Duration hinges that vary by poutcome × contact (piecewise linear intercept)
        params["dur_splits"] = jnp.array(dur_splits_init)
        # Base intercept per cell
        params["cell_base"] = jnp.zeros((n_cats["poutcome"], n_cats["contact"]))
        # Duration base slope per cell
        params["cell_dur_base"] = jnp.zeros((n_cats["poutcome"], n_cats["contact"]))
        # Duration hinge slopes per cell
        params["cell_dur_hinges"] = jnp.zeros((n_cats["poutcome"], n_cats["contact"], n_splits_lattice))

        # === REGRESSOR LEVEL (beta) ===
        # For each feature: splits + base beta + hinge slopes, varying by poutcome × contact
        for i, col in enumerate(numeric_cols):
            params[f"reg_splits_{col}"] = jnp.array(reg_splits_init[col])
            # Base beta per cell
            params[f"beta_base_{col}"] = jnp.zeros((n_cats["poutcome"], n_cats["contact"]))
            # Hinge slopes for beta per cell (how beta changes at each hinge)
            params[f"beta_hinges_{col}"] = jnp.zeros((n_cats["poutcome"], n_cats["contact"], n_splits_reg))

        total_params = sum(p.size for p in params.values())
        print(f"  Total params: {total_params}")

        X_train_j = jnp.array(X_train_s)
        y_train_j = jnp.array(y_train)

        cat_train_j = {col: jnp.array(cat_bins[col][train_idx]) for col in cat_cols}

        def loss_fn(params):
            pout_idx = cat_train_j["poutcome"]
            cont_idx = cat_train_j["contact"]

            # === LATTICE LEVEL ===
            # Categorical main effects
            logits = jnp.zeros(N_train)
            for col in cat_cols:
                logits = logits + params[f"cat_{col}"][cat_train_j[col]]

            # Cat × Cat
            logits = logits + params["pout_cont"][pout_idx, cont_idx]

            # Cell intercept with duration hinges
            # intercept[cell](dur) = base[cell] + dur_base[cell] * dur + sum_k dur_hinge_k[cell] * relu(dur - split_k)
            dur_x = X_train_j[:, dur_idx]
            dur_splits = params["dur_splits"]

            cell_base = params["cell_base"][pout_idx, cont_idx]  # (N,)
            cell_dur_base = params["cell_dur_base"][pout_idx, cont_idx]  # (N,)
            cell_dur_hinges = params["cell_dur_hinges"][pout_idx, cont_idx, :]  # (N, n_splits_lattice)

            dur_relu = jax.nn.relu(dur_x[:, None] - dur_splits[None, :])  # (N, n_splits_lattice)
            intercept = cell_base + cell_dur_base * dur_x + jnp.sum(dur_relu * cell_dur_hinges, axis=-1)
            logits = logits + intercept

            # === REGRESSOR LEVEL ===
            # For each feature: beta[cell](x_i) * x_i where beta is piecewise linear
            for i, col in enumerate(numeric_cols):
                x_i = X_train_j[:, i]
                splits_i = params[f"reg_splits_{col}"]

                beta_base = params[f"beta_base_{col}"][pout_idx, cont_idx]  # (N,)
                beta_hinges = params[f"beta_hinges_{col}"][pout_idx, cont_idx, :]  # (N, n_splits_reg)

                # beta(x) = beta_base + sum_k beta_hinge_k * relu(x - split_k)
                x_relu = jax.nn.relu(x_i[:, None] - splits_i[None, :])  # (N, n_splits_reg)
                beta_i = beta_base + jnp.sum(x_relu * beta_hinges, axis=-1)

                # Contribution: beta(x_i) * x_i
                logits = logits + beta_i * x_i

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # L2 regularization
            l2_reg = 0.0
            for key, val in params.items():
                if "splits" not in key:
                    l2_reg += 0.01 * jnp.sum(val**2)

            # Split ordering penalty
            split_penalty = 0.0
            split_penalty += jnp.sum(jax.nn.relu(params["dur_splits"][:-1] - params["dur_splits"][1:] + 0.05))
            for col in numeric_cols:
                splits = params[f"reg_splits_{col}"]
                split_penalty += jnp.sum(jax.nn.relu(splits[:-1] - splits[1:] + 0.05))

            return bce + l2_reg / N_train + 0.1 * split_penalty

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.001, peak_value=0.01, warmup_steps=1000,
            decay_steps=14000, end_value=0.001
        )
        opt = optax.adam(learning_rate=schedule)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for i in range(15001):
            params, opt_state, loss = step(params, opt_state)
            if i % 3000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}")

        # === EVALUATE ON TEST ===
        X_test_j = jnp.array(X_test_s)
        cat_test_j = {col: jnp.array(cat_bins[col][test_idx]) for col in cat_cols}

        pout_idx_test = cat_test_j["poutcome"]
        cont_idx_test = cat_test_j["contact"]

        logits_test = jnp.zeros(len(y_test))
        for col in cat_cols:
            logits_test = logits_test + params[f"cat_{col}"][cat_test_j[col]]

        logits_test = logits_test + params["pout_cont"][pout_idx_test, cont_idx_test]

        dur_x_test = X_test_j[:, dur_idx]
        dur_splits = params["dur_splits"]

        cell_base = params["cell_base"][pout_idx_test, cont_idx_test]
        cell_dur_base = params["cell_dur_base"][pout_idx_test, cont_idx_test]
        cell_dur_hinges = params["cell_dur_hinges"][pout_idx_test, cont_idx_test, :]

        dur_relu_test = jax.nn.relu(dur_x_test[:, None] - dur_splits[None, :])
        intercept_test = cell_base + cell_dur_base * dur_x_test + jnp.sum(dur_relu_test * cell_dur_hinges, axis=-1)
        logits_test = logits_test + intercept_test

        for i, col in enumerate(numeric_cols):
            x_i = X_test_j[:, i]
            splits_i = params[f"reg_splits_{col}"]

            beta_base = params[f"beta_base_{col}"][pout_idx_test, cont_idx_test]
            beta_hinges = params[f"beta_hinges_{col}"][pout_idx_test, cont_idx_test, :]

            x_relu = jax.nn.relu(x_i[:, None] - splits_i[None, :])
            beta_i = beta_base + jnp.sum(x_relu * beta_hinges, axis=-1)
            logits_test = logits_test + beta_i * x_i

        probs = 1 / (1 + jnp.exp(-logits_test))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"  AUC = {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\nOURS (EBM-inspired v8): {mean_auc:.4f} +/- {std_auc:.4f}")
    print("Baselines: LR=0.907, LGBM=0.935, EBM=0.934")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bank_ebm_v8()

"""EBM-inspired v3: Piecewise linear regressors with learned splits per feature."""
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


def soft_bin_probs(x, split_points, temperature=0.3):
    """Compute soft bin assignment probabilities."""
    n_splits = len(split_points)

    above = jax.nn.sigmoid((x[:, None] - split_points[None, :]) / temperature)
    below = 1 - above

    probs = []
    probs.append(below[:, 0])

    cum_above = above[:, 0]
    for k in range(1, n_splits):
        probs.append(cum_above * below[:, k])
        cum_above = cum_above * above[:, k]

    probs.append(cum_above)
    return jnp.stack(probs, axis=-1)


def piecewise_linear_contribution(x, split_points, bin_intercepts, bin_slopes, temperature=0.3):
    """Compute piecewise linear function f(x) with learned splits.

    Within each bin, f(x) = intercept_k + slope_k * (x - midpoint_k)
    We use soft bin assignment for differentiability.
    """
    n_splits = len(split_points)
    n_bins = n_splits + 1

    # Compute soft bin probabilities
    bin_probs = soft_bin_probs(x, split_points, temperature)  # (N, n_bins)

    # Compute bin midpoints for local coordinate
    # First bin: midpoint at -inf side, use first split as reference
    # Last bin: midpoint at +inf side, use last split as reference
    # Middle bins: average of adjacent splits

    # For simplicity, compute local x coordinate as (x - split_k) for bin k
    # But we need soft assignment, so we'll compute weighted sum

    # Contribution from each bin: intercept + slope * x (using global x for simplicity)
    # This is simpler and still allows piecewise linear behavior
    per_bin_contrib = bin_intercepts[None, :] + bin_slopes[None, :] * x[:, None]  # (N, n_bins)

    # Weighted sum over bins
    return jnp.sum(bin_probs * per_bin_contrib, axis=-1)


def run_bank_ebm_v3():
    """Bank with EBM-inspired piecewise linear regressors."""
    print("BANK - EBM-Inspired v3: Piecewise linear regressors per feature")
    print("="*70)

    df = load_bank_data()
    y = (df["y"] == "yes").astype(int).values
    N = len(y)
    print(f"N = {N}, pos_rate = {y.mean():.3f}")

    numeric_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    X_numeric = df[numeric_cols].values.astype(np.float32)
    X_numeric = np.nan_to_num(X_numeric, nan=0.0)

    cat_encoders = {}
    cat_bins = {}
    for col in ["poutcome", "contact", "housing", "month", "job", "marital", "education"]:
        le = LabelEncoder().fit(df[col].astype(str))
        cat_encoders[col] = le
        cat_bins[col] = le.transform(df[col].astype(str))

    n_cats = {col: len(le.classes_) for col, le in cat_encoders.items()}

    print(f"\nOrder 1: Categorical main effects + piecewise linear continuous")
    print(f"Order 2: Cat×Cont with piecewise linear + Cat×Cat")
    print(f"Regressors: Piecewise linear per feature, slopes vary by poutcome×contact")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    n_splits = 10  # 11 bins per feature

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(N), y)):
        print(f"\nFold {fold_idx + 1}/5:")

        X_train_num = X_numeric[train_idx]
        X_test_num = X_numeric[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_num)
        X_test_s = scaler.transform(X_test_num)

        y_train, y_test = y[train_idx], y[test_idx]
        N_train = len(y_train)
        n_features = len(numeric_cols)

        # Initialize split points at percentiles
        cont_splits_init = {}
        for i, col in enumerate(numeric_cols):
            percs = np.linspace(100/(n_splits+1), 100 - 100/(n_splits+1), n_splits)
            cont_splits_init[col] = np.percentile(X_train_s[:, i], percs)

        params = {"global_intercept": jnp.array(0.0)}

        # Order 1: Categorical main effects
        for col in cat_encoders:
            params[f"main_{col}"] = jnp.zeros(n_cats[col])

        # Order 1: Piecewise linear per continuous feature (intercepts + slopes)
        for i, col in enumerate(numeric_cols):
            params[f"splits_{col}"] = jnp.array(cont_splits_init[col])
            params[f"pwl_int_{col}"] = jnp.zeros(n_splits + 1)  # bin intercepts
            params[f"pwl_slope_{col}"] = jnp.zeros(n_splits + 1)  # bin slopes

        # Order 2: Key Cat×Cont interactions with piecewise linear
        # poutcome × duration (piecewise)
        params["pout_dur_int"] = jnp.zeros((n_cats["poutcome"], n_splits + 1))
        params["pout_dur_slope"] = jnp.zeros((n_cats["poutcome"], n_splits + 1))

        # contact × duration (piecewise)
        params["cont_dur_int"] = jnp.zeros((n_cats["contact"], n_splits + 1))
        params["cont_dur_slope"] = jnp.zeros((n_cats["contact"], n_splits + 1))

        # housing × duration (piecewise)
        params["hous_dur_int"] = jnp.zeros((n_cats["housing"], n_splits + 1))
        params["hous_dur_slope"] = jnp.zeros((n_cats["housing"], n_splits + 1))

        # Order 2: Cat×Cat
        params["pout_cont_inter"] = jnp.zeros((n_cats["poutcome"], n_cats["contact"]))
        params["pout_hous_inter"] = jnp.zeros((n_cats["poutcome"], n_cats["housing"]))

        # Piecewise linear regressors with slopes varying by poutcome × contact
        # For each feature: learned splits + per-bin base slopes + poutcome/contact adjustments
        for i, col in enumerate(numeric_cols):
            params[f"reg_splits_{col}"] = jnp.array(cont_splits_init[col])
            params[f"reg_slope_{col}"] = jnp.zeros(n_splits + 1)  # base per-bin slopes
            params[f"reg_slope_pout_{col}"] = jnp.zeros((n_cats["poutcome"], n_splits + 1))
            params[f"reg_slope_cont_{col}"] = jnp.zeros((n_cats["contact"], n_splits + 1))

        total_params = sum(p.size for p in params.values())
        print(f"  Total params: {total_params}")

        X_train_j = jnp.array(X_train_s)
        y_train_j = jnp.array(y_train)

        cat_train = {col: jnp.array(cat_bins[col][train_idx]) for col in cat_encoders}
        cat_test = {col: jnp.array(cat_bins[col][test_idx]) for col in cat_encoders}

        temperature = 0.3

        def loss_fn(params):
            # Order 0: Global intercept
            logits = params["global_intercept"] * jnp.ones(N_train)

            # Order 1: Categorical main effects
            for col in cat_encoders:
                logits = logits + params[f"main_{col}"][cat_train[col]]

            # Order 1: Piecewise linear continuous main effects
            for i, col in enumerate(numeric_cols):
                contrib = piecewise_linear_contribution(
                    X_train_j[:, i],
                    params[f"splits_{col}"],
                    params[f"pwl_int_{col}"],
                    params[f"pwl_slope_{col}"],
                    temperature
                )
                logits = logits + contrib

            # Order 2: Cat×Cont with piecewise linear
            pout_idx = cat_train["poutcome"]
            cont_idx = cat_train["contact"]
            hous_idx = cat_train["housing"]
            dur_col_idx = numeric_cols.index("duration")
            dur_x = X_train_j[:, dur_col_idx]
            dur_splits = params["splits_duration"]
            dur_probs = soft_bin_probs(dur_x, dur_splits, temperature)

            # poutcome × duration
            pout_dur_int = params["pout_dur_int"][pout_idx, :]
            pout_dur_slope = params["pout_dur_slope"][pout_idx, :]
            per_bin = pout_dur_int + pout_dur_slope * dur_x[:, None]
            logits = logits + jnp.sum(dur_probs * per_bin, axis=-1)

            # contact × duration
            cont_dur_int = params["cont_dur_int"][cont_idx, :]
            cont_dur_slope = params["cont_dur_slope"][cont_idx, :]
            per_bin = cont_dur_int + cont_dur_slope * dur_x[:, None]
            logits = logits + jnp.sum(dur_probs * per_bin, axis=-1)

            # housing × duration
            hous_dur_int = params["hous_dur_int"][hous_idx, :]
            hous_dur_slope = params["hous_dur_slope"][hous_idx, :]
            per_bin = hous_dur_int + hous_dur_slope * dur_x[:, None]
            logits = logits + jnp.sum(dur_probs * per_bin, axis=-1)

            # Cat×Cat interactions
            logits = logits + params["pout_cont_inter"][pout_idx, cont_idx]
            logits = logits + params["pout_hous_inter"][pout_idx, hous_idx]

            # Piecewise linear regressors with varying slopes
            for i, col in enumerate(numeric_cols):
                x_i = X_train_j[:, i]
                reg_splits = params[f"reg_splits_{col}"]
                reg_probs = soft_bin_probs(x_i, reg_splits, temperature)

                # Base slopes + poutcome/contact adjustments
                base_slopes = params[f"reg_slope_{col}"]
                pout_slopes = params[f"reg_slope_pout_{col}"][pout_idx, :]
                cont_slopes = params[f"reg_slope_cont_{col}"][cont_idx, :]
                total_slopes = base_slopes[None, :] + pout_slopes + cont_slopes

                # Per-bin contribution: slope * x
                per_bin_contrib = total_slopes * x_i[:, None]
                logits = logits + jnp.sum(reg_probs * per_bin_contrib, axis=-1)

            # BCE loss
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # L2 regularization
            l2_reg = 0.0
            for key, val in params.items():
                if "splits" not in key:
                    l2_reg += 0.01 * jnp.sum(val**2)

            # Split ordering penalty
            split_penalty = 0.0
            for col in numeric_cols:
                for prefix in ["splits_", "reg_splits_"]:
                    key = f"{prefix}{col}"
                    if key in params:
                        splits = params[key]
                        split_penalty += jnp.sum(jax.nn.relu(splits[:-1] - splits[1:] + 0.05))

            return bce + l2_reg / N_train + 0.1 * split_penalty

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0005, peak_value=0.005, warmup_steps=2000,
            decay_steps=18000, end_value=0.0005
        )
        opt = optax.adam(learning_rate=schedule)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for i in range(20001):
            params, opt_state, loss = step(params, opt_state)
            if i % 4000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}")

        # Evaluate on test
        X_test_j = jnp.array(X_test_s)

        logits_test = params["global_intercept"] * jnp.ones(len(y_test))

        for col in cat_encoders:
            logits_test = logits_test + params[f"main_{col}"][cat_test[col]]

        for i, col in enumerate(numeric_cols):
            contrib = piecewise_linear_contribution(
                X_test_j[:, i],
                params[f"splits_{col}"],
                params[f"pwl_int_{col}"],
                params[f"pwl_slope_{col}"],
                temperature
            )
            logits_test = logits_test + contrib

        pout_idx_test = cat_test["poutcome"]
        cont_idx_test = cat_test["contact"]
        hous_idx_test = cat_test["housing"]
        dur_x_test = X_test_j[:, numeric_cols.index("duration")]
        dur_probs_test = soft_bin_probs(dur_x_test, params["splits_duration"], temperature)

        pout_dur_int = params["pout_dur_int"][pout_idx_test, :]
        pout_dur_slope = params["pout_dur_slope"][pout_idx_test, :]
        per_bin = pout_dur_int + pout_dur_slope * dur_x_test[:, None]
        logits_test = logits_test + jnp.sum(dur_probs_test * per_bin, axis=-1)

        cont_dur_int = params["cont_dur_int"][cont_idx_test, :]
        cont_dur_slope = params["cont_dur_slope"][cont_idx_test, :]
        per_bin = cont_dur_int + cont_dur_slope * dur_x_test[:, None]
        logits_test = logits_test + jnp.sum(dur_probs_test * per_bin, axis=-1)

        hous_dur_int = params["hous_dur_int"][hous_idx_test, :]
        hous_dur_slope = params["hous_dur_slope"][hous_idx_test, :]
        per_bin = hous_dur_int + hous_dur_slope * dur_x_test[:, None]
        logits_test = logits_test + jnp.sum(dur_probs_test * per_bin, axis=-1)

        logits_test = logits_test + params["pout_cont_inter"][pout_idx_test, cont_idx_test]
        logits_test = logits_test + params["pout_hous_inter"][pout_idx_test, hous_idx_test]

        for i, col in enumerate(numeric_cols):
            x_i = X_test_j[:, i]
            reg_splits = params[f"reg_splits_{col}"]
            reg_probs = soft_bin_probs(x_i, reg_splits, temperature)

            base_slopes = params[f"reg_slope_{col}"]
            pout_slopes = params[f"reg_slope_pout_{col}"][pout_idx_test, :]
            cont_slopes = params[f"reg_slope_cont_{col}"][cont_idx_test, :]
            total_slopes = base_slopes[None, :] + pout_slopes + cont_slopes

            per_bin_contrib = total_slopes * x_i[:, None]
            logits_test = logits_test + jnp.sum(reg_probs * per_bin_contrib, axis=-1)

        probs = 1 / (1 + jnp.exp(-logits_test))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"  AUC = {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\nOURS (EBM-inspired v3): {mean_auc:.4f} +/- {std_auc:.4f}")
    print("Baselines: LR=0.907, LGBM=0.935, EBM=0.934")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bank_ebm_v3()

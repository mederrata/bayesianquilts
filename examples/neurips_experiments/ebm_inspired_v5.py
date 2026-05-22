"""EBM-inspired v5: Bagging + two-stage fitting like EBM."""
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


def train_single_model(X_train_s, y_train, cat_train, numeric_cols, cat_encoders, n_cats, bootstrap_idx=None, seed=0):
    """Train a single model, optionally on a bootstrap sample."""
    if bootstrap_idx is not None:
        X_train_s = X_train_s[bootstrap_idx]
        y_train = y_train[bootstrap_idx]
        cat_train = {k: v[bootstrap_idx] for k, v in cat_train.items()}

    N_train = len(y_train)
    n_splits = 15

    # Initialize splits at percentiles
    cont_splits_init = {}
    for i, col in enumerate(numeric_cols):
        percs = np.linspace(100/(n_splits+1), 100 - 100/(n_splits+1), n_splits)
        cont_splits_init[col] = np.percentile(X_train_s[:, i], percs)

    params = {"global_intercept": jnp.array(0.0)}

    # Categorical main effects
    for col in cat_encoders:
        params[f"cat_{col}"] = jnp.zeros(n_cats[col])

    # Continuous main effects
    for col in numeric_cols:
        params[f"splits_{col}"] = jnp.array(cont_splits_init[col])
        params[f"cont_int_{col}"] = jnp.zeros(n_splits + 1)
        params[f"cont_slope_{col}"] = jnp.zeros(n_splits + 1)

    # Interactions (added in stage 2)
    params["pout_dur_int"] = jnp.zeros((n_cats["poutcome"], n_splits + 1))
    params["pout_dur_slope"] = jnp.zeros((n_cats["poutcome"], n_splits + 1))
    params["cont_dur_int"] = jnp.zeros((n_cats["contact"], n_splits + 1))
    params["cont_dur_slope"] = jnp.zeros((n_cats["contact"], n_splits + 1))
    params["pout_cont"] = jnp.zeros((n_cats["poutcome"], n_cats["contact"]))

    X_train_j = jnp.array(X_train_s)
    y_train_j = jnp.array(y_train)
    cat_train_j = {k: jnp.array(v) for k, v in cat_train.items()}

    temperature = 0.3

    def loss_fn_stage1(params):
        """Stage 1: Main effects only."""
        logits = params["global_intercept"] * jnp.ones(N_train)

        for col in cat_encoders:
            logits = logits + params[f"cat_{col}"][cat_train_j[col]]

        for i, col in enumerate(numeric_cols):
            bin_probs = soft_bin_probs(X_train_j[:, i], params[f"splits_{col}"], temperature)
            per_bin = params[f"cont_int_{col}"][None, :] + params[f"cont_slope_{col}"][None, :] * X_train_j[:, i:i+1]
            logits = logits + jnp.sum(bin_probs * per_bin, axis=-1)

        bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

        l2_reg = 0.0
        for key, val in params.items():
            if "splits" not in key and "pout_dur" not in key and "cont_dur" not in key and "pout_cont" not in key:
                l2_reg += 0.01 * jnp.sum(val**2)

        split_penalty = 0.0
        for col in numeric_cols:
            splits = params[f"splits_{col}"]
            split_penalty += jnp.sum(jax.nn.relu(splits[:-1] - splits[1:] + 0.05))

        return bce + l2_reg / N_train + 0.1 * split_penalty

    def loss_fn_stage2(params):
        """Stage 2: All effects including interactions."""
        logits = params["global_intercept"] * jnp.ones(N_train)

        for col in cat_encoders:
            logits = logits + params[f"cat_{col}"][cat_train_j[col]]

        for i, col in enumerate(numeric_cols):
            bin_probs = soft_bin_probs(X_train_j[:, i], params[f"splits_{col}"], temperature)
            per_bin = params[f"cont_int_{col}"][None, :] + params[f"cont_slope_{col}"][None, :] * X_train_j[:, i:i+1]
            logits = logits + jnp.sum(bin_probs * per_bin, axis=-1)

        # Interactions
        dur_idx = numeric_cols.index("duration")
        dur_x = X_train_j[:, dur_idx]
        dur_probs = soft_bin_probs(dur_x, params["splits_duration"], temperature)

        pout_idx = cat_train_j["poutcome"]
        cont_idx = cat_train_j["contact"]

        per_bin = params["pout_dur_int"][pout_idx, :] + params["pout_dur_slope"][pout_idx, :] * dur_x[:, None]
        logits = logits + jnp.sum(dur_probs * per_bin, axis=-1)

        per_bin = params["cont_dur_int"][cont_idx, :] + params["cont_dur_slope"][cont_idx, :] * dur_x[:, None]
        logits = logits + jnp.sum(dur_probs * per_bin, axis=-1)

        logits = logits + params["pout_cont"][pout_idx, cont_idx]

        bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

        l2_reg = 0.0
        for key, val in params.items():
            if "splits" not in key:
                l2_reg += 0.01 * jnp.sum(val**2)

        split_penalty = 0.0
        for col in numeric_cols:
            splits = params[f"splits_{col}"]
            split_penalty += jnp.sum(jax.nn.relu(splits[:-1] - splits[1:] + 0.05))

        return bce + l2_reg / N_train + 0.1 * split_penalty

    # Stage 1: Main effects
    schedule1 = optax.warmup_cosine_decay_schedule(
        init_value=0.001, peak_value=0.01, warmup_steps=500,
        decay_steps=4500, end_value=0.001
    )
    opt1 = optax.adam(learning_rate=schedule1)
    opt_state1 = opt1.init(params)

    @jax.jit
    def step1(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn_stage1)(params)
        updates, opt_state = opt1.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    for i in range(5001):
        params, opt_state1, loss = step1(params, opt_state1)

    # Stage 2: Add interactions
    schedule2 = optax.warmup_cosine_decay_schedule(
        init_value=0.0005, peak_value=0.005, warmup_steps=500,
        decay_steps=4500, end_value=0.0005
    )
    opt2 = optax.adam(learning_rate=schedule2)
    opt_state2 = opt2.init(params)

    @jax.jit
    def step2(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn_stage2)(params)
        updates, opt_state = opt2.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    for i in range(5001):
        params, opt_state2, loss = step2(params, opt_state2)

    return params


def predict(params, X_test_s, cat_test, numeric_cols):
    """Generate predictions from trained model."""
    temperature = 0.3
    N_test = X_test_s.shape[0]
    n_splits = 15

    X_test_j = jnp.array(X_test_s)
    cat_test_j = {k: jnp.array(v) for k, v in cat_test.items()}

    logits = params["global_intercept"] * jnp.ones(N_test)

    for col in cat_test_j:
        if f"cat_{col}" in params:
            logits = logits + params[f"cat_{col}"][cat_test_j[col]]

    for i, col in enumerate(numeric_cols):
        if f"splits_{col}" in params:
            bin_probs = soft_bin_probs(X_test_j[:, i], params[f"splits_{col}"], temperature)
            per_bin = params[f"cont_int_{col}"][None, :] + params[f"cont_slope_{col}"][None, :] * X_test_j[:, i:i+1]
            logits = logits + jnp.sum(bin_probs * per_bin, axis=-1)

    dur_idx = numeric_cols.index("duration")
    dur_x = X_test_j[:, dur_idx]
    dur_probs = soft_bin_probs(dur_x, params["splits_duration"], temperature)

    pout_idx = cat_test_j["poutcome"]
    cont_idx = cat_test_j["contact"]

    per_bin = params["pout_dur_int"][pout_idx, :] + params["pout_dur_slope"][pout_idx, :] * dur_x[:, None]
    logits = logits + jnp.sum(dur_probs * per_bin, axis=-1)

    per_bin = params["cont_dur_int"][cont_idx, :] + params["cont_dur_slope"][cont_idx, :] * dur_x[:, None]
    logits = logits + jnp.sum(dur_probs * per_bin, axis=-1)

    logits = logits + params["pout_cont"][pout_idx, cont_idx]

    probs = 1 / (1 + jnp.exp(-logits))
    return np.array(probs)


def run_bank_ebm_v5():
    """Bank with EBM-inspired bagging + two-stage."""
    print("BANK - EBM-Inspired v5: Bagging + two-stage fitting")
    print("="*70)

    df = load_bank_data()
    y = (df["y"] == "yes").astype(int).values
    N = len(y)
    print(f"N = {N}, pos_rate = {y.mean():.3f}")

    numeric_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    X_numeric = df[numeric_cols].values.astype(np.float32)
    X_numeric = np.nan_to_num(X_numeric, nan=0.0)

    cat_cols = ["poutcome", "contact", "housing", "month", "job", "marital", "education"]
    cat_encoders = {}
    cat_bins = {}
    for col in cat_cols:
        le = LabelEncoder().fit(df[col].astype(str))
        cat_encoders[col] = le
        cat_bins[col] = le.transform(df[col].astype(str))

    n_cats = {col: len(le.classes_) for col, le in cat_encoders.items()}

    n_bags = 5  # Number of bagged models
    print(f"\nTwo-stage fitting + bagging ({n_bags} models)")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(N), y)):
        print(f"\nFold {fold_idx + 1}/5:")

        X_train_num = X_numeric[train_idx]
        X_test_num = X_numeric[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_num)
        X_test_s = scaler.transform(X_test_num)

        y_train, y_test = y[train_idx], y[test_idx]
        N_train = len(y_train)

        cat_train = {col: cat_bins[col][train_idx] for col in cat_cols}
        cat_test = {col: cat_bins[col][test_idx] for col in cat_cols}

        # Train multiple bagged models
        all_probs = []
        rng = np.random.RandomState(42 + fold_idx)

        for bag_idx in range(n_bags):
            print(f"  Training bag {bag_idx + 1}/{n_bags}...")

            # Bootstrap sample
            bootstrap_idx = rng.choice(N_train, N_train, replace=True)

            params = train_single_model(
                X_train_s, y_train, cat_train, numeric_cols,
                cat_encoders, n_cats, bootstrap_idx, seed=bag_idx
            )

            probs = predict(params, X_test_s, cat_test, numeric_cols)
            all_probs.append(probs)

        # Average predictions
        avg_probs = np.mean(all_probs, axis=0)

        auc = roc_auc_score(y_test, avg_probs)
        aucs.append(auc)
        print(f"  AUC = {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\nOURS (EBM-inspired v5): {mean_auc:.4f} +/- {std_auc:.4f}")
    print("Baselines: LR=0.907, LGBM=0.935, EBM=0.934")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bank_ebm_v5()

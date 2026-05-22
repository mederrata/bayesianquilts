"""EBM-inspired v2: More features, actual regressors, cyclic training."""
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
    """Compute soft bin assignment probabilities.

    More efficient implementation using cumulative products.
    """
    n_splits = len(split_points)
    n_bins = n_splits + 1
    N = len(x)

    # Sigmoid of being above each split
    above = jax.nn.sigmoid((x[:, None] - split_points[None, :]) / temperature)
    below = 1 - above

    # Bin probabilities using cumulative logic
    # bin_k = prod(above[:k]) * below[k] for k < n_splits
    # bin_last = prod(above[:])

    probs = []
    # First bin: below split 0
    probs.append(below[:, 0])

    # Middle bins: above splits 0..k-1, below split k
    cum_above = above[:, 0]
    for k in range(1, n_splits):
        probs.append(cum_above * below[:, k])
        cum_above = cum_above * above[:, k]

    # Last bin: above all splits
    probs.append(cum_above)

    return jnp.stack(probs, axis=-1)


def run_bank_ebm_v2():
    """Bank with fuller EBM-inspired approach."""
    print("BANK - EBM-Inspired v2: Full features + regressors + cyclic")
    print("="*70)

    df = load_bank_data()
    y = (df["y"] == "yes").astype(int).values
    N = len(y)
    print(f"N = {N}, pos_rate = {y.mean():.3f}")

    # Numeric features
    numeric_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    X_numeric = df[numeric_cols].values.astype(np.float32)
    X_numeric = np.nan_to_num(X_numeric, nan=0.0)

    # Categoricals
    cat_encoders = {}
    cat_bins = {}
    for col in ["poutcome", "contact", "housing", "month", "job", "marital", "education"]:
        le = LabelEncoder().fit(df[col].astype(str))
        cat_encoders[col] = le
        cat_bins[col] = le.transform(df[col].astype(str))

    n_cats = {col: len(le.classes_) for col, le in cat_encoders.items()}

    print(f"\nOrder 1: Main effects with learned splits for continuous")
    print(f"Order 2: Cat×Cat and Cat×Cont interactions")

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

        # Number of splits per continuous feature
        n_splits = 15  # 16 bins - like EBM's fine binning

        # Initialize split points at percentiles for each feature
        cont_splits_init = {}
        for i, col in enumerate(numeric_cols):
            percs = np.linspace(100/(n_splits+1), 100 - 100/(n_splits+1), n_splits)
            cont_splits_init[col] = np.percentile(X_train_s[:, i], percs)

        # Build parameter dict
        params = {"global_intercept": jnp.array(0.0)}

        # Order 1: Main effects
        # - Categorical: one value per level
        for col in cat_encoders:
            params[f"main_{col}"] = jnp.zeros(n_cats[col])

        # - Continuous: learned splits + per-bin values
        for i, col in enumerate(numeric_cols):
            params[f"splits_{col}"] = jnp.array(cont_splits_init[col])
            params[f"main_{col}"] = jnp.zeros(n_splits + 1)  # per-bin intercepts
            params[f"slope_{col}"] = jnp.zeros(n_splits + 1)  # per-bin slopes

        # Order 2: Key interactions
        # poutcome × duration (most important)
        params["pout_dur_inter"] = jnp.zeros((n_cats["poutcome"], n_splits + 1))
        # contact × duration
        params["cont_dur_inter"] = jnp.zeros((n_cats["contact"], n_splits + 1))
        # poutcome × contact
        params["pout_cont_inter"] = jnp.zeros((n_cats["poutcome"], n_cats["contact"]))
        # housing × duration
        params["hous_dur_inter"] = jnp.zeros((n_cats["housing"], n_splits + 1))

        # Regressors: beta coefficients that vary by poutcome × contact
        n_features = len(numeric_cols)
        params["beta_base"] = jnp.zeros(n_features)
        params["beta_pout"] = jnp.zeros((n_cats["poutcome"], n_features))
        params["beta_cont"] = jnp.zeros((n_cats["contact"], n_features))

        total_params = sum(p.size for p in params.values())
        print(f"  Total params: {total_params}")

        # Convert to JAX arrays
        X_train_j = jnp.array(X_train_s)
        y_train_j = jnp.array(y_train)

        cat_train = {col: jnp.array(cat_bins[col][train_idx]) for col in cat_encoders}
        cat_test = {col: jnp.array(cat_bins[col][test_idx]) for col in cat_encoders}

        temperature = 0.3

        def loss_fn(params):
            # Compute soft bin assignments for all continuous features
            cont_bin_probs = {}
            for i, col in enumerate(numeric_cols):
                cont_bin_probs[col] = soft_bin_probs(X_train_j[:, i], params[f"splits_{col}"], temperature)

            # Order 0: Global intercept
            logits = params["global_intercept"] * jnp.ones(N_train)

            # Order 1: Categorical main effects
            for col in cat_encoders:
                logits = logits + params[f"main_{col}"][cat_train[col]]

            # Order 1: Continuous main effects
            for i, col in enumerate(numeric_cols):
                bin_vals = params[f"main_{col}"]
                logits = logits + jnp.sum(cont_bin_probs[col] * bin_vals[None, :], axis=-1)

            # Order 2: Interactions
            # poutcome × duration
            pout_idx = cat_train["poutcome"]
            dur_probs = cont_bin_probs["duration"]
            pout_dur = params["pout_dur_inter"][pout_idx, :]  # (N, n_bins)
            logits = logits + jnp.sum(dur_probs * pout_dur, axis=-1)

            # contact × duration
            cont_idx = cat_train["contact"]
            cont_dur = params["cont_dur_inter"][cont_idx, :]
            logits = logits + jnp.sum(dur_probs * cont_dur, axis=-1)

            # housing × duration
            hous_idx = cat_train["housing"]
            hous_dur = params["hous_dur_inter"][hous_idx, :]
            logits = logits + jnp.sum(dur_probs * hous_dur, axis=-1)

            # poutcome × contact
            logits = logits + params["pout_cont_inter"][pout_idx, cont_idx]

            # Regressors: beta varies by poutcome and contact
            beta = params["beta_base"] + params["beta_pout"][pout_idx, :] + params["beta_cont"][cont_idx, :]
            logits = logits + jnp.sum(X_train_j * beta, axis=-1)

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
                splits = params[f"splits_{col}"]
                split_penalty += jnp.sum(jax.nn.relu(splits[:-1] - splits[1:] + 0.05))

            return bce + l2_reg / N_train + 0.1 * split_penalty

        # Low learning rate, many iterations (like EBM)
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

        cont_bin_probs_test = {}
        for i, col in enumerate(numeric_cols):
            cont_bin_probs_test[col] = soft_bin_probs(X_test_j[:, i], params[f"splits_{col}"], temperature)

        logits_test = params["global_intercept"] * jnp.ones(len(y_test))

        for col in cat_encoders:
            logits_test = logits_test + params[f"main_{col}"][cat_test[col]]

        for i, col in enumerate(numeric_cols):
            bin_vals = params[f"main_{col}"]
            logits_test = logits_test + jnp.sum(cont_bin_probs_test[col] * bin_vals[None, :], axis=-1)

        pout_idx_test = cat_test["poutcome"]
        cont_idx_test = cat_test["contact"]
        hous_idx_test = cat_test["housing"]
        dur_probs_test = cont_bin_probs_test["duration"]

        logits_test = logits_test + jnp.sum(dur_probs_test * params["pout_dur_inter"][pout_idx_test, :], axis=-1)
        logits_test = logits_test + jnp.sum(dur_probs_test * params["cont_dur_inter"][cont_idx_test, :], axis=-1)
        logits_test = logits_test + jnp.sum(dur_probs_test * params["hous_dur_inter"][hous_idx_test, :], axis=-1)
        logits_test = logits_test + params["pout_cont_inter"][pout_idx_test, cont_idx_test]

        beta_test = params["beta_base"] + params["beta_pout"][pout_idx_test, :] + params["beta_cont"][cont_idx_test, :]
        logits_test = logits_test + jnp.sum(X_test_j * beta_test, axis=-1)

        probs = 1 / (1 + jnp.exp(-logits_test))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"  AUC = {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\nOURS (EBM-inspired v2): {mean_auc:.4f} +/- {std_auc:.4f}")
    print("Baselines: LR=0.907, LGBM=0.935, EBM=0.934")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bank_ebm_v2()

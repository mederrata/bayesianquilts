"""EBM-inspired v4: Cyclic coordinate descent - train one feature at a time."""
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


def run_bank_ebm_v4():
    """Bank with EBM-inspired cyclic coordinate descent."""
    print("BANK - EBM-Inspired v4: Cyclic coordinate descent")
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

    print(f"\nCyclic training: update each feature component separately")
    print(f"Like EBM's boosting rounds but with gradient descent")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    n_splits = 15  # 16 bins

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

        # Initialize splits at percentiles
        cont_splits_init = {}
        for i, col in enumerate(numeric_cols):
            percs = np.linspace(100/(n_splits+1), 100 - 100/(n_splits+1), n_splits)
            cont_splits_init[col] = np.percentile(X_train_s[:, i], percs)

        # All parameters - will train cyclically
        params = {"global_intercept": jnp.array(0.0)}

        # Categorical main effects
        for col in cat_cols:
            params[f"cat_{col}"] = jnp.zeros(n_cats[col])

        # Continuous main effects with learned splits
        for col in numeric_cols:
            params[f"splits_{col}"] = jnp.array(cont_splits_init[col])
            params[f"cont_int_{col}"] = jnp.zeros(n_splits + 1)
            params[f"cont_slope_{col}"] = jnp.zeros(n_splits + 1)

        # Key interactions
        params["pout_dur_int"] = jnp.zeros((n_cats["poutcome"], n_splits + 1))
        params["pout_dur_slope"] = jnp.zeros((n_cats["poutcome"], n_splits + 1))
        params["cont_dur_int"] = jnp.zeros((n_cats["contact"], n_splits + 1))
        params["cont_dur_slope"] = jnp.zeros((n_cats["contact"], n_splits + 1))
        params["pout_cont"] = jnp.zeros((n_cats["poutcome"], n_cats["contact"]))

        total_params = sum(p.size for p in params.values())
        print(f"  Total params: {total_params}")

        X_train_j = jnp.array(X_train_s)
        y_train_j = jnp.array(y_train)

        cat_train = {col: jnp.array(cat_bins[col][train_idx]) for col in cat_cols}
        cat_test = {col: jnp.array(cat_bins[col][test_idx]) for col in cat_cols}

        temperature = 0.3

        def compute_logits(params, X, cat_idx):
            N_batch = X.shape[0]
            logits = params["global_intercept"] * jnp.ones(N_batch)

            # Categorical main effects
            for col in cat_cols:
                logits = logits + params[f"cat_{col}"][cat_idx[col]]

            # Continuous main effects
            for i, col in enumerate(numeric_cols):
                bin_probs = soft_bin_probs(X[:, i], params[f"splits_{col}"], temperature)
                per_bin = params[f"cont_int_{col}"][None, :] + params[f"cont_slope_{col}"][None, :] * X[:, i:i+1]
                logits = logits + jnp.sum(bin_probs * per_bin, axis=-1)

            # Interactions
            dur_idx = numeric_cols.index("duration")
            dur_x = X[:, dur_idx]
            dur_probs = soft_bin_probs(dur_x, params["splits_duration"], temperature)

            pout_idx = cat_idx["poutcome"]
            cont_idx = cat_idx["contact"]

            per_bin = params["pout_dur_int"][pout_idx, :] + params["pout_dur_slope"][pout_idx, :] * dur_x[:, None]
            logits = logits + jnp.sum(dur_probs * per_bin, axis=-1)

            per_bin = params["cont_dur_int"][cont_idx, :] + params["cont_dur_slope"][cont_idx, :] * dur_x[:, None]
            logits = logits + jnp.sum(dur_probs * per_bin, axis=-1)

            logits = logits + params["pout_cont"][pout_idx, cont_idx]

            return logits

        def make_loss_fn(active_keys):
            """Loss for updating only specific parameter groups."""
            def loss_fn(active_params, frozen_params):
                # Merge params
                full_params = {**frozen_params}
                for k in active_keys:
                    full_params[k] = active_params[k]

                logits = compute_logits(full_params, X_train_j, cat_train)
                bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

                # L2 regularization only on active params
                l2_reg = 0.0
                for k in active_keys:
                    if "splits" not in k:
                        l2_reg += 0.01 * jnp.sum(active_params[k]**2)

                # Split ordering penalty
                split_penalty = 0.0
                for k in active_keys:
                    if "splits" in k:
                        splits = active_params[k]
                        split_penalty += jnp.sum(jax.nn.relu(splits[:-1] - splits[1:] + 0.05))

                return bce + l2_reg / N_train + 0.1 * split_penalty

            return loss_fn

        # Define feature groups for cyclic updates
        feature_groups = [
            ["global_intercept"],
        ]
        # Categorical main effects
        for col in cat_cols:
            feature_groups.append([f"cat_{col}"])
        # Continuous main effects (update splits + effects together)
        for col in numeric_cols:
            feature_groups.append([f"splits_{col}", f"cont_int_{col}", f"cont_slope_{col}"])
        # Interactions
        feature_groups.append(["pout_dur_int", "pout_dur_slope"])
        feature_groups.append(["cont_dur_int", "cont_dur_slope"])
        feature_groups.append(["pout_cont"])

        print(f"  {len(feature_groups)} feature groups for cyclic updates")

        # Low learning rate like EBM
        lr = 0.01

        # Cyclic training: multiple rounds
        n_rounds = 30
        steps_per_group = 100

        for round_idx in range(n_rounds):
            round_loss = 0.0

            for group_keys in feature_groups:
                # Extract active and frozen params
                active_params = {k: params[k] for k in group_keys}
                frozen_params = {k: v for k, v in params.items() if k not in group_keys}

                loss_fn = make_loss_fn(group_keys)
                opt = optax.adam(learning_rate=lr)
                opt_state = opt.init(active_params)

                @jax.jit
                def step(active_params, opt_state, frozen_params):
                    loss, grads = jax.value_and_grad(loss_fn)(active_params, frozen_params)
                    updates, opt_state = opt.update(grads, opt_state, active_params)
                    return optax.apply_updates(active_params, updates), opt_state, loss

                for _ in range(steps_per_group):
                    active_params, opt_state, loss = step(active_params, opt_state, frozen_params)

                # Update global params
                for k in group_keys:
                    params[k] = active_params[k]

                round_loss = loss

            if round_idx % 5 == 0:
                print(f"    Round {round_idx}: loss = {round_loss:.4f}")

        # Evaluate on test
        X_test_j = jnp.array(X_test_s)
        logits_test = compute_logits(params, X_test_j, cat_test)
        probs = 1 / (1 + jnp.exp(-logits_test))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"  AUC = {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\nOURS (EBM-inspired v4): {mean_auc:.4f} +/- {std_auc:.4f}")
    print("Baselines: LR=0.907, LGBM=0.935, EBM=0.934")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bank_ebm_v4()

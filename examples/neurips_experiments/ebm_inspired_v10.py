"""EBM-inspired v10: Cyclic/boosting-style training."""
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
    """f(x) = sum_k slopes[k] * relu(x - splits[k])"""
    hinges = jax.nn.relu(x[:, None] - splits[None, :])
    return jnp.sum(hinges * slopes[None, :], axis=-1)


def run_bank_ebm_v10():
    """Bank with cyclic/boosting-style training like EBM."""
    print("BANK - EBM-Inspired v10: Cyclic training (EBM-style boosting)")
    print("="*70)

    df = load_bank_data()
    y = (df["y"] == "yes").astype(int).values
    N = len(y)
    print(f"N = {N}, pos_rate = {y.mean():.3f}")

    numeric_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    X_numeric = df[numeric_cols].values.astype(np.float32)
    X_numeric = np.nan_to_num(X_numeric, nan=0.0)

    cat_cols = ["poutcome", "contact", "housing", "month", "job"]
    cat_encoders = {}
    cat_bins = {}
    for col in cat_cols:
        le = LabelEncoder().fit(df[col].astype(str))
        cat_encoders[col] = le
        cat_bins[col] = le.transform(df[col].astype(str))

    n_cats = {col: len(le.classes_) for col, le in cat_encoders.items()}

    print(f"\nCyclic training: update feature groups one at a time")
    print(f"Round-robin through: categoricals, continuous, interactions")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    n_splits = 20

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(N), y)):
        print(f"\nFold {fold_idx + 1}/5:")

        X_train_num = X_numeric[train_idx]
        X_test_num = X_numeric[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_num)
        X_test_s = scaler.transform(X_test_num)

        y_train, y_test = y[train_idx], y[test_idx]
        N_train = len(y_train)

        splits_init = {}
        for i, col in enumerate(numeric_cols):
            percs = np.linspace(100/(n_splits+1), 100 - 100/(n_splits+1), n_splits)
            splits_init[col] = np.percentile(X_train_s[:, i], percs)

        params = {"global_intercept": jnp.array(0.0)}

        for col in cat_cols:
            params[f"cat_{col}"] = jnp.zeros(n_cats[col])

        for col in numeric_cols:
            params[f"base_{col}"] = jnp.array(0.0)
            params[f"splits_{col}"] = jnp.array(splits_init[col])
            params[f"slopes_{col}"] = jnp.zeros(n_splits)

        params["pout_cont"] = jnp.zeros((n_cats["poutcome"], n_cats["contact"]))
        params["pout_hous"] = jnp.zeros((n_cats["poutcome"], n_cats["housing"]))
        params["cont_hous"] = jnp.zeros((n_cats["contact"], n_cats["housing"]))
        params["pout_month"] = jnp.zeros((n_cats["poutcome"], n_cats["month"]))

        params["pout_dur_base"] = jnp.zeros(n_cats["poutcome"])
        params["pout_dur_slopes"] = jnp.zeros((n_cats["poutcome"], n_splits))

        params["cont_dur_base"] = jnp.zeros(n_cats["contact"])
        params["cont_dur_slopes"] = jnp.zeros((n_cats["contact"], n_splits))

        params["hous_dur_base"] = jnp.zeros(n_cats["housing"])
        params["hous_dur_slopes"] = jnp.zeros((n_cats["housing"], n_splits))

        params["month_dur_base"] = jnp.zeros(n_cats["month"])
        params["month_dur_slopes"] = jnp.zeros((n_cats["month"], n_splits))

        params["pout_bal_base"] = jnp.zeros(n_cats["poutcome"])
        params["pout_bal_slopes"] = jnp.zeros((n_cats["poutcome"], n_splits))

        params["cont_bal_base"] = jnp.zeros(n_cats["contact"])
        params["cont_bal_slopes"] = jnp.zeros((n_cats["contact"], n_splits))

        total_params = sum(p.size for p in params.values())
        print(f"  Total params: {total_params}")

        X_train_j = jnp.array(X_train_s)
        y_train_j = jnp.array(y_train)

        cat_train_j = {col: jnp.array(cat_bins[col][train_idx]) for col in cat_cols}

        dur_idx = numeric_cols.index("duration")
        bal_idx = numeric_cols.index("balance")

        # Define parameter groups for cyclic training
        cat_keys = frozenset(["global_intercept"] + [f"cat_{col}" for col in cat_cols])
        cont_keys = frozenset([f"base_{col}" for col in numeric_cols] +
                              [f"splits_{col}" for col in numeric_cols] +
                              [f"slopes_{col}" for col in numeric_cols])
        int_keys = frozenset([
            "pout_cont", "pout_hous", "cont_hous", "pout_month",
            "pout_dur_base", "pout_dur_slopes",
            "cont_dur_base", "cont_dur_slopes",
            "hous_dur_base", "hous_dur_slopes",
            "month_dur_base", "month_dur_slopes",
            "pout_bal_base", "pout_bal_slopes",
            "cont_bal_base", "cont_bal_slopes",
        ])

        def compute_logits(params):
            logits = params["global_intercept"] * jnp.ones(N_train)

            for col in cat_cols:
                logits = logits + params[f"cat_{col}"][cat_train_j[col]]

            for i, col in enumerate(numeric_cols):
                x = X_train_j[:, i]
                contrib = params[f"base_{col}"] * x
                contrib = contrib + piecewise_linear_relu(x, params[f"splits_{col}"], params[f"slopes_{col}"])
                logits = logits + contrib

            pout_idx = cat_train_j["poutcome"]
            cont_idx = cat_train_j["contact"]
            hous_idx = cat_train_j["housing"]
            month_idx = cat_train_j["month"]

            logits = logits + params["pout_cont"][pout_idx, cont_idx]
            logits = logits + params["pout_hous"][pout_idx, hous_idx]
            logits = logits + params["cont_hous"][cont_idx, hous_idx]
            logits = logits + params["pout_month"][pout_idx, month_idx]

            dur_x = X_train_j[:, dur_idx]
            dur_splits = params["splits_duration"]
            dur_hinges = jax.nn.relu(dur_x[:, None] - dur_splits[None, :])

            bal_x = X_train_j[:, bal_idx]
            bal_splits = params["splits_balance"]
            bal_hinges = jax.nn.relu(bal_x[:, None] - bal_splits[None, :])

            pout_dur = params["pout_dur_base"][pout_idx] * dur_x
            pout_dur = pout_dur + jnp.sum(dur_hinges * params["pout_dur_slopes"][pout_idx, :], axis=-1)
            logits = logits + pout_dur

            cont_dur = params["cont_dur_base"][cont_idx] * dur_x
            cont_dur = cont_dur + jnp.sum(dur_hinges * params["cont_dur_slopes"][cont_idx, :], axis=-1)
            logits = logits + cont_dur

            hous_dur = params["hous_dur_base"][hous_idx] * dur_x
            hous_dur = hous_dur + jnp.sum(dur_hinges * params["hous_dur_slopes"][hous_idx, :], axis=-1)
            logits = logits + hous_dur

            month_dur = params["month_dur_base"][month_idx] * dur_x
            month_dur = month_dur + jnp.sum(dur_hinges * params["month_dur_slopes"][month_idx, :], axis=-1)
            logits = logits + month_dur

            pout_bal = params["pout_bal_base"][pout_idx] * bal_x
            pout_bal = pout_bal + jnp.sum(bal_hinges * params["pout_bal_slopes"][pout_idx, :], axis=-1)
            logits = logits + pout_bal

            cont_bal = params["cont_bal_base"][cont_idx] * bal_x
            cont_bal = cont_bal + jnp.sum(bal_hinges * params["cont_bal_slopes"][cont_idx, :], axis=-1)
            logits = logits + cont_bal

            return logits

        def loss_fn(params):
            logits = compute_logits(params)
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

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.001, peak_value=0.01, warmup_steps=200,
            decay_steps=800, end_value=0.001
        )
        opt = optax.adam(learning_rate=schedule)
        opt_state = opt.init(params)

        # Create masked step functions for each group
        def make_masked_step(active_keys):
            @jax.jit
            def step_fn(params, opt_state):
                loss, grads = jax.value_and_grad(loss_fn)(params)
                masked_grads = {k: (v if k in active_keys else jnp.zeros_like(v))
                               for k, v in grads.items()}
                updates, new_opt_state = opt.update(masked_grads, opt_state, params)
                return optax.apply_updates(params, updates), new_opt_state, loss
            return step_fn

        step_cat = make_masked_step(cat_keys)
        step_cont = make_masked_step(cont_keys)
        step_int = make_masked_step(int_keys)

        # Cyclic training: rotate through parameter groups
        n_cycles = 15
        steps_per_group = 1000

        step_fns = [
            ("cat", step_cat),
            ("cont", step_cont),
            ("int", step_int),
        ]

        for cycle in range(n_cycles):
            for group_name, step_fn in step_fns:
                for i in range(steps_per_group):
                    params, opt_state, loss = step_fn(params, opt_state)

            if cycle % 3 == 0:
                print(f"    Cycle {cycle}: loss = {loss:.4f}")

        # Final joint optimization
        print(f"    Final joint optimization...")

        @jax.jit
        def step_all(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for i in range(3000):
            params, opt_state, loss = step_all(params, opt_state)

        print(f"    Final loss = {loss:.4f}")

        # Evaluate
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

        pout_idx_test = cat_test_j["poutcome"]
        cont_idx_test = cat_test_j["contact"]
        hous_idx_test = cat_test_j["housing"]
        month_idx_test = cat_test_j["month"]

        logits_test = logits_test + params["pout_cont"][pout_idx_test, cont_idx_test]
        logits_test = logits_test + params["pout_hous"][pout_idx_test, hous_idx_test]
        logits_test = logits_test + params["cont_hous"][cont_idx_test, hous_idx_test]
        logits_test = logits_test + params["pout_month"][pout_idx_test, month_idx_test]

        dur_x_test = X_test_j[:, dur_idx]
        dur_hinges_test = jax.nn.relu(dur_x_test[:, None] - params["splits_duration"][None, :])

        bal_x_test = X_test_j[:, bal_idx]
        bal_hinges_test = jax.nn.relu(bal_x_test[:, None] - params["splits_balance"][None, :])

        logits_test = logits_test + params["pout_dur_base"][pout_idx_test] * dur_x_test
        logits_test = logits_test + jnp.sum(dur_hinges_test * params["pout_dur_slopes"][pout_idx_test, :], axis=-1)

        logits_test = logits_test + params["cont_dur_base"][cont_idx_test] * dur_x_test
        logits_test = logits_test + jnp.sum(dur_hinges_test * params["cont_dur_slopes"][cont_idx_test, :], axis=-1)

        logits_test = logits_test + params["hous_dur_base"][hous_idx_test] * dur_x_test
        logits_test = logits_test + jnp.sum(dur_hinges_test * params["hous_dur_slopes"][hous_idx_test, :], axis=-1)

        logits_test = logits_test + params["month_dur_base"][month_idx_test] * dur_x_test
        logits_test = logits_test + jnp.sum(dur_hinges_test * params["month_dur_slopes"][month_idx_test, :], axis=-1)

        logits_test = logits_test + params["pout_bal_base"][pout_idx_test] * bal_x_test
        logits_test = logits_test + jnp.sum(bal_hinges_test * params["pout_bal_slopes"][pout_idx_test, :], axis=-1)

        logits_test = logits_test + params["cont_bal_base"][cont_idx_test] * bal_x_test
        logits_test = logits_test + jnp.sum(bal_hinges_test * params["cont_bal_slopes"][cont_idx_test, :], axis=-1)

        probs = 1 / (1 + jnp.exp(-logits_test))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"  AUC = {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\nOURS (EBM-inspired v10): {mean_auc:.4f} +/- {std_auc:.4f}")
    print("Baselines: LR=0.907, LGBM=0.935, EBM=0.934")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bank_ebm_v10()

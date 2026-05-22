"""Compare Ours #1 (original quilted) vs Ours #2 (ReLU hinges) across UCI datasets."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
import jax
import jax.numpy as jnp
import optax
import zipfile
import urllib.request
from io import BytesIO

# ============================================================================
# Data Loading
# ============================================================================

def load_bank():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
    cache_path = Path("data/bank/bank-full.csv")
    if cache_path.exists():
        df = pd.read_csv(cache_path, sep=";")
        if "y" in df.columns:
            return df, "y"
        cache_path.unlink()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        with zipfile.ZipFile(BytesIO(response.read())) as z:
            with z.open("bank-full.csv") as f:
                df = pd.read_csv(f, sep=";")
                df.to_csv(cache_path, sep=";", index=False)
                return df, "y"


def load_adult():
    cache_path = Path("data/adult/adult.csv")
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        return df, "income"

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ["age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "sex",
               "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
    df = pd.read_csv(url, names=columns, skipinitialspace=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df, "income"


def load_german():
    cache_path = Path("data/german/german.csv")
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        return df, "target"

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    columns = [f"A{i}" for i in range(1, 21)] + ["target"]
    df = pd.read_csv(url, sep=r'\s+', names=columns)
    df["target"] = (df["target"] == 2).astype(int)  # 2 = bad credit
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df, "target"


def load_phoneme():
    data = fetch_openml(data_id=1489, as_frame=True, parser='auto')
    df = data.data.copy()
    df["target"] = (data.target == "1").astype(int)
    return df, "target"


def load_electricity():
    data = fetch_openml(data_id=151, as_frame=True, parser='auto')
    df = data.data.copy()
    df["target"] = (data.target == "UP").astype(int)
    return df, "target"


def load_taiwan():
    data = fetch_openml(data_id=42477, as_frame=True, parser='auto')
    df = data.data.copy()
    df["target"] = data.target.astype(int)
    return df, "target"


# ============================================================================
# Ours #2: ReLU Hinges with Cyclic Training
# ============================================================================

def piecewise_linear_relu(x, splits, slopes):
    """f(x) = sum_k slopes[k] * relu(x - splits[k])"""
    hinges = jax.nn.relu(x[:, None] - splits[None, :])
    return jnp.sum(hinges * slopes[None, :], axis=-1)


def run_ours_v2(X_numeric, cat_bins, y, numeric_cols, cat_cols, n_cats,
                n_splits=20, n_cycles=10, steps_per_group=500, final_steps=2000,
                l2_reg=0.01):
    """Ours #2: ReLU hinges with cyclic training."""

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    N = len(y)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(N), y)):
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

        # Continuous main effects: ReLU hinges
        for col in numeric_cols:
            params[f"base_{col}"] = jnp.array(0.0)
            params[f"splits_{col}"] = jnp.array(splits_init[col])
            params[f"slopes_{col}"] = jnp.zeros(n_splits)

        # Cat × Cat interactions (top 2 categoricals if available)
        if len(cat_cols) >= 2:
            top_cats = cat_cols[:2]
            params["cat_int_01"] = jnp.zeros((n_cats[top_cats[0]], n_cats[top_cats[1]]))

        # Cat × Cont interactions (top categorical × top continuous)
        if len(cat_cols) >= 1 and len(numeric_cols) >= 1:
            top_cat = cat_cols[0]
            top_cont = numeric_cols[0]  # Usually most important
            params[f"catcont_{top_cat}_{top_cont}_base"] = jnp.zeros(n_cats[top_cat])
            params[f"catcont_{top_cat}_{top_cont}_slopes"] = jnp.zeros((n_cats[top_cat], n_splits))

        X_train_j = jnp.array(X_train_s)
        y_train_j = jnp.array(y_train)
        cat_train_j = {col: jnp.array(cat_bins[col][train_idx]) for col in cat_cols}

        # Define parameter groups
        cat_keys = frozenset(["global_intercept"] + [f"cat_{col}" for col in cat_cols])
        cont_keys = frozenset([f"base_{col}" for col in numeric_cols] +
                              [f"splits_{col}" for col in numeric_cols] +
                              [f"slopes_{col}" for col in numeric_cols])
        int_keys = frozenset([k for k in params.keys() if k not in cat_keys and k not in cont_keys])

        def loss_fn(params):
            logits = params["global_intercept"] * jnp.ones(N_train)

            # Categorical main effects
            for col in cat_cols:
                logits = logits + params[f"cat_{col}"][cat_train_j[col]]

            # Continuous main effects
            for i, col in enumerate(numeric_cols):
                x = X_train_j[:, i]
                contrib = params[f"base_{col}"] * x
                contrib = contrib + piecewise_linear_relu(x, params[f"splits_{col}"], params[f"slopes_{col}"])
                logits = logits + contrib

            # Cat × Cat interaction
            if len(cat_cols) >= 2:
                top_cats = cat_cols[:2]
                logits = logits + params["cat_int_01"][cat_train_j[top_cats[0]], cat_train_j[top_cats[1]]]

            # Cat × Cont interaction
            if len(cat_cols) >= 1 and len(numeric_cols) >= 1:
                top_cat = cat_cols[0]
                top_cont = numeric_cols[0]
                cat_idx = cat_train_j[top_cat]
                cont_x = X_train_j[:, 0]
                cont_splits = params[f"splits_{top_cont}"]
                cont_hinges = jax.nn.relu(cont_x[:, None] - cont_splits[None, :])

                catcont = params[f"catcont_{top_cat}_{top_cont}_base"][cat_idx] * cont_x
                catcont = catcont + jnp.sum(cont_hinges * params[f"catcont_{top_cat}_{top_cont}_slopes"][cat_idx, :], axis=-1)
                logits = logits + catcont

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # L2 regularization
            l2 = 0.0
            for key, val in params.items():
                if "splits" not in key:
                    l2 += l2_reg * jnp.sum(val**2)

            # Split ordering penalty
            split_penalty = 0.0
            for col in numeric_cols:
                splits = params[f"splits_{col}"]
                split_penalty += jnp.sum(jax.nn.relu(splits[:-1] - splits[1:] + 0.05))

            return bce + l2 / N_train + 0.1 * split_penalty

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.001, peak_value=0.01, warmup_steps=200,
            decay_steps=800, end_value=0.001
        )
        opt = optax.adam(learning_rate=schedule)
        opt_state = opt.init(params)

        # Cyclic training
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

        for cycle in range(n_cycles):
            for step_fn in [step_cat, step_cont, step_int]:
                for _ in range(steps_per_group):
                    params, opt_state, loss = step_fn(params, opt_state)

        # Final joint optimization
        @jax.jit
        def step_all(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for _ in range(final_steps):
            params, opt_state, loss = step_all(params, opt_state)

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

        if len(cat_cols) >= 2:
            top_cats = cat_cols[:2]
            logits_test = logits_test + params["cat_int_01"][cat_test_j[top_cats[0]], cat_test_j[top_cats[1]]]

        if len(cat_cols) >= 1 and len(numeric_cols) >= 1:
            top_cat = cat_cols[0]
            top_cont = numeric_cols[0]
            cat_idx = cat_test_j[top_cat]
            cont_x = X_test_j[:, 0]
            cont_splits = params[f"splits_{top_cont}"]
            cont_hinges = jax.nn.relu(cont_x[:, None] - cont_splits[None, :])

            catcont = params[f"catcont_{top_cat}_{top_cont}_base"][cat_idx] * cont_x
            catcont = catcont + jnp.sum(cont_hinges * params[f"catcont_{top_cat}_{top_cont}_slopes"][cat_idx, :], axis=-1)
            logits_test = logits_test + catcont

        probs = 1 / (1 + jnp.exp(-logits_test))
        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    Fold {fold_idx+1}: AUC = {auc:.4f}")

    return np.mean(aucs), np.std(aucs)


# ============================================================================
# Ours #1: Original Quilted (hard bins, decomposed parameters)
# ============================================================================

def run_ours_v1(X_numeric, cat_bins, y, numeric_cols, cat_cols, n_cats,
                n_bins=8, n_epochs=10000, l2_reg=0.01):
    """Ours #1: Original quilted with hard bins and decomposed parameters."""

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    N = len(y)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(N), y)):
        X_train_num = X_numeric[train_idx]
        X_test_num = X_numeric[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_num)
        X_test_s = scaler.transform(X_test_num)

        y_train, y_test = y[train_idx], y[test_idx]
        N_train = len(y_train)

        # Hard bin continuous features
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

        # Cat × Cat interaction (top 2)
        if len(cat_cols) >= 2:
            top_cats = cat_cols[:2]
            params["cat_int_01"] = jnp.zeros((n_cats[top_cats[0]], n_cats[top_cats[1]]))

        # Cat × Cont interaction (top categorical × top continuous bin)
        if len(cat_cols) >= 1 and len(numeric_cols) >= 1:
            top_cat = cat_cols[0]
            top_cont = numeric_cols[0]
            params[f"catcont_{top_cat}_{top_cont}"] = jnp.zeros((n_cats[top_cat], n_bins))

        y_train_j = jnp.array(y_train)
        cat_train_j = {col: jnp.array(cat_bins[col][train_idx]) for col in cat_cols}
        cont_train_j = {col: jnp.array(cont_bins_train[col]) for col in numeric_cols}

        def loss_fn(params):
            logits = params["global_intercept"] * jnp.ones(N_train)

            for col in cat_cols:
                logits = logits + params[f"cat_{col}"][cat_train_j[col]]

            for col in numeric_cols:
                logits = logits + params[f"cont_{col}"][cont_train_j[col]]

            if len(cat_cols) >= 2:
                top_cats = cat_cols[:2]
                logits = logits + params["cat_int_01"][cat_train_j[top_cats[0]], cat_train_j[top_cats[1]]]

            if len(cat_cols) >= 1 and len(numeric_cols) >= 1:
                top_cat = cat_cols[0]
                top_cont = numeric_cols[0]
                logits = logits + params[f"catcont_{top_cat}_{top_cont}"][cat_train_j[top_cat], cont_train_j[top_cont]]

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2 = 0.0
            for val in params.values():
                l2 += l2_reg * jnp.sum(val**2)

            return bce + l2 / N_train

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.001, peak_value=0.02, warmup_steps=500,
            decay_steps=n_epochs - 500, end_value=0.001
        )
        opt = optax.adam(learning_rate=schedule)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for i in range(n_epochs):
            params, opt_state, loss = step(params, opt_state)

        # Evaluate
        cat_test_j = {col: jnp.array(cat_bins[col][test_idx]) for col in cat_cols}
        cont_test_j = {col: jnp.array(cont_bins_test[col]) for col in numeric_cols}

        logits_test = params["global_intercept"] * jnp.ones(len(y_test))

        for col in cat_cols:
            logits_test = logits_test + params[f"cat_{col}"][cat_test_j[col]]

        for col in numeric_cols:
            logits_test = logits_test + params[f"cont_{col}"][cont_test_j[col]]

        if len(cat_cols) >= 2:
            top_cats = cat_cols[:2]
            logits_test = logits_test + params["cat_int_01"][cat_test_j[top_cats[0]], cat_test_j[top_cats[1]]]

        if len(cat_cols) >= 1 and len(numeric_cols) >= 1:
            top_cat = cat_cols[0]
            top_cont = numeric_cols[0]
            logits_test = logits_test + params[f"catcont_{top_cat}_{top_cont}"][cat_test_j[top_cat], cont_test_j[top_cont]]

        probs = 1 / (1 + jnp.exp(-logits_test))
        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    Fold {fold_idx+1}: AUC = {auc:.4f}")

    return np.mean(aucs), np.std(aucs)


# ============================================================================
# Dataset Configurations
# ============================================================================

DATASETS = {
    "bank": {
        "loader": load_bank,
        "numeric_cols": ["age", "balance", "duration", "campaign", "pdays", "previous"],
        "cat_cols": ["poutcome", "contact", "housing", "month"],
    },
    "adult": {
        "loader": load_adult,
        "numeric_cols": ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"],
        "cat_cols": ["education", "marital_status", "workclass", "occupation"],
    },
    "german": {
        "loader": load_german,
        "numeric_cols": ["A2", "A5", "A8", "A11", "A13", "A16", "A18"],
        "cat_cols": ["A1", "A3", "A4", "A6", "A7"],
    },
    "phoneme": {
        "loader": load_phoneme,
        "numeric_cols": ["V1", "V2", "V3", "V4", "V5"],
        "cat_cols": [],
    },
    "electricity": {
        "loader": load_electricity,
        "numeric_cols": ["date", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"],
        "cat_cols": ["day"],
    },
}


def run_dataset(name, config):
    print(f"\n{'='*70}")
    print(f"Dataset: {name.upper()}")
    print(f"{'='*70}")

    df, target_col = config["loader"]()

    # Prepare target
    target_vals = df[target_col].astype(str).str.strip()
    unique_vals = target_vals.unique()
    # Map to binary: use "yes", ">50K", "1", "2" as positive class
    positive_vals = ["yes", ">50K", "1", "2", "UP"]
    if any(pv in unique_vals for pv in positive_vals):
        pos_val = [pv for pv in positive_vals if pv in unique_vals][0]
        y = (target_vals == pos_val).astype(int).values
    else:
        y = (target_vals == unique_vals[1]).astype(int).values

    N = len(y)
    print(f"N = {N}, pos_rate = {y.mean():.3f}")

    numeric_cols = config["numeric_cols"]
    cat_cols = config["cat_cols"]

    # Prepare numeric features
    X_numeric = np.zeros((N, len(numeric_cols)), dtype=np.float32)
    for i, col in enumerate(numeric_cols):
        if col in df.columns:
            X_numeric[:, i] = pd.to_numeric(df[col], errors='coerce').fillna(0).values
    X_numeric = np.nan_to_num(X_numeric, nan=0.0)

    # Prepare categorical features
    cat_encoders = {}
    cat_bins = {}
    n_cats = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder().fit(df[col].astype(str))
            cat_encoders[col] = le
            cat_bins[col] = le.transform(df[col].astype(str))
            n_cats[col] = len(le.classes_)

    # Filter to cols that exist
    cat_cols = [c for c in cat_cols if c in cat_bins]

    print(f"\nRunning Ours #1 (hard bins)...")
    v1_mean, v1_std = run_ours_v1(X_numeric, cat_bins, y, numeric_cols, cat_cols, n_cats)
    print(f"  Ours #1: {v1_mean:.4f} +/- {v1_std:.4f}")

    print(f"\nRunning Ours #2 (ReLU hinges + cyclic)...")
    v2_mean, v2_std = run_ours_v2(X_numeric, cat_bins, y, numeric_cols, cat_cols, n_cats)
    print(f"  Ours #2: {v2_mean:.4f} +/- {v2_std:.4f}")

    return {
        "dataset": name,
        "ours_v1_mean": v1_mean,
        "ours_v1_std": v1_std,
        "ours_v2_mean": v2_mean,
        "ours_v2_std": v2_std,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Run specific dataset")
    args = parser.parse_args()

    results = []

    if args.dataset:
        datasets = {args.dataset: DATASETS[args.dataset]}
    else:
        datasets = DATASETS

    for name, config in datasets.items():
        try:
            result = run_dataset(name, config)
            results.append(result)
        except Exception as e:
            print(f"Error on {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Dataset':<15} {'Ours #1':<20} {'Ours #2':<20}")
    print("-"*55)
    for r in results:
        v1 = f"{r['ours_v1_mean']:.4f} +/- {r['ours_v1_std']:.4f}"
        v2 = f"{r['ours_v2_mean']:.4f} +/- {r['ours_v2_std']:.4f}"
        print(f"{r['dataset']:<15} {v1:<20} {v2:<20}")


if __name__ == "__main__":
    main()

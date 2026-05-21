"""EBM-inspired lattice: categorical grid + learned continuous splits at each order."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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


def soft_bin(x, split_points, temperature=1.0):
    """Soft binning using sigmoid for differentiable bin assignment.

    Returns soft bin weights (N, n_bins) that sum to 1.
    """
    # split_points: (n_splits,) learnable parameters
    # x: (N,) feature values
    n_splits = len(split_points)
    n_bins = n_splits + 1

    # Compute sigmoid probabilities of being above each split
    # Shape: (N, n_splits)
    above_split = jax.nn.sigmoid((x[:, None] - split_points[None, :]) / temperature)

    # Bin k probability = P(above split k-1) * P(below split k)
    # First bin: P(below split 0)
    # Last bin: P(above split n-1)
    # Middle bins: P(above split k-1) * P(below split k)

    below_split = 1 - above_split

    # Build bin probabilities
    bin_probs = jnp.zeros((len(x), n_bins))

    # First bin: below all splits up to 0
    bin_probs = bin_probs.at[:, 0].set(below_split[:, 0])

    # Middle bins
    for k in range(1, n_bins - 1):
        bin_probs = bin_probs.at[:, k].set(above_split[:, k-1] * below_split[:, k])

    # Last bin: above last split
    bin_probs = bin_probs.at[:, n_bins - 1].set(above_split[:, n_splits - 1])

    return bin_probs


def run_bank_ebm_inspired():
    """Bank with EBM-inspired learnable splits."""
    print("BANK - EBM-Inspired: Categorical grid + learned continuous splits")
    print("="*70)

    df = load_bank_data()
    y = (df["y"] == "yes").astype(int).values
    N = len(y)
    print(f"N = {N}, pos_rate = {y.mean():.3f}")

    # Numeric features - will learn splits for these
    numeric_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    X_numeric = df[numeric_cols].values.astype(np.float32)
    X_numeric = np.nan_to_num(X_numeric, nan=0.0)

    # Categorical features - fixed grid
    le_poutcome = LabelEncoder().fit(df["poutcome"].astype(str))
    poutcome_bins = le_poutcome.transform(df["poutcome"].astype(str))
    n_poutcome = len(le_poutcome.classes_)

    le_contact = LabelEncoder().fit(df["contact"].astype(str))
    contact_bins = le_contact.transform(df["contact"].astype(str))
    n_contact = len(le_contact.classes_)

    le_housing = LabelEncoder().fit(df["housing"].astype(str))
    housing_bins = le_housing.transform(df["housing"].astype(str))
    n_housing = len(le_housing.classes_)

    le_month = LabelEncoder().fit(df["month"].astype(str))
    month_bins = le_month.transform(df["month"].astype(str))
    n_month = len(le_month.classes_)

    print(f"\nLearning continuous splits within categorical grid")
    print(f"Using cyclic coordinate descent like EBM")

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

        # Duration is the most important continuous feature
        dur_idx = numeric_cols.index("duration")
        duration_train = X_train_s[:, dur_idx]
        duration_test = X_test_s[:, dur_idx]

        # Initialize split points at percentiles
        n_splits = 7  # 8 bins
        init_splits = np.percentile(duration_train, np.linspace(100/(n_splits+1), 100 - 100/(n_splits+1), n_splits))

        # Categorical grid: poutcome × contact × housing
        cat_train_idx = np.stack([
            poutcome_bins[train_idx],
            contact_bins[train_idx],
            housing_bins[train_idx],
        ], axis=-1)
        cat_test_idx = np.stack([
            poutcome_bins[test_idx],
            contact_bins[test_idx],
            housing_bins[test_idx],
        ], axis=-1)

        # For each categorical cell, we have:
        # 1. A base intercept
        # 2. Per-bin coefficients for duration (learned splits)
        # 3. Linear slope within each bin

        n_cat_cells = n_poutcome * n_contact * n_housing
        n_dur_bins = n_splits + 1

        # Parameters:
        # - split_points: (n_splits,) - learned split locations
        # - cell_intercepts: (n_poutcome, n_contact, n_housing) - base for each cell
        # - cell_dur_effects: (n_poutcome, n_contact, n_housing, n_dur_bins) - duration effect per cell per bin
        # - cell_dur_slopes: (n_poutcome, n_contact, n_housing, n_dur_bins) - slope within bin

        params = {
            "split_points": jnp.array(init_splits),
            "cell_intercepts": jnp.zeros((n_poutcome, n_contact, n_housing)),
            "cell_dur_effects": jnp.zeros((n_poutcome, n_contact, n_housing, n_dur_bins)),
            "cell_dur_slopes": jnp.zeros((n_poutcome, n_contact, n_housing, n_dur_bins)),
            "global_intercept": jnp.array(0.0),
        }

        # Also add main effects for categoricals
        params["poutcome_main"] = jnp.zeros(n_poutcome)
        params["contact_main"] = jnp.zeros(n_contact)
        params["housing_main"] = jnp.zeros(n_housing)
        params["month_main"] = jnp.zeros(n_month)

        total_params = sum(p.size for p in params.values())
        print(f"  Total params: {total_params} (including {n_splits} learnable splits)")

        duration_train_j = jnp.array(duration_train)
        y_train_j = jnp.array(y_train)
        cat_train_j = jnp.array(cat_train_idx)
        month_train_j = jnp.array(month_bins[train_idx])

        temperature = 0.5  # Controls sharpness of soft bins

        def loss_fn(params):
            # Soft bin assignment for duration
            bin_probs = soft_bin(duration_train_j, params["split_points"], temperature)

            # Look up cell-specific effects
            # cell_intercepts[poutcome, contact, housing]
            cell_int = params["cell_intercepts"][cat_train_j[:, 0], cat_train_j[:, 1], cat_train_j[:, 2]]

            # cell_dur_effects[poutcome, contact, housing, :] -> (N, n_dur_bins)
            cell_dur = params["cell_dur_effects"][cat_train_j[:, 0], cat_train_j[:, 1], cat_train_j[:, 2], :]

            # Weighted sum over bins
            dur_effect = jnp.sum(bin_probs * cell_dur, axis=-1)

            # Main effects
            pout_main = params["poutcome_main"][cat_train_j[:, 0]]
            cont_main = params["contact_main"][cat_train_j[:, 1]]
            hous_main = params["housing_main"][cat_train_j[:, 2]]
            mont_main = params["month_main"][month_train_j]

            logits = (params["global_intercept"] + cell_int + dur_effect +
                     pout_main + cont_main + hous_main + mont_main)

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # Regularization
            l2_reg = 0.0
            l2_reg += 0.01 * jnp.sum(params["cell_intercepts"]**2)
            l2_reg += 0.01 * jnp.sum(params["cell_dur_effects"]**2)
            l2_reg += 0.1 * jnp.sum(params["cell_dur_slopes"]**2)
            l2_reg += 0.01 * jnp.sum(params["poutcome_main"]**2)
            l2_reg += 0.01 * jnp.sum(params["contact_main"]**2)
            l2_reg += 0.01 * jnp.sum(params["housing_main"]**2)
            l2_reg += 0.01 * jnp.sum(params["month_main"]**2)

            # Encourage ordered splits
            split_order_penalty = jnp.sum(jax.nn.relu(params["split_points"][:-1] - params["split_points"][1:] + 0.1))

            return bce + l2_reg / N_train + 0.1 * split_order_penalty

        # Use low learning rate like EBM
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.001, peak_value=0.01, warmup_steps=1000,
            decay_steps=9000, end_value=0.001
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
                sorted_splits = jnp.sort(params["split_points"])
                print(f"    Step {i}: loss = {loss:.4f}, splits range = [{sorted_splits[0]:.2f}, {sorted_splits[-1]:.2f}]")

        # Evaluate on test
        duration_test_j = jnp.array(duration_test)
        cat_test_j = jnp.array(cat_test_idx)
        month_test_j = jnp.array(month_bins[test_idx])

        bin_probs_test = soft_bin(duration_test_j, params["split_points"], temperature)
        cell_int_test = params["cell_intercepts"][cat_test_j[:, 0], cat_test_j[:, 1], cat_test_j[:, 2]]
        cell_dur_test = params["cell_dur_effects"][cat_test_j[:, 0], cat_test_j[:, 1], cat_test_j[:, 2], :]
        dur_effect_test = jnp.sum(bin_probs_test * cell_dur_test, axis=-1)

        pout_main_test = params["poutcome_main"][cat_test_j[:, 0]]
        cont_main_test = params["contact_main"][cat_test_j[:, 1]]
        hous_main_test = params["housing_main"][cat_test_j[:, 2]]
        mont_main_test = params["month_main"][month_test_j]

        logits_test = (params["global_intercept"] + cell_int_test + dur_effect_test +
                      pout_main_test + cont_main_test + hous_main_test + mont_main_test)
        probs = 1 / (1 + jnp.exp(-logits_test))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"  AUC = {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\nOURS (EBM-inspired): {mean_auc:.4f} +/- {std_auc:.4f}")
    print("Baselines: LR=0.907, LGBM=0.935, EBM=0.934")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bank_ebm_inspired()

"""Bank Marketing v1: Apply Adult learnings - multiple additive decompositions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from itertools import combinations
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def within_bin_normalize(values, bin_edges):
    values = np.asarray(values, dtype=float)
    full_edges = np.concatenate([[values.min()], bin_edges, [values.max()]])
    bins = np.digitize(values, bin_edges)
    lower = full_edges[bins]
    upper = full_edges[bins + 1]
    width = np.where(upper - lower == 0, 1.0, upper - lower)
    return np.clip((values - lower) / width, 0, 1)


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


def run_bank():
    print("BANK V1 - Multiple additive decompositions (Adult pattern)")
    print("="*70)

    df = load_bank_data()
    y = (df["y"] == "yes").astype(int).values
    N = len(y)
    print(f"N = {N}, pos_rate = {y.mean():.3f}")

    # Numeric features
    numeric_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    X_numeric = df[numeric_cols].values.astype(np.float32)
    X_numeric = np.nan_to_num(X_numeric, nan=0.0)

    # Bin key numeric features
    duration = X_numeric[:, numeric_cols.index("duration")]
    age = X_numeric[:, numeric_cols.index("age")]
    balance = X_numeric[:, numeric_cols.index("balance")]
    campaign = X_numeric[:, numeric_cols.index("campaign")]

    dur_edges = np.percentile(duration, [12.5, 25, 37.5, 50, 62.5, 75, 87.5])
    age_edges = np.percentile(age, [12.5, 25, 37.5, 50, 62.5, 75, 87.5])
    bal_edges = np.percentile(balance[balance > 0], [25, 50, 75]) if np.sum(balance > 0) > 0 else np.array([100, 500, 1000])
    camp_edges = np.array([1, 2, 3, 5])

    dur_bins = np.digitize(duration, dur_edges)
    age_bins = np.digitize(age, age_edges)
    bal_bins = np.zeros(len(balance), dtype=int)
    bal_bins[balance > 0] = 1 + np.digitize(balance[balance > 0], bal_edges)
    camp_bins = np.digitize(campaign, camp_edges)

    dur_norm = within_bin_normalize(duration, dur_edges)
    age_norm = within_bin_normalize(age, age_edges)

    # Encode categoricals
    le_poutcome = LabelEncoder().fit(df["poutcome"].astype(str))
    poutcome_bins = le_poutcome.transform(df["poutcome"].astype(str))
    n_poutcome = len(le_poutcome.classes_)

    le_month = LabelEncoder().fit(df["month"].astype(str))
    month_bins = le_month.transform(df["month"].astype(str))
    n_month = len(le_month.classes_)

    le_contact = LabelEncoder().fit(df["contact"].astype(str))
    contact_bins = le_contact.transform(df["contact"].astype(str))
    n_contact = len(le_contact.classes_)

    le_housing = LabelEncoder().fit(df["housing"].astype(str))
    housing_bins = le_housing.transform(df["housing"].astype(str))
    n_housing = len(le_housing.classes_)

    le_job = LabelEncoder().fit(df["job"].astype(str))
    job_bins = le_job.transform(df["job"].astype(str))
    n_job = len(le_job.classes_)

    print(f"\n8 bins for duration/age, 5 for balance, 5 for campaign")
    print(f"Multiple additive decompositions like Adult")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(N), y)):
        print(f"\nFold {fold_idx + 1}/5:")

        # Build features
        X_train_num = X_numeric[train_idx]
        X_test_num = X_numeric[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_num)
        X_test_s = scaler.transform(X_test_num)

        # Add pairwise interactions for top features
        n_top = 4
        pairwise_train = [X_train_s[:, i] * X_train_s[:, j] for i, j in combinations(range(n_top), 2)]
        pairwise_test = [X_test_s[:, i] * X_test_s[:, j] for i, j in combinations(range(n_top), 2)]

        # Add normalized features
        X_train = np.concatenate([
            X_train_s, np.stack(pairwise_train, axis=1),
            dur_norm[train_idx:train_idx+1].T if len(train_idx) == 1 else dur_norm[train_idx].reshape(-1, 1),
            age_norm[train_idx:train_idx+1].T if len(train_idx) == 1 else age_norm[train_idx].reshape(-1, 1),
        ], axis=1)
        X_test = np.concatenate([
            X_test_s, np.stack(pairwise_test, axis=1),
            dur_norm[test_idx].reshape(-1, 1),
            age_norm[test_idx].reshape(-1, 1),
        ], axis=1)

        y_train, y_test = y[train_idx], y[test_idx]
        N_train = len(y_train)
        n_features = X_train.shape[1]

        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        lr.fit(X_train, y_train)
        lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
        print(f"  n_features={n_features}, LR baseline AUC: {lr_auc:.4f}")

        # Multiple additive decompositions (Adult pattern)
        # Base: poutcome × contact × housing × duration (main drivers)
        dims_base = [
            Dimension("poutcome", n_poutcome),
            Dimension("contact", n_contact),
            Dimension("housing", n_housing),
            Dimension("duration", 8),
        ]
        decomp_base = Decomposed(interactions=Interactions(dimensions=dims_base), param_shape=[1], name="base")

        # Month × duration (seasonality interacts with call length)
        dims_month_dur = [Dimension("month", n_month), Dimension("duration", 8)]
        decomp_month_dur = Decomposed(interactions=Interactions(dimensions=dims_month_dur), param_shape=[1], name="month_dur")

        # Job × age (job type varies with age)
        dims_job_age = [Dimension("job", n_job), Dimension("age", 8)]
        decomp_job_age = Decomposed(interactions=Interactions(dimensions=dims_job_age), param_shape=[1], name="job_age")

        # Beta lattice: poutcome × contact (simple)
        dims_beta = [Dimension("poutcome", n_poutcome), Dimension("contact", n_contact)]
        decomp_beta = Decomposed(interactions=Interactions(dimensions=dims_beta), param_shape=[n_features], name="beta")

        train_idx_base = np.stack([
            poutcome_bins[train_idx], contact_bins[train_idx],
            housing_bins[train_idx], dur_bins[train_idx],
        ], axis=-1)
        test_idx_base = np.stack([
            poutcome_bins[test_idx], contact_bins[test_idx],
            housing_bins[test_idx], dur_bins[test_idx],
        ], axis=-1)

        train_idx_month_dur = np.stack([month_bins[train_idx], dur_bins[train_idx]], axis=-1)
        test_idx_month_dur = np.stack([month_bins[test_idx], dur_bins[test_idx]], axis=-1)

        train_idx_job_age = np.stack([job_bins[train_idx], age_bins[train_idx]], axis=-1)
        test_idx_job_age = np.stack([job_bins[test_idx], age_bins[test_idx]], axis=-1)

        train_idx_beta = np.stack([poutcome_bins[train_idx], contact_bins[train_idx]], axis=-1)
        test_idx_beta = np.stack([poutcome_bins[test_idx], contact_bins[test_idx]], axis=-1)

        decomps = {
            "base": decomp_base,
            "month_dur": decomp_month_dur,
            "job_age": decomp_job_age,
            "beta": decomp_beta,
        }

        prior_scales = {}
        active = {}
        for key, decomp in decomps.items():
            prior_scales[key] = decomp.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)
            if key in ["month_dur", "job_age"]:
                active[key] = [n for n in decomp._tensor_parts.keys() if decomp.component_order(n) == 2]
            else:
                active[key] = [n for n in decomp._tensor_parts.keys() if decomp.component_order(n) <= 2]

        params = {}
        for key, decomp in decomps.items():
            params[key] = {n: jnp.zeros(decomp._tensor_part_shapes[n]) for n in active[key]}

        # Main effects for interaction-only terms
        params["month_main"] = jnp.zeros(n_month)
        params["job_main"] = jnp.zeros(n_job)
        params["age_main"] = jnp.zeros(8)

        total_params = sum(sum(p.size for p in d.values()) if isinstance(d, dict) else d.size for d in params.values())
        print(f"  Total params: {total_params}")

        train_indices = {
            "base": jnp.array(train_idx_base),
            "month_dur": jnp.array(train_idx_month_dur),
            "job_age": jnp.array(train_idx_job_age),
            "beta": jnp.array(train_idx_beta),
        }
        X_train_j = jnp.array(X_train)
        y_train_j = jnp.array(y_train)

        scale_mult = 50.0

        def loss_fn(params):
            base_vals = decomp_base.lookup_flat(train_indices["base"], params["base"])[:, 0]
            month_dur_vals = decomp_month_dur.lookup_flat(train_indices["month_dur"], params["month_dur"])[:, 0]
            job_age_vals = decomp_job_age.lookup_flat(train_indices["job_age"], params["job_age"])[:, 0]

            month_main = params["month_main"][month_bins[train_idx]]
            job_main = params["job_main"][job_bins[train_idx]]
            age_main = params["age_main"][age_bins[train_idx]]

            intercept = base_vals + month_dur_vals + job_age_vals + month_main + job_main + age_main

            beta_vals = decomp_beta.lookup_flat(train_indices["beta"], params["beta"])
            logits = jnp.sum(X_train_j * beta_vals, axis=-1) + intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_reg = 0.0
            for key in ["base", "month_dur", "job_age", "beta"]:
                for name, p in params[key].items():
                    scale = prior_scales[key].get(name, 1.0)
                    l2_reg += 0.5 * jnp.sum(p**2) / ((scale * scale_mult)**2 + 1e-8)

            l2_reg += 0.5 * jnp.sum(params["month_main"]**2)
            l2_reg += 0.5 * jnp.sum(params["job_main"]**2)
            l2_reg += 0.5 * jnp.sum(params["age_main"]**2)

            return bce + l2_reg / N_train

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.001, peak_value=0.02, warmup_steps=500,
            decay_steps=5500, end_value=0.001
        )
        opt = optax.adam(learning_rate=schedule)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for i in range(6001):
            params, opt_state, loss = step(params, opt_state)
            if i % 1000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}")

        # Evaluate
        test_indices = {
            "base": jnp.array(test_idx_base),
            "month_dur": jnp.array(test_idx_month_dur),
            "job_age": jnp.array(test_idx_job_age),
            "beta": jnp.array(test_idx_beta),
        }
        X_test_j = jnp.array(X_test)

        base_vals = decomp_base.lookup_flat(test_indices["base"], params["base"])[:, 0]
        month_dur_vals = decomp_month_dur.lookup_flat(test_indices["month_dur"], params["month_dur"])[:, 0]
        job_age_vals = decomp_job_age.lookup_flat(test_indices["job_age"], params["job_age"])[:, 0]

        month_main = params["month_main"][month_bins[test_idx]]
        job_main = params["job_main"][job_bins[test_idx]]
        age_main = params["age_main"][age_bins[test_idx]]

        intercept = base_vals + month_dur_vals + job_age_vals + month_main + job_main + age_main

        beta_vals = decomp_beta.lookup_flat(test_indices["beta"], params["beta"])
        probs = 1 / (1 + jnp.exp(-(jnp.sum(X_test_j * beta_vals, axis=-1) + intercept)))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"  AUC = {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\nOURS: {mean_auc:.4f} +/- {std_auc:.4f}")
    print("Baselines: LR=0.907, LGBM=0.935, EBM=0.934")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bank()

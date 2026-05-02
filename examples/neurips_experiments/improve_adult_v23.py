"""Adult v23: Add hours to intercept lattice (age×hours interaction)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
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


def ordinal_smoothness_penalty(params: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    diffs = jnp.diff(params, axis=axis)
    return jnp.sum(diffs ** 2)


def run_adult():
    print("ADULT V23 - Add hours to intercept lattice")
    print("="*70)

    data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    df = data.frame
    y = (df["class"].astype(str) == ">50K").astype(int).values
    N = len(y)
    print(f"N = {N}, pos_rate = {y.mean():.3f}")

    # Categorical features for one-hot
    cats = ["workclass", "relationship", "native-country"]

    # Raw continuous values (drop fnlwgt - it's sampling weight, not predictive)
    age = df["age"].values.astype(float)
    edu_num = df["education-num"].values.astype(float)
    capital_gain = df["capital-gain"].values.astype(float)
    capital_loss = df["capital-loss"].values.astype(float)
    hours = df["hours-per-week"].values.astype(float)

    # Fixed bin edges
    age_edges = np.array([25, 35, 45, 55, 65])
    edu_edges = np.array([9, 11, 14])
    hours_edges = np.array([35, 40, 50])

    # Capital gain bins
    cg_nonzero = capital_gain[capital_gain > 0]
    cg_edges = np.percentile(cg_nonzero, [50, 90]) if len(cg_nonzero) > 0 else np.array([1, 2])

    # Compute all bins
    age_bins = np.digitize(age, age_edges)
    edu_bins = np.digitize(edu_num, edu_edges)
    hours_bins = np.digitize(hours, hours_edges)
    capital_gain_bins = np.zeros(len(capital_gain), dtype=int)
    capital_gain_bins[capital_gain > 0] = 1 + np.digitize(capital_gain[capital_gain > 0], cg_edges)
    capital_loss_bins = (capital_loss > 0).astype(int)

    # Race as features
    race = df["race"].values.astype(str)
    race_bins = np.where(race == "White", 0, np.where(race == "Asian-Pac-Islander", 1, 2))

    # Sex
    sex = df["sex"].values.astype(str)
    sex_bins = np.where(sex == "Male", 0, 1)

    # Marital status
    le_marital = LabelEncoder().fit(df["marital-status"].astype(str))
    marital_bins = le_marital.transform(df["marital-status"].astype(str))
    n_marital = len(le_marital.classes_)

    # Group occupation
    occupation = df["occupation"].values.astype(str)
    occ_groups = {
        "Exec-managerial": 0, "Prof-specialty": 0,
        "Tech-support": 1, "Sales": 1,
        "Craft-repair": 2, "Protective-serv": 2, "Transport-moving": 2,
        "Adm-clerical": 3, "Machine-op-inspct": 3, "Farming-fishing": 3,
        "Other-service": 4, "Priv-house-serv": 4, "Handlers-cleaners": 4, "Armed-Forces": 4, "?": 4
    }
    occupation_bins = np.array([occ_groups.get(o, 4) for o in occupation])
    n_occ = 5

    # Within-bin normalized
    age_norm = within_bin_normalize(age, age_edges)
    edu_norm = within_bin_normalize(edu_num, edu_edges)
    hours_norm = within_bin_normalize(hours, hours_edges)
    cg_norm = np.zeros(len(capital_gain))
    if len(cg_nonzero) > 0:
        cg_norm[capital_gain > 0] = within_bin_normalize(capital_gain[capital_gain > 0], cg_edges)
    cl_nonzero = capital_loss[capital_loss > 0]
    cl_norm = np.zeros(len(capital_loss))
    if len(cl_nonzero) > 0:
        cl_norm[capital_loss > 0] = (capital_loss[capital_loss > 0] - cl_nonzero.min()) / (cl_nonzero.max() - cl_nonzero.min() + 1e-8)

    # Intercept lattice: 8 dimensions including hours
    # age(6) × marital(7) × edu(4) × cg(4) × cl(2) × sex(2) × occ(5) × hours(4) = 53760 cells
    print(f"Intercept lattice: age(6) × marital({n_marital}) × edu(4) × cg(4) × cl(2) × sex(2) × occ({n_occ}) × hours(4)")
    print(f"  = {6 * n_marital * 4 * 4 * 2 * 2 * n_occ * 4} cells")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(N), y)):
        print(f"\nFold {fold_idx + 1}/5:")

        # Numeric features: raw + normalized (no fnlwgt)
        X_numeric = np.column_stack([
            age, edu_num, capital_gain, capital_loss, hours,
            age_norm, edu_norm, hours_norm, cg_norm, cl_norm,
        ])

        # Standardize numeric
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_numeric[train_idx])
        X_test_num = scaler.transform(X_numeric[test_idx])

        # One-hot encode categoricals
        enc = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        X_train_cat = enc.fit_transform(df.iloc[train_idx][cats].astype(str))
        X_test_cat = enc.transform(df.iloc[test_idx][cats].astype(str))

        # Add race as one-hot features
        X_train_race = np.eye(3)[race_bins[train_idx]][:, 1:]
        X_test_race = np.eye(3)[race_bins[test_idx]][:, 1:]

        # Add full occupation one-hot
        le_occ = LabelEncoder().fit(df["occupation"].astype(str))
        occ_full = le_occ.transform(df["occupation"].astype(str))
        X_train_occ = np.eye(len(le_occ.classes_))[occ_full[train_idx]][:, 1:]
        X_test_occ = np.eye(len(le_occ.classes_))[occ_full[test_idx]][:, 1:]

        X_train_base = np.concatenate([X_train_num, X_train_cat, X_train_race, X_train_occ], axis=1)
        X_test_base = np.concatenate([X_test_num, X_test_cat, X_test_race, X_test_occ], axis=1)

        # Add pairwise interactions for top numeric features
        top_idx = [2, 3, 0, 1, 4]  # capital_gain, capital_loss, age, edu, hours
        pairwise_train = []
        pairwise_test = []
        for i, j in combinations(range(5), 2):
            pairwise_train.append(X_train_num[:, top_idx[i]] * X_train_num[:, top_idx[j]])
            pairwise_test.append(X_test_num[:, top_idx[i]] * X_test_num[:, top_idx[j]])

        X_train = np.concatenate([X_train_base, np.stack(pairwise_train, axis=1)], axis=1)
        X_test = np.concatenate([X_test_base, np.stack(pairwise_test, axis=1)], axis=1)

        y_train, y_test = y[train_idx], y[test_idx]
        N_train = len(y_train)
        n_features = X_train.shape[1]

        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        lr.fit(X_train, y_train)
        lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
        print(f"  n_features={n_features}, LR baseline AUC: {lr_auc:.4f}")

        # Intercept lattice dimensions (8 dims with hours)
        dims_int = [
            Dimension("age", 6),
            Dimension("marital", n_marital),
            Dimension("education", 4),
            Dimension("capital_gain", 4),
            Dimension("capital_loss", 2),
            Dimension("sex", 2),
            Dimension("occupation", n_occ),
            Dimension("hours", 4),
        ]

        # Simple beta lattice (just global beta since hours is now in intercept)
        dims_beta = [Dimension("marital", n_marital)]

        interactions_int = Interactions(dimensions=dims_int)
        interactions_beta = Interactions(dimensions=dims_beta)
        decomp_int = Decomposed(interactions=interactions_int, param_shape=[1], name="intercept")
        decomp_beta = Decomposed(interactions=interactions_beta, param_shape=[n_features], name="beta")

        # Build index arrays
        train_idx_int = np.stack([
            age_bins[train_idx],
            marital_bins[train_idx],
            edu_bins[train_idx],
            capital_gain_bins[train_idx],
            capital_loss_bins[train_idx],
            sex_bins[train_idx],
            occupation_bins[train_idx],
            hours_bins[train_idx],
        ], axis=-1)
        test_idx_int = np.stack([
            age_bins[test_idx],
            marital_bins[test_idx],
            edu_bins[test_idx],
            capital_gain_bins[test_idx],
            capital_loss_bins[test_idx],
            sex_bins[test_idx],
            occupation_bins[test_idx],
            hours_bins[test_idx],
        ], axis=-1)
        train_idx_beta = np.stack([marital_bins[train_idx]], axis=-1)
        test_idx_beta = np.stack([marital_bins[test_idx]], axis=-1)

        # Order 2 for intercept (53760 cells needs regularization)
        max_order_int = 2
        max_order_beta = 1
        print(f"  Using orders: intercept={max_order_int}, beta={max_order_beta}")

        prior_scales_int = decomp_int.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True
        )
        prior_scales_beta = decomp_beta.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True
        )

        active_int = [n for n in decomp_int._tensor_parts.keys()
                      if decomp_int.component_order(n) <= max_order_int]
        active_beta = [n for n in decomp_beta._tensor_parts.keys()
                       if decomp_beta.component_order(n) <= max_order_beta]

        print(f"  Intercept: {len(active_int)} components, Beta: {len(active_beta)} components")

        params = {
            "intercept": {n: jnp.zeros(decomp_int._tensor_part_shapes[n]) for n in active_int},
            "beta": {n: jnp.zeros(decomp_beta._tensor_part_shapes[n]) for n in active_beta}
        }

        train_idx_int_j = jnp.array(train_idx_int)
        train_idx_beta_j = jnp.array(train_idx_beta)
        X_train_j = jnp.array(X_train)
        y_train_j = jnp.array(y_train)

        ordinal_int = {"age", "education", "capital_gain", "occupation", "hours"}

        scale_mult = 50.0
        smooth_wt = 0.3

        def loss_fn(params):
            int_vals = decomp_int.lookup_flat(train_idx_int_j, params["intercept"])
            beta_vals = decomp_beta.lookup_flat(train_idx_beta_j, params["beta"])
            logits = jnp.sum(X_train_j * beta_vals, axis=-1) + int_vals[:, 0]
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_int = 0.0
            for n, p in params["intercept"].items():
                scale = prior_scales_int.get(n, 1.0)
                l2_int += 0.5 * jnp.sum(p ** 2) / ((scale * scale_mult) ** 2 + 1e-8)

            l2_beta = 0.0
            for n, p in params["beta"].items():
                scale = prior_scales_beta.get(n, 1.0)
                l2_beta += 0.5 * jnp.sum(p ** 2) / ((scale * scale_mult) ** 2 + 1e-8)

            smooth_penalty = 0.0
            for n, p in params["intercept"].items():
                order = decomp_int.component_order(n)
                if order > 0:
                    for axis, dim_name in enumerate(n.split("_")):
                        if dim_name in ordinal_int and p.shape[axis] > 1:
                            smooth_penalty += ordinal_smoothness_penalty(p, axis=axis)

            reg = (l2_int + l2_beta) / N_train + smooth_wt * smooth_penalty / N_train
            return bce + reg

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.001, peak_value=0.02, warmup_steps=500,
            decay_steps=4500, end_value=0.001
        )
        opt = optax.adam(learning_rate=schedule)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for i in range(5001):
            params, opt_state, loss = step(params, opt_state)
            if i % 1000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}")

        test_idx_int_j = jnp.array(test_idx_int)
        test_idx_beta_j = jnp.array(test_idx_beta)
        X_test_j = jnp.array(X_test)

        int_vals = decomp_int.lookup_flat(test_idx_int_j, params["intercept"])
        beta_vals = decomp_beta.lookup_flat(test_idx_beta_j, params["beta"])
        probs = 1 / (1 + jnp.exp(-(jnp.sum(X_test_j * beta_vals, axis=-1) + int_vals[:, 0])))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"  AUC = {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\nOURS: {mean_auc:.4f} +/- {std_auc:.4f}")
    print("Baselines: LR=0.907, LGBM=0.929, EBM=0.930")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_adult()

"""Adult v24: Multiple additive decomposed terms for hours interactions.

Key insight: pay = hourly_wage × hours
- occupation, education, workclass predict hourly wage
- So we want occupation×hours, education×hours, workclass×hours interactions
- Use separate additive decomposed terms instead of one giant lattice
"""
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
    print("ADULT V24 - Multiple additive decomposed terms for hours interactions")
    print("="*70)
    print("pay = hourly_wage × hours")
    print("→ occupation×hours, education×hours, workclass×hours interactions")
    print("="*70)

    data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    df = data.frame
    y = (df["class"].astype(str) == ">50K").astype(int).values
    N = len(y)
    print(f"N = {N}, pos_rate = {y.mean():.3f}")

    # Raw continuous values
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

    # Race
    race = df["race"].values.astype(str)
    race_bins = np.where(race == "White", 0, np.where(race == "Asian-Pac-Islander", 1, 2))

    # Sex
    sex = df["sex"].values.astype(str)
    sex_bins = np.where(sex == "Male", 0, 1)

    # Marital status
    le_marital = LabelEncoder().fit(df["marital-status"].astype(str))
    marital_bins = le_marital.transform(df["marital-status"].astype(str))
    n_marital = len(le_marital.classes_)

    # Occupation (full - 15 categories)
    le_occ = LabelEncoder().fit(df["occupation"].astype(str))
    occupation_bins = le_occ.transform(df["occupation"].astype(str))
    n_occ = len(le_occ.classes_)

    # Workclass (8 categories)
    le_workclass = LabelEncoder().fit(df["workclass"].astype(str))
    workclass_bins = le_workclass.transform(df["workclass"].astype(str))
    n_workclass = len(le_workclass.classes_)

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

    print(f"\nDecomposition structure:")
    print(f"  Base intercept: age(6) × marital({n_marital}) × cg(4) × cl(2) × sex(2) = {6*n_marital*4*2*2} cells")
    print(f"  Occupation × hours: {n_occ} × 4 = {n_occ * 4} cells")
    print(f"  Education × hours: 4 × 4 = 16 cells")
    print(f"  Workclass × hours: {n_workclass} × 4 = {n_workclass * 4} cells")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(N), y)):
        print(f"\nFold {fold_idx + 1}/5:")

        # Numeric features
        X_numeric = np.column_stack([
            age, edu_num, capital_gain, capital_loss, hours,
            age_norm, edu_norm, hours_norm, cg_norm, cl_norm,
        ])

        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_numeric[train_idx])
        X_test_num = scaler.transform(X_numeric[test_idx])

        # One-hot for remaining categoricals
        cats = ["relationship", "native-country"]
        enc = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        X_train_cat = enc.fit_transform(df.iloc[train_idx][cats].astype(str))
        X_test_cat = enc.transform(df.iloc[test_idx][cats].astype(str))

        # Race one-hot
        X_train_race = np.eye(3)[race_bins[train_idx]][:, 1:]
        X_test_race = np.eye(3)[race_bins[test_idx]][:, 1:]

        X_train = np.concatenate([X_train_num, X_train_cat, X_train_race], axis=1)
        X_test = np.concatenate([X_test_num, X_test_cat, X_test_race], axis=1)

        y_train, y_test = y[train_idx], y[test_idx]
        N_train = len(y_train)
        n_features = X_train.shape[1]

        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        lr.fit(X_train, y_train)
        lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
        print(f"  n_features={n_features}, LR baseline AUC: {lr_auc:.4f}")

        # Build multiple decomposed components

        # 1. Base intercept lattice (demographics + capital)
        dims_base = [
            Dimension("age", 6),
            Dimension("marital", n_marital),
            Dimension("capital_gain", 4),
            Dimension("capital_loss", 2),
            Dimension("sex", 2),
        ]
        decomp_base = Decomposed(interactions=Interactions(dimensions=dims_base), param_shape=[1], name="base")

        # 2. Occupation × hours (hourly wage interaction)
        dims_occ_hours = [Dimension("occupation", n_occ), Dimension("hours", 4)]
        decomp_occ_hours = Decomposed(interactions=Interactions(dimensions=dims_occ_hours), param_shape=[1], name="occ_hours")

        # 3. Education × hours
        dims_edu_hours = [Dimension("education", 4), Dimension("hours", 4)]
        decomp_edu_hours = Decomposed(interactions=Interactions(dimensions=dims_edu_hours), param_shape=[1], name="edu_hours")

        # 4. Workclass × hours
        dims_work_hours = [Dimension("workclass", n_workclass), Dimension("hours", 4)]
        decomp_work_hours = Decomposed(interactions=Interactions(dimensions=dims_work_hours), param_shape=[1], name="work_hours")

        # 5. Marital × hours (for beta - how coefficients vary)
        dims_beta = [Dimension("marital", n_marital), Dimension("hours", 4)]
        decomp_beta = Decomposed(interactions=Interactions(dimensions=dims_beta), param_shape=[n_features], name="beta")

        # Build index arrays
        train_idx_base = np.stack([
            age_bins[train_idx],
            marital_bins[train_idx],
            capital_gain_bins[train_idx],
            capital_loss_bins[train_idx],
            sex_bins[train_idx],
        ], axis=-1)
        test_idx_base = np.stack([
            age_bins[test_idx],
            marital_bins[test_idx],
            capital_gain_bins[test_idx],
            capital_loss_bins[test_idx],
            sex_bins[test_idx],
        ], axis=-1)

        train_idx_occ_hours = np.stack([occupation_bins[train_idx], hours_bins[train_idx]], axis=-1)
        test_idx_occ_hours = np.stack([occupation_bins[test_idx], hours_bins[test_idx]], axis=-1)

        train_idx_edu_hours = np.stack([edu_bins[train_idx], hours_bins[train_idx]], axis=-1)
        test_idx_edu_hours = np.stack([edu_bins[test_idx], hours_bins[test_idx]], axis=-1)

        train_idx_work_hours = np.stack([workclass_bins[train_idx], hours_bins[train_idx]], axis=-1)
        test_idx_work_hours = np.stack([workclass_bins[test_idx], hours_bins[test_idx]], axis=-1)

        train_idx_beta = np.stack([marital_bins[train_idx], hours_bins[train_idx]], axis=-1)
        test_idx_beta = np.stack([marital_bins[test_idx], hours_bins[test_idx]], axis=-1)

        # Use order 2 for base, full order for 2-way interactions
        max_order_base = 2
        max_order_2way = 2  # Full order for 2-way is 2

        prior_scales_base = decomp_base.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)
        prior_scales_occ = decomp_occ_hours.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)
        prior_scales_edu = decomp_edu_hours.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)
        prior_scales_work = decomp_work_hours.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)
        prior_scales_beta = decomp_beta.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)

        active_base = [n for n in decomp_base._tensor_parts.keys() if decomp_base.component_order(n) <= max_order_base]
        active_occ = [n for n in decomp_occ_hours._tensor_parts.keys() if decomp_occ_hours.component_order(n) <= max_order_2way]
        active_edu = [n for n in decomp_edu_hours._tensor_parts.keys() if decomp_edu_hours.component_order(n) <= max_order_2way]
        active_work = [n for n in decomp_work_hours._tensor_parts.keys() if decomp_work_hours.component_order(n) <= max_order_2way]
        active_beta = [n for n in decomp_beta._tensor_parts.keys() if decomp_beta.component_order(n) <= max_order_2way]

        total_components = len(active_base) + len(active_occ) + len(active_edu) + len(active_work) + len(active_beta)
        print(f"  Total components: {total_components} (base={len(active_base)}, occ×h={len(active_occ)}, edu×h={len(active_edu)}, work×h={len(active_work)}, beta={len(active_beta)})")

        params = {
            "base": {n: jnp.zeros(decomp_base._tensor_part_shapes[n]) for n in active_base},
            "occ_hours": {n: jnp.zeros(decomp_occ_hours._tensor_part_shapes[n]) for n in active_occ},
            "edu_hours": {n: jnp.zeros(decomp_edu_hours._tensor_part_shapes[n]) for n in active_edu},
            "work_hours": {n: jnp.zeros(decomp_work_hours._tensor_part_shapes[n]) for n in active_work},
            "beta": {n: jnp.zeros(decomp_beta._tensor_part_shapes[n]) for n in active_beta},
        }

        # JAX arrays
        train_idx_base_j = jnp.array(train_idx_base)
        train_idx_occ_hours_j = jnp.array(train_idx_occ_hours)
        train_idx_edu_hours_j = jnp.array(train_idx_edu_hours)
        train_idx_work_hours_j = jnp.array(train_idx_work_hours)
        train_idx_beta_j = jnp.array(train_idx_beta)
        X_train_j = jnp.array(X_train)
        y_train_j = jnp.array(y_train)

        ordinal_dims = {"age", "education", "capital_gain", "hours"}
        scale_mult = 50.0
        smooth_wt = 0.3

        def loss_fn(params):
            # Sum all intercept contributions
            base_vals = decomp_base.lookup_flat(train_idx_base_j, params["base"])[:, 0]
            occ_vals = decomp_occ_hours.lookup_flat(train_idx_occ_hours_j, params["occ_hours"])[:, 0]
            edu_vals = decomp_edu_hours.lookup_flat(train_idx_edu_hours_j, params["edu_hours"])[:, 0]
            work_vals = decomp_work_hours.lookup_flat(train_idx_work_hours_j, params["work_hours"])[:, 0]

            intercept = base_vals + occ_vals + edu_vals + work_vals

            beta_vals = decomp_beta.lookup_flat(train_idx_beta_j, params["beta"])
            logits = jnp.sum(X_train_j * beta_vals, axis=-1) + intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # L2 regularization
            l2_reg = 0.0
            for name, p in params["base"].items():
                scale = prior_scales_base.get(name, 1.0)
                l2_reg += 0.5 * jnp.sum(p ** 2) / ((scale * scale_mult) ** 2 + 1e-8)
            for name, p in params["occ_hours"].items():
                scale = prior_scales_occ.get(name, 1.0)
                l2_reg += 0.5 * jnp.sum(p ** 2) / ((scale * scale_mult) ** 2 + 1e-8)
            for name, p in params["edu_hours"].items():
                scale = prior_scales_edu.get(name, 1.0)
                l2_reg += 0.5 * jnp.sum(p ** 2) / ((scale * scale_mult) ** 2 + 1e-8)
            for name, p in params["work_hours"].items():
                scale = prior_scales_work.get(name, 1.0)
                l2_reg += 0.5 * jnp.sum(p ** 2) / ((scale * scale_mult) ** 2 + 1e-8)
            for name, p in params["beta"].items():
                scale = prior_scales_beta.get(name, 1.0)
                l2_reg += 0.5 * jnp.sum(p ** 2) / ((scale * scale_mult) ** 2 + 1e-8)

            # Ordinal smoothness on hours dimension
            smooth_penalty = 0.0
            for name, p in params["occ_hours"].items():
                if "hours" in name and p.shape[-2] > 1:  # hours is second dim
                    smooth_penalty += ordinal_smoothness_penalty(p, axis=-2)
            for name, p in params["edu_hours"].items():
                if "hours" in name and p.shape[-2] > 1:
                    smooth_penalty += ordinal_smoothness_penalty(p, axis=-2)
            for name, p in params["work_hours"].items():
                if "hours" in name and p.shape[-2] > 1:
                    smooth_penalty += ordinal_smoothness_penalty(p, axis=-2)

            reg = l2_reg / N_train + smooth_wt * smooth_penalty / N_train
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

        # Evaluate
        test_idx_base_j = jnp.array(test_idx_base)
        test_idx_occ_hours_j = jnp.array(test_idx_occ_hours)
        test_idx_edu_hours_j = jnp.array(test_idx_edu_hours)
        test_idx_work_hours_j = jnp.array(test_idx_work_hours)
        test_idx_beta_j = jnp.array(test_idx_beta)
        X_test_j = jnp.array(X_test)

        base_vals = decomp_base.lookup_flat(test_idx_base_j, params["base"])[:, 0]
        occ_vals = decomp_occ_hours.lookup_flat(test_idx_occ_hours_j, params["occ_hours"])[:, 0]
        edu_vals = decomp_edu_hours.lookup_flat(test_idx_edu_hours_j, params["edu_hours"])[:, 0]
        work_vals = decomp_work_hours.lookup_flat(test_idx_work_hours_j, params["work_hours"])[:, 0]
        intercept = base_vals + occ_vals + edu_vals + work_vals

        beta_vals = decomp_beta.lookup_flat(test_idx_beta_j, params["beta"])
        probs = 1 / (1 + jnp.exp(-(jnp.sum(X_test_j * beta_vals, axis=-1) + intercept)))

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

"""Adult v29: Finer bins (8) + stage-wise boosting.

Fit components sequentially, each to residuals of previous.
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


def run_adult():
    print("ADULT V29 - Finer bins (8) + stage-wise boosting")
    print("="*70)

    data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    df = data.frame
    y = (df["class"].astype(str) == ">50K").astype(int).values
    N = len(y)
    print(f"N = {N}, pos_rate = {y.mean():.3f}")

    age = df["age"].values.astype(float)
    edu_num = df["education-num"].values.astype(float)
    capital_gain = df["capital-gain"].values.astype(float)
    capital_loss = df["capital-loss"].values.astype(float)
    hours = df["hours-per-week"].values.astype(float)

    # FINER bins - 8 bins for continuous variables (like EBM)
    age_edges = np.percentile(age, [12.5, 25, 37.5, 50, 62.5, 75, 87.5])  # 8 bins
    edu_edges = np.array([8, 9, 10, 11, 12, 13, 14])  # 8 bins based on education-num
    hours_edges = np.percentile(hours, [12.5, 25, 37.5, 50, 62.5, 75, 87.5])  # 8 bins

    cg_nonzero = capital_gain[capital_gain > 0]
    # 5 bins for capital gain: 0, then 4 quantile bins
    cg_edges = np.percentile(cg_nonzero, [25, 50, 75]) if len(cg_nonzero) > 0 else np.array([1, 2, 3])

    age_bins = np.digitize(age, age_edges)
    edu_bins = np.digitize(edu_num, edu_edges)
    hours_bins = np.digitize(hours, hours_edges)
    capital_gain_bins = np.zeros(len(capital_gain), dtype=int)
    capital_gain_bins[capital_gain > 0] = 1 + np.digitize(capital_gain[capital_gain > 0], cg_edges)
    capital_loss_bins = (capital_loss > 0).astype(int)

    race = df["race"].values.astype(str)
    race_bins = np.where(race == "White", 0, np.where(race == "Asian-Pac-Islander", 1, 2))

    sex = df["sex"].values.astype(str)
    sex_bins = np.where(sex == "Male", 0, 1)

    le_marital = LabelEncoder().fit(df["marital-status"].astype(str))
    marital_bins = le_marital.transform(df["marital-status"].astype(str))
    n_marital = len(le_marital.classes_)

    le_occ = LabelEncoder().fit(df["occupation"].astype(str))
    occupation_bins = le_occ.transform(df["occupation"].astype(str))
    n_occ = len(le_occ.classes_)

    le_workclass = LabelEncoder().fit(df["workclass"].astype(str))
    workclass_bins = le_workclass.transform(df["workclass"].astype(str))
    n_workclass = len(le_workclass.classes_)

    le_relationship = LabelEncoder().fit(df["relationship"].astype(str))
    relationship_bins = le_relationship.transform(df["relationship"].astype(str))
    n_relationship = len(le_relationship.classes_)

    age_norm = within_bin_normalize(age, age_edges)
    edu_norm = within_bin_normalize(edu_num, edu_edges)
    hours_norm = within_bin_normalize(hours, hours_edges)
    cg_norm = np.zeros(len(capital_gain))
    if len(cg_nonzero) > 0:
        cg_norm[capital_gain > 0] = within_bin_normalize(capital_gain[capital_gain > 0], cg_edges)

    print(f"\nFiner binning: age(8), edu(8), hours(8), cg(5), cl(2)")
    print(f"Stage-wise boosting: base → interactions → more interactions")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(N), y)):
        print(f"\nFold {fold_idx + 1}/5:")

        X_numeric = np.column_stack([
            age, edu_num, capital_gain, capital_loss, hours,
            age_norm, edu_norm, hours_norm, cg_norm,
        ])

        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_numeric[train_idx])
        X_test_num = scaler.transform(X_numeric[test_idx])

        cats = ["occupation", "education", "workclass", "relationship", "native-country"]
        enc = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        X_train_cat = enc.fit_transform(df.iloc[train_idx][cats].astype(str))
        X_test_cat = enc.transform(df.iloc[test_idx][cats].astype(str))

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

        # ============================================================
        # STAGE 1: Base lattice (demographics + capital)
        # ============================================================
        dims_base = [
            Dimension("age", 8),
            Dimension("marital", n_marital),
            Dimension("capital_gain", 5),
            Dimension("capital_loss", 2),
            Dimension("sex", 2),
        ]
        decomp_base = Decomposed(interactions=Interactions(dimensions=dims_base), param_shape=[1], name="base")

        train_idx_base = np.stack([
            age_bins[train_idx], marital_bins[train_idx], capital_gain_bins[train_idx],
            capital_loss_bins[train_idx], sex_bins[train_idx],
        ], axis=-1)
        test_idx_base = np.stack([
            age_bins[test_idx], marital_bins[test_idx], capital_gain_bins[test_idx],
            capital_loss_bins[test_idx], sex_bins[test_idx],
        ], axis=-1)

        prior_scales_base = decomp_base.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)
        active_base = [n for n in decomp_base._tensor_parts.keys() if decomp_base.component_order(n) <= 2]

        params_base = {n: jnp.zeros(decomp_base._tensor_part_shapes[n]) for n in active_base}
        # Also fit global beta
        params_beta = jnp.zeros(n_features)

        train_idx_base_j = jnp.array(train_idx_base)
        X_train_j = jnp.array(X_train)
        y_train_j = jnp.array(y_train)

        scale_mult = 50.0

        def loss_stage1(params_base, params_beta):
            base_vals = decomp_base.lookup_flat(train_idx_base_j, params_base)[:, 0]
            logits = jnp.sum(X_train_j * params_beta, axis=-1) + base_vals
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)
            l2_base = sum(0.5 * jnp.sum(p**2) / ((prior_scales_base.get(n, 1.0) * scale_mult)**2 + 1e-8)
                         for n, p in params_base.items())
            l2_beta = 0.5 * jnp.sum(params_beta**2)
            return bce + (l2_base + l2_beta) / N_train

        opt1 = optax.adam(0.01)
        opt_state1 = opt1.init((params_base, params_beta))

        @jax.jit
        def step1(params_base, params_beta, opt_state):
            loss, grads = jax.value_and_grad(loss_stage1, argnums=(0, 1))(params_base, params_beta)
            updates, opt_state = opt1.update(grads, opt_state, (params_base, params_beta))
            params_base = optax.apply_updates(params_base, updates[0])
            params_beta = optax.apply_updates(params_beta, updates[1])
            return params_base, params_beta, opt_state, loss

        print("  Stage 1: Fitting base lattice + global beta...")
        for i in range(3001):
            params_base, params_beta, opt_state1, loss = step1(params_base, params_beta, opt_state1)
            if i % 1000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}")

        # Get predictions from stage 1
        base_preds_train = decomp_base.lookup_flat(train_idx_base_j, params_base)[:, 0]
        base_preds_train += jnp.sum(X_train_j * params_beta, axis=-1)

        # ============================================================
        # STAGE 2: Add interaction terms (fit to residuals)
        # ============================================================
        # Interaction: occupation × hours
        dims_occ_hours = [Dimension("occupation", n_occ), Dimension("hours", 8)]
        decomp_occ_hours = Decomposed(interactions=Interactions(dimensions=dims_occ_hours), param_shape=[1], name="occ_hours")

        # Interaction: education × hours
        dims_edu_hours = [Dimension("education", 8), Dimension("hours", 8)]
        decomp_edu_hours = Decomposed(interactions=Interactions(dimensions=dims_edu_hours), param_shape=[1], name="edu_hours")

        # Interaction: relationship × capital_gain
        dims_rel_cg = [Dimension("relationship", n_relationship), Dimension("capital_gain", 5)]
        decomp_rel_cg = Decomposed(interactions=Interactions(dimensions=dims_rel_cg), param_shape=[1], name="rel_cg")

        train_idx_occ_hours = np.stack([occupation_bins[train_idx], hours_bins[train_idx]], axis=-1)
        train_idx_edu_hours = np.stack([edu_bins[train_idx], hours_bins[train_idx]], axis=-1)
        train_idx_rel_cg = np.stack([relationship_bins[train_idx], capital_gain_bins[train_idx]], axis=-1)

        train_idx_occ_hours_j = jnp.array(train_idx_occ_hours)
        train_idx_edu_hours_j = jnp.array(train_idx_edu_hours)
        train_idx_rel_cg_j = jnp.array(train_idx_rel_cg)

        # Only 2-way interactions (order 2)
        active_occ_hours = [n for n in decomp_occ_hours._tensor_parts.keys() if decomp_occ_hours.component_order(n) == 2]
        active_edu_hours = [n for n in decomp_edu_hours._tensor_parts.keys() if decomp_edu_hours.component_order(n) == 2]
        active_rel_cg = [n for n in decomp_rel_cg._tensor_parts.keys() if decomp_rel_cg.component_order(n) == 2]

        params_occ_hours = {n: jnp.zeros(decomp_occ_hours._tensor_part_shapes[n]) for n in active_occ_hours}
        params_edu_hours = {n: jnp.zeros(decomp_edu_hours._tensor_part_shapes[n]) for n in active_edu_hours}
        params_rel_cg = {n: jnp.zeros(decomp_rel_cg._tensor_part_shapes[n]) for n in active_rel_cg}

        prior_occ = decomp_occ_hours.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)
        prior_edu = decomp_edu_hours.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)
        prior_rel = decomp_rel_cg.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)

        shrinkage = 0.5  # Boosting shrinkage

        def loss_stage2(params_occ, params_edu, params_rel):
            occ_vals = decomp_occ_hours.lookup_flat(train_idx_occ_hours_j, params_occ)[:, 0]
            edu_vals = decomp_edu_hours.lookup_flat(train_idx_edu_hours_j, params_edu)[:, 0]
            rel_vals = decomp_rel_cg.lookup_flat(train_idx_rel_cg_j, params_rel)[:, 0]

            # Add to base predictions (frozen)
            logits = base_preds_train + shrinkage * (occ_vals + edu_vals + rel_vals)
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_occ = sum(0.5 * jnp.sum(p**2) / ((prior_occ.get(n, 1.0) * scale_mult)**2 + 1e-8) for n, p in params_occ.items())
            l2_edu = sum(0.5 * jnp.sum(p**2) / ((prior_edu.get(n, 1.0) * scale_mult)**2 + 1e-8) for n, p in params_edu.items())
            l2_rel = sum(0.5 * jnp.sum(p**2) / ((prior_rel.get(n, 1.0) * scale_mult)**2 + 1e-8) for n, p in params_rel.items())

            return bce + (l2_occ + l2_edu + l2_rel) / N_train

        opt2 = optax.adam(0.01)
        opt_state2 = opt2.init((params_occ_hours, params_edu_hours, params_rel_cg))

        @jax.jit
        def step2(params_occ, params_edu, params_rel, opt_state):
            loss, grads = jax.value_and_grad(loss_stage2, argnums=(0, 1, 2))(params_occ, params_edu, params_rel)
            updates, opt_state = opt2.update(grads, opt_state, (params_occ, params_edu, params_rel))
            params_occ = optax.apply_updates(params_occ, updates[0])
            params_edu = optax.apply_updates(params_edu, updates[1])
            params_rel = optax.apply_updates(params_rel, updates[2])
            return params_occ, params_edu, params_rel, opt_state, loss

        print("  Stage 2: Fitting interaction terms to residuals...")
        for i in range(2001):
            params_occ_hours, params_edu_hours, params_rel_cg, opt_state2, loss = step2(
                params_occ_hours, params_edu_hours, params_rel_cg, opt_state2)
            if i % 1000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}")

        # ============================================================
        # Evaluate
        # ============================================================
        test_idx_base_j = jnp.array(test_idx_base)
        test_idx_occ_hours = np.stack([occupation_bins[test_idx], hours_bins[test_idx]], axis=-1)
        test_idx_edu_hours = np.stack([edu_bins[test_idx], hours_bins[test_idx]], axis=-1)
        test_idx_rel_cg = np.stack([relationship_bins[test_idx], capital_gain_bins[test_idx]], axis=-1)

        test_idx_occ_hours_j = jnp.array(test_idx_occ_hours)
        test_idx_edu_hours_j = jnp.array(test_idx_edu_hours)
        test_idx_rel_cg_j = jnp.array(test_idx_rel_cg)
        X_test_j = jnp.array(X_test)

        base_preds = decomp_base.lookup_flat(test_idx_base_j, params_base)[:, 0]
        base_preds += jnp.sum(X_test_j * params_beta, axis=-1)

        occ_preds = decomp_occ_hours.lookup_flat(test_idx_occ_hours_j, params_occ_hours)[:, 0]
        edu_preds = decomp_edu_hours.lookup_flat(test_idx_edu_hours_j, params_edu_hours)[:, 0]
        rel_preds = decomp_rel_cg.lookup_flat(test_idx_rel_cg_j, params_rel_cg)[:, 0]

        logits = base_preds + shrinkage * (occ_preds + edu_preds + rel_preds)
        probs = 1 / (1 + jnp.exp(-logits))

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

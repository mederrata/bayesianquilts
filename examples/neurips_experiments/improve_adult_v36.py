"""Adult v36: Two-stage fitting (fit base, then fit residuals)."""
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
    print("ADULT V36 - Two-stage fitting (base + residual)")
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

    age_edges = np.percentile(age, [12.5, 25, 37.5, 50, 62.5, 75, 87.5])
    edu_edges = np.array([8, 9, 10, 11, 12, 13, 14])
    hours_edges = np.percentile(hours, [12.5, 25, 37.5, 50, 62.5, 75, 87.5])

    cg_nonzero = capital_gain[capital_gain > 0]
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

    le_relationship = LabelEncoder().fit(df["relationship"].astype(str))
    relationship_bins = le_relationship.transform(df["relationship"].astype(str))
    n_relationship = len(le_relationship.classes_)

    age_norm = within_bin_normalize(age, age_edges)
    edu_norm = within_bin_normalize(edu_num, edu_edges)
    hours_norm = within_bin_normalize(hours, hours_edges)
    cg_norm = np.zeros(len(capital_gain))
    if len(cg_nonzero) > 0:
        cg_norm[capital_gain > 0] = within_bin_normalize(capital_gain[capital_gain > 0], cg_edges)

    print(f"\nTwo-stage: base model + residual correction")

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

        # Stage 1: Base model
        dims_base = [
            Dimension("age", 8),
            Dimension("marital", n_marital),
            Dimension("capital_gain", 5),
            Dimension("capital_loss", 2),
            Dimension("sex", 2),
        ]
        decomp_base = Decomposed(interactions=Interactions(dimensions=dims_base), param_shape=[1], name="base")

        dims_occ_hours = [Dimension("occupation", n_occ), Dimension("hours", 8)]
        decomp_occ_hours = Decomposed(interactions=Interactions(dimensions=dims_occ_hours), param_shape=[1], name="occ_hours")

        dims_edu_hours = [Dimension("education", 8), Dimension("hours", 8)]
        decomp_edu_hours = Decomposed(interactions=Interactions(dimensions=dims_edu_hours), param_shape=[1], name="edu_hours")

        dims_rel_cg = [Dimension("relationship", n_relationship), Dimension("capital_gain", 5)]
        decomp_rel_cg = Decomposed(interactions=Interactions(dimensions=dims_rel_cg), param_shape=[1], name="rel_cg")

        dims_beta = [Dimension("marital", n_marital), Dimension("hours", 8)]
        decomp_beta = Decomposed(interactions=Interactions(dimensions=dims_beta), param_shape=[n_features], name="beta")

        train_idx_base = np.stack([
            age_bins[train_idx], marital_bins[train_idx], capital_gain_bins[train_idx],
            capital_loss_bins[train_idx], sex_bins[train_idx],
        ], axis=-1)
        test_idx_base = np.stack([
            age_bins[test_idx], marital_bins[test_idx], capital_gain_bins[test_idx],
            capital_loss_bins[test_idx], sex_bins[test_idx],
        ], axis=-1)

        train_idx_occ_hours = np.stack([occupation_bins[train_idx], hours_bins[train_idx]], axis=-1)
        test_idx_occ_hours = np.stack([occupation_bins[test_idx], hours_bins[test_idx]], axis=-1)

        train_idx_edu_hours = np.stack([edu_bins[train_idx], hours_bins[train_idx]], axis=-1)
        test_idx_edu_hours = np.stack([edu_bins[test_idx], hours_bins[test_idx]], axis=-1)

        train_idx_rel_cg = np.stack([relationship_bins[train_idx], capital_gain_bins[train_idx]], axis=-1)
        test_idx_rel_cg = np.stack([relationship_bins[test_idx], capital_gain_bins[test_idx]], axis=-1)

        train_idx_beta = np.stack([marital_bins[train_idx], hours_bins[train_idx]], axis=-1)
        test_idx_beta = np.stack([marital_bins[test_idx], hours_bins[test_idx]], axis=-1)

        decomps = {
            "base": decomp_base,
            "occ_hours": decomp_occ_hours,
            "edu_hours": decomp_edu_hours,
            "rel_cg": decomp_rel_cg,
            "beta": decomp_beta,
        }

        prior_scales = {}
        active = {}
        for key, decomp in decomps.items():
            prior_scales[key] = decomp.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)
            if key in ["occ_hours", "edu_hours", "rel_cg"]:
                active[key] = [n for n in decomp._tensor_parts.keys() if decomp.component_order(n) == 2]
            else:
                active[key] = [n for n in decomp._tensor_parts.keys() if decomp.component_order(n) <= 2]

        params = {}
        for key, decomp in decomps.items():
            params[key] = {n: jnp.zeros(decomp._tensor_part_shapes[n]) for n in active[key]}

        params["occ_main"] = jnp.zeros(n_occ)
        params["edu_main"] = jnp.zeros(8)
        params["hours_main"] = jnp.zeros(8)
        params["rel_main"] = jnp.zeros(n_relationship)

        train_indices = {
            "base": jnp.array(train_idx_base),
            "occ_hours": jnp.array(train_idx_occ_hours),
            "edu_hours": jnp.array(train_idx_edu_hours),
            "rel_cg": jnp.array(train_idx_rel_cg),
            "beta": jnp.array(train_idx_beta),
        }
        X_train_j = jnp.array(X_train)
        y_train_j = jnp.array(y_train)

        scale_mult = 50.0

        def compute_logits(params, indices, X):
            base_vals = decomp_base.lookup_flat(indices["base"], params["base"])[:, 0]
            occ_vals = decomp_occ_hours.lookup_flat(indices["occ_hours"], params["occ_hours"])[:, 0]
            edu_vals = decomp_edu_hours.lookup_flat(indices["edu_hours"], params["edu_hours"])[:, 0]
            rel_vals = decomp_rel_cg.lookup_flat(indices["rel_cg"], params["rel_cg"])[:, 0]

            occ_main = params["occ_main"][occupation_bins[train_idx if len(indices["base"]) == N_train else test_idx]]
            edu_main = params["edu_main"][edu_bins[train_idx if len(indices["base"]) == N_train else test_idx]]
            hours_main = params["hours_main"][hours_bins[train_idx if len(indices["base"]) == N_train else test_idx]]
            rel_main = params["rel_main"][relationship_bins[train_idx if len(indices["base"]) == N_train else test_idx]]

            intercept = base_vals + occ_vals + edu_vals + rel_vals + occ_main + edu_main + hours_main + rel_main
            beta_vals = decomp_beta.lookup_flat(indices["beta"], params["beta"])
            return jnp.sum(X * beta_vals, axis=-1) + intercept

        def loss_fn(params):
            logits = compute_logits(params, train_indices, X_train_j)
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_reg = 0.0
            for key in ["base", "occ_hours", "edu_hours", "rel_cg", "beta"]:
                for name, p in params[key].items():
                    scale = prior_scales[key].get(name, 1.0)
                    l2_reg += 0.5 * jnp.sum(p**2) / ((scale * scale_mult)**2 + 1e-8)

            l2_reg += 0.5 * jnp.sum(params["occ_main"]**2)
            l2_reg += 0.5 * jnp.sum(params["edu_main"]**2)
            l2_reg += 0.5 * jnp.sum(params["hours_main"]**2)
            l2_reg += 0.5 * jnp.sum(params["rel_main"]**2)

            return bce + l2_reg / N_train

        # Stage 1 training
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

        print("  Stage 1: Base model")
        for i in range(6001):
            params, opt_state, loss = step(params, opt_state)
            if i % 2000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}")

        # Get stage 1 predictions
        train_logits_s1 = np.array(compute_logits(params, train_indices, X_train_j))
        train_probs_s1 = 1 / (1 + np.exp(-train_logits_s1))

        # Stage 2: Residual model - learn corrections based on age×education (EBM's top)
        print("  Stage 2: Residual correction (age×edu)")
        dims_age_edu = [Dimension("age", 8), Dimension("education", 8)]
        decomp_age_edu = Decomposed(interactions=Interactions(dimensions=dims_age_edu), param_shape=[1], name="age_edu")

        train_idx_age_edu = np.stack([age_bins[train_idx], edu_bins[train_idx]], axis=-1)
        test_idx_age_edu = np.stack([age_bins[test_idx], edu_bins[test_idx]], axis=-1)

        prior_scales_ae = decomp_age_edu.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)
        active_ae = [n for n in decomp_age_edu._tensor_parts.keys() if decomp_age_edu.component_order(n) == 2]

        params2 = {n: jnp.zeros(decomp_age_edu._tensor_part_shapes[n]) for n in active_ae}
        train_indices_ae = jnp.array(train_idx_age_edu)

        # Residual: working points
        working_weights = train_probs_s1 * (1 - train_probs_s1) + 1e-6
        residuals = (y_train - train_probs_s1) / working_weights

        def loss_fn2(params2):
            corrections = decomp_age_edu.lookup_flat(train_indices_ae, params2)[:, 0]
            # Weighted least squares on residuals
            loss = jnp.mean(working_weights * (residuals - corrections)**2)

            l2_reg = 0.0
            for name, p in params2.items():
                scale = prior_scales_ae.get(name, 1.0)
                l2_reg += 0.5 * jnp.sum(p**2) / ((scale * scale_mult)**2 + 1e-8)

            return loss + l2_reg / N_train

        schedule2 = optax.warmup_cosine_decay_schedule(
            init_value=0.001, peak_value=0.02, warmup_steps=200,
            decay_steps=1800, end_value=0.001
        )
        opt2 = optax.adam(learning_rate=schedule2)
        opt_state2 = opt2.init(params2)

        @jax.jit
        def step2(params2, opt_state2):
            loss, grads = jax.value_and_grad(loss_fn2)(params2)
            updates, opt_state2 = opt2.update(grads, opt_state2, params2)
            return optax.apply_updates(params2, updates), opt_state2, loss

        for i in range(2001):
            params2, opt_state2, loss = step2(params2, opt_state2)
            if i % 1000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}")

        # Test predictions
        test_indices = {
            "base": jnp.array(test_idx_base),
            "occ_hours": jnp.array(test_idx_occ_hours),
            "edu_hours": jnp.array(test_idx_edu_hours),
            "rel_cg": jnp.array(test_idx_rel_cg),
            "beta": jnp.array(test_idx_beta),
        }
        X_test_j = jnp.array(X_test)

        # Compute s1 logits on test (fixing the index issue)
        base_vals = decomp_base.lookup_flat(test_indices["base"], params["base"])[:, 0]
        occ_vals = decomp_occ_hours.lookup_flat(test_indices["occ_hours"], params["occ_hours"])[:, 0]
        edu_vals = decomp_edu_hours.lookup_flat(test_indices["edu_hours"], params["edu_hours"])[:, 0]
        rel_vals = decomp_rel_cg.lookup_flat(test_indices["rel_cg"], params["rel_cg"])[:, 0]

        occ_main = params["occ_main"][occupation_bins[test_idx]]
        edu_main = params["edu_main"][edu_bins[test_idx]]
        hours_main = params["hours_main"][hours_bins[test_idx]]
        rel_main = params["rel_main"][relationship_bins[test_idx]]

        intercept = base_vals + occ_vals + edu_vals + rel_vals + occ_main + edu_main + hours_main + rel_main
        beta_vals = decomp_beta.lookup_flat(test_indices["beta"], params["beta"])
        test_logits_s1 = jnp.sum(X_test_j * beta_vals, axis=-1) + intercept

        # Add stage 2 corrections
        test_indices_ae = jnp.array(test_idx_age_edu)
        corrections = decomp_age_edu.lookup_flat(test_indices_ae, params2)[:, 0]

        test_logits = test_logits_s1 + corrections
        probs = 1 / (1 + jnp.exp(-test_logits))

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

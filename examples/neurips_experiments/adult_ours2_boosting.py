"""Adult Ours #2 with Boosting: fit residuals iteratively like EBM."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
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


def run_adult_boosting():
    """Adult Ours #2 with boosting."""
    print("ADULT - OURS #2 WITH BOOSTING")
    print("="*70)

    data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    df = data.frame
    y = (df["class"].astype(str) == ">50K").astype(int).values
    N = len(y)
    print(f"N = {N}, pos_rate = {y.mean():.3f}")

    # Same feature engineering as v30
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

    race = df["race"].values.astype(str)
    race_bins = np.where(race == "White", 0, np.where(race == "Asian-Pac-Islander", 1, 2))

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

        # Build lattices (same as v30)
        dims_base = [
            Dimension("age", 8), Dimension("marital", n_marital),
            Dimension("capital_gain", 5), Dimension("capital_loss", 2), Dimension("sex", 2),
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

        # Index arrays
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
            "base": decomp_base, "occ_hours": decomp_occ_hours,
            "edu_hours": decomp_edu_hours, "rel_cg": decomp_rel_cg, "beta": decomp_beta,
        }
        train_indices = {
            "base": jnp.array(train_idx_base), "occ_hours": jnp.array(train_idx_occ_hours),
            "edu_hours": jnp.array(train_idx_edu_hours), "rel_cg": jnp.array(train_idx_rel_cg),
            "beta": jnp.array(train_idx_beta),
        }
        test_indices = {
            "base": jnp.array(test_idx_base), "occ_hours": jnp.array(test_idx_occ_hours),
            "edu_hours": jnp.array(test_idx_edu_hours), "rel_cg": jnp.array(test_idx_rel_cg),
            "beta": jnp.array(test_idx_beta),
        }

        # Get prior scales
        prior_scales = {}
        active = {}
        for key, decomp in decomps.items():
            prior_scales[key] = decomp.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)
            if key in ["occ_hours", "edu_hours", "rel_cg"]:
                active[key] = [n for n in decomp._tensor_parts.keys() if decomp.component_order(n) == 2]
            else:
                active[key] = [n for n in decomp._tensor_parts.keys() if decomp.component_order(n) <= 2]

        # Initialize parameters
        params = {}
        for key, decomp in decomps.items():
            params[key] = {n: jnp.zeros(decomp._tensor_part_shapes[n]) for n in active[key]}

        params["occ_main"] = jnp.zeros(n_occ)
        params["edu_main"] = jnp.zeros(8)
        params["hours_main"] = jnp.zeros(8)
        params["rel_main"] = jnp.zeros(n_relationship)

        X_train_j = jnp.array(X_train)
        X_test_j = jnp.array(X_test)
        y_train_j = jnp.array(y_train)

        scale_mult = 50.0
        learning_rate_boost = 0.1
        n_boost_rounds = 10

        # BOOSTING: fit components iteratively
        accumulated_logits = jnp.zeros(N_train)

        for boost_round in range(n_boost_rounds):
            current_probs = 1 / (1 + jnp.exp(-accumulated_logits))
            residuals = y_train_j - current_probs

            # Fit each intercept component to residuals
            for key in ["base", "occ_hours", "edu_hours", "rel_cg"]:
                decomp = decomps[key]
                idx = train_indices[key]

                params_round = {n: jnp.zeros(decomp._tensor_part_shapes[n]) for n in active[key]}

                def make_loss(decomp, idx, active_names, prior_sc):
                    def loss_fn(p):
                        vals = decomp.lookup_flat(idx, p)[:, 0]
                        mse = jnp.mean((residuals - vals)**2)
                        l2 = sum(0.5 * jnp.sum(p[n]**2) / ((prior_sc.get(n, 1.0) * scale_mult)**2 + 1e-8)
                                for n in active_names)
                        return mse + l2 / N_train
                    return loss_fn

                loss_fn = make_loss(decomp, idx, active[key], prior_scales[key])
                opt = optax.adam(0.02)
                opt_state = opt.init(params_round)

                @jax.jit
                def step(p, opt_state):
                    loss, grads = jax.value_and_grad(loss_fn)(p)
                    updates, opt_state = opt.update(grads, opt_state, p)
                    return optax.apply_updates(p, updates), opt_state, loss

                for _ in range(100):
                    params_round, opt_state, _ = step(params_round, opt_state)

                # Update with shrinkage
                for n in active[key]:
                    params[key][n] = params[key][n] + learning_rate_boost * params_round[n]

            # Update accumulated logits
            intercept = jnp.zeros(N_train)
            for key in ["base", "occ_hours", "edu_hours", "rel_cg"]:
                intercept = intercept + decomps[key].lookup_flat(train_indices[key], params[key])[:, 0]
            accumulated_logits = intercept

            # Fit beta to residuals
            current_probs = 1 / (1 + jnp.exp(-accumulated_logits))
            residuals = y_train_j - current_probs

            beta_round = {n: jnp.zeros(decomps["beta"]._tensor_part_shapes[n]) for n in active["beta"]}

            def loss_beta(p):
                beta_vals = decomps["beta"].lookup_flat(train_indices["beta"], p)
                pred = jnp.sum(X_train_j * beta_vals, axis=-1)
                mse = jnp.mean((residuals - pred)**2)
                l2 = sum(0.5 * jnp.sum(p[n]**2) / ((prior_scales["beta"].get(n, 1.0) * scale_mult)**2 + 1e-8)
                        for n in active["beta"])
                return mse + l2 / N_train

            opt_b = optax.adam(0.01)
            opt_state_b = opt_b.init(beta_round)

            @jax.jit
            def step_b(p, opt_state):
                loss, grads = jax.value_and_grad(loss_beta)(p)
                updates, opt_state = opt_b.update(grads, opt_state, p)
                return optax.apply_updates(p, updates), opt_state, loss

            for _ in range(100):
                beta_round, opt_state_b, _ = step_b(beta_round, opt_state_b)

            for n in active["beta"]:
                params["beta"][n] = params["beta"][n] + learning_rate_boost * beta_round[n]

            # Update accumulated
            beta_vals = decomps["beta"].lookup_flat(train_indices["beta"], params["beta"])
            accumulated_logits = intercept + jnp.sum(X_train_j * beta_vals, axis=-1)

        print(f"  Completed {n_boost_rounds} boosting rounds")

        # Final joint refinement
        print("  Final joint refinement...")

        def loss_joint(params):
            intercept = jnp.zeros(N_train)
            for key in ["base", "occ_hours", "edu_hours", "rel_cg"]:
                intercept = intercept + decomps[key].lookup_flat(train_indices[key], params[key])[:, 0]

            intercept = intercept + params["occ_main"][occupation_bins[train_idx]]
            intercept = intercept + params["edu_main"][edu_bins[train_idx]]
            intercept = intercept + params["hours_main"][hours_bins[train_idx]]
            intercept = intercept + params["rel_main"][relationship_bins[train_idx]]

            beta_vals = decomps["beta"].lookup_flat(train_indices["beta"], params["beta"])
            logits = jnp.sum(X_train_j * beta_vals, axis=-1) + intercept

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_reg = 0.0
            for key in ["base", "occ_hours", "edu_hours", "rel_cg", "beta"]:
                for n, p in params[key].items():
                    scale = prior_scales[key].get(n, 1.0)
                    l2_reg += 0.5 * jnp.sum(p**2) / ((scale * scale_mult)**2 + 1e-8)

            l2_reg += 0.5 * jnp.sum(params["occ_main"]**2)
            l2_reg += 0.5 * jnp.sum(params["edu_main"]**2)
            l2_reg += 0.5 * jnp.sum(params["hours_main"]**2)
            l2_reg += 0.5 * jnp.sum(params["rel_main"]**2)

            return bce + l2_reg / N_train

        opt_joint = optax.adam(0.005)
        opt_state_joint = opt_joint.init(params)

        @jax.jit
        def step_joint(params, opt_state):
            loss, grads = jax.value_and_grad(loss_joint)(params)
            updates, opt_state = opt_joint.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for i in range(2000):
            params, opt_state_joint, loss = step_joint(params, opt_state_joint)

        # Evaluate
        intercept = jnp.zeros(len(y_test))
        for key in ["base", "occ_hours", "edu_hours", "rel_cg"]:
            intercept = intercept + decomps[key].lookup_flat(test_indices[key], params[key])[:, 0]

        intercept = intercept + params["occ_main"][occupation_bins[test_idx]]
        intercept = intercept + params["edu_main"][edu_bins[test_idx]]
        intercept = intercept + params["hours_main"][hours_bins[test_idx]]
        intercept = intercept + params["rel_main"][relationship_bins[test_idx]]

        beta_vals = decomps["beta"].lookup_flat(test_indices["beta"], params["beta"])
        logits = jnp.sum(X_test_j * beta_vals, axis=-1) + intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"  AUC = {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\nOURS #2 (boosting): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"OURS #1 (v30): 0.9154 +/- 0.0025")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_adult_boosting()

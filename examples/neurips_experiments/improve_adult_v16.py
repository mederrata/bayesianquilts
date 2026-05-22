"""Adult v16: Add race(3) and sex(2) to intercept lattice."""
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
    """Transform continuous values to 0-1 based on position within their bin."""
    values = np.asarray(values, dtype=float)
    full_edges = np.concatenate([[values.min()], bin_edges, [values.max()]])
    bins = np.digitize(values, bin_edges)

    lower = full_edges[bins]
    upper = full_edges[bins + 1]
    width = np.where(upper - lower == 0, 1.0, upper - lower)

    return np.clip((values - lower) / width, 0, 1)


def run_adult():
    print("ADULT V16 - Add race(3) and sex(2) to intercept lattice")

    data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    df = data.frame
    y = (df["class"].astype(str) == ">50K").astype(int).values

    # Categorical features for one-hot (excluding race/sex since they're in lattice)
    cats = ["workclass", "education", "marital-status", "occupation", "relationship", "native-country"]

    # Raw continuous values
    age = df["age"].values.astype(float)
    fnlwgt = df["fnlwgt"].values.astype(float)
    edu_num = df["education-num"].values.astype(float)
    capital_gain = df["capital-gain"].values.astype(float)
    capital_loss = df["capital-loss"].values.astype(float)
    hours = df["hours-per-week"].values.astype(float)

    # Bin edges
    age_edges = np.array([25, 35, 45, 55, 65])
    edu_edges = np.array([9, 11, 14])
    hours_edges = np.array([35, 40, 50])

    # Bin indices
    age_bins = np.digitize(age, age_edges)
    edu_bins = np.digitize(edu_num, edu_edges)
    hours_bins = np.digitize(hours, hours_edges)

    # Capital gain bins (4 bins: zero, low, mid, high)
    cg_nonzero = capital_gain[capital_gain > 0]
    cg_edges = np.percentile(cg_nonzero, [50, 90]) if len(cg_nonzero) > 0 else np.array([1, 2])
    capital_gain_bins = np.zeros(len(capital_gain), dtype=int)
    capital_gain_bins[capital_gain > 0] = 1 + np.digitize(capital_gain[capital_gain > 0], cg_edges)

    # Capital loss bins (2 bins: zero vs non-zero - huge signal!)
    capital_loss_bins = (capital_loss > 0).astype(int)

    # Race: white=0, asian=1, other=2
    race = df["race"].values.astype(str)
    race_bins = np.where(race == "White", 0,
                np.where(race == "Asian-Pac-Islander", 1, 2))

    # Sex: male=0, female=1
    sex = df["sex"].values.astype(str)
    sex_bins = np.where(sex == "Male", 0, 1)

    # Within-bin normalized
    age_norm = within_bin_normalize(age, age_edges)
    edu_norm = within_bin_normalize(edu_num, edu_edges)
    hours_norm = within_bin_normalize(hours, hours_edges)
    cg_norm = np.zeros(len(capital_gain))
    if len(cg_nonzero) > 0:
        cg_norm[capital_gain > 0] = within_bin_normalize(capital_gain[capital_gain > 0], cg_edges)
    # Capital loss: normalize within non-zero bin
    cl_nonzero = capital_loss[capital_loss > 0]
    cl_norm = np.zeros(len(capital_loss))
    if len(cl_nonzero) > 0:
        cl_norm[capital_loss > 0] = (capital_loss[capital_loss > 0] - cl_nonzero.min()) / (cl_nonzero.max() - cl_nonzero.min() + 1e-8)

    # Numeric features: raw + normalized
    X_numeric = np.column_stack([
        age, fnlwgt, edu_num, capital_gain, capital_loss, hours,
        age_norm, edu_norm, hours_norm, cg_norm, cl_norm,
    ])

    print("Intercept lattice: age(6) × marital(7) × edu(4) × cg(4) × cl(2) × race(3) × sex(2)")
    print(f"  = {6*7*4*4*2*3*2} cells")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_numeric, y)):
        print(f"Fold {fold_idx + 1}/5:", end=" ")

        X_train_num = StandardScaler().fit_transform(X_numeric[train_idx])
        X_test_num = StandardScaler().fit(X_numeric[train_idx]).transform(X_numeric[test_idx])

        enc = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        X_train_cat = enc.fit_transform(df.iloc[train_idx][cats].astype(str))
        X_test_cat = enc.transform(df.iloc[test_idx][cats].astype(str))

        X_train = np.concatenate([X_train_num, X_train_cat], axis=1)
        X_test = np.concatenate([X_test_num, X_test_cat], axis=1)
        y_train, y_test = y[train_idx], y[test_idx]

        N_train = len(y_train)
        n_features = X_train.shape[1]

        # Label encoder for marital
        le_marital = LabelEncoder().fit(df["marital-status"].astype(str))
        train_marital = le_marital.transform(df.iloc[train_idx]["marital-status"].astype(str))
        test_marital = le_marital.transform(df.iloc[test_idx]["marital-status"].astype(str))
        n_marital = len(le_marital.classes_)

        # Intercept: age × marital × edu × cg × cl × race × sex (7 dims)
        dims_int = [
            Dimension("age", 6),
            Dimension("marital", n_marital),
            Dimension("education", 4),
            Dimension("capital_gain", 4),
            Dimension("capital_loss", 2),
            Dimension("race", 3),
            Dimension("sex", 2),
        ]

        # Beta: marital × hours
        dims_beta = [Dimension("marital", n_marital), Dimension("hours", 4)]

        interactions_int = Interactions(dimensions=dims_int)
        interactions_beta = Interactions(dimensions=dims_beta)
        decomp_int = Decomposed(interactions=interactions_int, param_shape=[1], name="intercept")
        decomp_beta = Decomposed(interactions=interactions_beta, param_shape=[n_features], name="beta")

        prior_scales_int = decomp_int.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)
        prior_scales_beta = decomp_beta.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)

        train_idx_int = jnp.stack([
            jnp.array(age_bins[train_idx]),
            jnp.array(train_marital),
            jnp.array(edu_bins[train_idx]),
            jnp.array(capital_gain_bins[train_idx]),
            jnp.array(capital_loss_bins[train_idx]),
            jnp.array(race_bins[train_idx]),
            jnp.array(sex_bins[train_idx]),
        ], axis=-1)
        test_idx_int = jnp.stack([
            jnp.array(age_bins[test_idx]),
            jnp.array(test_marital),
            jnp.array(edu_bins[test_idx]),
            jnp.array(capital_gain_bins[test_idx]),
            jnp.array(capital_loss_bins[test_idx]),
            jnp.array(race_bins[test_idx]),
            jnp.array(sex_bins[test_idx]),
        ], axis=-1)
        train_idx_beta = jnp.stack([jnp.array(train_marital), jnp.array(hours_bins[train_idx])], axis=-1)
        test_idx_beta = jnp.stack([jnp.array(test_marital), jnp.array(hours_bins[test_idx])], axis=-1)

        X_train_j, X_test_j, y_train_j = jnp.array(X_train), jnp.array(X_test), jnp.array(y_train)

        # Order 3 for intercept (6 dims), order 2 for beta
        active_int = [n for n in decomp_int._tensor_parts.keys() if decomp_int.component_order(n) <= 3]
        active_beta = [n for n in decomp_beta._tensor_parts.keys() if decomp_beta.component_order(n) <= 2]

        if fold_idx == 0:
            print(f"\n    Intercept: {len(active_int)} components (order ≤ 3)")
            print(f"    n_features: {n_features}, N_train: {N_train}")

        params = {"intercept": {n: jnp.zeros(decomp_int._tensor_part_shapes[n]) for n in active_int},
                  "beta": {n: jnp.zeros(decomp_beta._tensor_part_shapes[n]) for n in active_beta}}

        scale_multiplier = 50.0

        def loss_fn(params):
            int_vals = decomp_int.lookup_flat(train_idx_int, params["intercept"])
            beta_vals = decomp_beta.lookup_flat(train_idx_beta, params["beta"])
            logits = jnp.sum(X_train_j * beta_vals, axis=-1) + int_vals[:, 0]
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)
            l2_int = sum(0.5 * jnp.sum(p ** 2) / ((prior_scales_int.get(n, 1.0) * scale_multiplier) ** 2 + 1e-8) for n, p in params["intercept"].items())
            l2_beta = sum(0.5 * jnp.sum(p ** 2) / ((prior_scales_beta.get(n, 1.0) * scale_multiplier) ** 2 + 1e-8) for n, p in params["beta"].items())
            return bce + (l2_int + l2_beta) / N_train

        opt = optax.adam(0.01)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for i in range(5001):
            params, opt_state, loss = step(params, opt_state)

        int_vals = decomp_int.lookup_flat(test_idx_int, params["intercept"])
        beta_vals = decomp_beta.lookup_flat(test_idx_beta, params["beta"])
        probs = 1 / (1 + jnp.exp(-(jnp.sum(X_test_j * beta_vals, axis=-1) + int_vals[:, 0])))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"AUC = {auc:.4f}")

    print(f"\nOURS: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print("Baselines: LR=0.907, LGBM=0.929, EBM=0.930")
    return np.mean(aucs), np.std(aucs)

if __name__ == "__main__":
    run_adult()

"""Automatic interaction detection like EBM's FAST algorithm.

Uses mutual information and conditional dependence tests to find
important pairwise interactions, then incorporates them into the model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import fetch_openml
from itertools import combinations
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def compute_interaction_scores(X_binned, y, feature_names, n_bins=10):
    """Compute interaction importance scores for all pairs.

    Uses residual mutual information: I(X1, X2; Y) - I(X1; Y) - I(X2; Y)
    Higher values indicate stronger interaction effects.
    """
    n_features = X_binned.shape[1]
    scores = {}

    # Compute main effect MI for each feature
    main_mi = {}
    for i, name in enumerate(feature_names):
        main_mi[name] = mutual_info_classif(X_binned[:, i:i+1], y, discrete_features=True)[0]

    # Compute interaction MI for each pair
    for (i, name_i), (j, name_j) in combinations(enumerate(feature_names), 2):
        # Joint feature: bin(i) * n_bins + bin(j)
        joint = X_binned[:, i] * n_bins + X_binned[:, j]
        joint_mi = mutual_info_classif(joint.reshape(-1, 1), y, discrete_features=True)[0]

        # Interaction score = joint MI - sum of main MIs
        interaction_mi = joint_mi - main_mi[name_i] - main_mi[name_j]
        scores[(name_i, name_j)] = max(0, interaction_mi)  # Clip negative values

    return scores, main_mi


def select_top_interactions(scores, k=5):
    """Select top-k interactions by score."""
    sorted_pairs = sorted(scores.items(), key=lambda x: -x[1])
    return [pair for pair, score in sorted_pairs[:k]]


def run_german_auto_interactions():
    """German Credit with automatic interaction detection."""
    print("\n" + "="*60)
    print("GERMAN CREDIT - OURS #2 WITH AUTO INTERACTION DETECTION")
    print("="*60)

    data = fetch_openml(data_id=31, as_frame=True, parser="auto")
    df = data.frame

    # All features
    cat_cols = ["checking_status", "credit_history", "purpose", "savings_status",
                "employment", "personal_status", "other_parties", "property_magnitude",
                "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"]
    num_cols = ["duration", "credit_amount", "installment_commitment", "residence_since",
                "age", "existing_credits", "num_dependents"]

    y = (df["class"].astype(str) == "good").astype(int).values
    N = len(y)
    print(f"  N = {N}, pos_rate = {y.mean():.3f}")

    # Encode all features for interaction detection
    all_features = []
    feature_names = []

    for col in cat_cols:
        le = LabelEncoder()
        all_features.append(le.fit_transform(df[col].astype(str)))
        feature_names.append(col)

    for col in num_cols:
        vals = df[col].values.astype(float)
        bins = np.percentile(vals, np.linspace(0, 100, 6)[1:-1])
        all_features.append(np.digitize(vals, bins))
        feature_names.append(col)

    X_binned = np.stack(all_features, axis=1)

    # Detect important interactions
    print("\n  Detecting important interactions...")
    interaction_scores, main_mi = compute_interaction_scores(X_binned, y, feature_names)

    print("\n  Top main effects:")
    sorted_main = sorted(main_mi.items(), key=lambda x: -x[1])[:5]
    for name, mi in sorted_main:
        print(f"    {name}: {mi:.4f}")

    print("\n  Top interactions:")
    top_interactions = select_top_interactions(interaction_scores, k=5)
    for (f1, f2), score in [(pair, interaction_scores[pair]) for pair in top_interactions]:
        print(f"    {f1} × {f2}: {score:.4f}")

    # Use detected interactions for model
    # Build lattice with top 2 main effects + add interaction terms
    top_main = [name for name, _ in sorted_main[:2]]
    print(f"\n  Using main effects: {top_main}")
    print(f"  Using interactions: {top_interactions[:3]}")

    # One-hot encode categoricals for regression
    ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    X_cat_ohe = ohe.fit_transform(df[cat_cols].astype(str))
    X_num = df[num_cols].values.astype(np.float32)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_binned, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        scaler = StandardScaler()
        X_num_train = scaler.fit_transform(X_num[train_idx])
        X_num_test = scaler.transform(X_num[test_idx])

        X_train = np.hstack([X_cat_ohe[train_idx], X_num_train])
        X_test = np.hstack([X_cat_ohe[test_idx], X_num_test])

        y_train, y_test = y[train_idx], y[test_idx]
        N_train = len(y_train)
        n_features = X_train.shape[1]

        # Build main lattice from top 2 main effects
        dims_main = []
        main_indices_train = []
        main_indices_test = []

        for fname in top_main:
            idx = feature_names.index(fname)
            n_levels = len(np.unique(X_binned[:, idx]))
            dims_main.append(Dimension(fname, n_levels))
            main_indices_train.append(X_binned[train_idx, idx])
            main_indices_test.append(X_binned[test_idx, idx])

        decomp_main = Decomposed(
            interactions=Interactions(dimensions=dims_main),
            param_shape=[1], name="main"
        )

        train_idx_main = jnp.stack([jnp.array(idx) for idx in main_indices_train], axis=-1)
        test_idx_main = jnp.stack([jnp.array(idx) for idx in main_indices_test], axis=-1)

        # Build interaction lattices from detected pairs
        interaction_decomps = []
        interaction_train_indices = []
        interaction_test_indices = []

        for f1, f2 in top_interactions[:3]:
            idx1 = feature_names.index(f1)
            idx2 = feature_names.index(f2)
            n1 = len(np.unique(X_binned[:, idx1]))
            n2 = len(np.unique(X_binned[:, idx2]))

            dims_int = [Dimension(f1, n1), Dimension(f2, n2)]
            decomp_int = Decomposed(
                interactions=Interactions(dimensions=dims_int),
                param_shape=[1], name=f"{f1}_{f2}"
            )
            interaction_decomps.append(decomp_int)

            train_int = jnp.stack([jnp.array(X_binned[train_idx, idx1]),
                                   jnp.array(X_binned[train_idx, idx2])], axis=-1)
            test_int = jnp.stack([jnp.array(X_binned[test_idx, idx1]),
                                  jnp.array(X_binned[test_idx, idx2])], axis=-1)
            interaction_train_indices.append(train_int)
            interaction_test_indices.append(test_int)

        # Theory-based regularization
        prior_scales_main = decomp_main.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True
        )
        prior_scales_int = [
            d.generalization_preserving_scales(noise_scale=1.0, total_n=N_train, c=0.5, per_component=True)
            for d in interaction_decomps
        ]

        # Initialize parameters
        active_main = [n for n in decomp_main._tensor_parts.keys() if decomp_main.component_order(n) <= 1]
        active_int = [
            [n for n in d._tensor_parts.keys() if d.component_order(n) == 2]
            for d in interaction_decomps
        ]

        params = {
            "main": {n: jnp.zeros(decomp_main._tensor_part_shapes[n]) for n in active_main},
            "beta": jnp.zeros(n_features),
        }
        for i, d in enumerate(interaction_decomps):
            params[f"int_{i}"] = {n: jnp.zeros(d._tensor_part_shapes[n]) for n in active_int[i]}

        X_train_j = jnp.array(X_train)
        X_test_j = jnp.array(X_test)
        y_train_j = jnp.array(y_train)

        scale_mult = 50.0

        def loss_fn(params):
            # Main effects
            main_vals = decomp_main.lookup_flat(train_idx_main, params["main"])[:, 0]

            # Interaction effects
            int_vals = jnp.zeros(N_train)
            for i, d in enumerate(interaction_decomps):
                int_vals = int_vals + d.lookup_flat(interaction_train_indices[i], params[f"int_{i}"])[:, 0]

            # Linear
            linear = jnp.sum(X_train_j * params["beta"], axis=-1)

            logits = main_vals + int_vals + linear
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # L2 regularization with theory-based scales
            l2_main = sum(0.5 * jnp.sum(params["main"][n]**2) / ((prior_scales_main.get(n, 1.0) * scale_mult)**2 + 1e-8)
                         for n in active_main)

            l2_int = 0.0
            for i, d in enumerate(interaction_decomps):
                for n in active_int[i]:
                    l2_int += 0.5 * jnp.sum(params[f"int_{i}"][n]**2) / ((prior_scales_int[i].get(n, 1.0) * scale_mult)**2 + 1e-8)

            tau_beta = 0.1
            l2_beta = 0.5 * jnp.sum(params["beta"]**2) / (tau_beta**2)

            return bce + (l2_main + l2_int + l2_beta) / N_train

        opt = optax.adam(0.01)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for i in range(3000):
            params, opt_state, loss = step(params, opt_state)

        # Evaluate
        main_vals = decomp_main.lookup_flat(test_idx_main, params["main"])[:, 0]
        int_vals = jnp.zeros(len(y_test))
        for i, d in enumerate(interaction_decomps):
            int_vals = int_vals + d.lookup_flat(interaction_test_indices[i], params[f"int_{i}"])[:, 0]
        linear = jnp.sum(X_test_j * params["beta"], axis=-1)

        logits = main_vals + int_vals + linear
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS #2 (auto interactions): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  OURS #1 (v7): 0.7883 +/- 0.0211")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_german_auto_interactions()

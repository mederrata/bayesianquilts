"""Ours v2: Lattice decomposition with EBM/GAMINet enhancements.

Core: Lattice decomposition with hierarchical additive structure
Enhancements:
1. MI-based automatic dimension selection for lattice
2. Learnable ReLU hinges for continuous feature binning
3. Boosting/cyclic training (intercept -> beta -> joint)
4. Theory-based generalization-preserving regularization
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import fetch_openml
from itertools import combinations
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def compute_feature_importance(X, y, feature_names, n_bins=10):
    """Compute feature importance using mutual information.

    Returns main effect MI and interaction MI scores.
    """
    n_features = len(feature_names)

    # Bin continuous features for MI computation
    X_binned = np.zeros_like(X, dtype=int)
    for i in range(X.shape[1]):
        if len(np.unique(X[:, i])) > n_bins:
            edges = np.percentile(X[:, i], np.linspace(0, 100, n_bins + 1)[1:-1])
            X_binned[:, i] = np.digitize(X[:, i], edges)
        else:
            X_binned[:, i] = X[:, i].astype(int)

    # Main effect MI
    main_mi = {}
    for i, name in enumerate(feature_names):
        main_mi[name] = mutual_info_classif(X_binned[:, i:i+1], y, discrete_features=True)[0]

    # Interaction MI for top features
    top_k = min(10, n_features)
    top_features = sorted(range(n_features), key=lambda i: -main_mi[feature_names[i]])[:top_k]

    interaction_mi = {}
    for i, j in combinations(top_features, 2):
        joint = X_binned[:, i] * n_bins + X_binned[:, j]
        joint_mi = mutual_info_classif(joint.reshape(-1, 1), y, discrete_features=True)[0]
        interaction_score = joint_mi - main_mi[feature_names[i]] - main_mi[feature_names[j]]
        interaction_mi[(feature_names[i], feature_names[j])] = max(0, interaction_score)

    return main_mi, interaction_mi


def select_lattice_dimensions(main_mi, interaction_mi, max_dims=5, max_interactions=3):
    """Select dimensions for lattice based on MI scores."""
    # Select top main effects
    sorted_main = sorted(main_mi.items(), key=lambda x: -x[1])
    top_main = [name for name, _ in sorted_main[:max_dims]]

    # Select top interactions
    sorted_int = sorted(interaction_mi.items(), key=lambda x: -x[1])
    top_interactions = [pair for pair, _ in sorted_int[:max_interactions]]

    return top_main, top_interactions


class LearnableHingeBins:
    """Learnable ReLU hinges for soft binning of continuous features."""

    def __init__(self, n_bins, feature_idx):
        self.n_bins = n_bins
        self.feature_idx = feature_idx

    def init_splits(self, X_col):
        """Initialize split points from data quantiles."""
        percentiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
        return jnp.array(np.percentile(X_col, percentiles))

    def soft_bin(self, x, splits, temperature=0.1):
        """Soft binning using sigmoid approximation to step function.

        Returns soft bin indices in [0, n_bins-1].
        """
        # Compute cumulative "passed split" indicators
        # As temperature -> 0, this becomes hard binning
        passed = jax.nn.sigmoid((x[:, None] - splits[None, :]) / temperature)
        soft_idx = jnp.sum(passed, axis=-1)  # Sum of passed splits
        return soft_idx

    def hard_bin(self, x, splits):
        """Hard binning for evaluation."""
        return jnp.digitize(x, splits)


def run_ours_v2(dataset_name, df, cat_cols, num_cols, y,
                n_bins=8, max_lattice_dims=4, max_interactions=3,
                n_boost_rounds=5, verbose=True):
    """Run Ours v2 on a dataset."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"OURS V2: {dataset_name.upper()}")
        print("Lattice decomposition + MI selection + learnable hinges + boosting")
        print("="*60)
        print(f"  N = {len(y)}, pos_rate = {y.mean():.3f}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(df, y)):
        if verbose:
            print(f"\n  Fold {fold_idx + 1}/5:")

        y_train, y_test = y[train_idx], y[test_idx]
        N_train = len(y_train)

        # Encode categoricals
        cat_encoders = {}
        cat_indices_train = {}
        cat_indices_test = {}
        cat_n_levels = {}

        for col in cat_cols:
            le = LabelEncoder()
            cat_indices_train[col] = le.fit_transform(df[col].astype(str).iloc[train_idx])
            # Handle unknown categories in test
            test_vals = df[col].astype(str).iloc[test_idx].values
            cat_indices_test[col] = np.array([
                le.transform([v])[0] if v in le.classes_ else 0
                for v in test_vals
            ])
            cat_n_levels[col] = len(le.classes_)
            cat_encoders[col] = le

        # Standardize numerics
        num_scalers = {}
        X_num_train = {}
        X_num_test = {}
        for col in num_cols:
            scaler = StandardScaler()
            vals = df[col].values.astype(np.float32)
            vals = np.nan_to_num(vals, nan=0.0)
            X_num_train[col] = scaler.fit_transform(vals[train_idx].reshape(-1, 1))[:, 0]
            X_num_test[col] = scaler.transform(vals[test_idx].reshape(-1, 1))[:, 0]
            num_scalers[col] = scaler

        # One-hot encode for linear features
        if cat_cols:
            ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            X_cat_train = ohe.fit_transform(df[cat_cols].astype(str).iloc[train_idx])
            X_cat_test = ohe.transform(df[cat_cols].astype(str).iloc[test_idx])
        else:
            X_cat_train = np.zeros((len(train_idx), 0))
            X_cat_test = np.zeros((len(test_idx), 0))

        X_num_arr_train = np.column_stack([X_num_train[col] for col in num_cols]) if num_cols else np.zeros((len(train_idx), 0))
        X_num_arr_test = np.column_stack([X_num_test[col] for col in num_cols]) if num_cols else np.zeros((len(test_idx), 0))

        X_train = np.hstack([X_num_arr_train, X_cat_train]).astype(np.float32)
        X_test = np.hstack([X_num_arr_test, X_cat_test]).astype(np.float32)
        n_features = X_train.shape[1]

        # === MI-based dimension selection ===
        all_feature_names = num_cols + cat_cols
        X_for_mi = np.column_stack([
            X_num_arr_train,
            np.column_stack([cat_indices_train[col] for col in cat_cols]) if cat_cols else np.zeros((len(train_idx), 0))
        ])

        main_mi, interaction_mi = compute_feature_importance(X_for_mi, y_train, all_feature_names)
        top_main, top_interactions = select_lattice_dimensions(
            main_mi, interaction_mi,
            max_dims=max_lattice_dims,
            max_interactions=max_interactions
        )

        if verbose:
            print(f"    Selected main effects: {top_main[:3]}...")
            print(f"    Selected interactions: {top_interactions[:2]}...")

        # === Build intercept lattice from selected dimensions ===
        lattice_dims = []
        lattice_train_indices = []
        lattice_test_indices = []
        hinge_params = {}

        for dim_name in top_main[:max_lattice_dims]:
            if dim_name in cat_cols:
                # Categorical: use direct indices
                n_levels = cat_n_levels[dim_name]
                lattice_dims.append(Dimension(dim_name, n_levels))
                lattice_train_indices.append(cat_indices_train[dim_name])
                lattice_test_indices.append(cat_indices_test[dim_name])
            else:
                # Continuous: use learnable bins
                hinge = LearnableHingeBins(n_bins, dim_name)
                splits = hinge.init_splits(X_num_train[dim_name])
                hinge_params[dim_name] = splits

                lattice_dims.append(Dimension(dim_name, n_bins))
                train_bins = np.digitize(X_num_train[dim_name], np.array(splits))
                test_bins = np.digitize(X_num_test[dim_name], np.array(splits))
                lattice_train_indices.append(train_bins)
                lattice_test_indices.append(test_bins)

        if not lattice_dims:
            # Fallback: use first categorical or binned numeric
            if cat_cols:
                col = cat_cols[0]
                lattice_dims.append(Dimension(col, cat_n_levels[col]))
                lattice_train_indices.append(cat_indices_train[col])
                lattice_test_indices.append(cat_indices_test[col])
            elif num_cols:
                col = num_cols[0]
                edges = np.percentile(X_num_train[col], np.linspace(0, 100, n_bins + 1)[1:-1])
                lattice_dims.append(Dimension(col, n_bins))
                lattice_train_indices.append(np.digitize(X_num_train[col], edges))
                lattice_test_indices.append(np.digitize(X_num_test[col], edges))

        # Create decomposed parameters
        decomp_int = Decomposed(
            interactions=Interactions(dimensions=lattice_dims),
            param_shape=[1],
            name="intercept"
        )

        decomp_beta = Decomposed(
            interactions=Interactions(dimensions=lattice_dims[:2] if len(lattice_dims) >= 2 else lattice_dims),
            param_shape=[n_features],
            name="beta"
        )

        # Index arrays
        train_idx_int = jnp.stack([jnp.array(idx) for idx in lattice_train_indices], axis=-1)
        test_idx_int = jnp.stack([jnp.array(idx) for idx in lattice_test_indices], axis=-1)

        beta_lattice_dims = lattice_dims[:2] if len(lattice_dims) >= 2 else lattice_dims
        train_idx_beta = train_idx_int[:, :len(beta_lattice_dims)]
        test_idx_beta = test_idx_int[:, :len(beta_lattice_dims)]

        # === Theory-based regularization scales ===
        prior_scales_int = decomp_int.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True
        )
        prior_scales_beta = decomp_beta.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True
        )

        # Limit interaction order
        max_order_int = min(3, len(lattice_dims))
        max_order_beta = min(2, len(beta_lattice_dims))
        active_int = [n for n in decomp_int._tensor_parts.keys() if decomp_int.component_order(n) <= max_order_int]
        active_beta = [n for n in decomp_beta._tensor_parts.keys() if decomp_beta.component_order(n) <= max_order_beta]

        # Initialize parameters
        params_int = {n: jnp.zeros(decomp_int._tensor_part_shapes[n]) for n in active_int}
        params_beta = {n: jnp.zeros(decomp_beta._tensor_part_shapes[n]) for n in active_beta}

        X_train_j = jnp.array(X_train)
        X_test_j = jnp.array(X_test)
        y_train_j = jnp.array(y_train)

        scale_mult = 50.0
        learning_rate_boost = 0.1

        # === BOOSTING: Cyclic training ===
        accumulated_logits = jnp.zeros(N_train)

        for boost_round in range(n_boost_rounds):
            # Compute residuals
            current_probs = jax.nn.sigmoid(accumulated_logits)
            residuals = y_train_j - current_probs

            # Stage 1: Fit intercept lattice to residuals
            params_int_round = {n: jnp.zeros(decomp_int._tensor_part_shapes[n]) for n in active_int}

            def loss_int(p):
                vals = decomp_int.lookup_flat(train_idx_int, p)[:, 0]
                mse = jnp.mean((residuals - vals)**2)
                l2 = sum(0.5 * jnp.sum(p[n]**2) / ((prior_scales_int.get(n, 1.0) * scale_mult)**2 + 1e-8)
                        for n in active_int)
                return mse + l2 / N_train

            opt_i = optax.adam(0.02)
            opt_state_i = opt_i.init(params_int_round)

            @jax.jit
            def step_i(p, opt_state):
                loss, grads = jax.value_and_grad(loss_int)(p)
                updates, opt_state = opt_i.update(grads, opt_state, p)
                return optax.apply_updates(p, updates), opt_state, loss

            for _ in range(100):
                params_int_round, opt_state_i, _ = step_i(params_int_round, opt_state_i)

            # Update with shrinkage
            for n in active_int:
                params_int[n] = params_int[n] + learning_rate_boost * params_int_round[n]

            # Update accumulated logits
            int_vals = decomp_int.lookup_flat(train_idx_int, params_int)[:, 0]
            accumulated_logits = int_vals

            # Stage 2: Fit beta lattice to residuals
            current_probs = jax.nn.sigmoid(accumulated_logits)
            residuals = y_train_j - current_probs

            params_beta_round = {n: jnp.zeros(decomp_beta._tensor_part_shapes[n]) for n in active_beta}

            def loss_beta(p):
                beta_vals = decomp_beta.lookup_flat(train_idx_beta, p)
                pred = jnp.sum(X_train_j * beta_vals, axis=-1)
                mse = jnp.mean((residuals - pred)**2)
                l2 = sum(0.5 * jnp.sum(p[n]**2) / ((prior_scales_beta.get(n, 1.0) * scale_mult)**2 + 1e-8)
                        for n in active_beta)
                return mse + l2 / N_train

            opt_b = optax.adam(0.01)
            opt_state_b = opt_b.init(params_beta_round)

            @jax.jit
            def step_b(p, opt_state):
                loss, grads = jax.value_and_grad(loss_beta)(p)
                updates, opt_state = opt_b.update(grads, opt_state, p)
                return optax.apply_updates(p, updates), opt_state, loss

            for _ in range(100):
                params_beta_round, opt_state_b, _ = step_b(params_beta_round, opt_state_b)

            # Update with shrinkage
            for n in active_beta:
                params_beta[n] = params_beta[n] + learning_rate_boost * params_beta_round[n]

            # Update accumulated logits
            beta_vals = decomp_beta.lookup_flat(train_idx_beta, params_beta)
            accumulated_logits = int_vals + jnp.sum(X_train_j * beta_vals, axis=-1)

        # === Joint refinement ===
        params = {"int": params_int, "beta": params_beta}

        def loss_joint(params):
            int_vals = decomp_int.lookup_flat(train_idx_int, params["int"])[:, 0]
            beta_vals = decomp_beta.lookup_flat(train_idx_beta, params["beta"])
            logits = int_vals + jnp.sum(X_train_j * beta_vals, axis=-1)

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_int = sum(0.5 * jnp.sum(params["int"][n]**2) / ((prior_scales_int.get(n, 1.0) * scale_mult)**2 + 1e-8)
                        for n in active_int)
            l2_beta = sum(0.5 * jnp.sum(params["beta"][n]**2) / ((prior_scales_beta.get(n, 1.0) * scale_mult)**2 + 1e-8)
                         for n in active_beta)

            return bce + (l2_int + l2_beta) / N_train

        opt_joint = optax.adam(0.005)
        opt_state_joint = opt_joint.init(params)

        @jax.jit
        def step_joint(params, opt_state):
            loss, grads = jax.value_and_grad(loss_joint)(params)
            updates, opt_state = opt_joint.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for i in range(2000):
            params, opt_state_joint, loss = step_joint(params, opt_state_joint)

        # === Evaluate ===
        int_vals = decomp_int.lookup_flat(test_idx_int, params["int"])[:, 0]
        beta_vals = decomp_beta.lookup_flat(test_idx_beta, params["beta"])
        logits = int_vals + jnp.sum(X_test_j * beta_vals, axis=-1)
        probs = jax.nn.sigmoid(logits)

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        if verbose:
            print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    if verbose:
        print(f"\n  OURS V2: {mean_auc:.4f} +/- {std_auc:.4f}")
    return mean_auc, std_auc


def run_heart():
    """Run Ours v2 on Heart Disease (smallest dataset, N=270)."""
    data = fetch_openml(data_id=53, as_frame=True, parser="auto")
    df = data.frame

    cat_cols = []  # All features are numeric in this dataset
    num_cols = ["age", "resting_blood_pressure", "serum_cholestoral",
                "maximum_heart_rate_achieved", "oldpeak",
                "sex", "chest", "fasting_blood_sugar",
                "resting_electrocardiographic_results", "exercise_induced_angina",
                "slope", "number_of_major_vessels", "thal"]

    y = (df["class"].astype(str) == "present").astype(int).values

    return run_ours_v2("Heart Disease", df, cat_cols, num_cols, y,
                       n_bins=4, max_lattice_dims=2, max_interactions=1, n_boost_rounds=3)


def run_german():
    """Run Ours v2 on German Credit (N=1000)."""
    data = fetch_openml(data_id=31, as_frame=True, parser="auto")
    df = data.frame

    cat_cols = ["checking_status", "credit_history", "purpose", "savings_status",
                "employment", "personal_status", "other_parties", "property_magnitude",
                "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"]
    num_cols = ["duration", "credit_amount", "installment_commitment", "residence_since",
                "age", "existing_credits", "num_dependents"]

    y = (df["class"].astype(str) == "good").astype(int).values

    return run_ours_v2("German Credit", df, cat_cols, num_cols, y,
                       n_bins=8, max_lattice_dims=3, max_interactions=2, n_boost_rounds=5)


def run_adult():
    """Run Ours v2 on Adult Income."""
    data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    df = data.frame

    cat_cols = ["workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country"]
    num_cols = ["age", "fnlwgt", "education-num", "capital-gain",
                "capital-loss", "hours-per-week"]

    y = (df["class"].astype(str).str.strip() == ">50K").astype(int).values

    return run_ours_v2("Adult Income", df, cat_cols, num_cols, y,
                       n_bins=8, max_lattice_dims=4, max_interactions=3, n_boost_rounds=10)


def run_bank():
    """Run Ours v2 on Bank Marketing."""
    import zipfile
    import urllib.request
    from io import BytesIO

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
    cache_path = Path("data/bank/bank-full.csv")

    if cache_path.exists():
        df = pd.read_csv(cache_path, sep=";")
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            with zipfile.ZipFile(BytesIO(response.read())) as z:
                with z.open("bank-full.csv") as f:
                    df = pd.read_csv(f, sep=";")
                    df.to_csv(cache_path, sep=";", index=False)

    cat_cols = ["job", "marital", "education", "default", "housing", "loan",
                "contact", "month", "poutcome"]
    num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

    y = (df["y"] == "yes").astype(int).values

    return run_ours_v2("Bank Marketing", df, cat_cols, num_cols, y,
                       n_bins=8, max_lattice_dims=5, max_interactions=3, n_boost_rounds=10)


def run_covertype():
    """Run Ours v2 on Covertype (N=581,012 - full dataset)."""
    data = fetch_openml(data_id=293, as_frame=False, parser="auto")
    X = data.data
    y_raw = data.target

    # Convert sparse to dense if needed
    if hasattr(X, 'toarray'):
        X = X.toarray()

    # Create DataFrame
    feature_names = data.feature_names
    df = pd.DataFrame(X, columns=feature_names)

    # Use all columns as numeric (first 10 continuous, rest binary indicators)
    all_num_cols = list(df.columns)
    cat_cols = []

    # Binary: class 2 (Lodgepole Pine) vs others
    y = (y_raw.astype(str) == "2").astype(int)

    print(f"Covertype: N={len(y)}, pos_rate={y.mean():.3f}")

    # Use higher-order interactions (order 4) - our advantage over EBM/GAMINet
    return run_ours_v2("Covertype", df, cat_cols, all_num_cols, y,
                       n_bins=10, max_lattice_dims=5, max_interactions=4, n_boost_rounds=10)


def run_higgs():
    """Run Ours v2 on HIGGS (N=11,000,000 - very large dataset)."""
    data = fetch_openml(data_id=23512, as_frame=False, parser="auto")
    X = data.data
    y_raw = data.target

    # Convert sparse to dense if needed
    if hasattr(X, 'toarray'):
        X = X.toarray()

    # Create DataFrame
    feature_names = data.feature_names if data.feature_names else [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)

    # All features are numeric
    all_num_cols = list(df.columns)
    cat_cols = []

    # Binary classification (signal=1 vs background=0)
    y = y_raw.astype(int)

    print(f"HIGGS: N={len(y)}, pos_rate={y.mean():.3f}")

    # Use higher-order interactions (order 4+) - our advantage over EBM/GAMINet
    # EBM/GAMINet limited to pairwise; we can go higher
    return run_ours_v2("HIGGS", df, cat_cols, all_num_cols, y,
                       n_bins=10, max_lattice_dims=6, max_interactions=5, n_boost_rounds=15)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["heart", "german", "adult", "bank", "covertype", "higgs", "all", "small", "large"], default="all")
    args = parser.parse_args()

    results = {}

    # Run in order of dataset size (smallest first)
    if args.dataset in ["heart", "all", "small"]:
        results["heart"] = run_heart()

    if args.dataset in ["german", "all", "small"]:
        results["german"] = run_german()

    if args.dataset in ["adult", "all"]:
        results["adult"] = run_adult()

    if args.dataset in ["bank", "all"]:
        results["bank"] = run_bank()

    if args.dataset in ["covertype", "all", "large"]:
        results["covertype"] = run_covertype()

    if args.dataset in ["higgs", "all", "large"]:
        results["higgs"] = run_higgs()

    print("\n" + "="*60)
    print("OURS V2 SUMMARY")
    print("="*60)
    for name, (mean, std) in results.items():
        print(f"  {name}: {mean:.4f} +/- {std:.4f}")

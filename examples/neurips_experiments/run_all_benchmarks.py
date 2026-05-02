"""Comprehensive UCI benchmarks with proper preprocessing and model persistence."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
import jax.numpy as jnp

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    HAS_EBM = True
except ImportError:
    HAS_EBM = False

from sklearn.neural_network import MLPClassifier

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def lookup_params(decomp, interaction_indices, params):
    """Lookup parameters at interaction indices."""
    return decomp.lookup_flat(interaction_indices, params)


def fit_logistic_model(data, decomp, max_order=2, prior_scales=None, sparse=True, l1_weight=0.01, n_steps=5000):
    """Fit logistic regression with decomposed parameters."""
    import jax
    import jax.numpy as jnp
    import optax

    active_components = [
        name for name in decomp._tensor_parts.keys()
        if decomp.component_order(name) <= max_order
    ]

    params = {
        name: jnp.zeros(decomp._tensor_part_shapes[name])
        for name in active_components
    }
    params["_intercept"] = jnp.zeros(1)

    dim_names = [d.name for d in decomp._interactions._dimensions]
    interaction_indices = jnp.stack(
        [jnp.array(data[name]) for name in dim_names],
        axis=-1
    )

    X = jnp.array(data["X"])
    y = jnp.array(data["y"])
    N = len(y)

    def loss_fn(params):
        intercept = params.get("_intercept", jnp.zeros(1))
        model_params = {k: v for k, v in params.items() if k != "_intercept"}

        beta = lookup_params(decomp, interaction_indices, model_params)
        logits = jnp.sum(X * beta, axis=-1) + intercept[0]

        bce = jnp.mean(jnp.logaddexp(0, logits) - y * logits)

        l2_reg = 0.0
        for name, param in model_params.items():
            scale = prior_scales.get(name, 1.0) if prior_scales else 1.0
            l2_reg += 0.5 * jnp.sum(param ** 2) / ((scale * 10) ** 2)

        l1_reg = 0.0
        if sparse:
            for name, param in model_params.items():
                order = decomp.component_order(name)
                if order > 0:
                    l1_reg += l1_weight * order * jnp.sum(jnp.abs(param))

        return bce + l2_reg / N + l1_reg / N

    opt = optax.adam(0.01)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, state):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, new_state = opt.update(grads, state, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_state, loss

    for i in range(n_steps):
        params, opt_state, loss = step(params, opt_state)
        if i % 1000 == 0:
            print(f"    Step {i}: loss = {loss:.4f}", flush=True)

    return params


def evaluate_model(data, decomp, params, max_order=2):
    """Evaluate model on test data."""
    import jax.numpy as jnp

    X = jnp.array(data["X"])
    y = np.array(data["y"])

    dim_names = [d.name for d in decomp._interactions._dimensions]
    interaction_indices = jnp.stack(
        [jnp.array(data[name]) for name in dim_names],
        axis=-1
    )

    intercept = params.get("_intercept", jnp.zeros(1))
    model_params = {k: v for k, v in params.items() if k != "_intercept"}

    beta = lookup_params(decomp, interaction_indices, model_params)
    logits = jnp.sum(X * beta, axis=-1) + intercept[0]
    probs = 1 / (1 + jnp.exp(-logits))

    auc = roc_auc_score(y, np.array(probs))
    return {"auc": auc, "probs": np.array(probs)}


def run_baselines(X, y, cat_mask, n_splits=5):
    """Run baseline models with proper one-hot encoding."""
    results = {}

    cat_cols = np.where(cat_mask)[0].tolist()
    num_cols = np.where(~cat_mask)[0].tolist()

    if cat_cols:
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols)
        ])
    else:
        preprocessor = StandardScaler()

    models = {
        "LR-L1": LogisticRegression(max_iter=1000, C=1.0, penalty='l1', solver='saga'),
        "RF": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True)
    }
    if HAS_XGB:
        models["XGB"] = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric="logloss")
    if HAS_LGB:
        models["LGBM"] = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
    if HAS_EBM:
        models["EBM"] = ExplainableBoostingClassifier(random_state=42, n_jobs=1)

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for name, model in models.items():
        aucs = []
        for train_idx, test_idx in kfold.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if cat_cols:
                pipe = Pipeline([("prep", preprocessor), ("clf", model)])
            else:
                pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])

            pipe.fit(X_train, y_train)
            probs = pipe.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, probs))

        results[name] = {"mean": np.mean(aucs), "std": np.std(aucs)}
        print(f"  {name}: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}", flush=True)

    return results


def run_german_credit():
    """German Credit dataset with categorical features."""
    print("\n" + "="*60)
    print("GERMAN CREDIT (N=1000)")
    print("="*60)

    data = fetch_openml(data_id=31, as_frame=True, parser="auto")
    df = data.frame

    cat_cols = ["checking_status", "credit_history", "purpose", "savings_status",
                "employment", "personal_status", "other_parties", "property_magnitude",
                "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"]
    num_cols = ["duration", "credit_amount", "installment_commitment", "residence_since",
                "age", "existing_credits", "num_dependents"]

    y = (df["class"].astype(str) == "good").astype(int).values

    X_num = df[num_cols].values.astype(np.float32)
    X_cat_encoded = []
    cat_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_cat_encoded.append(le.fit_transform(df[col].astype(str)))
        cat_encoders[col] = le
    X_cat = np.column_stack(X_cat_encoded)

    X_all = np.hstack([X_num, X_cat])
    cat_mask = np.array([False]*len(num_cols) + [True]*len(cat_cols))

    print("\nBaselines (with one-hot encoding):")
    baseline_results = run_baselines(X_all, y, cat_mask)

    print("\nOurs (piecewise linear):")
    # PCA on one-hot encoded categoricals + numeric features
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import OneHotEncoder

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_ohe = ohe.fit_transform(df[cat_cols].astype(str))

    n_pca_cat = 6  # Reduce categoricals to 6 PCA components
    n_bins = 3  # Fewer bins for small N
    n_lat = 3  # Only 3 lattice dimensions
    n_reg = 8  # Use top 8 features for regression

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    fold_models = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_num, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")
        X_train_num, X_test_num = X_num[train_idx], X_num[test_idx]
        X_train_cat, X_test_cat = X_cat_ohe[train_idx], X_cat_ohe[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale numeric features
        scaler = StandardScaler()
        X_train_num_s = scaler.fit_transform(X_train_num)
        X_test_num_s = scaler.transform(X_test_num)

        # PCA on categorical one-hot (fit on train only)
        pca = PCA(n_components=n_pca_cat)
        X_train_cat_pca = pca.fit_transform(X_train_cat)
        X_test_cat_pca = pca.transform(X_test_cat)

        # Combine: numeric + PCA(categorical)
        X_train_all = np.hstack([X_train_num_s, X_train_cat_pca])
        X_test_all = np.hstack([X_test_num_s, X_test_cat_pca])
        n_features = X_train_all.shape[1]

        # Use LR to find important features
        lr = LogisticRegression(max_iter=1000, C=1.0, penalty='l1', solver='saga')
        lr.fit(X_train_all, y_train)
        importance = np.abs(lr.coef_[0])
        top_idx = np.argsort(importance)[::-1]
        reg_idx = top_idx[:n_reg]  # Top features for regression
        lat_idx = top_idx[:n_lat]  # Top features for lattice

        X_train_sub = X_train_all[:, reg_idx]
        X_test_sub = X_test_all[:, reg_idx]

        dimensions = []
        train_indices = {}
        test_indices = {}

        for i, idx in enumerate(lat_idx):
            percentiles = np.linspace(0, 100, n_bins + 1)
            edges = np.percentile(X_train_all[:, idx], percentiles)
            train_indices[f"f{i}"] = np.clip(
                np.digitize(X_train_all[:, idx], edges[1:-1]), 0, n_bins - 1
            )
            test_indices[f"f{i}"] = np.clip(
                np.digitize(X_test_all[:, idx], edges[1:-1]), 0, n_bins - 1
            )
            dimensions.append(Dimension(f"f{i}", n_bins))

        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(interactions=interactions, param_shape=[n_reg], name="beta")

        train_data = {"X": X_train_sub, "y": y_train, **train_indices}
        test_data = {"X": X_test_sub, "y": y_test, **test_indices}

        prior_scales = decomp.generalization_preserving_scales(
            noise_scale=1.0, total_n=len(train_idx), c=0.3, per_component=True
        )

        params = fit_logistic_model(
            train_data, decomp, max_order=2,
            prior_scales=prior_scales, sparse=True, l1_weight=0.01, n_steps=3000
        )

        metrics = evaluate_model(test_data, decomp, params)
        aucs.append(metrics["auc"])
        print(f"    Fold AUC: {metrics['auc']:.4f}")

        fold_models.append({
            "params": {k: np.array(v) for k, v in params.items()},
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "top_idx": list(top_idx),
            "fold_idx": fold_idx
        })

    our_mean, our_std = np.mean(aucs), np.std(aucs)
    print(f"\n  OURS: {our_mean:.4f} +/- {our_std:.4f}")

    print("\n  Training final model on all data...")
    scaler = StandardScaler()
    X_all_num_s = scaler.fit_transform(X_num)

    pca_final = PCA(n_components=n_pca_cat)
    X_all_cat_pca = pca_final.fit_transform(X_cat_ohe)
    X_all_features = np.hstack([X_all_num_s, X_all_cat_pca])
    n_features = X_all_features.shape[1]

    # Use LR to find important features on all data
    lr_final = LogisticRegression(max_iter=1000, C=1.0, penalty='l1', solver='saga')
    lr_final.fit(X_all_features, y)
    importance_final = np.abs(lr_final.coef_[0])
    top_idx_final = np.argsort(importance_final)[::-1]
    reg_idx_final = top_idx_final[:n_reg]
    lat_idx_final = top_idx_final[:n_lat]

    X_all_sub = X_all_features[:, reg_idx_final]

    dimensions = []
    all_indices = {}

    for i, idx in enumerate(lat_idx_final):
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(X_all_features[:, idx], percentiles)
        all_indices[f"f{i}"] = np.clip(
            np.digitize(X_all_features[:, idx], edges[1:-1]), 0, n_bins - 1
        )
        dimensions.append(Dimension(f"f{i}", n_bins))

    interactions = Interactions(dimensions=dimensions)
    decomp = Decomposed(interactions=interactions, param_shape=[n_reg], name="beta")

    all_data = {"X": X_all_sub, "y": y, **all_indices}
    prior_scales = decomp.generalization_preserving_scales(
        noise_scale=1.0, total_n=len(y), c=0.3, per_component=True
    )

    final_params = fit_logistic_model(
        all_data, decomp, max_order=2,
        prior_scales=prior_scales, sparse=True, l1_weight=0.01, n_steps=3000
    )

    model_data = {
        "dataset": "german_credit",
        "cv_aucs": aucs,
        "cv_mean": our_mean,
        "cv_std": our_std,
        "baseline_results": baseline_results,
        "fold_models": fold_models,
        "final_model": {
            "params": {k: np.array(v) for k, v in final_params.items()},
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        "dimensions": [(d.name, d.cardinality) for d in dimensions],
        "n_pca_cat": n_pca_cat,
        "n_reg": n_reg,
        "n_lat": n_lat,
        "reg_idx": list(reg_idx_final),
        "lat_idx": list(lat_idx_final),
        "cat_encoders": {col: list(le.classes_) for col, le in cat_encoders.items()},
        "num_cols": num_cols,
        "decomp_info": {
            "tensor_parts": list(decomp._tensor_parts.keys()),
            "tensor_part_shapes": {k: v for k, v in decomp._tensor_part_shapes.items()},
        }
    }

    save_path = RESULTS_DIR / "german_credit_full.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  Saved to {save_path}")

    return our_mean, our_std, baseline_results


def run_adult():
    """Adult income dataset."""
    print("\n" + "="*60)
    print("ADULT (N=48842)")
    print("="*60)

    data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    df = data.frame

    cat_cols = ["workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country"]
    num_cols = ["age", "fnlwgt", "education-num", "capital-gain",
                "capital-loss", "hours-per-week"]

    y = (df["class"].astype(str).str.strip() == ">50K").astype(int).values

    X_num = df[num_cols].values.astype(np.float32)
    X_cat_encoded = []
    cat_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_cat_encoded.append(le.fit_transform(df[col].astype(str)))
        cat_encoders[col] = le
    X_cat = np.column_stack(X_cat_encoded)

    X_all = np.hstack([X_num, X_cat])
    cat_mask = np.array([False]*len(num_cols) + [True]*len(cat_cols))

    print(f"N = {len(y)}, Class balance: {y.mean():.3f}")

    print("\nBaselines (with one-hot encoding):")
    baseline_results = run_baselines(X_all, y, cat_mask)

    print("\nOurs (piecewise linear):")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import OneHotEncoder

    # One-hot encode ALL categoricals
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_ohe = ohe.fit_transform(df[cat_cols].astype(str))

    # Adult is larger (N=48842), can use more features
    n_pca_cat = 20
    n_bins = 5
    n_lat = 8
    n_reg = 25

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    fold_models = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_num, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")
        X_train_num, X_test_num = X_num[train_idx], X_num[test_idx]
        X_train_cat, X_test_cat = X_cat_ohe[train_idx], X_cat_ohe[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_num_s = scaler.fit_transform(X_train_num)
        X_test_num_s = scaler.transform(X_test_num)

        pca = PCA(n_components=n_pca_cat)
        X_train_cat_pca = pca.fit_transform(X_train_cat)
        X_test_cat_pca = pca.transform(X_test_cat)

        X_train_all = np.hstack([X_train_num_s, X_train_cat_pca])
        X_test_all = np.hstack([X_test_num_s, X_test_cat_pca])

        # Use LR to find important features
        lr = LogisticRegression(max_iter=1000, C=1.0, penalty='l1', solver='saga')
        lr.fit(X_train_all, y_train)
        importance = np.abs(lr.coef_[0])
        top_idx = np.argsort(importance)[::-1]
        reg_idx = top_idx[:n_reg]
        lat_idx = top_idx[:n_lat]

        X_train_sub = X_train_all[:, reg_idx]
        X_test_sub = X_test_all[:, reg_idx]

        dimensions = []
        train_indices = {}
        test_indices = {}

        for i, idx in enumerate(lat_idx):
            percentiles = np.linspace(0, 100, n_bins + 1)
            edges = np.percentile(X_train_all[:, idx], percentiles)
            train_indices[f"f{i}"] = np.clip(
                np.digitize(X_train_all[:, idx], edges[1:-1]), 0, n_bins - 1
            )
            test_indices[f"f{i}"] = np.clip(
                np.digitize(X_test_all[:, idx], edges[1:-1]), 0, n_bins - 1
            )
            dimensions.append(Dimension(f"f{i}", n_bins))

        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(interactions=interactions, param_shape=[n_reg], name="beta")

        train_data = {"X": X_train_sub, "y": y_train, **train_indices}
        test_data = {"X": X_test_sub, "y": y_test, **test_indices}

        prior_scales = decomp.generalization_preserving_scales(
            noise_scale=1.0, total_n=len(train_idx), c=0.3, per_component=True
        )

        params = fit_logistic_model(
            train_data, decomp, max_order=2,
            prior_scales=prior_scales, sparse=True, l1_weight=0.01, n_steps=3000
        )

        metrics = evaluate_model(test_data, decomp, params)
        aucs.append(metrics["auc"])
        print(f"    Fold AUC: {metrics['auc']:.4f}")

        fold_models.append({
            "params": {k: np.array(v) for k, v in params.items()},
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "pca_components": pca.components_,
            "reg_idx": list(reg_idx),
            "fold_idx": fold_idx
        })

    our_mean, our_std = np.mean(aucs), np.std(aucs)
    print(f"\n  OURS: {our_mean:.4f} +/- {our_std:.4f}")

    print("\n  Training final model on all data...")
    scaler = StandardScaler()
    X_all_num_s = scaler.fit_transform(X_num)

    pca_final = PCA(n_components=n_pca_cat)
    X_all_cat_pca = pca_final.fit_transform(X_cat_ohe)
    X_all_features = np.hstack([X_all_num_s, X_all_cat_pca])

    lr_final = LogisticRegression(max_iter=1000, C=1.0, penalty='l1', solver='saga')
    lr_final.fit(X_all_features, y)
    importance_final = np.abs(lr_final.coef_[0])
    top_idx_final = np.argsort(importance_final)[::-1]
    reg_idx_final = top_idx_final[:n_reg]
    lat_idx_final = top_idx_final[:n_lat]

    X_all_sub = X_all_features[:, reg_idx_final]

    dimensions = []
    all_indices = {}

    for i, idx in enumerate(lat_idx_final):
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(X_all_features[:, idx], percentiles)
        all_indices[f"f{i}"] = np.clip(
            np.digitize(X_all_features[:, idx], edges[1:-1]), 0, n_bins - 1
        )
        dimensions.append(Dimension(f"f{i}", n_bins))

    interactions = Interactions(dimensions=dimensions)
    decomp = Decomposed(interactions=interactions, param_shape=[n_reg], name="beta")

    all_data = {"X": X_all_sub, "y": y, **all_indices}
    prior_scales = decomp.generalization_preserving_scales(
        noise_scale=1.0, total_n=len(y), c=0.3, per_component=True
    )

    final_params = fit_logistic_model(
        all_data, decomp, max_order=2,
        prior_scales=prior_scales, sparse=True, l1_weight=0.01, n_steps=3000
    )

    model_data = {
        "dataset": "adult",
        "cv_aucs": aucs,
        "cv_mean": our_mean,
        "cv_std": our_std,
        "baseline_results": baseline_results,
        "fold_models": fold_models,
        "final_model": {
            "params": {k: np.array(v) for k, v in final_params.items()},
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "pca_components": pca_final.components_,
        },
        "dimensions": [(d.name, d.cardinality) for d in dimensions],
        "n_pca_cat": n_pca_cat,
        "n_reg": n_reg,
        "n_lat": n_lat,
        "reg_idx": list(reg_idx_final),
        "lat_idx": list(lat_idx_final),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }

    save_path = RESULTS_DIR / "adult_full.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  Saved to {save_path}")

    return our_mean, our_std, baseline_results


def run_madelon():
    """Madelon dataset (synthetic, high-dimensional)."""
    print("\n" + "="*60)
    print("MADELON (N=2600, p=500)")
    print("="*60)

    data = fetch_openml(data_id=1485, as_frame=True, parser="auto")
    df = data.frame

    X = df.drop(columns=["Class"]).values.astype(np.float32)
    y = (df["Class"].astype(str) == "2").astype(int).values

    print(f"N = {len(y)}, p = {X.shape[1]}")

    cat_mask = np.array([False] * X.shape[1])

    print("\nBaselines:")
    baseline_results = run_baselines(X, y, cat_mask)

    print("\nOurs (piecewise linear):")

    n_reg = 15
    n_lat = 10
    n_bins = 5

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    fold_models = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Fit LR on training fold only for feature selection
        lr = LogisticRegression(max_iter=1000, C=1.0, penalty='l1', solver='saga')
        lr.fit(X_train_s, y_train)
        coefs = np.abs(lr.coef_[0])
        top_idx = np.argsort(coefs)[::-1]
        reg_features = top_idx[:n_reg]
        lattice_features = top_idx[:n_lat]

        X_train_sub = X_train_s[:, reg_features]
        X_test_sub = X_test_s[:, reg_features]

        dimensions = []
        train_indices = {}
        test_indices = {}

        for i, feat in enumerate(lattice_features):
            percentiles = np.linspace(0, 100, n_bins + 1)
            edges = np.percentile(X_train_s[:, feat], percentiles)
            train_indices[f"f{i}"] = np.clip(
                np.digitize(X_train_s[:, feat], edges[1:-1]), 0, n_bins - 1
            )
            test_indices[f"f{i}"] = np.clip(
                np.digitize(X_test_s[:, feat], edges[1:-1]), 0, n_bins - 1
            )
            dimensions.append(Dimension(f"f{i}", n_bins))

        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(interactions=interactions, param_shape=[n_reg], name="beta")

        train_data = {"X": X_train_sub, "y": y_train, **train_indices}
        test_data = {"X": X_test_sub, "y": y_test, **test_indices}

        prior_scales = decomp.generalization_preserving_scales(
            noise_scale=1.0, total_n=len(train_idx), c=0.3, per_component=True
        )

        params = fit_logistic_model(
            train_data, decomp, max_order=2,
            prior_scales=prior_scales, sparse=True, l1_weight=0.01, n_steps=5000
        )

        metrics = evaluate_model(test_data, decomp, params)
        aucs.append(metrics["auc"])
        print(f"    Fold AUC: {metrics['auc']:.4f}")

        fold_models.append({
            "params": {k: np.array(v) for k, v in params.items()},
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "fold_idx": fold_idx
        })

    our_mean, our_std = np.mean(aucs), np.std(aucs)
    print(f"\n  OURS: {our_mean:.4f} +/- {our_std:.4f}")

    print("\n  Training final model on all data...")
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X)

    # Fit LR on all data for final model feature selection
    lr_final = LogisticRegression(max_iter=1000, C=1.0, penalty='l1', solver='saga')
    lr_final.fit(X_all_s, y)
    coefs_final = np.abs(lr_final.coef_[0])
    top_idx_final = np.argsort(coefs_final)[::-1]
    reg_features_final = top_idx_final[:n_reg]
    lattice_features_final = top_idx_final[:n_lat]

    X_all_sub = X_all_s[:, reg_features_final]

    dimensions = []
    all_indices = {}

    for i, feat in enumerate(lattice_features_final):
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(X_all_s[:, feat], percentiles)
        all_indices[f"f{i}"] = np.clip(
            np.digitize(X_all_s[:, feat], edges[1:-1]), 0, n_bins - 1
        )
        dimensions.append(Dimension(f"f{i}", n_bins))

    interactions = Interactions(dimensions=dimensions)
    decomp = Decomposed(interactions=interactions, param_shape=[n_reg], name="beta")

    all_data = {"X": X_all_sub, "y": y, **all_indices}
    prior_scales = decomp.generalization_preserving_scales(
        noise_scale=1.0, total_n=len(y), c=0.3, per_component=True
    )

    final_params = fit_logistic_model(
        all_data, decomp, max_order=2,
        prior_scales=prior_scales, sparse=True, l1_weight=0.01, n_steps=5000
    )

    model_data = {
        "dataset": "madelon",
        "cv_aucs": aucs,
        "cv_mean": our_mean,
        "cv_std": our_std,
        "baseline_results": baseline_results,
        "fold_models": fold_models,
        "final_model": {
            "params": {k: np.array(v) for k, v in final_params.items()},
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        "dimensions": [(d.name, d.cardinality) for d in dimensions],
        "reg_features": list(reg_features_final),
        "lattice_features": list(lattice_features_final),
        "lr_coefs": coefs_final,
        "n_reg": n_reg,
        "n_lat": n_lat,
        "n_bins": n_bins,
    }

    save_path = RESULTS_DIR / "madelon_full.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  Saved to {save_path}")

    return our_mean, our_std, baseline_results


def run_bioresponse():
    """Bioresponse dataset."""
    print("\n" + "="*60)
    print("BIORESPONSE (N=3751, p=1776)")
    print("="*60)

    data = fetch_openml(data_id=4134, as_frame=True, parser="auto")
    df = data.frame

    X = df.drop(columns=["target"]).values.astype(np.float32)
    y = df["target"].astype(int).values

    print(f"N = {len(y)}, p = {X.shape[1]}")

    cat_mask = np.array([False] * X.shape[1])

    print("\nBaselines:")
    baseline_results = run_baselines(X, y, cat_mask)

    print("\nOurs (piecewise linear):")

    n_reg = 100
    n_lat = 10
    n_bins = 4

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    fold_models = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Fit LR on training fold only for feature selection
        lr = LogisticRegression(max_iter=1000, C=1.0, penalty='l1', solver='saga')
        lr.fit(X_train_s, y_train)
        coefs = np.abs(lr.coef_[0])
        top_idx = np.argsort(coefs)[::-1]
        reg_features = top_idx[:n_reg]
        lattice_features = top_idx[:n_lat]

        X_train_sub = X_train_s[:, reg_features]
        X_test_sub = X_test_s[:, reg_features]

        dimensions = []
        train_indices = {}
        test_indices = {}

        for i, feat in enumerate(lattice_features):
            percentiles = np.linspace(0, 100, n_bins + 1)
            edges = np.percentile(X_train_s[:, feat], percentiles)
            train_indices[f"f{i}"] = np.clip(
                np.digitize(X_train_s[:, feat], edges[1:-1]), 0, n_bins - 1
            )
            test_indices[f"f{i}"] = np.clip(
                np.digitize(X_test_s[:, feat], edges[1:-1]), 0, n_bins - 1
            )
            dimensions.append(Dimension(f"f{i}", n_bins))

        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(interactions=interactions, param_shape=[n_reg], name="beta")

        train_data = {"X": X_train_sub, "y": y_train, **train_indices}
        test_data = {"X": X_test_sub, "y": y_test, **test_indices}

        prior_scales = decomp.generalization_preserving_scales(
            noise_scale=1.0, total_n=len(train_idx), c=0.3, per_component=True
        )

        params = fit_logistic_model(
            train_data, decomp, max_order=2,
            prior_scales=prior_scales, sparse=True, l1_weight=0.01, n_steps=5000
        )

        metrics = evaluate_model(test_data, decomp, params)
        aucs.append(metrics["auc"])
        print(f"    Fold AUC: {metrics['auc']:.4f}")

        fold_models.append({
            "params": {k: np.array(v) for k, v in params.items()},
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "fold_idx": fold_idx
        })

    our_mean, our_std = np.mean(aucs), np.std(aucs)
    print(f"\n  OURS: {our_mean:.4f} +/- {our_std:.4f}")

    print("\n  Training final model on all data...")
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X)

    # Fit LR on all data for final model feature selection
    lr_final = LogisticRegression(max_iter=1000, C=1.0, penalty='l1', solver='saga')
    lr_final.fit(X_all_s, y)
    coefs_final = np.abs(lr_final.coef_[0])
    top_idx_final = np.argsort(coefs_final)[::-1]
    reg_features_final = top_idx_final[:n_reg]
    lattice_features_final = top_idx_final[:n_lat]

    X_all_sub = X_all_s[:, reg_features_final]

    dimensions = []
    all_indices = {}

    for i, feat in enumerate(lattice_features_final):
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(X_all_s[:, feat], percentiles)
        all_indices[f"f{i}"] = np.clip(
            np.digitize(X_all_s[:, feat], edges[1:-1]), 0, n_bins - 1
        )
        dimensions.append(Dimension(f"f{i}", n_bins))

    interactions = Interactions(dimensions=dimensions)
    decomp = Decomposed(interactions=interactions, param_shape=[n_reg], name="beta")

    all_data = {"X": X_all_sub, "y": y, **all_indices}
    prior_scales = decomp.generalization_preserving_scales(
        noise_scale=1.0, total_n=len(y), c=0.3, per_component=True
    )

    final_params = fit_logistic_model(
        all_data, decomp, max_order=2,
        prior_scales=prior_scales, sparse=True, l1_weight=0.01, n_steps=5000
    )

    model_data = {
        "dataset": "bioresponse",
        "cv_aucs": aucs,
        "cv_mean": our_mean,
        "cv_std": our_std,
        "baseline_results": baseline_results,
        "fold_models": fold_models,
        "final_model": {
            "params": {k: np.array(v) for k, v in final_params.items()},
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        "dimensions": [(d.name, d.cardinality) for d in dimensions],
        "reg_features": list(reg_features_final),
        "lattice_features": list(lattice_features_final),
        "lattice_features": list(lattice_features),
        "n_reg": n_reg,
        "n_lat": n_lat,
        "n_bins": n_bins,
    }

    save_path = RESULTS_DIR / "bioresponse_full.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  Saved to {save_path}")

    return our_mean, our_std, baseline_results


if __name__ == "__main__":
    print("UCI Benchmark Suite with Proper Preprocessing")
    print("=" * 60)

    all_results = {}

    gc_ours, gc_std, gc_base = run_german_credit()
    all_results["german_credit"] = {"ours": (gc_ours, gc_std), "baselines": gc_base}

    mad_ours, mad_std, mad_base = run_madelon()
    all_results["madelon"] = {"ours": (mad_ours, mad_std), "baselines": mad_base}

    bio_ours, bio_std, bio_base = run_bioresponse()
    all_results["bioresponse"] = {"ours": (bio_ours, bio_std), "baselines": bio_base}

    adult_ours, adult_std, adult_base = run_adult()
    all_results["adult"] = {"ours": (adult_ours, adult_std), "baselines": adult_base}

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for dataset, results in all_results.items():
        ours_mean, ours_std = results["ours"]
        print(f"\n{dataset.upper()}:")
        print(f"  OURS: {ours_mean:.4f} +/- {ours_std:.4f}")
        for name, vals in results["baselines"].items():
            print(f"  {name}: {vals['mean']:.4f} +/- {vals['std']:.4f}")

    summary_path = RESULTS_DIR / "all_results.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nAll results saved to {summary_path}")

"""Adult with RICH lattice - use full complexity, trust gen-preserving regularization."""
from uci_benchmarks import *
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from itertools import combinations
import xgboost as xgb
import lightgbm as lgb

df = download_dataset("adult", "data/uci")
config = UCI_DATASETS["adult"]

numeric_cols = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
cat_cols = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex"]

print(f"N = {len(df)}")

# Prepare features
X_num = df[numeric_cols].values.astype(np.float32)
X_num = np.nan_to_num(X_num, nan=0.0)
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = encoder.fit_transform(df[cat_cols].astype(str))

# Add pairwise on numeric
pairwise = [X_num_scaled[:, i] * X_num_scaled[:, j] for i, j in combinations(range(X_num_scaled.shape[1]), 2)]
X_full = np.concatenate([X_num_scaled, X_cat, np.stack(pairwise, axis=1)], axis=1)

y = (df[config["target"]].str.strip() == ">50K").astype(int).values

print(f"X shape: {X_full.shape}, Class balance: {y.mean():.3f}")

# Baselines
print("\n=== Baselines (5-fold CV) ===")
X_baseline = np.concatenate([X_num_scaled, X_cat], axis=1)

for name, model in [
    ("LR", LogisticRegression(C=1.0, max_iter=1000)),
    ("RF", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
    ("XGB", xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="logloss")),
    ("LGBM", lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)),
]:
    aucs = []
    for train_idx, test_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_baseline, y):
        model.fit(X_baseline[train_idx], y[train_idx])
        aucs.append(roc_auc_score(y[test_idx], model.predict_proba(X_baseline[test_idx])[:, 1]))
    print(f"{name}: AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

# Ours - incrementally add complexity from best config
print("\n=== Ours (incremental complexity) ===")

def test_config(cat_cols_use, num_bins_list, max_order=2):
    dimensions = []
    factor_indices = {}

    for col in cat_cols_use:
        le = LabelEncoder()
        indices = le.fit_transform(df[col].astype(str))
        factor_indices[col] = indices
        dimensions.append(Dimension(col, len(le.classes_)))

    for col, n_bins in num_bins_list:
        col_idx = numeric_cols.index(col)
        vals = X_num_scaled[:, col_idx]
        ranks = np.argsort(np.argsort(vals))
        factor_indices[f"bin_{col}"] = (ranks * n_bins // len(ranks)).astype(int)
        dimensions.append(Dimension(f"bin_{col}", n_bins))

    data_local = {"X": X_full, "y": y, **factor_indices}
    interactions = Interactions(dimensions=dimensions)
    decomp = Decomposed(interactions=interactions, param_shape=[X_full.shape[1]], name="beta")

    aucs = []
    for train_idx, test_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_full, y):
        train_data, test_data = create_train_test_split(data_local, train_idx, test_idx)
        prior_scales = decomp.generalization_preserving_scales(noise_scale=1.0, total_n=len(train_idx), c=0.3, per_component=True)
        params = fit_logistic_model(train_data, decomp, max_order=max_order, prior_scales=prior_scales, sparse=True, l1_weight=0.01, n_steps=5000)
        metrics = evaluate_model(test_data, decomp, params)
        aucs.append(metrics["auc"])
    return np.mean(aucs), np.std(aucs)

# Full categorical complement, refine continuous bins
# Start coarse, progressively refine
ALL_CATS = ["education", "marital_status", "occupation", "relationship"]  # Key 4 cats

configs = [
    # Coarse bins (3)
    (ALL_CATS, [("age", 3), ("capital_gain", 3), ("hours_per_week", 3)], "4cat + 3x3bins"),
    # Medium bins (5)
    (ALL_CATS, [("age", 5), ("capital_gain", 5), ("hours_per_week", 5)], "4cat + 3x5bins"),
    # Fine bins (8)
    (ALL_CATS, [("age", 8), ("capital_gain", 8), ("hours_per_week", 8)], "4cat + 3x8bins"),
    # Very fine bins (10)
    (ALL_CATS, [("age", 10), ("capital_gain", 10), ("hours_per_week", 10)], "4cat + 3x10bins"),
    # Add more continuous
    (ALL_CATS, [("age", 8), ("capital_gain", 8), ("hours_per_week", 8), ("education_num", 5)], "4cat + 4num"),
]

for cats, nums, desc in configs:
    try:
        auc, std = test_config(cats, nums, max_order=2)
        print(f"{desc}: AUC = {auc:.4f} +/- {std:.4f}")
    except Exception as e:
        print(f"{desc}: ERROR - {e}")

data = {"X": X_full, "y": y, **factor_indices}
interactions = Interactions(dimensions=dimensions)
decomp = Decomposed(interactions=interactions, param_shape=[X_full.shape[1]], name="beta")

aucs = []
for train_idx, test_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_full, y):
    train_data, test_data = create_train_test_split(data, train_idx, test_idx)
    prior_scales = decomp.generalization_preserving_scales(noise_scale=1.0, total_n=len(train_idx), c=0.3, per_component=True)
    # Use order 2 - trust the regularization
    params = fit_logistic_model(train_data, decomp, max_order=2, prior_scales=prior_scales, sparse=True, l1_weight=0.01, n_steps=5000)
    metrics = evaluate_model(test_data, decomp, params)
    aucs.append(metrics["auc"])

print(f"\nOurs: AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

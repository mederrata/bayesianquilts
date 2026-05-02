"""Re-run Table 1 baselines with proper 5-fold CV to get real values."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("Warning: lightgbm not installed")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: xgboost not installed")

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    HAS_EBM = True
except ImportError:
    HAS_EBM = False
    print("Warning: interpret not installed")


def run_baselines(X, y, dataset_name):
    """Run all baselines with proper 5-fold CV."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"N={len(y)}, p={X.shape[1]}, pos_rate={y.mean():.3f}")
    print(f"{'='*60}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # Logistic Regression L1
    print("\nRunning LR-L1...")
    aucs = []
    for train_idx, test_idx in kfold.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = LogisticRegression(penalty='l1', solver='saga', max_iter=1000, C=1.0)
        model.fit(X_train, y[train_idx])
        probs = model.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y[test_idx], probs))
    results['LR-L1'] = {'mean': np.mean(aucs), 'std': np.std(aucs)}
    print(f"  LR-L1: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    # Random Forest
    print("Running RF...")
    aucs = []
    for train_idx, test_idx in kfold.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y[train_idx])
        probs = model.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y[test_idx], probs))
    results['RF'] = {'mean': np.mean(aucs), 'std': np.std(aucs)}
    print(f"  RF: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    # MLP
    print("Running MLP...")
    aucs = []
    for train_idx, test_idx in kfold.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        model.fit(X_train, y[train_idx])
        probs = model.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y[test_idx], probs))
    results['MLP'] = {'mean': np.mean(aucs), 'std': np.std(aucs)}
    print(f"  MLP: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    # LightGBM
    if HAS_LGBM:
        print("Running LGBM...")
        aucs = []
        for train_idx, test_idx in kfold.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
            model.fit(X_train, y[train_idx])
            probs = model.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y[test_idx], probs))
        results['LGBM'] = {'mean': np.mean(aucs), 'std': np.std(aucs)}
        print(f"  LGBM: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    # XGBoost
    if HAS_XGB:
        print("Running XGB...")
        aucs = []
        for train_idx, test_idx in kfold.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y[train_idx])
            probs = model.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y[test_idx], probs))
        results['XGB'] = {'mean': np.mean(aucs), 'std': np.std(aucs)}
        print(f"  XGB: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    # EBM
    if HAS_EBM:
        print("Running EBM...")
        aucs = []
        for train_idx, test_idx in kfold.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            model = ExplainableBoostingClassifier(random_state=42, n_jobs=-1)
            model.fit(X_train, y[train_idx])
            probs = model.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y[test_idx], probs))
        results['EBM'] = {'mean': np.mean(aucs), 'std': np.std(aucs)}
        print(f"  EBM: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    return results


def load_dataset(name):
    """Load a dataset from OpenML."""
    configs = {
        'adult': 1590,
        'bank': 1461,  # bank-marketing
        'taiwan': 42477,
        'bioresponse': 4134,
        'phoneme': 1489,
        'electricity': 151,
        'german': 31,  # credit-g
    }

    print(f"\nLoading {name}...")
    data = fetch_openml(data_id=configs[name], as_frame=True, parser="auto")
    df = data.frame.copy()

    # Handle target
    target_col = data.target_names[0] if hasattr(data, 'target_names') else 'target'
    if target_col not in df.columns:
        target_col = df.columns[-1]

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode target to binary
    if y.dtype == object or str(y.dtype) == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    else:
        y = y.values.astype(int)

    # Encode categoricals first (before fillna to avoid category issues)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        X_cat = X[cat_cols].astype(str).fillna('missing')
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        X_cat_enc = ohe.fit_transform(X_cat)
        X_num = X.drop(columns=cat_cols).fillna(0).values.astype(float)
        X = np.concatenate([X_num, X_cat_enc], axis=1)
    else:
        X = X.fillna(0).values.astype(float)

    # Final NaN check
    X = np.nan_to_num(X, nan=0.0)

    # For high-dim datasets, use PCA
    if X.shape[1] > 100:
        print(f"  Applying PCA (p={X.shape[1]} -> 50)")
        pca = PCA(n_components=50)
        X = pca.fit_transform(X)

    return X, y


def main():
    all_results = {}

    # Run baselines for each dataset
    for dataset in ['adult', 'bank', 'taiwan', 'bioresponse', 'phoneme', 'electricity', 'german']:
        try:
            X, y = load_dataset(dataset)
            results = run_baselines(X, y, dataset)
            all_results[dataset] = {'baselines': results}
        except Exception as e:
            print(f"Error on {dataset}: {e}")
            continue

    # Save results
    output_path = Path(__file__).parent / "results" / "table1_baselines_fresh.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - Table 1 Baselines (Fresh 5-Fold CV)")
    print("="*80)
    for dataset, data in all_results.items():
        print(f"\n{dataset}:")
        for method, vals in data['baselines'].items():
            print(f"  {method}: {vals['mean']:.4f} +/- {vals['std']:.4f}")


if __name__ == "__main__":
    main()

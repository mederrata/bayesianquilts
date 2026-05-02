"""Run EBM baseline on Bank data with same preprocessing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

def load_bank_data():
    import zipfile
    import urllib.request
    from io import BytesIO

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
    cache_path = Path("data/bank/bank-full.csv")
    if cache_path.exists():
        df = pd.read_csv(cache_path, sep=";")
        if "y" in df.columns:
            return df
        cache_path.unlink()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        with zipfile.ZipFile(BytesIO(response.read())) as z:
            with z.open("bank-full.csv") as f:
                df = pd.read_csv(f, sep=";")
                df.to_csv(cache_path, sep=";", index=False)
                return df


def run_baselines():
    print("BANK - Baseline comparison")
    print("="*70)

    df = load_bank_data()
    y = (df["y"] == "yes").astype(int).values
    N = len(y)
    print(f"N = {N}, pos_rate = {y.mean():.3f}")

    # Prepare features
    numeric_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    cat_cols = ["job", "marital", "education", "default", "housing", "loan",
                "contact", "month", "poutcome"]

    X_numeric = df[numeric_cols].values.astype(np.float32)
    X_numeric = np.nan_to_num(X_numeric, nan=0.0)

    # Encode categoricals
    X_cat = np.zeros((N, len(cat_cols)), dtype=np.int32)
    for i, col in enumerate(cat_cols):
        le = LabelEncoder().fit(df[col].astype(str))
        X_cat[:, i] = le.transform(df[col].astype(str))

    X = np.hstack([X_numeric, X_cat])

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Try EBM
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        print("\nRunning EBM...")
        ebm_aucs = []
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            ebm = ExplainableBoostingClassifier(random_state=42)
            ebm.fit(X_train, y_train)
            probs = ebm.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
            ebm_aucs.append(auc)
            print(f"  Fold {fold_idx+1}: AUC = {auc:.4f}")
        print(f"EBM: {np.mean(ebm_aucs):.4f} +/- {np.std(ebm_aucs):.4f}")
    except ImportError:
        print("EBM not available")

    # Try LightGBM
    try:
        import lightgbm as lgb
        print("\nRunning LightGBM...")
        lgbm_aucs = []
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
            lgbm_aucs.append(auc)
            print(f"  Fold {fold_idx+1}: AUC = {auc:.4f}")
        print(f"LGBM: {np.mean(lgbm_aucs):.4f} +/- {np.std(lgbm_aucs):.4f}")
    except ImportError:
        print("LightGBM not available")

    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    print("\nRunning Logistic Regression...")
    lr_aucs = []
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_s, y_train)
        probs = lr.predict_proba(X_test_s)[:, 1]
        auc = roc_auc_score(y_test, probs)
        lr_aucs.append(auc)
        print(f"  Fold {fold_idx+1}: AUC = {auc:.4f}")
    print(f"LR: {np.mean(lr_aucs):.4f} +/- {np.std(lr_aucs):.4f}")


if __name__ == "__main__":
    run_baselines()

"""Run EBM on missing datasets."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml

from interpret.glassbox import ExplainableBoostingClassifier


def run_ebm_cv(X, y, name):
    """Run EBM with 5-fold CV."""
    print(f"\n{name}")
    print("="*50)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        ebm = ExplainableBoostingClassifier(random_state=42)
        ebm.fit(X_train, y_train)

        probs = ebm.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        aucs.append(auc)
        print(f"  Fold {fold_idx+1}: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"  EBM: {mean_auc:.4f} +/- {std_auc:.4f}")
    return mean_auc, std_auc


def heart_disease():
    data = fetch_openml(data_id=53, as_frame=True, parser="auto")
    df = data.frame
    X = df.drop(columns=["class"]).values.astype(np.float32)
    y = (df["class"].astype(str) == "present").astype(int).values
    return run_ebm_cv(X, y, "Heart Disease (N=270)")


def spambase():
    data = fetch_openml(data_id=44, as_frame=True, parser="auto")
    df = data.frame
    X = df.drop(columns=["class"]).values.astype(np.float32)
    y = (df["class"].astype(str) == "1").astype(int).values
    return run_ebm_cv(X, y, "Spambase (N=4601)")


def phoneme():
    data = fetch_openml(data_id=1489, as_frame=True, parser="auto")
    df = data.frame
    X = df.drop(columns=["Class"]).values.astype(np.float32)
    y = (df["Class"].astype(str) == "1").astype(int).values
    return run_ebm_cv(X, y, "Phoneme (N=5404)")


def taiwan_credit():
    data = fetch_openml(data_id=42477, as_frame=True, parser="auto")
    df = data.frame
    target_col = [c for c in df.columns if 'default' in c.lower() or 'target' in c.lower()][0]
    X = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].astype(int).values
    return run_ebm_cv(X, y, "Taiwan Credit (N=30000)")


def bank_marketing():
    data = fetch_openml(data_id=1461, as_frame=True, parser="auto")
    df = data.frame
    # One-hot encode categoricals
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'Class']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if cat_cols:
        ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        X_cat = ohe.fit_transform(df[cat_cols].astype(str))
        X_num = df[num_cols].values.astype(np.float32)
        X = np.hstack([X_num, X_cat])
    else:
        X = df[num_cols].values.astype(np.float32)

    y = (df["Class"].astype(str) == "2").astype(int).values
    return run_ebm_cv(X, y, "Bank Marketing (N=45211)")


if __name__ == "__main__":
    results = {}

    results["heart_disease"] = heart_disease()
    results["spambase"] = spambase()
    results["phoneme"] = phoneme()
    # results["taiwan_credit"] = taiwan_credit()  # Large, skip for now
    # results["bank_marketing"] = bank_marketing()  # Large, skip for now

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for name, (mean, std) in results.items():
        print(f"  {name}: {mean:.4f} +/- {std:.4f}")

"""GAMINet baseline using the official gaminet package.

GAMINet: Generalized Additive Model with Structured Interactions
    y = β₀ + Σᵢ fᵢ(xᵢ) + Σᵢⱼ fᵢⱼ(xᵢ, xⱼ)

Reference: Yang et al. (2021) "GAMI-Net: An Explainable Neural Network
based on Generalized Additive Models with Structured Interactions"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml

# Import GAMINet
from gaminet import GAMINet
from gaminet.utils import get_interaction_list


def run_gaminet_dataset(name, df, cat_cols, num_cols, y, verbose=True):
    """Run GAMINet on a dataset."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"GAMINET: {name.upper()}")
        print("="*60)
        print(f"  N = {len(y)}, pos_rate = {y.mean():.3f}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(df, y)):
        if verbose:
            print(f"\n  Fold {fold_idx + 1}/5:")

        y_train, y_test = y[train_idx], y[test_idx]

        # Prepare features - GAMINet handles categorical encoding internally
        X_train_list = []
        X_test_list = []
        meta_info = {}

        # Numeric features with scalers
        for col in num_cols:
            vals = df[col].values.astype(np.float32)
            # Handle NaN values
            vals = np.nan_to_num(vals, nan=0.0)
            scaler = MinMaxScaler()
            vals_train = scaler.fit_transform(vals[train_idx].reshape(-1, 1))
            vals_test = scaler.transform(vals[test_idx].reshape(-1, 1))
            # Clip to [0, 1] to avoid issues with extreme values
            vals_train = np.clip(vals_train, 0, 1)
            vals_test = np.clip(vals_test, 0, 1)
            X_train_list.append(vals_train)
            X_test_list.append(vals_test)
            meta_info[col] = {"type": "continuous", "scaler": scaler}

        # Categorical features
        for col in cat_cols:
            # Convert to string and fill missing
            col_vals = df[col].astype(str).values
            col_vals = np.where(pd.isna(df[col].values) | (col_vals == 'nan'), "_missing_", col_vals)
            le = LabelEncoder()
            le.fit(col_vals[train_idx])
            # Handle unknown categories in test set
            vals_train = le.transform(col_vals[train_idx]).reshape(-1, 1).astype(np.float32)
            # For test, map unknown to 0
            test_vals = col_vals[test_idx]
            vals_test = np.zeros(len(test_vals), dtype=np.float32)
            for i, v in enumerate(test_vals):
                if v in le.classes_:
                    vals_test[i] = le.transform([v])[0]
            vals_test = vals_test.reshape(-1, 1)
            X_train_list.append(vals_train)
            X_test_list.append(vals_test)
            n_categories = len(le.classes_)
            meta_info[col] = {"type": "categorical", "values": list(range(n_categories))}

        # Add target to meta_info
        meta_info["target"] = {"type": "target"}

        X_train = np.hstack(X_train_list).astype(np.float32)
        X_test = np.hstack(X_test_list).astype(np.float32)
        y_train_arr = y_train.astype(np.float32).reshape(-1, 1)
        y_test_arr = y_test.astype(np.float32).reshape(-1, 1)

        # Create feature names
        feature_names = num_cols + cat_cols

        # Get interaction candidates
        interact_num = min(10, len(feature_names) * (len(feature_names) - 1) // 2)

        # Build GAMINet model
        model = GAMINet(
            meta_info=meta_info,
            interact_num=interact_num,
            subnet_arch=[40, 40],
            interact_arch=[20, 20],
            task_type="Classification",
            main_effect_epochs=2000,
            interaction_epochs=2000,
            tuning_epochs=500,
            lr_bp=[1e-3, 1e-4, 1e-5],
            early_stop_thres=[50, 50, 50],
            heredity=True,
            loss_threshold=0.01,
            reg_clarity=0.1,
            verbose=False,
            val_ratio=0.2,
            random_state=fold_idx,
        )

        # Fit model
        model.fit(X_train, y_train_arr)

        # Predict - GAMINet returns logits for classification
        preds = model.predict(X_test)
        if preds.ndim > 1:
            preds = preds[:, 0]
        # Convert logits to probabilities
        probs = 1 / (1 + np.exp(-preds))

        auc = roc_auc_score(y_test, probs)
        aucs.append(auc)
        if verbose:
            print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    if verbose:
        print(f"\n  GAMINet: {mean_auc:.4f} +/- {std_auc:.4f}")
    return mean_auc, std_auc


def run_heart():
    """Run GAMINet on Heart Disease (smallest, N=270)."""
    data = fetch_openml(data_id=53, as_frame=True, parser="auto")
    df = data.frame

    cat_cols = []  # All numeric
    num_cols = ["age", "resting_blood_pressure", "serum_cholestoral",
                "maximum_heart_rate_achieved", "oldpeak",
                "sex", "chest", "fasting_blood_sugar",
                "resting_electrocardiographic_results", "exercise_induced_angina",
                "slope", "number_of_major_vessels", "thal"]

    y = (df["class"].astype(str) == "present").astype(int).values

    return run_gaminet_dataset("Heart Disease", df, cat_cols, num_cols, y)


def run_german():
    """Run GAMINet on German Credit."""
    data = fetch_openml(data_id=31, as_frame=True, parser="auto")
    df = data.frame

    cat_cols = ["checking_status", "credit_history", "purpose", "savings_status",
                "employment", "personal_status", "other_parties", "property_magnitude",
                "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"]
    num_cols = ["duration", "credit_amount", "installment_commitment", "residence_since",
                "age", "existing_credits", "num_dependents"]

    y = (df["class"].astype(str) == "good").astype(int).values

    return run_gaminet_dataset("German Credit", df, cat_cols, num_cols, y)


def run_adult():
    """Run GAMINet on Adult Income."""
    data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    df = data.frame

    cat_cols = ["workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country"]
    num_cols = ["age", "fnlwgt", "education-num", "capital-gain",
                "capital-loss", "hours-per-week"]

    y = (df["class"].astype(str).str.strip() == ">50K").astype(int).values

    return run_gaminet_dataset("Adult Income", df, cat_cols, num_cols, y)


def run_bank():
    """Run GAMINet on Bank Marketing."""
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

    return run_gaminet_dataset("Bank Marketing", df, cat_cols, num_cols, y)


def run_covertype():
    """Run GAMINet on Covertype (N=581,012 - full dataset)."""
    data = fetch_openml(data_id=293, as_frame=False, parser="auto")
    X = data.data
    y_raw = data.target

    # Convert sparse to dense if needed
    if hasattr(X, 'toarray'):
        X = X.toarray()

    # Create DataFrame
    feature_names = data.feature_names
    df = pd.DataFrame(X, columns=feature_names)

    # All columns numeric
    num_cols = list(df.columns)
    cat_cols = []

    # Binary: class 2 (Lodgepole Pine) vs others
    y = (y_raw.astype(str) == "2").astype(int)

    print(f"Covertype: N={len(y)}, pos_rate={y.mean():.3f}")

    return run_gaminet_dataset("Covertype", df, cat_cols, num_cols, y)


def run_higgs():
    """Run GAMINet on HIGGS (N=98,050 on OpenML)."""
    data = fetch_openml(data_id=23512, as_frame=False, parser="auto")
    X = data.data
    y_raw = data.target

    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.nan_to_num(X, nan=0.0)

    feature_names = data.feature_names if data.feature_names else [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)

    num_cols = list(df.columns)
    cat_cols = []

    y = (y_raw == '1').astype(int)

    print(f"HIGGS: N={len(y)}, pos_rate={y.mean():.3f}")

    return run_gaminet_dataset("HIGGS", df, cat_cols, num_cols, y)


def run_madelon():
    """Run GAMINet on Madelon (N=2600, p=500)."""
    data = fetch_openml(data_id=1485, as_frame=False, parser="auto")
    X = data.data
    y_raw = data.target

    if hasattr(X, 'toarray'):
        X = X.toarray()

    feature_names = data.feature_names if data.feature_names else [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)

    num_cols = list(df.columns)
    cat_cols = []

    y = (y_raw == '1').astype(int)

    return run_gaminet_dataset("Madelon", df, cat_cols, num_cols, y)


def run_bioresponse():
    """Run GAMINet on Bioresponse (N=3751, p=1776)."""
    data = fetch_openml(data_id=4134, as_frame=False, parser="auto")
    X = data.data
    y_raw = data.target

    if hasattr(X, 'toarray'):
        X = X.toarray()

    feature_names = data.feature_names if data.feature_names else [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)

    num_cols = list(df.columns)
    cat_cols = []

    y = (y_raw == '1').astype(int)

    return run_gaminet_dataset("Bioresponse", df, cat_cols, num_cols, y)


def run_spambase():
    """Run GAMINet on Spambase (N=4601, p=57)."""
    data = fetch_openml(data_id=44, as_frame=False, parser="auto")
    X = data.data
    y_raw = data.target

    if hasattr(X, 'toarray'):
        X = X.toarray()

    feature_names = data.feature_names if data.feature_names else [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)

    num_cols = list(df.columns)
    cat_cols = []

    y = (y_raw == '1').astype(int)

    return run_gaminet_dataset("Spambase", df, cat_cols, num_cols, y)


def run_phoneme():
    """Run GAMINet on Phoneme (N=5404, p=5)."""
    data = fetch_openml(data_id=1489, as_frame=False, parser="auto")
    X = data.data
    y_raw = data.target

    if hasattr(X, 'toarray'):
        X = X.toarray()

    feature_names = data.feature_names if data.feature_names else [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)

    num_cols = list(df.columns)
    cat_cols = []

    y = (y_raw == '1').astype(int)

    return run_gaminet_dataset("Phoneme", df, cat_cols, num_cols, y)


def run_taiwan():
    """Run GAMINet on Taiwan Credit (N=30000, p=23)."""
    data = fetch_openml(data_id=42477, as_frame=True, parser="auto")
    df = data.frame

    cat_cols = ["x2", "x3", "x4"]  # SEX, EDUCATION, MARRIAGE
    num_cols = ["x1", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
                "x12", "x13", "x14", "x15", "x16", "x17",
                "x18", "x19", "x20", "x21", "x22", "x23"]

    y = df["y"].astype(int).values

    return run_gaminet_dataset("Taiwan Credit", df, cat_cols, num_cols, y)


def run_electricity():
    """Run GAMINet on Electricity (N=45312, p=8)."""
    data = fetch_openml(data_id=151, as_frame=True, parser="auto")
    df = data.frame

    cat_cols = ["day"]
    num_cols = ["date", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]

    y = (df["class"] == "UP").astype(int).values

    return run_gaminet_dataset("Electricity", df, cat_cols, num_cols, y)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[
        "heart", "german", "madelon", "bioresponse", "spambase", "phoneme",
        "taiwan", "bank", "electricity", "adult", "higgs", "covertype",
        "all", "small", "medium", "large"
    ], default="all")
    args = parser.parse_args()

    results = {}

    # In order of Table 2 (by sample size)
    if args.dataset in ["heart", "all", "small"]:
        results["heart"] = run_heart()

    if args.dataset in ["german", "all", "small"]:
        results["german"] = run_german()

    if args.dataset in ["madelon", "all", "small"]:
        results["madelon"] = run_madelon()

    if args.dataset in ["bioresponse", "all", "medium"]:
        results["bioresponse"] = run_bioresponse()

    if args.dataset in ["spambase", "all", "medium"]:
        results["spambase"] = run_spambase()

    if args.dataset in ["phoneme", "all", "medium"]:
        results["phoneme"] = run_phoneme()

    if args.dataset in ["taiwan", "all", "medium"]:
        results["taiwan"] = run_taiwan()

    if args.dataset in ["bank", "all"]:
        results["bank"] = run_bank()

    if args.dataset in ["electricity", "all"]:
        results["electricity"] = run_electricity()

    if args.dataset in ["adult", "all"]:
        results["adult"] = run_adult()

    if args.dataset in ["higgs", "all", "large"]:
        results["higgs"] = run_higgs()

    if args.dataset in ["covertype", "all", "large"]:
        results["covertype"] = run_covertype()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, (mean, std) in results.items():
        print(f"  {name}: {mean:.4f} +/- {std:.4f}")

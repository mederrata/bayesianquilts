#!/usr/bin/env python3
"""
HIGGS: Logistic Regression with main effects + all pairwise interactions.
p=28, n=98050, pairwise=378, total=406 features.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

print("Loading HIGGS...", flush=True)
higgs = fetch_openml(data_id=23512, as_frame=False, parser='auto')
X_full, y_full = higgs.data, higgs.target
X_full = np.nan_to_num(X_full, nan=0.0)
y_full = (y_full.astype(float) > 0.5).astype(int)

N, n_features = X_full.shape
print(f"Dataset: {N} samples, {n_features} features")
print(f"Pairwise features: {n_features * (n_features - 1) // 2}")
print(f"Total features: {n_features + n_features * (n_features - 1) // 2}")

results = {}

# 1. LR with main effects only (baseline)
print("\n" + "="*60)
print("LR: Main effects only")
print("="*60)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []

for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_full, y_full)):
    X_train, X_test = X_full[train_idx], X_full[test_idx]
    y_train, y_test = y_full[train_idx], y_full[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
    lr.fit(X_train_s, y_train)
    probs = lr.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, probs)
    aucs.append(auc)
    print(f"  Fold {fold_idx + 1}: AUC={auc:.4f}")

results['lr_main'] = (np.mean(aucs), np.std(aucs))
print(f"\n  LR main effects: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

# 2. LR with main + pairwise (interaction_only=True keeps it to degree 2)
print("\n" + "="*60)
print("LR: Main + Pairwise interactions")
print("="*60)

aucs = []

for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_full, y_full)):
    X_train, X_test = X_full[train_idx], X_full[test_idx]
    y_train, y_test = y_full[train_idx], y_full[test_idx]

    # Scale then add pairwise
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_s)
    X_test_poly = poly.transform(X_test_s)

    print(f"  Fold {fold_idx + 1}: {X_train_poly.shape[1]} features...", flush=True)

    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    lr.fit(X_train_poly, y_train)
    probs = lr.predict_proba(X_test_poly)[:, 1]
    auc = roc_auc_score(y_test, probs)
    aucs.append(auc)
    print(f"  Fold {fold_idx + 1}: AUC={auc:.4f}")

results['lr_pairwise'] = (np.mean(aucs), np.std(aucs))
print(f"\n  LR main + pairwise: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

# 3. LR with CV for regularization
print("\n" + "="*60)
print("LR-CV: Main + Pairwise with tuned regularization")
print("="*60)

aucs = []

for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_full, y_full)):
    X_train, X_test = X_full[train_idx], X_full[test_idx]
    y_train, y_test = y_full[train_idx], y_full[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_s)
    X_test_poly = poly.transform(X_test_s)

    print(f"  Fold {fold_idx + 1}: fitting LR-CV...", flush=True)

    # Tune C with inner CV
    lr_cv = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0],
        cv=3,
        max_iter=1000,
        solver='lbfgs',
        scoring='roc_auc'
    )
    lr_cv.fit(X_train_poly, y_train)
    probs = lr_cv.predict_proba(X_test_poly)[:, 1]
    auc = roc_auc_score(y_test, probs)
    aucs.append(auc)
    print(f"  Fold {fold_idx + 1}: AUC={auc:.4f}, best C={lr_cv.C_[0]:.2f}")

results['lr_cv_pairwise'] = (np.mean(aucs), np.std(aucs))
print(f"\n  LR-CV main + pairwise: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

# 4. Top 8 features + their pairwise (smaller feature set)
print("\n" + "="*60)
print("LR: Top 8 features + their pairwise")
print("="*60)

# Get top features from LR
scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X_full)
lr_full = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
lr_full.fit(X_scaled, y_full)
coef_abs = np.abs(lr_full.coef_.flatten())
sorted_idx = np.argsort(-coef_abs)
top8 = sorted_idx[:8].tolist()
print(f"  Top 8 features: {top8}")

aucs = []

for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_full, y_full)):
    X_train, X_test = X_full[train_idx][:, top8], X_full[test_idx][:, top8]
    y_train, y_test = y_full[train_idx], y_full[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_s)
    X_test_poly = poly.transform(X_test_s)

    lr = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
    lr.fit(X_train_poly, y_train)
    probs = lr.predict_proba(X_test_poly)[:, 1]
    auc = roc_auc_score(y_test, probs)
    aucs.append(auc)
    print(f"  Fold {fold_idx + 1}: AUC={auc:.4f}")

results['lr_top8_pair'] = (np.mean(aucs), np.std(aucs))
print(f"\n  LR top8 + pairwise: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

print(f"\n{'='*70}")
print("HIGGS LR + PAIRWISE RESULTS:")
print(f"  Our decomp best:  0.7715 (5d order-2)")
print(f"  EBM target:       0.803")
print(f"{'='*70}")
for name, (mean, std) in sorted(results.items(), key=lambda x: -x[1][0]):
    marker = " **BEST**" if mean >= max(r[0] for r in results.values()) else ""
    print(f"  {name}: {mean:.4f} +/- {std:.4f}{marker}")
print(f"{'='*70}")

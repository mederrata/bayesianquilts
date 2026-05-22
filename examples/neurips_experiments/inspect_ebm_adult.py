"""Inspect EBM on Adult to see what interactions it learns."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
from interpret.glassbox import ExplainableBoostingClassifier

def run_ebm():
    print("Inspecting EBM on Adult dataset")
    print("="*70)

    data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    df = data.frame.copy()
    y = (df["class"].astype(str) == ">50K").astype(int).values

    # Drop target and prepare features
    feature_cols = [c for c in df.columns if c != "class"]
    X = df[feature_cols].copy()

    # Label encode categoricals
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training EBM with interactions...")
    ebm = ExplainableBoostingClassifier(
        interactions=10,  # Allow 10 pairwise interactions
        random_state=42,
        n_jobs=-1
    )
    ebm.fit(X_train, y_train)

    probs = ebm.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"EBM AUC: {auc:.4f}")

    # Get feature importances
    print("\n" + "="*70)
    print("Feature Importances (top 20)")
    print("="*70)

    term_names = ebm.term_names_
    term_scores = ebm.term_importances()

    # Sort by importance
    sorted_idx = np.argsort(term_scores)[::-1]

    for i in sorted_idx[:20]:
        print(f"  {term_scores[i]:8.4f}  {term_names[i]}")

    # Focus on interactions
    print("\n" + "="*70)
    print("Interactions Only")
    print("="*70)
    for i in sorted_idx:
        name = term_names[i]
        if " x " in name:
            print(f"  {term_scores[i]:8.4f}  {name}")

    # Detailed look at top interactions
    print("\n" + "="*70)
    print("Detailed Interaction Analysis")
    print("="*70)

    # Get global explanations
    global_exp = ebm.explain_global()

    for i in sorted_idx[:30]:
        name = term_names[i]
        if " x " in name:
            print(f"\n{name}: importance={term_scores[i]:.4f}")
            # Get the contribution shape
            term_idx = list(term_names).index(name)
            scores = ebm.term_scores_[term_idx]
            print(f"  Shape: {scores.shape}")
            print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    return ebm


if __name__ == "__main__":
    ebm = run_ebm()

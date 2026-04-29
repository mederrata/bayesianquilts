#!/usr/bin/env python3
"""Aggregate UCI benchmark results and generate LaTeX table."""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


DATASET_ORDER = [
    "german", "adult", "bank", "taiwan", "heart",
    "bioresponse", "spambase", "mushroom", "phoneme", "electricity"
]

DATASET_NAMES = {
    "german": "German Credit",
    "adult": "Adult",
    "bank": "Bank Marketing",
    "taiwan": "Taiwan Credit",
    "heart": "Heart Disease",
    "bioresponse": "Bioresponse",
    "spambase": "Spambase",
    "mushroom": "Mushroom",
    "phoneme": "Phoneme",
    "electricity": "Electricity",
}

METHOD_ORDER = ["LR", "random_forest", "xgboost", "lightgbm", "mlp", "ebm", "sparse"]


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all results from JSON files."""
    all_data = []
    for json_file in results_dir.glob("*/results.json"):
        dataset = json_file.parent.name
        with open(json_file) as f:
            data = json.load(f)
        for row in data:
            row["dataset"] = dataset
            all_data.append(row)
    return pd.DataFrame(all_data)


def aggregate_by_method(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by dataset and method."""
    agg = df.groupby(["dataset", "method"]).agg({
        "test_auc": ["mean", "std"],
        "test_accuracy": ["mean", "std"],
    }).reset_index()
    agg.columns = ["dataset", "method", "auc_mean", "auc_std", "acc_mean", "acc_std"]
    return agg


def format_cell(mean: float, std: float, is_best: bool = False, is_second: bool = False) -> str:
    """Format a table cell with mean±std."""
    val = f"{mean:.3f}"
    if is_best:
        return f"\\textbf{{{val}}}"
    elif is_second:
        return f"\\underline{{{val}}}"
    return val


def generate_latex_table(agg: pd.DataFrame) -> str:
    """Generate LaTeX table from aggregated results."""
    lines = []

    for dataset in DATASET_ORDER:
        if dataset not in agg["dataset"].values:
            continue

        subset = agg[agg["dataset"] == dataset]

        row_values = {}
        for _, row in subset.iterrows():
            method = row["method"]
            if method in METHOD_ORDER:
                row_values[method] = row["auc_mean"]

        # Find best and second-best
        sorted_methods = sorted(row_values.items(), key=lambda x: x[1], reverse=True)
        best = sorted_methods[0][0] if sorted_methods else None
        second = sorted_methods[1][0] if len(sorted_methods) > 1 else None

        # Format cells
        cells = []
        for method in METHOD_ORDER:
            if method in row_values:
                is_best = (method == best)
                is_second = (method == second)
                cells.append(format_cell(row_values[method], 0, is_best, is_second))
            else:
                cells.append("-")

        # Generate row
        name = DATASET_NAMES.get(dataset, dataset)
        line = f"        {name} & " + " & ".join(cells) + " \\\\"
        lines.append(line)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate UCI benchmark results")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory containing result subdirectories")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for LaTeX table (default: stdout)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist")
        return

    df = load_results(results_dir)
    if df.empty:
        print("No results found")
        return

    print(f"Loaded {len(df)} result rows from {df['dataset'].nunique()} datasets")

    agg = aggregate_by_method(df)

    # Print summary
    print("\n=== Summary by Dataset ===")
    for dataset in DATASET_ORDER:
        subset = agg[agg["dataset"] == dataset]
        if subset.empty:
            continue
        print(f"\n{DATASET_NAMES.get(dataset, dataset)}:")
        for _, row in subset.sort_values("auc_mean", ascending=False).iterrows():
            print(f"  {row['method']:20s}: {row['auc_mean']:.4f} ± {row['auc_std']:.4f}")

    # Generate LaTeX
    latex = generate_latex_table(agg)
    print("\n=== LaTeX Table Rows ===")
    print(latex)

    if args.output:
        with open(args.output, "w") as f:
            f.write(latex)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

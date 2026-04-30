#!/usr/bin/env python3
"""
Run all NeurIPS experiments for generalization-preserving regularization.

Usage:
    python run_all.py --quick           # Quick test run
    python run_all.py --output_dir results/  # Full experiment suite

Reference:
    Chang (2025), "A renormalization-group inspired hierarchical Bayesian
    framework for piecewise linear regression models"
"""

import argparse
import os
import sys
from pathlib import Path


def run_all_experiments(output_dir: str, quick: bool = False):
    """Run all experiments in sequence."""
    from . import (
        synthetic_validation,
        rg_flow_verification,
        uci_benchmarks,
        hierarchical_prediction,
        ablation_studies,
        interpretability_demo,
        scalability,
    )

    experiments = [
        ("Synthetic Validation", synthetic_validation, "synthetic_validation"),
        ("RG Flow Verification", rg_flow_verification, "rg_flow"),
        ("UCI Benchmarks", uci_benchmarks, "uci_benchmarks"),
        ("Hierarchical Prediction", hierarchical_prediction, "hierarchical_prediction"),
        ("Ablation Studies", ablation_studies, "ablation_studies"),
        ("Interpretability Demo", interpretability_demo, "interpretability"),
        ("Scalability Analysis", scalability, "scalability"),
    ]

    results_summary = {}

    for name, module, subdir in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}\n")

        exp_output_dir = str(Path(output_dir) / subdir)

        try:
            if hasattr(module, "ExperimentConfig"):
                if quick:
                    if name == "Synthetic Validation":
                        config = module.ExperimentConfig(
                            n_replications=5,
                            n_obs_values=[1000, 5000],
                            rho_values=[0.3, 0.5],
                            output_dir=exp_output_dir,
                        )
                    elif name == "RG Flow Verification":
                        config = module.ExperimentConfig(
                            n_replications=5,
                            effect_sizes=[0.3, 0.7],
                            output_dir=exp_output_dir,
                        )
                    elif name == "UCI Benchmarks":
                        config = module.ExperimentConfig(
                            n_folds=3,
                            n_replications=2,
                            max_order=1,
                            datasets=["german"],
                            output_dir=exp_output_dir,
                        )
                    elif name == "Hierarchical Prediction":
                        config = module.ExperimentConfig(
                            n_replications=3,
                            max_order=1,
                            n_users_subsample=200,
                            n_items_subsample=100,
                            output_dir=exp_output_dir,
                        )
                    elif name == "Ablation Studies":
                        config = module.ExperimentConfig(
                            n_replications=5,
                            max_order=2,
                            c_values=[0.25, 0.5, 1.0],
                            noise_misspec_factors=[0.75, 1.0, 1.5],
                            output_dir=exp_output_dir,
                        )
                    elif name == "Scalability Analysis":
                        config = module.ExperimentConfig(
                            n_obs_values=[1000, 10000],
                            d_factors_values=[2, 3, 4],
                            L_levels_values=[3, 5, 10],
                            output_dir=exp_output_dir,
                        )
                    else:
                        config = module.ExperimentConfig(output_dir=exp_output_dir)
                else:
                    config = module.ExperimentConfig(output_dir=exp_output_dir)

                module.run_full_experiment(config)
            elif name == "Interpretability Demo":
                module.run_interpretability_demo(exp_output_dir)
            else:
                print(f"Warning: Unknown experiment structure for {name}")
                continue

            results_summary[name] = "Success"
            print(f"\n{name}: Completed successfully")

        except Exception as e:
            results_summary[name] = f"Failed: {str(e)}"
            print(f"\n{name}: Failed with error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for name, status in results_summary.items():
        print(f"  {name}: {status}")

    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all NeurIPS experiments for generalization-preserving regularization"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Base output directory for all experiment results"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick versions of all experiments"
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        choices=[
            "synthetic", "rg_flow", "uci", "hierarchical",
            "ablation", "interpretability", "scalability"
        ],
        help="Run only a specific experiment"
    )
    args = parser.parse_args()

    if args.experiment:
        from . import (
            synthetic_validation,
            rg_flow_verification,
            uci_benchmarks,
            hierarchical_prediction,
            ablation_studies,
            interpretability_demo,
            scalability,
        )

        exp_map = {
            "synthetic": (synthetic_validation, "synthetic_validation"),
            "rg_flow": (rg_flow_verification, "rg_flow"),
            "uci": (uci_benchmarks, "uci_benchmarks"),
            "hierarchical": (hierarchical_prediction, "hierarchical_prediction"),
            "ablation": (ablation_studies, "ablation_studies"),
            "interpretability": (interpretability_demo, "interpretability"),
            "scalability": (scalability, "scalability"),
        }

        module, subdir = exp_map[args.experiment]
        exp_output_dir = str(Path(args.output_dir) / subdir)

        if args.experiment == "interpretability":
            module.run_interpretability_demo(exp_output_dir)
        else:
            if args.quick:
                sys.argv = ["", "--quick", "--output_dir", exp_output_dir]
            else:
                sys.argv = ["", "--output_dir", exp_output_dir]
            module.main()
    else:
        run_all_experiments(args.output_dir, args.quick)


if __name__ == "__main__":
    main()

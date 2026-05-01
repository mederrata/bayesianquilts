"""NeurIPS experiments for generalization-preserving regularization.

Experiments:
    1. synthetic_validation: Validates generalization-preserving scaling on synthetic data
    2. rg_flow_verification: Verifies RG flow interpretation and fixed point prediction
    3. uci_benchmarks: Benchmarks on UCI classification datasets
    4. hierarchical_prediction: Real-world hierarchical prediction (MovieLens)
    5. ablation_studies: Systematic ablation of regularization components
    6. interpretability_demo: Demonstrates interpretability benefits
    7. scalability: Computational scalability analysis

Usage:
    python -m bayesianquilts.examples.neurips_experiments.synthetic_validation --quick
    python -m bayesianquilts.examples.neurips_experiments.run_all --quick
"""

from . import synthetic_validation
from . import rg_flow_verification
from . import uci_benchmarks
from . import hierarchical_prediction
from . import ablation_studies
from . import interpretability_demo
from . import scalability

__all__ = [
    "synthetic_validation",
    "rg_flow_verification",
    "uci_benchmarks",
    "hierarchical_prediction",
    "ablation_studies",
    "interpretability_demo",
    "scalability",
]

# forbidden_sets/experiments/__init__.py
"""
Example experiments: Reproduce the paper's key results.

This module provides ready-to-run experiments that reproduce
the key findings from the research papers:

Experiments A-C (from Paper 1 - "Learning by Elimination"):
- Experiment A: Polynomial memory growth without aliasing
- Experiment B: Exponential failure under aliasing
- Experiment C: Recovery via history augmentation

Experiments D-H (from Paper 2 - "Constraint Accumulation"):
- Experiment D: Representation sufficiency
- Experiment E: Diameter scaling
- Experiment F: False-positive safety
- Experiment G: Comparison with R-MAX (conceptual)
- Experiment H: Robustness across seeds

All experiments are deterministic and reproducible.
"""

from forbidden_sets.experiments.polynomial_growth import (
    run_polynomial_scaling_experiment,
    ExperimentA,
)
from forbidden_sets.experiments.aliasing_stress import (
    run_aliasing_stress_experiment,
    ExperimentB,
)
from forbidden_sets.experiments.history_recovery import (
    run_history_recovery_experiment,
    ExperimentC,
)
from forbidden_sets.experiments.representation import (
    run_representation_experiment,
    run_diameter_scaling_experiment,
    run_safety_experiment,
    run_robustness_experiment,
    ExperimentD,
    ExperimentE,
    ExperimentF,
    ExperimentH,
)

__all__ = [
    # Paper 1 experiments
    "run_polynomial_scaling_experiment",
    "run_aliasing_stress_experiment",
    "run_history_recovery_experiment",
    "ExperimentA",
    "ExperimentB",
    "ExperimentC",
    # Paper 2 experiments
    "run_representation_experiment",
    "run_diameter_scaling_experiment",
    "run_safety_experiment",
    "run_robustness_experiment",
    "ExperimentD",
    "ExperimentE",
    "ExperimentF",
    "ExperimentH",
]

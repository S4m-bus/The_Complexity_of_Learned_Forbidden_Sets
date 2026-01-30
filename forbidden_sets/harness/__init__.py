# forbidden_sets/harness/__init__.py
"""
Experimental harness: Tools for running controlled experiments.

This module provides:
- DeterministicRollout: Execute deterministic episodes
- ExperimentRunner: Run complete experiments with metric collection
- ExperimentConfig: Configuration for experiments
- Plotting utilities for log-log analysis

All experiments are designed for reproducibility and determinism.
"""

from forbidden_sets.harness.rollout import DeterministicRollout, EpisodeResult
from forbidden_sets.harness.experiment import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
)
from forbidden_sets.harness.plotting import (
    plot_memory_growth,
    plot_aliasing_comparison,
    plot_feasibility_curve,
)

__all__ = [
    "DeterministicRollout",
    "EpisodeResult",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    "plot_memory_growth",
    "plot_aliasing_comparison",
    "plot_feasibility_curve",
]

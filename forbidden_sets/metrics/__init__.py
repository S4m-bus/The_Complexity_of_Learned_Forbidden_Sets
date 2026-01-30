# forbidden_sets/metrics/__init__.py
"""
Metrics module: Measurement tools for theoretical analysis.

This module provides metrics designed for testing theoretical claims:
- ConstraintSizeTracker: Track |F| growth over episodes
- FalseEliminationTracker: Detect incorrect eliminations
- FeasibilityTracker: Track success/failure rates
- SeedInvarianceChecker: Verify determinism across seeds

IMPORTANT: This module deliberately does NOT track:
- Reward curves
- Regret
- Value function accuracy

These would be inappropriate for elimination-based learning.
"""

from forbidden_sets.metrics.constraint_size import ConstraintSizeTracker
from forbidden_sets.metrics.false_elimination import FalseEliminationTracker
from forbidden_sets.metrics.feasibility import FeasibilityTracker
from forbidden_sets.metrics.seed_invariance import SeedInvarianceChecker

__all__ = [
    "ConstraintSizeTracker",
    "FalseEliminationTracker",
    "FeasibilityTracker",
    "SeedInvarianceChecker",
]

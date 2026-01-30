# forbidden_sets/core/__init__.py
"""
Core module: Types, invariants, and error handling.

This module provides the fundamental building blocks that enforce
theoretical constraints throughout the library.
"""

from forbidden_sets.core.types import (
    State,
    Observation,
    Action,
    Transition,
    ForbiddenPair,
    Trajectory,
    EpisodeOutcome,
)

from forbidden_sets.core.errors import (
    TheoreticalViolationError,
    StochasticTransitionError,
    ValueEstimationAttemptError,
    HiddenMemoryError,
    NonMonotonicUpdateError,
)

from forbidden_sets.core.invariants import (
    enforce_determinism,
    enforce_policy_determinism,
    enforce_monotonicity,
    forbid_value_estimation,
    forbid_stochastic_reward,
)

__all__ = [
    # Types
    "State",
    "Observation",
    "Action",
    "Transition",
    "ForbiddenPair",
    "Trajectory",
    "EpisodeOutcome",
    # Errors
    "TheoreticalViolationError",
    "StochasticTransitionError",
    "ValueEstimationAttemptError",
    "HiddenMemoryError",
    "NonMonotonicUpdateError",
    # Invariants
    "enforce_determinism",
    "enforce_policy_determinism",
    "enforce_monotonicity",
    "forbid_value_estimation",
    "forbid_stochastic_reward",
]

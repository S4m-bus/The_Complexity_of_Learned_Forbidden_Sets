# forbidden_sets/__init__.py
"""
Learned Forbidden Sets: A Theoretical RL Research Library

This library provides infrastructure for experimentally isolating structural
causes of exponential complexity in deterministic reinforcement learning
under partial observability.

This is a THEORY-FIRST library designed for:
- Testing impossibility and tractability claims
- Measuring memory growth (not reward maximization)
- Demonstrating polynomial vs exponential behavior

NON-NEGOTIABLE CONSTRAINTS (structurally enforced):
- Deterministic transitions only
- Deterministic policies only
- No stochastic rewards
- No value functions (Q, V, advantage)
- No probabilistic belief states
- No neural networks or function approximation
- No exploration bonuses, optimism, or entropy
- No hidden memory (all memory must be explicit)

Example:
    >>> from forbidden_sets import CorridorMDP, ForbiddenSetAgent, ExperimentRunner
    >>> env = CorridorMDP(diameter=20, num_actions=4)
    >>> agent = ForbiddenSetAgent()
    >>> runner = ExperimentRunner()
    >>> result = runner.run(env, agent, num_episodes=100)
    >>> print(f"Forbidden set size: {result.final_forbidden_set_size}")
"""

__version__ = "0.1.0"
__author__ = "Theoretical RL Research"

# Core types
from forbidden_sets.core.types import (
    State,
    Observation,
    Action,
    Transition,
    ForbiddenPair,
    Trajectory,
    EpisodeOutcome,
)

# Errors
from forbidden_sets.core.errors import (
    TheoreticalViolationError,
    StochasticTransitionError,
    ValueEstimationAttemptError,
    HiddenMemoryError,
    NonMonotonicUpdateError,
)

# Environments
from forbidden_sets.environments.base import DeterministicMDP
from forbidden_sets.environments.corridor import CorridorMDP
from forbidden_sets.environments.conflicting_graph import ConflictingGraphMDP
from forbidden_sets.environments.observation import ObservationMapping

# Agents
from forbidden_sets.agents.base import Agent
from forbidden_sets.agents.stateless import StatelessAgent
from forbidden_sets.agents.history import FiniteHistoryAgent
from forbidden_sets.agents.forbidden import ForbiddenSetAgent

# Metrics
from forbidden_sets.metrics.constraint_size import ConstraintSizeTracker
from forbidden_sets.metrics.false_elimination import FalseEliminationTracker
from forbidden_sets.metrics.feasibility import FeasibilityTracker
from forbidden_sets.metrics.seed_invariance import SeedInvarianceChecker

# Harness
from forbidden_sets.harness.rollout import DeterministicRollout
from forbidden_sets.harness.experiment import ExperimentRunner, ExperimentConfig, ExperimentResult

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
    # Environments
    "DeterministicMDP",
    "CorridorMDP",
    "ConflictingGraphMDP",
    "ObservationMapping",
    # Agents
    "Agent",
    "StatelessAgent",
    "FiniteHistoryAgent",
    "ForbiddenSetAgent",
    # Metrics
    "ConstraintSizeTracker",
    "FalseEliminationTracker",
    "FeasibilityTracker",
    "SeedInvarianceChecker",
    # Harness
    "DeterministicRollout",
    "ExperimentRunner",
    "ExperimentConfig",
    "ExperimentResult",
]

# forbidden_sets/core/types.py
"""
Fundamental type definitions for the theoretical RL library.

All types are designed to:
1. Express theoretical concepts explicitly
2. Enable structural enforcement of constraints
3. Be fully inspectable (no hidden state)

These types form the vocabulary of the research framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NewType, Tuple, List, FrozenSet, Optional


# =============================================================================
# Primitive Types
# =============================================================================

# State: The latent/true state of the environment (not directly observed)
# This is the "ground truth" that may be hidden by aliasing
State = NewType('State', int)

# Observation: What the agent actually sees (may be aliased)
# Multiple distinct States may map to the same Observation
Observation = NewType('Observation', int)

# Action: A discrete action the agent can take
# The action space is finite and explicitly enumerable
Action = NewType('Action', int)


# =============================================================================
# Composite Types
# =============================================================================

@dataclass(frozen=True, slots=True)
class Transition:
    """
    A deterministic transition: (s, a) → s'
    
    This is the fundamental unit of environment dynamics.
    Frozen to ensure immutability (transitions cannot change).
    
    Attributes:
        state: The source state
        action: The action taken
        next_state: The resulting state (deterministic)
    """
    state: State
    action: Action
    next_state: State
    
    def __repr__(self) -> str:
        return f"({self.state}, {self.action}) → {self.next_state}"


@dataclass(frozen=True, slots=True)
class ForbiddenPair:
    """
    An observation-action pair that has been forbidden.
    
    Once an (observation, action) pair is added to the forbidden set,
    it can never be removed (monotonicity constraint).
    
    We record additional metadata for analysis:
    - When it was forbidden (episode number)
    - The true state that caused the failure
    
    This enables detection of false eliminations (when the action was
    actually optimal for some state that maps to this observation).
    
    Attributes:
        observation: The observed state when failure occurred
        action: The action that led to failure
        episode_forbidden: Episode number when this was forbidden
        true_state: The actual latent state (for analysis only)
    """
    observation: Observation
    action: Action
    episode_forbidden: int
    true_state: State
    
    def __repr__(self) -> str:
        return f"Forbidden(o={self.observation}, a={self.action}, ep={self.episode_forbidden})"


class EpisodeOutcome(Enum):
    """
    Possible outcomes of an episode.
    
    We explicitly distinguish:
    - SUCCESS: Agent reached a goal state
    - FAILURE: Agent reached a terminal failure state
    - INFEASIBLE: No legal actions available (all forbidden)
    
    Note: There is no "timeout" or "truncation" — episodes are 
    deterministic and always terminate.
    """
    SUCCESS = auto()
    FAILURE = auto()
    INFEASIBLE = auto()


@dataclass(frozen=True, slots=True)
class Step:
    """
    A single step in a trajectory.
    
    Records the complete information at each step for analysis.
    
    Attributes:
        state: True latent state
        observation: What the agent observed
        action: Action taken
        next_state: Resulting state
    """
    state: State
    observation: Observation
    action: Action
    next_state: State


@dataclass(slots=True)
class Trajectory:
    """
    A complete episode trajectory.
    
    Contains the full sequence of steps and the final outcome.
    This is mutable during episode execution but should be
    frozen after completion.
    
    Attributes:
        steps: List of steps taken
        outcome: How the episode ended
        start_state: Initial state
    """
    start_state: State
    steps: List[Step] = field(default_factory=list)
    outcome: Optional[EpisodeOutcome] = None
    
    def add_step(self, step: Step) -> None:
        """Add a step to the trajectory."""
        self.steps.append(step)
    
    def finalize(self, outcome: EpisodeOutcome) -> None:
        """Mark the trajectory as complete with given outcome."""
        self.outcome = outcome
    
    @property
    def length(self) -> int:
        """Number of steps taken."""
        return len(self.steps)
    
    @property
    def final_state(self) -> Optional[State]:
        """The last state reached, if any steps were taken."""
        if self.steps:
            return self.steps[-1].next_state
        return self.start_state
    
    @property
    def observations(self) -> Tuple[Observation, ...]:
        """All observations in order."""
        return tuple(step.observation for step in self.steps)
    
    @property
    def actions(self) -> Tuple[Action, ...]:
        """All actions in order."""
        return tuple(step.action for step in self.steps)
    
    @property
    def state_action_pairs(self) -> FrozenSet[Tuple[State, Action]]:
        """All unique (state, action) pairs visited."""
        return frozenset((step.state, step.action) for step in self.steps)
    
    @property
    def observation_action_pairs(self) -> FrozenSet[Tuple[Observation, Action]]:
        """All unique (observation, action) pairs visited."""
        return frozenset((step.observation, step.action) for step in self.steps)


# =============================================================================
# History Keys (for finite-history agents)
# =============================================================================

# A history key is a tuple of past observations
# The length determines the "memory depth" of the agent
HistoryKey = Tuple[Observation, ...]


def make_history_key(observations: List[Observation], depth: int) -> HistoryKey:
    """
    Create a history key from recent observations.
    
    Args:
        observations: List of observations (most recent last)
        depth: How many observations to include
        
    Returns:
        A tuple of the last `depth` observations (or fewer if not enough)
    """
    if depth <= 0:
        return ()
    return tuple(observations[-depth:]) if observations else ()

# forbidden_sets/metrics/false_elimination.py
"""
False elimination tracking: Detect incorrect constraint additions.

A false elimination occurs when forbidding (observation, action)
eliminates an action that is optimal for at least one state
that maps to that observation.

This is a critical failure mode under aliasing:
- State s1 and s2 both map to observation o
- Action a is optimal for s1, action b is optimal for s2
- If the agent fails in s2 with action a, it forbids (o, a)
- This prevents optimal behavior in s1 forever

This tracker identifies and counts such false eliminations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Tuple, FrozenSet, TYPE_CHECKING

from forbidden_sets.core.types import (
    Observation,
    Action,
    State,
    ForbiddenPair,
)

if TYPE_CHECKING:
    from forbidden_sets.environments.base import DeterministicMDP
    from forbidden_sets.environments.observation import ObservationMapping


@dataclass
class FalseEliminationRecord:
    """Record of a single false elimination."""
    observation: Observation
    action: Action
    forbidden_at_episode: int
    caused_by_state: State           # State where failure occurred
    optimal_for_states: FrozenSet[State]  # States where action was optimal


@dataclass
class FalseEliminationTracker:
    """
    Track false eliminations during learning.
    
    A false elimination is when forbidding (o, a) eliminates an
    action that was optimal for at least one state in φ^{-1}(o).
    
    This is a permanent, irrecoverable error that prevents the
    agent from ever learning the optimal policy.
    
    Example:
        >>> tracker = FalseEliminationTracker(env, obs_mapping)
        >>> # After agent forbids a pair:
        >>> is_false = tracker.check_elimination(forbidden_pair)
        >>> print(tracker.false_elimination_rate)
    """
    
    # Records of false eliminations
    false_eliminations: List[FalseEliminationRecord] = field(default_factory=list)
    
    # All eliminations (both true and false)
    total_eliminations: int = 0
    
    # Cache of optimal actions per state
    _optimal_cache: dict = field(default_factory=dict, repr=False)
    
    def check_and_record(
        self,
        forbidden_pair: ForbiddenPair,
        env: "DeterministicMDP",
        obs_mapping: "ObservationMapping"
    ) -> bool:
        """
        Check if a newly forbidden pair is a false elimination.
        
        Args:
            forbidden_pair: The pair being forbidden
            env: The environment
            obs_mapping: The observation mapping
            
        Returns:
            True if this is a false elimination
        """
        self.total_eliminations += 1
        
        obs = forbidden_pair.observation
        action = forbidden_pair.action
        caused_by = forbidden_pair.true_state
        
        # Get all states that map to this observation
        aliased_states = obs_mapping.get_aliased_states(obs)
        
        # Find states where this action is optimal
        optimal_for: Set[State] = set()
        for state in aliased_states:
            if not env.is_terminal(state):
                if env.is_optimal_action(state, action):
                    optimal_for.add(state)
        
        # If action is optimal for any aliased state, it's a false elimination
        if optimal_for:
            record = FalseEliminationRecord(
                observation=obs,
                action=action,
                forbidden_at_episode=forbidden_pair.episode_forbidden,
                caused_by_state=caused_by,
                optimal_for_states=frozenset(optimal_for),
            )
            self.false_eliminations.append(record)
            return True
        
        return False
    
    def check_elimination(
        self,
        observation: Observation,
        action: Action,
        env: "DeterministicMDP",
        obs_mapping: "ObservationMapping"
    ) -> bool:
        """
        Check if forbidding (obs, action) would be a false elimination.
        
        This is a simpler interface that doesn't record.
        
        Args:
            observation: The observation
            action: The action
            env: The environment
            obs_mapping: The observation mapping
            
        Returns:
            True if this would be a false elimination
        """
        aliased_states = obs_mapping.get_aliased_states(observation)
        
        for state in aliased_states:
            if not env.is_terminal(state):
                if env.is_optimal_action(state, action):
                    return True
        
        return False
    
    @property
    def false_elimination_count(self) -> int:
        """Number of false eliminations."""
        return len(self.false_eliminations)
    
    @property
    def true_elimination_count(self) -> int:
        """Number of correct eliminations."""
        return self.total_eliminations - len(self.false_eliminations)
    
    @property
    def false_elimination_rate(self) -> float:
        """
        Fraction of eliminations that are false.
        
        0.0 means no false eliminations (ideal)
        1.0 means all eliminations were false (worst case)
        """
        if self.total_eliminations == 0:
            return 0.0
        return len(self.false_eliminations) / self.total_eliminations
    
    def get_affected_states(self) -> FrozenSet[State]:
        """
        Get all states affected by false eliminations.
        
        These are states where optimal behavior is now impossible.
        """
        affected: Set[State] = set()
        for record in self.false_eliminations:
            affected.update(record.optimal_for_states)
        return frozenset(affected)
    
    def get_affected_observations(self) -> FrozenSet[Observation]:
        """Get observations that have false eliminations."""
        return frozenset(r.observation for r in self.false_eliminations)
    
    def is_policy_feasible(
        self,
        env: "DeterministicMDP",
        obs_mapping: "ObservationMapping"
    ) -> bool:
        """
        Check if all states still have their optimal action available.
        
        Returns:
            True if no false eliminations have occurred
        """
        return len(self.false_eliminations) == 0
    
    def summary(self) -> str:
        """Generate a summary string."""
        return (
            f"FalseEliminationTracker:\n"
            f"  Total eliminations: {self.total_eliminations}\n"
            f"  False eliminations: {self.false_elimination_count}\n"
            f"  True eliminations: {self.true_elimination_count}\n"
            f"  False rate: {self.false_elimination_rate:.2%}\n"
            f"  Affected states: {len(self.get_affected_states())}"
        )
    
    def reset(self) -> None:
        """Reset all tracking."""
        self.false_eliminations.clear()
        self.total_eliminations = 0


def compute_aliasing_conflict_potential(
    env: "DeterministicMDP",
    obs_mapping: "ObservationMapping"
) -> int:
    """
    Count the number of observation-action pairs that would
    be false eliminations if forbidden.
    
    This measures the "conflict potential" of an aliasing scheme.
    Higher values mean more opportunities for false elimination.
    
    Args:
        env: The environment
        obs_mapping: The observation mapping
        
    Returns:
        Number of (observation, action) pairs that are optimal
        for at least one aliased state
    """
    conflict_pairs: Set[Tuple[Observation, Action]] = set()
    
    for state in env.states:
        if env.is_terminal(state):
            continue
        
        obs = obs_mapping.observe(state)
        optimal = env.optimal_action(state)
        
        # This pair is a potential false elimination target
        conflict_pairs.add((obs, optimal))
    
    return len(conflict_pairs)

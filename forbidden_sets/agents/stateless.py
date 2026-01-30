# forbidden_sets/agents/stateless.py
"""
Stateless agent: A reactive agent with no persistent memory.

This agent represents the baseline case where the policy is
purely reactive: π(o) depends only on the current observation,
with no memory of past observations or actions.

Key properties:
- Cannot disambiguate aliased observations
- Selects actions deterministically based on current forbidden set
- Memory is limited to the forbidden set itself
"""

from __future__ import annotations

from typing import FrozenSet, Optional, Set, Tuple, Dict, TYPE_CHECKING

from forbidden_sets.core.types import (
    Observation, 
    Action, 
    Trajectory, 
    EpisodeOutcome,
    ForbiddenPair,
)
from forbidden_sets.agents.base import Agent

if TYPE_CHECKING:
    from forbidden_sets.environments.base import DeterministicMDP
    from forbidden_sets.environments.observation import ObservationMapping


class StatelessAgent(Agent):
    """
    A stateless (reactive) agent using elimination-based learning.
    
    This agent:
    - Maintains a forbidden set F ⊆ O × A
    - Selects the lowest-indexed non-forbidden action (deterministic)
    - On failure, forbids the (observation, action) pair that failed
    
    This is the simplest elimination agent. It demonstrates:
    - Polynomial memory when representation is sufficient
    - Exponential blowup when states are aliased
    
    The action selection is deterministic: among non-forbidden actions,
    always select the one with the smallest index. This ensures
    reproducibility across runs.
    """
    
    def __init__(self, actions: FrozenSet[Action]):
        """
        Initialize the stateless agent.
        
        Args:
            actions: The complete action space
        """
        self._actions = actions
        self._sorted_actions = sorted(actions, key=lambda a: int(a))
        
        # The forbidden set: (observation, action) pairs
        self._forbidden: Set[Tuple[Observation, Action]] = set()
        
        # Detailed records for analysis
        self._forbidden_pairs: list[ForbiddenPair] = []
        self._episode_count = 0
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def memory(self) -> FrozenSet[Tuple[Observation, Action]]:
        """The complete forbidden set."""
        return frozenset(self._forbidden)
    
    @property
    def memory_size(self) -> int:
        """|F| - size of forbidden set."""
        return len(self._forbidden)
    
    @property
    def forbidden_set(self) -> FrozenSet[Tuple[Observation, Action]]:
        """Alias for memory - the forbidden set."""
        return self.memory
    
    @property
    def forbidden_pairs_detailed(self) -> list[ForbiddenPair]:
        """Detailed records of all forbidden pairs."""
        return list(self._forbidden_pairs)
    
    # =========================================================================
    # Core Methods
    # =========================================================================
    
    def select_action(
        self,
        observation: Observation,
        available_actions: FrozenSet[Action]
    ) -> Optional[Action]:
        """
        Select the lowest-indexed non-forbidden action.
        
        This is deterministic: same observation + forbidden set
        always produces the same action.
        
        Args:
            observation: Current observation
            available_actions: Actions to consider
            
        Returns:
            The selected action, or None if all forbidden
        """
        for action in self._sorted_actions:
            if action in available_actions:
                if (observation, action) not in self._forbidden:
                    return action
        return None  # All actions forbidden
    
    def update(
        self,
        trajectory: Trajectory,
        outcome: EpisodeOutcome,
        env: "DeterministicMDP",
        obs_mapping: "ObservationMapping"
    ) -> None:
        """
        Update forbidden set after an episode.
        
        On failure: forbid the (observation, action) pair from the
        last step before failure.
        
        Args:
            trajectory: Episode trajectory
            outcome: How episode ended
            env: The environment
            obs_mapping: Observation mapping
        """
        self._episode_count += 1
        
        if outcome == EpisodeOutcome.FAILURE and trajectory.steps:
            # Forbid the last action that led to failure
            last_step = trajectory.steps[-1]
            pair = (last_step.observation, last_step.action)
            
            if pair not in self._forbidden:
                self._forbidden.add(pair)
                
                # Record detailed info
                forbidden_pair = ForbiddenPair(
                    observation=last_step.observation,
                    action=last_step.action,
                    episode_forbidden=self._episode_count,
                    true_state=last_step.state
                )
                self._forbidden_pairs.append(forbidden_pair)
    
    def reset_episode(self) -> None:
        """Reset episodic state (nothing for stateless agent)."""
        pass  # No episodic state to reset
    
    def get_available_actions(
        self,
        observation: Observation,
        all_actions: FrozenSet[Action]
    ) -> FrozenSet[Action]:
        """Get actions not forbidden for this observation."""
        return frozenset(
            a for a in all_actions
            if (observation, a) not in self._forbidden
        )
    
    def is_action_forbidden(
        self,
        observation: Observation,
        action: Action
    ) -> bool:
        """Check if action is forbidden for observation."""
        return (observation, action) in self._forbidden
    
    def get_forbidden_pairs(self) -> FrozenSet[Tuple[Observation, Action]]:
        """Get all forbidden pairs."""
        return self.memory
    
    # =========================================================================
    # Analysis Methods
    # =========================================================================
    
    def get_forbidden_count_per_observation(self) -> Dict[Observation, int]:
        """Count of forbidden actions per observation."""
        counts: Dict[Observation, int] = {}
        for obs, _ in self._forbidden:
            counts[obs] = counts.get(obs, 0) + 1
        return counts
    
    def has_all_actions_forbidden(
        self,
        observation: Observation,
        all_actions: FrozenSet[Action]
    ) -> bool:
        """Check if all actions are forbidden for an observation."""
        return all((observation, a) in self._forbidden for a in all_actions)
    
    def reset(self) -> None:
        """Fully reset the agent (clear all learning)."""
        self._forbidden.clear()
        self._forbidden_pairs.clear()
        self._episode_count = 0
    
    def __repr__(self) -> str:
        return f"StatelessAgent(|F|={self.memory_size})"

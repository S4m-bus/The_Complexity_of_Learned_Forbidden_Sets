# forbidden_sets/agents/history.py
"""
Finite-history agent: Conditions on bounded observation history.

This agent addresses aliasing by using a limited history of past
observations as the memory key. The key insight from the paper is
that even single-step history (o_t, o_{t-1}) can collapse exponential
aliasing back to polynomial memory growth.

Key properties:
- Memory key is a tuple of recent observations
- History depth is explicit and bounded
- Can disambiguate states that differ in recent history
"""

from __future__ import annotations

from typing import FrozenSet, Optional, Set, Tuple, Dict, List, TYPE_CHECKING

from forbidden_sets.core.types import (
    Observation, 
    Action, 
    Trajectory, 
    EpisodeOutcome,
    HistoryKey,
    make_history_key,
)
from forbidden_sets.agents.base import Agent

if TYPE_CHECKING:
    from forbidden_sets.environments.base import DeterministicMDP
    from forbidden_sets.environments.observation import ObservationMapping


class FiniteHistoryAgent(Agent):
    """
    An agent that conditions on finite observation history.
    
    Instead of using just the current observation o as the key,
    this agent uses a tuple of recent observations:
    (o_{t-k}, ..., o_{t-1}, o_t)
    
    This provides limited temporal disambiguation without
    requiring full belief state tracking.
    
    Theoretical Significance:
    - depth=0: Equivalent to StatelessAgent
    - depth=1: Key is (o_{t-1}, o_t), resolves many aliasing cases
    - depth=k: Key is last k+1 observations
    
    The paper shows that depth=1 often suffices to restore
    polynomial memory growth under aliasing.
    """
    
    def __init__(
        self, 
        actions: FrozenSet[Action],
        history_depth: int = 1
    ):
        """
        Initialize the finite-history agent.
        
        Args:
            actions: The complete action space
            history_depth: Number of past observations to include
                          (0 = stateless, 1 = one-step history, etc.)
        """
        if history_depth < 0:
            raise ValueError(f"history_depth must be >= 0, got {history_depth}")
        
        self._actions = actions
        self._sorted_actions = sorted(actions, key=lambda a: int(a))
        self._history_depth = history_depth
        
        # Forbidden set keyed by history tuples
        # Key: (o_{t-k}, ..., o_t) → set of forbidden actions
        self._forbidden: Dict[HistoryKey, Set[Action]] = {}
        
        # Current episode's observation history
        self._current_history: List[Observation] = []
        
        # Episode tracking
        self._episode_count = 0
        
        # Detailed records for analysis
        self._forbidden_records: list[Tuple[HistoryKey, Action, int]] = []
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def history_depth(self) -> int:
        """The history depth k."""
        return self._history_depth
    
    @property
    def memory(self) -> Dict[HistoryKey, FrozenSet[Action]]:
        """
        The complete forbidden set structure.
        
        Returns a dict mapping history keys to their forbidden actions.
        """
        return {
            key: frozenset(actions) 
            for key, actions in self._forbidden.items()
        }
    
    @property
    def memory_size(self) -> int:
        """
        Total number of (history_key, action) pairs forbidden.
        
        This is Σ_{k} |F(k)| where F(k) is the forbidden set for key k.
        """
        return sum(len(actions) for actions in self._forbidden.values())
    
    @property
    def num_history_keys(self) -> int:
        """Number of distinct history keys with forbidden actions."""
        return len(self._forbidden)
    
    # =========================================================================
    # Core Methods
    # =========================================================================
    
    def _get_current_key(self) -> HistoryKey:
        """Get the history key for current state."""
        return make_history_key(self._current_history, self._history_depth + 1)
    
    def select_action(
        self,
        observation: Observation,
        available_actions: FrozenSet[Action]
    ) -> Optional[Action]:
        """
        Select the lowest-indexed non-forbidden action for current history.
        
        Args:
            observation: Current observation (will be added to history)
            available_actions: Actions to consider
            
        Returns:
            Selected action, or None if all forbidden
        """
        # Update history with current observation
        self._current_history.append(observation)
        
        # Get the history key
        key = self._get_current_key()
        
        # Get forbidden actions for this key
        forbidden_for_key = self._forbidden.get(key, set())
        
        # Select lowest-indexed non-forbidden action
        for action in self._sorted_actions:
            if action in available_actions and action not in forbidden_for_key:
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
        
        On failure: forbid the action under its history key.
        """
        self._episode_count += 1
        
        if outcome == EpisodeOutcome.FAILURE and trajectory.steps:
            # Reconstruct the history key at the point of failure
            observations = list(trajectory.observations)
            
            # The key at the last step
            key = make_history_key(observations, self._history_depth + 1)
            
            # The action that failed
            failed_action = trajectory.steps[-1].action
            
            # Add to forbidden set
            if key not in self._forbidden:
                self._forbidden[key] = set()
            
            if failed_action not in self._forbidden[key]:
                self._forbidden[key].add(failed_action)
                self._forbidden_records.append((key, failed_action, self._episode_count))
    
    def reset_episode(self) -> None:
        """Reset episode-specific state."""
        self._current_history = []
    
    def get_available_actions(
        self,
        observation: Observation,
        all_actions: FrozenSet[Action]
    ) -> FrozenSet[Action]:
        """Get actions not forbidden for current history key."""
        # Get the history key (including the given observation)
        temp_history = self._current_history + [observation]
        key = make_history_key(temp_history, self._history_depth + 1)
        
        forbidden_for_key = self._forbidden.get(key, set())
        return frozenset(a for a in all_actions if a not in forbidden_for_key)
    
    def is_action_forbidden_for_key(
        self,
        key: HistoryKey,
        action: Action
    ) -> bool:
        """Check if action is forbidden for a specific history key."""
        return action in self._forbidden.get(key, set())
    
    def get_forbidden_pairs(self) -> FrozenSet[Tuple[HistoryKey, Action]]:
        """Get all forbidden (key, action) pairs."""
        pairs = set()
        for key, actions in self._forbidden.items():
            for action in actions:
                pairs.add((key, action))
        return frozenset(pairs)
    
    # =========================================================================
    # Analysis Methods
    # =========================================================================
    
    def get_key_distribution(self) -> Dict[int, int]:
        """
        Distribution of forbidden actions per key.
        
        Returns:
            Dict mapping count → number of keys with that count
        """
        dist: Dict[int, int] = {}
        for count in (len(a) for a in self._forbidden.values()):
            dist[count] = dist.get(count, 0) + 1
        return dist
    
    def reset(self) -> None:
        """Fully reset the agent."""
        self._forbidden.clear()
        self._forbidden_records.clear()
        self._current_history = []
        self._episode_count = 0
    
    def __repr__(self) -> str:
        return (
            f"FiniteHistoryAgent(depth={self._history_depth}, "
            f"|F|={self.memory_size}, keys={self.num_history_keys})"
        )

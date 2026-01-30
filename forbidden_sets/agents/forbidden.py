# forbidden_sets/agents/forbidden.py
"""
Forbidden-set agent: The core elimination-based learning agent.

This is the primary agent studied in the theoretical framework.
It learns by maintaining and accumulating forbidden (observation, action)
pairs, never estimating values or beliefs.

Key theoretical properties:
- Monotonic: Constraints only accumulate, never removed
- Explicit: All memory is directly inspectable
- No value estimation: Pure elimination-based learning
- Deterministic: Action selection is fully deterministic
"""

from __future__ import annotations

from typing import FrozenSet, Optional, Set, Tuple, Dict, List, Union, TYPE_CHECKING
from dataclasses import dataclass, field

from forbidden_sets.core.types import (
    Observation, 
    Action, 
    Trajectory, 
    EpisodeOutcome,
    ForbiddenPair,
    State,
    HistoryKey,
    make_history_key,
)
from forbidden_sets.core.errors import NonMonotonicUpdateError
from forbidden_sets.agents.base import Agent

if TYPE_CHECKING:
    from forbidden_sets.environments.base import DeterministicMDP
    from forbidden_sets.environments.observation import ObservationMapping


# Type for the memory key: either an observation or a history tuple
MemoryKey = Union[Observation, HistoryKey]


@dataclass
class ForbiddenSetMemory:
    """
    Explicit, inspectable memory structure for forbidden sets.
    
    This dataclass exposes all internal state for analysis.
    Acts as the "memory" property of ForbiddenSetAgent.
    """
    # The core forbidden set: (key, action) pairs
    forbidden_pairs: FrozenSet[Tuple[MemoryKey, Action]] = field(default_factory=frozenset)
    
    # Detailed records of each forbidding event
    detailed_records: Tuple[ForbiddenPair, ...] = field(default_factory=tuple)
    
    # Organized by key for efficient lookup
    by_key: Dict[MemoryKey, FrozenSet[Action]] = field(default_factory=dict)
    
    # Episode when last update occurred
    last_update_episode: int = 0
    
    @property
    def size(self) -> int:
        """Total number of forbidden pairs."""
        return len(self.forbidden_pairs)


class ForbiddenSetAgent(Agent):
    """
    The core elimination-based learning agent.
    
    This agent learns by accumulating forbidden (key, action) pairs.
    The key can be either:
    - An observation (history_depth=0, stateless)
    - A history tuple (history_depth>0, history-augmented)
    
    Core Algorithm:
    1. On each step, select lowest-indexed non-forbidden action
    2. On failure, forbid the (key, action) pair permanently
    3. Repeat until policy is feasible or all actions forbidden
    
    Theoretical Properties:
    - |F| grows polynomially with diameter under sufficient representation
    - |F| can grow exponentially under aliasing without history
    - Single-step history often restores polynomial growth
    
    This agent is the experimental workhorse for testing the paper's claims.
    """
    
    def __init__(
        self,
        actions: FrozenSet[Action],
        history_depth: int = 0
    ):
        """
        Initialize the forbidden-set agent.
        
        Args:
            actions: The complete action space
            history_depth: How much history to include in keys
                          0 = stateless (key = observation)
                          k>0 = use last k+1 observations as key
        """
        if history_depth < 0:
            raise ValueError(f"history_depth must be >= 0, got {history_depth}")
        
        self._actions = actions
        self._sorted_actions = tuple(sorted(actions, key=lambda a: int(a)))
        self._history_depth = history_depth
        
        # Core forbidden set data structures
        self._forbidden_pairs: Set[Tuple[MemoryKey, Action]] = set()
        self._forbidden_by_key: Dict[MemoryKey, Set[Action]] = {}
        
        # Detailed records for analysis
        self._detailed_records: List[ForbiddenPair] = []
        
        # Monotonicity tracking
        self._previous_size = 0
        
        # Episode state
        self._current_history: List[Observation] = []
        self._episode_count = 0
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def memory(self) -> ForbiddenSetMemory:
        """
        Complete, inspectable memory structure.
        
        This is the primary interface for analyzing what the agent learned.
        """
        return ForbiddenSetMemory(
            forbidden_pairs=frozenset(self._forbidden_pairs),
            detailed_records=tuple(self._detailed_records),
            by_key={k: frozenset(v) for k, v in self._forbidden_by_key.items()},
            last_update_episode=self._episode_count,
        )
    
    @property
    def memory_size(self) -> int:
        """|F| - the primary metric for memory complexity."""
        return len(self._forbidden_pairs)
    
    @property
    def forbidden_set_size(self) -> int:
        """Alias for memory_size."""
        return self.memory_size
    
    @property
    def history_depth(self) -> int:
        """The history depth used for keys."""
        return self._history_depth
    
    @property
    def num_keys(self) -> int:
        """Number of distinct keys with forbidden actions."""
        return len(self._forbidden_by_key)
    
    # =========================================================================
    # Key Construction
    # =========================================================================
    
    def _make_key(self, observation: Observation) -> MemoryKey:
        """
        Construct the memory key for current state.
        
        If history_depth=0, key is just the observation.
        Otherwise, key is the tuple of recent observations.
        """
        if self._history_depth == 0:
            return observation
        else:
            # Include the current observation in the key
            full_history = self._current_history + [observation]
            return make_history_key(full_history, self._history_depth + 1)
    
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
        
        Deterministic: same key + forbidden set → same action.
        
        Args:
            observation: Current observation
            available_actions: Actions to choose from
            
        Returns:
            Selected action, or None if all are forbidden
        """
        # Update history
        self._current_history.append(observation)
        
        # Get the memory key
        key = self._make_key(observation)
        
        # Get actions forbidden for this key
        forbidden_for_key = self._forbidden_by_key.get(key, set())
        
        # Select lowest-indexed available, non-forbidden action
        for action in self._sorted_actions:
            if action in available_actions and action not in forbidden_for_key:
                return action
        
        return None  # All actions forbidden for this key
    
    def update(
        self,
        trajectory: Trajectory,
        outcome: EpisodeOutcome,
        env: "DeterministicMDP",
        obs_mapping: "ObservationMapping"
    ) -> None:
        """
        Update the forbidden set after an episode.
        
        On failure: forbid the (key, action) pair that led to failure.
        
        Enforces monotonicity: the forbidden set can only grow.
        """
        self._episode_count += 1
        
        if outcome == EpisodeOutcome.FAILURE and trajectory.steps:
            last_step = trajectory.steps[-1]
            
            # Reconstruct the key at the point of failure
            observations = list(trajectory.observations)
            
            if self._history_depth == 0:
                key: MemoryKey = last_step.observation
            else:
                key = make_history_key(observations, self._history_depth + 1)
            
            # The action that failed
            failed_action = last_step.action
            
            # Check monotonicity before update
            old_size = len(self._forbidden_pairs)
            
            # Add to forbidden set
            pair = (key, failed_action)
            if pair not in self._forbidden_pairs:
                self._forbidden_pairs.add(pair)
                
                if key not in self._forbidden_by_key:
                    self._forbidden_by_key[key] = set()
                self._forbidden_by_key[key].add(failed_action)
                
                # Record detailed info
                record = ForbiddenPair(
                    observation=last_step.observation,
                    action=failed_action,
                    episode_forbidden=self._episode_count,
                    true_state=last_step.state
                )
                self._detailed_records.append(record)
            
            # Verify monotonicity
            new_size = len(self._forbidden_pairs)
            if new_size < old_size:
                raise NonMonotonicUpdateError(
                    observation=last_step.observation,
                    action=failed_action,
                    message="Forbidden set shrank after update!"
                )
            
            self._previous_size = new_size
    
    def reset_episode(self) -> None:
        """Reset episode-specific state."""
        self._current_history = []
    
    def get_available_actions(
        self,
        observation: Observation,
        all_actions: FrozenSet[Action]
    ) -> FrozenSet[Action]:
        """Get actions not forbidden for the current key."""
        key = self._make_key(observation)
        forbidden_for_key = self._forbidden_by_key.get(key, set())
        return frozenset(a for a in all_actions if a not in forbidden_for_key)
    
    def is_action_forbidden(
        self,
        observation: Observation,
        action: Action
    ) -> bool:
        """Check if action is forbidden for the given observation's key."""
        key = self._make_key(observation)
        return action in self._forbidden_by_key.get(key, set())
    
    def get_forbidden_pairs(self) -> FrozenSet[Tuple[MemoryKey, Action]]:
        """Get all forbidden (key, action) pairs."""
        return frozenset(self._forbidden_pairs)
    
    # =========================================================================
    # Analysis Methods
    # =========================================================================
    
    def get_forbidden_count_per_key(self) -> Dict[MemoryKey, int]:
        """Count of forbidden actions per key."""
        return {k: len(v) for k, v in self._forbidden_by_key.items()}
    
    def get_growth_trajectory(self) -> List[Tuple[int, int]]:
        """
        Get the episode → |F| trajectory.
        
        Returns:
            List of (episode, forbidden_set_size) pairs
        """
        trajectory = []
        size = 0
        for i, record in enumerate(self._detailed_records):
            size += 1
            trajectory.append((record.episode_forbidden, size))
        return trajectory
    
    def get_detailed_records(self) -> List[ForbiddenPair]:
        """Get all detailed forbidding records."""
        return list(self._detailed_records)
    
    def has_all_actions_forbidden(self, key: MemoryKey) -> bool:
        """Check if all actions are forbidden for a key."""
        forbidden = self._forbidden_by_key.get(key, set())
        return len(forbidden) >= len(self._actions)
    
    def reset(self) -> None:
        """Fully reset the agent (clear all learning)."""
        self._forbidden_pairs.clear()
        self._forbidden_by_key.clear()
        self._detailed_records.clear()
        self._current_history = []
        self._episode_count = 0
        self._previous_size = 0
    
    def __repr__(self) -> str:
        if self._history_depth == 0:
            return f"ForbiddenSetAgent(|F|={self.memory_size})"
        else:
            return (
                f"ForbiddenSetAgent(depth={self._history_depth}, "
                f"|F|={self.memory_size}, keys={self.num_keys})"
            )

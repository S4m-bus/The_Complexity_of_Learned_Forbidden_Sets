# forbidden_sets/agents/base.py
"""
Base protocol and abstract class for agents.

All agents in this framework must:
1. Have deterministic policies
2. Have explicitly inspectable memory
3. Not perform value estimation

This module defines the interface that all agents must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, FrozenSet, Tuple, Optional, TYPE_CHECKING

from forbidden_sets.core.types import (
    Observation, 
    Action, 
    Trajectory, 
    EpisodeOutcome,
    ForbiddenPair,
)

if TYPE_CHECKING:
    from forbidden_sets.environments.base import DeterministicMDP
    from forbidden_sets.environments.observation import ObservationMapping


class Agent(ABC):
    """
    Abstract base class for all agents.
    
    Enforces:
    - Deterministic action selection
    - Explicit memory inspection
    - Learning only through elimination (no value estimation)
    
    Subclasses must implement:
    - memory: The agent's complete internal state
    - select_action: Deterministic action selection
    - update: Learning from episode outcomes
    - reset_episode: Prepare for new episode
    - available_actions: Actions not yet forbidden
    """
    
    # =========================================================================
    # Abstract Properties
    # =========================================================================
    
    @property
    @abstractmethod
    def memory(self) -> Any:
        """
        The agent's complete internal memory.
        
        This must be:
        - Externally accessible
        - Fully enumerable
        - Contain ALL learned information
        
        Subclasses should return a structure that represents
        everything the agent has learned.
        """
        ...
    
    @property
    @abstractmethod
    def memory_size(self) -> int:
        """
        Size of the agent's memory.
        
        For forbidden-set agents, this is |F|.
        This is the primary metric for memory complexity analysis.
        """
        ...
    
    # =========================================================================
    # Abstract Methods
    # =========================================================================
    
    @abstractmethod
    def select_action(
        self, 
        observation: Observation,
        available_actions: FrozenSet[Action]
    ) -> Optional[Action]:
        """
        Select an action deterministically.
        
        This method MUST be deterministic: same inputs → same output.
        
        Args:
            observation: The current observation
            available_actions: Set of actions to choose from
                             (may be filtered by forbidden set)
        
        Returns:
            A selected action, or None if no actions available
        """
        ...
    
    @abstractmethod
    def update(
        self,
        trajectory: Trajectory,
        outcome: EpisodeOutcome,
        env: "DeterministicMDP",
        obs_mapping: "ObservationMapping"
    ) -> None:
        """
        Update the agent after an episode.
        
        This is where learning happens. For elimination-based agents,
        this adds (observation, action) pairs to the forbidden set.
        
        Args:
            trajectory: The complete episode trajectory
            outcome: How the episode ended
            env: The environment (for optimal action lookup)
            obs_mapping: The observation mapping
        """
        ...
    
    @abstractmethod
    def reset_episode(self) -> None:
        """
        Reset episode-specific state.
        
        Called at the start of each episode.
        Should NOT clear learned knowledge, only episodic state.
        """
        ...
    
    @abstractmethod
    def get_available_actions(
        self,
        observation: Observation,
        all_actions: FrozenSet[Action]
    ) -> FrozenSet[Action]:
        """
        Get actions that are not forbidden for an observation.
        
        Args:
            observation: The current observation
            all_actions: The complete action set
            
        Returns:
            Set of actions still available (not forbidden)
        """
        ...
    
    # =========================================================================
    # Concrete Methods
    # =========================================================================
    
    def is_action_forbidden(
        self,
        observation: Observation,
        action: Action
    ) -> bool:
        """
        Check if an action is forbidden for an observation.
        
        Default implementation uses get_available_actions.
        """
        # This requires knowing all actions, which we may not have
        # Subclasses should override if they have direct forbidden set access
        return False  # Default: nothing is forbidden
    
    def get_forbidden_pairs(self) -> FrozenSet[Tuple[Observation, Action]]:
        """
        Get all currently forbidden (observation, action) pairs.
        
        Default implementation returns empty set.
        Subclasses with forbidden sets should override.
        """
        return frozenset()


class AgentFactory:
    """
    Factory for creating agents with consistent configuration.
    
    This helps ensure experiments use identically configured agents.
    """
    
    @staticmethod
    def create_stateless(actions: FrozenSet[Action]) -> "StatelessAgent":
        """Create a stateless random agent."""
        from forbidden_sets.agents.stateless import StatelessAgent
        return StatelessAgent(actions)
    
    @staticmethod
    def create_forbidden_set(
        actions: FrozenSet[Action],
        history_depth: int = 0
    ) -> "ForbiddenSetAgent":
        """Create a forbidden-set agent."""
        from forbidden_sets.agents.forbidden import ForbiddenSetAgent
        return ForbiddenSetAgent(actions, history_depth=history_depth)
    
    @staticmethod
    def create_history(
        actions: FrozenSet[Action],
        depth: int = 1
    ) -> "FiniteHistoryAgent":
        """Create a finite-history agent."""
        from forbidden_sets.agents.history import FiniteHistoryAgent
        return FiniteHistoryAgent(actions, history_depth=depth)

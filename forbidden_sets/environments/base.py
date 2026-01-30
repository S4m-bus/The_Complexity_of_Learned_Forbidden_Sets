# forbidden_sets/environments/base.py
"""
Base class and protocol for deterministic MDPs.

This module defines the interface that all environments must implement.
The design enforces:
- Deterministic transitions (no randomness)
- Explicit state enumeration
- Clear terminal state handling
- Inspectable optimal action structure
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple, FrozenSet, Optional

from forbidden_sets.core.types import State, Action, Observation
from forbidden_sets.core.errors import StochasticTransitionError
from forbidden_sets.core.invariants import enforce_determinism


class DeterministicMDP(ABC):
    """
    Abstract base class for deterministic Markov Decision Processes.
    
    Theoretical Model:
    - State space S is finite and explicitly enumerable
    - Action space A is finite
    - Transition T: S × A → S is deterministic
    - Terminal states are explicitly marked (failure or success)
    - Each non-terminal state has exactly one optimal action
    
    This class enforces:
    - No stochastic transitions
    - No reward functions (we track failure/success only)
    - Explicit observability of all structure
    
    Subclasses must implement:
    - states: The complete state space
    - actions: The complete action space
    - initial_state: Starting state for episodes
    - terminal_failures: States that end episodes in failure
    - goal_states: States that end episodes in success
    - _transition_impl: The actual transition logic
    - _optimal_action_impl: The optimal action for each state
    """
    
    def __init__(self):
        """Initialize and verify determinism."""
        self._transition_cache: Dict[Tuple[State, Action], State] = {}
        self._verified_determinism = False
    
    # =========================================================================
    # Abstract Properties (must be implemented by subclasses)
    # =========================================================================
    
    @property
    @abstractmethod
    def states(self) -> FrozenSet[State]:
        """
        The complete state space S.
        
        Must be finite and explicitly enumerable.
        """
        ...
    
    @property
    @abstractmethod
    def actions(self) -> FrozenSet[Action]:
        """
        The complete action space A.
        
        Must be finite and shared across all states.
        """
        ...
    
    @property
    @abstractmethod
    def initial_state(self) -> State:
        """
        The starting state for each episode.
        
        Must be a non-terminal state.
        """
        ...
    
    @property
    @abstractmethod
    def terminal_failures(self) -> FrozenSet[State]:
        """
        States that represent terminal failure.
        
        Entering these states ends the episode with FAILURE outcome.
        """
        ...
    
    @property
    @abstractmethod
    def goal_states(self) -> FrozenSet[State]:
        """
        States that represent successful completion.
        
        Entering these states ends the episode with SUCCESS outcome.
        """
        ...
    
    @abstractmethod
    def _transition_impl(self, state: State, action: Action) -> State:
        """
        Implementation of the transition function.
        
        This must be a pure, deterministic function.
        No randomness is allowed.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Next state (deterministic)
        """
        ...
    
    @abstractmethod
    def _optimal_action_impl(self, state: State) -> Action:
        """
        Return the unique optimal action for a state.
        
        The framework assumes each non-terminal state has exactly
        one optimal action that leads to eventual success.
        
        Args:
            state: The state to query
            
        Returns:
            The unique optimal action
        """
        ...
    
    # =========================================================================
    # Concrete Methods (with enforcement)
    # =========================================================================
    
    @property
    def diameter(self) -> int:
        """
        The diameter of the state-transition graph.
        
        This is the maximum shortest path length between any two states.
        For corridor-like environments, this is approximately |S|.
        """
        # Default: number of states (conservative upper bound)
        return len(self.states)
    
    @property
    def num_actions(self) -> int:
        """Number of actions."""
        return len(self.actions)
    
    @property
    def num_states(self) -> int:
        """Number of states."""
        return len(self.states)
    
    def transition(self, state: State, action: Action) -> State:
        """
        Execute a deterministic transition.
        
        This method wraps _transition_impl with caching and
        verification to ensure determinism is maintained.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Next state (deterministic, cached)
            
        Raises:
            StochasticTransitionError: If non-determinism is detected
        """
        key = (state, action)
        
        if key in self._transition_cache:
            # Verify determinism on cache hit
            new_result = self._transition_impl(state, action)
            cached_result = self._transition_cache[key]
            
            if new_result != cached_result:
                raise StochasticTransitionError(
                    state=state,
                    action=action,
                    observed_next_states=[cached_result, new_result]
                )
            return cached_result
        
        # First call: compute and cache
        result = self._transition_impl(state, action)
        self._transition_cache[key] = result
        return result
    
    def is_terminal(self, state: State) -> bool:
        """Check if a state is terminal (failure or success)."""
        return state in self.terminal_failures or state in self.goal_states
    
    def is_failure(self, state: State) -> bool:
        """Check if a state is a terminal failure."""
        return state in self.terminal_failures
    
    def is_success(self, state: State) -> bool:
        """Check if a state is a goal state."""
        return state in self.goal_states
    
    def optimal_action(self, state: State) -> Action:
        """
        Get the unique optimal action for a state.
        
        Args:
            state: The state to query
            
        Returns:
            The optimal action
            
        Raises:
            ValueError: If state is terminal (no action needed)
        """
        if self.is_terminal(state):
            raise ValueError(f"State {state} is terminal; no action needed.")
        return self._optimal_action_impl(state)
    
    def is_optimal_action(self, state: State, action: Action) -> bool:
        """
        Check if an action is optimal for a state.
        
        Args:
            state: The state to check
            action: The action to verify
            
        Returns:
            True iff action is the optimal action for state
        """
        if self.is_terminal(state):
            return False  # No action is meaningful for terminal states
        return action == self._optimal_action_impl(state)
    
    def get_all_optimal_pairs(self) -> FrozenSet[Tuple[State, Action]]:
        """
        Get all (state, optimal_action) pairs.
        
        This is the complete description of all correct behavior.
        """
        pairs = set()
        for state in self.states:
            if not self.is_terminal(state):
                pairs.add((state, self._optimal_action_impl(state)))
        return frozenset(pairs)
    
    def verify_determinism(self) -> None:
        """
        Verify that all transitions are deterministic.
        
        Calls each transition multiple times to check for consistency.
        This is expensive but provides strong guarantees.
        
        Raises:
            StochasticTransitionError: If any transition is non-deterministic
        """
        if self._verified_determinism:
            return
        
        enforce_determinism(
            transition_function=self._transition_impl,
            states=self.states,
            actions=self.actions,
            num_checks=3
        )
        self._verified_determinism = True
    
    def reset(self) -> State:
        """
        Reset the environment to initial state.
        
        Returns:
            The initial state
        """
        return self.initial_state
    
    def step(self, state: State, action: Action) -> Tuple[State, bool, bool]:
        """
        Take a step in the environment.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Tuple of (next_state, is_terminal, is_success)
        """
        next_state = self.transition(state, action)
        is_terminal = self.is_terminal(next_state)
        is_success = self.is_success(next_state)
        return next_state, is_terminal, is_success

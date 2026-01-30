# forbidden_sets/environments/corridor.py
"""
Corridor MDP: A simple linear environment for scaling experiments.

The corridor is a chain of states 0 → 1 → 2 → ... → N-1
where N-1 is the goal. Each state has K actions, exactly one
of which advances to the next state. All other actions lead
to immediate terminal failure.

This environment is used for:
- Experiment A: Polynomial memory scaling with diameter
- Testing forbidden set growth rates
- Baseline complexity measurements
"""

from __future__ import annotations

from typing import FrozenSet, Optional

from forbidden_sets.core.types import State, Action
from forbidden_sets.environments.base import DeterministicMDP


class CorridorMDP(DeterministicMDP):
    """
    A deterministic corridor environment.
    
    Structure:
    - States: 0, 1, 2, ..., diameter (diameter+2 total with failure state)
    - State 0: Initial state
    - State diameter: Goal state
    - State diameter+1: Terminal failure state
    - Actions: 0, 1, ..., num_actions-1
    
    Transitions:
    - From state s (0 ≤ s < diameter):
      - optimal_action(s): moves to s+1
      - any other action: moves to failure state
    - Goal and failure states are terminal
    
    The optimal action at state s is determined by a fixed pattern
    to ensure the environment is fully deterministic.
    
    Parameters:
        diameter: Length of the corridor (states 0 to diameter-1 before goal)
        num_actions: Number of available actions per state
        
    Example:
        >>> env = CorridorMDP(diameter=10, num_actions=4)
        >>> env.states
        frozenset({0, 1, 2, ..., 11})  # 0-9 corridor, 10 goal, 11 failure
        >>> env.initial_state
        State(0)
        >>> env.goal_states
        frozenset({State(10)})
    """
    
    def __init__(self, diameter: int = 10, num_actions: int = 4):
        """
        Initialize the corridor environment.
        
        Args:
            diameter: Number of steps to reach the goal (D in the paper)
            num_actions: Size of the action space (K in the paper)
            
        Raises:
            ValueError: If diameter < 1 or num_actions < 2
        """
        if diameter < 1:
            raise ValueError(f"diameter must be >= 1, got {diameter}")
        if num_actions < 2:
            raise ValueError(f"num_actions must be >= 2, got {num_actions}")
        
        self._diameter = diameter
        self._num_actions = num_actions
        
        # Precompute state sets
        # States 0 to diameter-1: corridor states
        # State diameter: goal state
        # State diameter+1: failure state
        self._goal_state = State(diameter)
        self._failure_state = State(diameter + 1)
        
        self._state_set = frozenset(
            State(s) for s in range(diameter + 2)
        )
        self._action_set = frozenset(
            Action(a) for a in range(num_actions)
        )
        
        # The optimal action at state s is: s mod num_actions
        # This creates a repeating pattern that's deterministic
        # and allows the corridor to have more states than actions
        
        super().__init__()
    
    # =========================================================================
    # Abstract Property Implementations
    # =========================================================================
    
    @property
    def states(self) -> FrozenSet[State]:
        """All states including goal and failure."""
        return self._state_set
    
    @property
    def actions(self) -> FrozenSet[Action]:
        """All available actions."""
        return self._action_set
    
    @property
    def initial_state(self) -> State:
        """Starting state is always 0."""
        return State(0)
    
    @property
    def terminal_failures(self) -> FrozenSet[State]:
        """The single failure state."""
        return frozenset({self._failure_state})
    
    @property
    def goal_states(self) -> FrozenSet[State]:
        """The single goal state."""
        return frozenset({self._goal_state})
    
    @property
    def diameter(self) -> int:
        """The corridor length (number of steps to goal)."""
        return self._diameter
    
    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================
    
    def _transition_impl(self, state: State, action: Action) -> State:
        """
        Deterministic transition function.
        
        - From corridor state with optimal action: advance
        - From corridor state with wrong action: fail
        - From terminal states: stay in place
        """
        # Terminal states are absorbing
        if state == self._goal_state:
            return self._goal_state
        if state == self._failure_state:
            return self._failure_state
        
        # Corridor states
        s = int(state)
        a = int(action)
        
        # Optimal action for state s is: s mod num_actions
        optimal = s % self._num_actions
        
        if a == optimal:
            # Correct action: advance
            return State(s + 1)
        else:
            # Wrong action: immediate failure
            return self._failure_state
    
    def _optimal_action_impl(self, state: State) -> Action:
        """
        Return the unique optimal action.
        
        At state s, the optimal action is s mod num_actions.
        """
        s = int(state)
        return Action(s % self._num_actions)
    
    # =========================================================================
    # Additional Methods
    # =========================================================================
    
    def get_optimal_sequence(self) -> list[Action]:
        """
        Get the complete sequence of optimal actions.
        
        Returns:
            List of actions from state 0 to goal
        """
        return [self._optimal_action_impl(State(s)) for s in range(self._diameter)]
    
    def distance_to_goal(self, state: State) -> int:
        """
        Number of steps from state to goal (if following optimal policy).
        
        Args:
            state: Current state
            
        Returns:
            Number of optimal steps needed, or -1 if terminal failure
        """
        s = int(state)
        if state == self._failure_state:
            return -1  # No path from failure
        if state == self._goal_state:
            return 0
        return self._diameter - s
    
    def __repr__(self) -> str:
        return f"CorridorMDP(diameter={self._diameter}, num_actions={self._num_actions})"

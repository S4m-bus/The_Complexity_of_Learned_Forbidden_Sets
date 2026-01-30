# forbidden_sets/environments/conflicting_graph.py
"""
Conflicting Graph MDP: An adversarial environment for aliasing stress tests.

This environment is designed to create worst-case scenarios for
elimination-based learning under observation aliasing. The key property
is that aliased states require CONFLICTING optimal actions.

When two states s1, s2 map to the same observation o, but have
different optimal actions a1 ≠ a2, a stateless agent cannot learn
the correct policy for both states.

This environment is used for:
- Experiment B: Aliasing stress test
- Demonstrating exponential blowup under insufficient representation
- Testing false elimination rates
"""

from __future__ import annotations

from typing import FrozenSet, Dict, Tuple

from forbidden_sets.core.types import State, Action
from forbidden_sets.environments.base import DeterministicMDP


class ConflictingGraphMDP(DeterministicMDP):
    """
    A deterministic MDP with adversarial aliasing structure.
    
    This environment creates pairs of states that:
    1. Map to the same observation
    2. Require different optimal actions
    
    Structure:
    - States: 0, 1, ..., 2*num_pairs + 1 (plus goal and failure)
    - Pairs of states (2i, 2i+1) are designed to conflict
    - The environment forms a graph where correct navigation
      requires distinguishing aliased states
    
    The key insight is that this creates a situation where:
    - If aliases are not resolved, at least half the actions
      will be incorrectly forbidden
    - Memory must scale with the number of conflicts
    
    Parameters:
        num_pairs: Number of conflicting state pairs
        num_actions: Number of actions per state
        base_alias_factor: Controls aliasing strength
    """
    
    def __init__(
        self, 
        num_pairs: int = 10, 
        num_actions: int = 4,
        base_alias_factor: int = 2
    ):
        """
        Initialize the conflicting graph environment.
        
        Args:
            num_pairs: Number of state pairs with conflicting optimal actions
            num_actions: Size of action space (at least 2)
            base_alias_factor: How many states share each observation
            
        Raises:
            ValueError: If parameters are invalid
        """
        if num_pairs < 1:
            raise ValueError(f"num_pairs must be >= 1, got {num_pairs}")
        if num_actions < 2:
            raise ValueError(f"num_actions must be >= 2, got {num_actions}")
        if base_alias_factor < 1:
            raise ValueError(f"base_alias_factor must be >= 1, got {base_alias_factor}")
        
        self._num_pairs = num_pairs
        self._num_actions = num_actions
        self._alias_factor = base_alias_factor
        
        # State layout:
        # 0 to 2*num_pairs - 1: paired states (pairs are 0-1, 2-3, 4-5, ...)
        # 2*num_pairs: goal state
        # 2*num_pairs + 1: failure state
        self._num_corridor_states = 2 * num_pairs
        self._goal_state = State(self._num_corridor_states)
        self._failure_state = State(self._num_corridor_states + 1)
        
        self._state_set = frozenset(
            State(s) for s in range(self._num_corridor_states + 2)
        )
        self._action_set = frozenset(
            Action(a) for a in range(num_actions)
        )
        
        # Precompute optimal actions for each state
        # States in a pair have DIFFERENT optimal actions
        self._optimal_actions: Dict[State, Action] = {}
        for pair_idx in range(num_pairs):
            state_a = State(2 * pair_idx)      # First state in pair
            state_b = State(2 * pair_idx + 1)  # Second state in pair
            
            # Assign different optimal actions
            # State A: action = pair_idx mod num_actions
            # State B: action = (pair_idx + 1) mod num_actions
            self._optimal_actions[state_a] = Action(pair_idx % num_actions)
            self._optimal_actions[state_b] = Action((pair_idx + 1) % num_actions)
        
        # Build transition graph
        self._transitions: Dict[Tuple[State, Action], State] = {}
        self._build_transitions()
        
        super().__init__()
    
    def _build_transitions(self) -> None:
        """Build the complete transition graph."""
        for state in self._state_set:
            s = int(state)
            
            # Terminal states are absorbing
            if state == self._goal_state or state == self._failure_state:
                for action in self._action_set:
                    self._transitions[(state, action)] = state
                continue
            
            # For corridor states
            optimal = self._optimal_actions[state]
            
            for action in self._action_set:
                if action == optimal:
                    # Correct: move toward goal
                    # States advance: 0→1→2→...→goal
                    if s + 1 >= self._num_corridor_states:
                        next_state = self._goal_state
                    else:
                        next_state = State(s + 1)
                else:
                    # Wrong: go to failure
                    next_state = self._failure_state
                
                self._transitions[(state, action)] = next_state
    
    # =========================================================================
    # Abstract Property Implementations
    # =========================================================================
    
    @property
    def states(self) -> FrozenSet[State]:
        """All states in the environment."""
        return self._state_set
    
    @property
    def actions(self) -> FrozenSet[Action]:
        """All available actions."""
        return self._action_set
    
    @property
    def initial_state(self) -> State:
        """Starting state."""
        return State(0)
    
    @property
    def terminal_failures(self) -> FrozenSet[State]:
        """Terminal failure states."""
        return frozenset({self._failure_state})
    
    @property
    def goal_states(self) -> FrozenSet[State]:
        """Goal states."""
        return frozenset({self._goal_state})
    
    @property
    def diameter(self) -> int:
        """Diameter of the state graph."""
        return self._num_corridor_states
    
    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================
    
    def _transition_impl(self, state: State, action: Action) -> State:
        """Look up the precomputed transition."""
        return self._transitions[(state, action)]
    
    def _optimal_action_impl(self, state: State) -> Action:
        """Return the optimal action for a non-terminal state."""
        return self._optimal_actions[state]
    
    # =========================================================================
    # Aliasing Analysis
    # =========================================================================
    
    def get_conflicting_pairs(self) -> list[Tuple[State, State, Action, Action]]:
        """
        Get all pairs of states with conflicting optimal actions.
        
        Returns:
            List of (state_a, state_b, optimal_a, optimal_b) tuples
            where state_a and state_b would be aliased under floor-division
            but have different optimal actions.
        """
        pairs = []
        for pair_idx in range(self._num_pairs):
            state_a = State(2 * pair_idx)
            state_b = State(2 * pair_idx + 1)
            opt_a = self._optimal_actions[state_a]
            opt_b = self._optimal_actions[state_b]
            if opt_a != opt_b:
                pairs.append((state_a, state_b, opt_a, opt_b))
        return pairs
    
    @property
    def num_conflicts(self) -> int:
        """Number of state pairs with conflicting optimal actions."""
        return len(self.get_conflicting_pairs())
    
    def __repr__(self) -> str:
        return (
            f"ConflictingGraphMDP("
            f"num_pairs={self._num_pairs}, "
            f"num_actions={self._num_actions}, "
            f"conflicts={self.num_conflicts})"
        )

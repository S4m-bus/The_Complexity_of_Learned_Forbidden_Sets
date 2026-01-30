# forbidden_sets/environments/observation.py
"""
Observation mapping with controlled aliasing.

This module provides the ObservationMapping class that implements
the many-to-one function φ: S → O from states to observations.

Key features:
- Explicit control over aliasing strength
- Full inspection of aliasing structure
- Support for analyzing false eliminations
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Callable, Optional
from dataclasses import dataclass

from forbidden_sets.core.types import State, Observation
from forbidden_sets.core.errors import AliasingViolationError


@dataclass(frozen=True)
class AliasingStats:
    """Statistics about the aliasing structure."""
    num_states: int
    num_observations: int
    max_alias_degree: int       # Maximum states per observation
    mean_alias_degree: float    # Average states per observation
    min_alias_degree: int       # Minimum states per observation
    
    @property
    def compression_ratio(self) -> float:
        """How much the state space is compressed."""
        return self.num_states / self.num_observations if self.num_observations > 0 else 0


class ObservationMapping:
    """
    Many-to-one observation mapping φ: S → O.
    
    This class implements the observation function that may alias
    multiple distinct states to the same observation. This is the
    core mechanism that creates partial observability.
    
    The aliasing structure is:
    - Deterministic (same state always maps to same observation)
    - Explicitly inspectable (can query inverse mapping)
    - Controllable (can construct with specific aliasing patterns)
    
    Example:
        # Create floor-division aliasing with factor 4
        # States 0,1,2,3 → Obs 0
        # States 4,5,6,7 → Obs 1
        # etc.
        mapping = ObservationMapping.floor_division(num_states=20, alias_factor=4)
    """
    
    def __init__(
        self,
        mapping: Dict[State, Observation],
        *,
        verify: bool = True
    ):
        """
        Create an observation mapping from an explicit dictionary.
        
        Args:
            mapping: State → Observation mapping (must be deterministic)
            verify: Whether to verify the mapping is valid
            
        Raises:
            AliasingViolationError: If mapping is invalid
        """
        self._mapping = dict(mapping)  # Copy to prevent external mutation
        self._inverse: Dict[Observation, FrozenSet[State]] = {}
        
        # Build inverse mapping
        for state, obs in self._mapping.items():
            if obs not in self._inverse:
                self._inverse[obs] = set()
            self._inverse[obs].add(state)
        
        # Freeze inverse sets
        self._inverse = {
            obs: frozenset(states) 
            for obs, states in self._inverse.items()
        }
        
        if verify:
            self._verify()
    
    def _verify(self) -> None:
        """Verify the mapping is valid (deterministic, complete)."""
        # All states must have exactly one observation
        # (this is guaranteed by the dict structure)
        pass  # The dict nature enforces this
    
    # =========================================================================
    # Core Operations
    # =========================================================================
    
    def observe(self, state: State) -> Observation:
        """
        Get the observation for a state.
        
        This is the forward mapping φ(s).
        
        Args:
            state: The state to observe
            
        Returns:
            The corresponding observation
            
        Raises:
            AliasingViolationError: If state is not in the mapping
        """
        if state not in self._mapping:
            raise AliasingViolationError(
                state=state,
                message=f"State {state} not in observation mapping"
            )
        return self._mapping[state]
    
    def __call__(self, state: State) -> Observation:
        """Shorthand for observe()."""
        return self.observe(state)
    
    def get_aliased_states(self, observation: Observation) -> FrozenSet[State]:
        """
        Get all states that map to an observation.
        
        This is the inverse mapping φ^{-1}(o).
        
        Args:
            observation: The observation to query
            
        Returns:
            Frozenset of all states that produce this observation
        """
        return self._inverse.get(observation, frozenset())
    
    # =========================================================================
    # Aliasing Analysis
    # =========================================================================
    
    @property
    def aliasing_degree(self) -> Dict[Observation, int]:
        """
        Number of states aliased to each observation.
        
        Returns:
            Dict mapping observation → number of aliased states
        """
        return {obs: len(states) for obs, states in self._inverse.items()}
    
    @property
    def stats(self) -> AliasingStats:
        """Compute statistics about the aliasing structure."""
        degrees = list(self.aliasing_degree.values())
        return AliasingStats(
            num_states=len(self._mapping),
            num_observations=len(self._inverse),
            max_alias_degree=max(degrees) if degrees else 0,
            mean_alias_degree=sum(degrees) / len(degrees) if degrees else 0,
            min_alias_degree=min(degrees) if degrees else 0,
        )
    
    @property
    def observations(self) -> FrozenSet[Observation]:
        """All observations in the mapping."""
        return frozenset(self._inverse.keys())
    
    @property
    def states(self) -> FrozenSet[State]:
        """All states in the mapping."""
        return frozenset(self._mapping.keys())
    
    def has_aliasing(self) -> bool:
        """Check if any observation has multiple states."""
        return any(len(states) > 1 for states in self._inverse.values())
    
    def is_identity(self) -> bool:
        """Check if this is the identity mapping (no aliasing)."""
        return all(len(states) == 1 for states in self._inverse.values())
    
    # =========================================================================
    # Conflict Detection (for false elimination analysis)
    # =========================================================================
    
    def has_action_conflict(
        self,
        observation: Observation,
        action_is_optimal: Callable[[State], bool]
    ) -> bool:
        """
        Check if an observation has conflicting optimal actions.
        
        An observation has a conflict if some aliased states have
        a given action as optimal and others don't.
        
        Args:
            observation: The observation to check
            action_is_optimal: Function that returns True if a given
                             action is optimal for a state
        
        Returns:
            True if there's at least one state where action is optimal
            and at least one where it isn't
        """
        states = self.get_aliased_states(observation)
        if len(states) <= 1:
            return False
        
        optimal_count = sum(1 for s in states if action_is_optimal(s))
        return 0 < optimal_count < len(states)
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def identity(cls, states: FrozenSet[State]) -> "ObservationMapping":
        """
        Create an identity mapping (no aliasing).
        
        Each state maps to itself: φ(s) = s
        
        Args:
            states: The state space
            
        Returns:
            An ObservationMapping with no aliasing
        """
        mapping = {s: Observation(s) for s in states}
        return cls(mapping)
    
    @classmethod
    def floor_division(
        cls, 
        num_states: int, 
        alias_factor: int
    ) -> "ObservationMapping":
        """
        Create floor-division aliasing: o = ⌊s / m⌋
        
        This groups consecutive states together.
        States 0 to m-1 → Observation 0
        States m to 2m-1 → Observation 1
        etc.
        
        Args:
            num_states: Number of states (0 to num_states-1)
            alias_factor: The divisor m (higher = more aliasing)
            
        Returns:
            An ObservationMapping with floor-division aliasing
        """
        if alias_factor < 1:
            raise ValueError("alias_factor must be >= 1")
        
        mapping = {
            State(s): Observation(s // alias_factor)
            for s in range(num_states)
        }
        return cls(mapping)
    
    @classmethod
    def modular(
        cls,
        num_states: int,
        alias_factor: int
    ) -> "ObservationMapping":
        """
        Create modular aliasing: o = s mod m
        
        This interleaves aliased states.
        States 0, m, 2m, ... → Observation 0
        States 1, m+1, 2m+1, ... → Observation 1
        etc.
        
        Args:
            num_states: Number of states
            alias_factor: The modulus m
            
        Returns:
            An ObservationMapping with modular aliasing
        """
        if alias_factor < 1:
            raise ValueError("alias_factor must be >= 1")
        
        mapping = {
            State(s): Observation(s % alias_factor)
            for s in range(num_states)
        }
        return cls(mapping)
    
    @classmethod
    def from_function(
        cls,
        states: FrozenSet[State],
        fn: Callable[[State], Observation]
    ) -> "ObservationMapping":
        """
        Create a mapping from an arbitrary function.
        
        Args:
            states: The state space
            fn: A deterministic function S → O
            
        Returns:
            An ObservationMapping based on the function
        """
        mapping = {s: fn(s) for s in states}
        return cls(mapping)


# =============================================================================
# Convenient Type Alias
# =============================================================================

# For passing observation mappings, can be either:
# - An ObservationMapping object
# - A callable that maps State → Observation
ObservationFunction = Callable[[State], Observation]

# forbidden_sets/core/errors.py
"""
Custom exceptions for theoretical constraint violations.

These exceptions are raised when code attempts to violate the
non-negotiable constraints of the theoretical framework.

All violations are treated as programming errors, not runtime
conditions to be handled — they indicate incorrect usage of the library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forbidden_sets.core.types import State, Action, Observation


class TheoreticalViolationError(Exception):
    """
    Base class for violations of theoretical assumptions.
    
    Any exception derived from this class indicates that code
    has attempted to violate a core assumption of the theoretical
    framework. These are unrecoverable errors.
    """
    pass


class StochasticTransitionError(TheoreticalViolationError):
    """
    Raised when a stochastic transition is detected.
    
    The framework requires fully deterministic transitions:
    T(s, a) must always return the same s' for the same inputs.
    
    This error is raised if:
    - A transition function returns different values for same input
    - Random number generation is detected in transition logic
    - Non-deterministic behavior is observed
    """
    
    def __init__(
        self, 
        state: "State", 
        action: "Action",
        observed_next_states: list,
        message: str = ""
    ):
        self.state = state
        self.action = action
        self.observed_next_states = observed_next_states
        
        default_msg = (
            f"Stochastic transition detected: T({state}, {action}) returned "
            f"multiple different states: {observed_next_states}. "
            f"Transitions must be deterministic."
        )
        super().__init__(message or default_msg)


class ValueEstimationAttemptError(TheoreticalViolationError):
    """
    Raised when code attempts value function estimation.
    
    This library explicitly forbids:
    - Q-functions
    - V-functions
    - Advantage functions
    - Any form of expected return estimation
    
    Learning must proceed purely through elimination of
    forbidden state-action pairs, not value estimation.
    """
    
    def __init__(self, attempted_operation: str = ""):
        msg = (
            "Value estimation is forbidden in this theoretical framework. "
            "Learning must proceed through elimination only, not value estimation. "
        )
        if attempted_operation:
            msg += f"Attempted operation: {attempted_operation}"
        super().__init__(msg)


class HiddenMemoryError(TheoreticalViolationError):
    """
    Raised when memory is not explicitly inspectable.
    
    All agent memory must be:
    - Externally accessible
    - Fully enumerable
    - Not hidden in closures, caches, or opaque objects
    
    This ensures all learning is transparent and analyzable.
    """
    
    def __init__(self, component: str = "", message: str = ""):
        default_msg = (
            "All memory must be explicitly inspectable. "
            "Hidden state, closures, or opaque memory structures are forbidden. "
        )
        if component:
            default_msg += f"Violation in component: {component}"
        super().__init__(message or default_msg)


class NonMonotonicUpdateError(TheoreticalViolationError):
    """
    Raised when a forbidden constraint is removed or modified.
    
    The forbidden set must be monotonically increasing:
    - Pairs can only be added, never removed
    - Once forbidden, a pair stays forbidden forever
    
    This ensures learning is irreversible and cumulative.
    """
    
    def __init__(
        self,
        observation: "Observation",
        action: "Action",
        message: str = ""
    ):
        self.observation = observation
        self.action = action
        
        default_msg = (
            f"Non-monotonic update detected: attempted to remove or modify "
            f"forbidden pair ({observation}, {action}). "
            f"The forbidden set must be monotonically increasing."
        )
        super().__init__(message or default_msg)


class PolicyNonDeterminismError(TheoreticalViolationError):
    """
    Raised when a policy exhibits non-deterministic behavior.
    
    Policies must be deterministic functions:
    π(o) must always return the same action for the same observation
    (given the same forbidden set state).
    """
    
    def __init__(
        self,
        observation: "Observation",
        actions_returned: list,
        message: str = ""
    ):
        self.observation = observation
        self.actions_returned = actions_returned
        
        default_msg = (
            f"Non-deterministic policy detected: π({observation}) returned "
            f"different actions: {actions_returned}. Policies must be deterministic."
        )
        super().__init__(message or default_msg)


class AliasingViolationError(TheoreticalViolationError):
    """
    Raised when observation mapping violates required properties.
    
    The observation mapping φ: S → O must be:
    - Deterministic (same state always maps to same observation)
    - Total (every state has an observation)
    """
    
    def __init__(self, state: "State", message: str = ""):
        self.state = state
        default_msg = f"Observation mapping violation for state {state}."
        super().__init__(message or default_msg)


class InfeasibleStateError(TheoreticalViolationError):
    """
    Raised when an agent reaches a state with no legal actions.
    
    This occurs when all actions have been forbidden for an observation.
    It is a valid experimental outcome (not a bug), but indicates
    that learning has failed for this observation.
    """
    
    def __init__(self, observation: "Observation", forbidden_actions: frozenset):
        self.observation = observation
        self.forbidden_actions = forbidden_actions
        
        msg = (
            f"Infeasible state: all actions forbidden for observation {observation}. "
            f"Forbidden actions: {forbidden_actions}"
        )
        super().__init__(msg)

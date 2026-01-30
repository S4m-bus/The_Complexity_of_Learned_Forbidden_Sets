# forbidden_sets/core/invariants.py
"""
Structural enforcement of theoretical assumptions.

These functions verify and enforce the core invariants that must
hold throughout the library. They are called at construction time
and during operations to catch violations early.

The philosophy is: if an invariant can be violated, it will be.
Therefore, we check proactively rather than assuming correctness.
"""

from __future__ import annotations

from typing import Dict, Tuple, Callable, Set, Any, NoReturn, FrozenSet

from forbidden_sets.core.types import State, Action, Observation, ForbiddenPair
from forbidden_sets.core.errors import (
    StochasticTransitionError,
    ValueEstimationAttemptError,
    NonMonotonicUpdateError,
    PolicyNonDeterminismError,
    HiddenMemoryError,
)


def enforce_determinism(
    transition_function: Callable[[State, Action], State],
    states: FrozenSet[State],
    actions: FrozenSet[Action],
    num_checks: int = 3
) -> None:
    """
    Verify that a transition function is deterministic.
    
    Checks that T(s, a) returns the same value on repeated calls.
    This catches accidental randomness in transition logic.
    
    Args:
        transition_function: The transition function to verify
        states: Set of states to check
        actions: Set of actions to check
        num_checks: Number of repeated calls per (s, a) pair
        
    Raises:
        StochasticTransitionError: If different results are observed
    """
    for state in states:
        for action in actions:
            results = []
            for _ in range(num_checks):
                try:
                    result = transition_function(state, action)
                    results.append(result)
                except Exception:
                    # If it raises, that's consistent behavior
                    results.append(None)
            
            # Check all results are identical
            if len(set(results)) > 1:
                raise StochasticTransitionError(
                    state=state,
                    action=action,
                    observed_next_states=results
                )


def enforce_policy_determinism(
    policy_function: Callable[[Observation], Action],
    observation: Observation,
    num_checks: int = 3
) -> None:
    """
    Verify that a policy is deterministic for a given observation.
    
    Checks that π(o) returns the same action on repeated calls.
    
    Args:
        policy_function: The policy to verify
        observation: The observation to test
        num_checks: Number of repeated calls
        
    Raises:
        PolicyNonDeterminismError: If different actions are returned
    """
    actions = []
    for _ in range(num_checks):
        action = policy_function(observation)
        actions.append(action)
    
    if len(set(actions)) > 1:
        raise PolicyNonDeterminismError(
            observation=observation,
            actions_returned=actions
        )


def enforce_monotonicity(
    old_forbidden_set: FrozenSet[Tuple[Observation, Action]],
    new_forbidden_set: FrozenSet[Tuple[Observation, Action]]
) -> None:
    """
    Verify that the forbidden set is monotonically increasing.
    
    The new set must be a superset of the old set — no pairs
    can be removed.
    
    Args:
        old_forbidden_set: The previous forbidden set
        new_forbidden_set: The updated forbidden set
        
    Raises:
        NonMonotonicUpdateError: If any pairs were removed
    """
    removed = old_forbidden_set - new_forbidden_set
    if removed:
        # Report the first removed pair
        obs, act = next(iter(removed))
        raise NonMonotonicUpdateError(
            observation=obs,
            action=act,
            message=f"Removed {len(removed)} pairs from forbidden set. First: ({obs}, {act})"
        )


def forbid_value_estimation() -> NoReturn:
    """
    Structural block: always raises if called.
    
    This function exists to be placed in code paths that would
    perform value estimation. Its mere presence serves as documentation
    and enforcement.
    
    Raises:
        ValueEstimationAttemptError: Always
    """
    raise ValueEstimationAttemptError(
        "This code path would perform value estimation, which is forbidden."
    )


def forbid_stochastic_reward() -> NoReturn:
    """
    Structural block: prevents stochastic reward usage.
    
    Raises:
        TheoreticalViolationError: Always
    """
    from forbidden_sets.core.errors import TheoreticalViolationError
    raise TheoreticalViolationError(
        "Stochastic rewards are forbidden. The framework uses only "
        "deterministic terminal success/failure outcomes."
    )


def verify_explicit_memory(agent: Any, required_attributes: Tuple[str, ...]) -> None:
    """
    Verify that an agent's memory is explicitly inspectable.
    
    Checks that the specified attributes exist and are accessible.
    
    Args:
        agent: The agent to verify
        required_attributes: Attribute names that must be accessible
        
    Raises:
        HiddenMemoryError: If any required attribute is missing or inaccessible
    """
    for attr in required_attributes:
        if not hasattr(agent, attr):
            raise HiddenMemoryError(
                component=type(agent).__name__,
                message=f"Required memory attribute '{attr}' is not accessible."
            )
        
        # Try to access it
        try:
            _ = getattr(agent, attr)
        except Exception as e:
            raise HiddenMemoryError(
                component=type(agent).__name__,
                message=f"Memory attribute '{attr}' raised exception on access: {e}"
            )


def verify_no_random_state(obj: Any) -> None:
    """
    Check that an object doesn't appear to contain random state.
    
    This is a heuristic check — it looks for common random state
    indicators but cannot catch all cases.
    
    Args:
        obj: Object to check
        
    Raises:
        TheoreticalViolationError: If random state is detected
    """
    from forbidden_sets.core.errors import TheoreticalViolationError
    
    suspicious_attrs = ['_random', 'rng', 'random_state', 'np_random', '_rng']
    
    for attr in suspicious_attrs:
        if hasattr(obj, attr):
            raise TheoreticalViolationError(
                f"Object {type(obj).__name__} contains suspicious random state "
                f"attribute: '{attr}'. Random state is forbidden."
            )


class DeterminismVerifier:
    """
    Context manager for verifying determinism of operations.
    
    Usage:
        with DeterminismVerifier() as v:
            result = some_operation()
            v.record(result)
        # Raises if recorded values differ
    
    This is useful for wrapping complex operations and verifying
    they produce consistent results.
    """
    
    def __init__(self, num_checks: int = 2):
        self.num_checks = num_checks
        self.results: list = []
    
    def __enter__(self) -> "DeterminismVerifier":
        return self
    
    def record(self, result: Any) -> None:
        """Record a result for comparison."""
        self.results.append(result)
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            return  # Let the exception propagate
        
        if len(self.results) > 1:
            first = self.results[0]
            for i, result in enumerate(self.results[1:], 2):
                if result != first:
                    from forbidden_sets.core.errors import TheoreticalViolationError
                    raise TheoreticalViolationError(
                        f"Non-determinism detected: result 1 = {first}, "
                        f"result {i} = {result}"
                    )

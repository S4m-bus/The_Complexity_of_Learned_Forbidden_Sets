# forbidden_sets/environments/__init__.py
"""
Environment module: Deterministic MDPs with controlled aliasing.

This module provides:
- DeterministicMDP: Base protocol for all environments
- ObservationMapping: Controlled many-to-one observation functions
- CorridorMDP: Simple linear environment for scaling experiments
- ConflictingGraphMDP: Adversarial environment for aliasing stress tests
"""

from forbidden_sets.environments.base import DeterministicMDP
from forbidden_sets.environments.observation import ObservationMapping
from forbidden_sets.environments.corridor import CorridorMDP
from forbidden_sets.environments.conflicting_graph import ConflictingGraphMDP

__all__ = [
    "DeterministicMDP",
    "ObservationMapping",
    "CorridorMDP",
    "ConflictingGraphMDP",
]

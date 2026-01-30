# forbidden_sets/agents/__init__.py
"""
Agent module: Learning agents with explicit memory structures.

This module provides:
- Agent: Base protocol for all agents
- StatelessAgent: Reactive agent with no memory
- FiniteHistoryAgent: Agent conditioned on bounded history
- ForbiddenSetAgent: Elimination-based learning agent

All agents enforce:
- Deterministic policies
- Explicit, inspectable memory
- No value estimation
"""

from forbidden_sets.agents.base import Agent
from forbidden_sets.agents.stateless import StatelessAgent
from forbidden_sets.agents.history import FiniteHistoryAgent
from forbidden_sets.agents.forbidden import ForbiddenSetAgent

__all__ = [
    "Agent",
    "StatelessAgent",
    "FiniteHistoryAgent",
    "ForbiddenSetAgent",
]

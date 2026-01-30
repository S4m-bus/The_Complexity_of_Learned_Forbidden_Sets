# forbidden_sets/harness/rollout.py
"""
Deterministic rollout engine: Execute episodes with full control.

This module provides the core episode execution logic.
All rollouts are deterministic:
- Same agent state + same environment state = same trajectory
- No random tie-breaking
- No hidden randomness

This ensures that experimental results are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING

from forbidden_sets.core.types import (
    State,
    Observation,
    Action,
    Trajectory,
    Step,
    EpisodeOutcome,
)
from forbidden_sets.core.errors import InfeasibleStateError

if TYPE_CHECKING:
    from forbidden_sets.environments.base import DeterministicMDP
    from forbidden_sets.environments.observation import ObservationMapping
    from forbidden_sets.agents.base import Agent


@dataclass
class EpisodeResult:
    """Complete result of a single episode."""
    
    # The trajectory taken
    trajectory: Trajectory
    
    # How the episode ended
    outcome: EpisodeOutcome
    
    # Episode metadata
    episode_number: int
    
    # Whether a new constraint was added
    constraint_added: bool = False
    
    # The constraint that was added (if any)
    added_pair: Optional[tuple] = None
    
    @property
    def length(self) -> int:
        """Number of steps taken."""
        return self.trajectory.length
    
    @property
    def succeeded(self) -> bool:
        """Did the episode reach the goal?"""
        return self.outcome == EpisodeOutcome.SUCCESS
    
    @property
    def failed(self) -> bool:
        """Did the episode hit terminal failure?"""
        return self.outcome == EpisodeOutcome.FAILURE
    
    @property
    def was_infeasible(self) -> bool:
        """Did the agent have no legal action?"""
        return self.outcome == EpisodeOutcome.INFEASIBLE


class DeterministicRollout:
    """
    Execute deterministic episodes in an environment.
    
    This class handles:
    - Starting episodes from initial state
    - Observing states through the observation mapping
    - Getting actions from the agent
    - Executing transitions
    - Recording trajectories
    - Updating the agent on episode end
    
    There is NO randomness in this class. All behavior is deterministic.
    
    Example:
        >>> rollout = DeterministicRollout(max_steps=100)
        >>> result = rollout.run_episode(env, agent, obs_mapping, episode_num=1)
        >>> print(result.outcome)
    """
    
    def __init__(self, max_steps: int = 1000):
        """
        Initialize the rollout engine.
        
        Args:
            max_steps: Maximum steps per episode (prevents infinite loops)
        """
        self._max_steps = max_steps
    
    def run_episode(
        self,
        env: "DeterministicMDP",
        agent: "Agent",
        obs_mapping: "ObservationMapping",
        episode_num: int = 0,
        update_agent: bool = True
    ) -> EpisodeResult:
        """
        Run a single episode.
        
        Args:
            env: The environment
            agent: The learning agent
            obs_mapping: Maps states to observations
            episode_num: Episode number (for logging)
            update_agent: Whether to update the agent after episode
            
        Returns:
            EpisodeResult with complete information
        """
        # Initialize
        agent.reset_episode()
        state = env.reset()
        trajectory = Trajectory(start_state=state)
        
        # Track initial forbidden set size
        initial_memory_size = agent.memory_size
        
        outcome = None
        
        for step_num in range(self._max_steps):
            # Get observation
            observation = obs_mapping.observe(state)
            
            # Get available actions (not forbidden)
            available = agent.get_available_actions(observation, env.actions)
            
            # Check for infeasibility
            if not available:
                outcome = EpisodeOutcome.INFEASIBLE
                trajectory.finalize(outcome)
                break
            
            # Select action (deterministic)
            action = agent.select_action(observation, available)
            
            if action is None:
                outcome = EpisodeOutcome.INFEASIBLE
                trajectory.finalize(outcome)
                break
            
            # Execute transition
            next_state = env.transition(state, action)
            
            # Record step
            step = Step(
                state=state,
                observation=observation,
                action=action,
                next_state=next_state,
            )
            trajectory.add_step(step)
            
            # Check for termination
            if env.is_success(next_state):
                outcome = EpisodeOutcome.SUCCESS
                trajectory.finalize(outcome)
                break
            
            if env.is_failure(next_state):
                outcome = EpisodeOutcome.FAILURE
                trajectory.finalize(outcome)
                break
            
            # Continue
            state = next_state
        
        # Handle timeout (shouldn't happen in well-designed environments)
        if outcome is None:
            # Treat timeout as failure
            outcome = EpisodeOutcome.FAILURE
            trajectory.finalize(outcome)
        
        # Update agent
        if update_agent:
            agent.update(trajectory, outcome, env, obs_mapping)
        
        # Check if constraint was added
        final_memory_size = agent.memory_size
        constraint_added = final_memory_size > initial_memory_size
        
        # Get the added pair if any
        added_pair = None
        if constraint_added and trajectory.steps:
            last_step = trajectory.steps[-1]
            added_pair = (last_step.observation, last_step.action)
        
        return EpisodeResult(
            trajectory=trajectory,
            outcome=outcome,
            episode_number=episode_num,
            constraint_added=constraint_added,
            added_pair=added_pair,
        )
    
    def run_episodes(
        self,
        env: "DeterministicMDP",
        agent: "Agent",
        obs_mapping: "ObservationMapping",
        num_episodes: int,
        start_episode: int = 0
    ) -> List[EpisodeResult]:
        """
        Run multiple episodes.
        
        Args:
            env: The environment
            agent: The learning agent
            obs_mapping: Observation mapping
            num_episodes: Number of episodes to run
            start_episode: Starting episode number
            
        Returns:
            List of EpisodeResults
        """
        results = []
        for i in range(num_episodes):
            episode_num = start_episode + i
            result = self.run_episode(env, agent, obs_mapping, episode_num)
            results.append(result)
        return results

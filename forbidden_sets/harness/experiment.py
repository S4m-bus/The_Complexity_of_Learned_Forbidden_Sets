# forbidden_sets/harness/experiment.py
"""
Experiment runner: Execute complete experiments with metric collection.

This module provides the high-level interface for running experiments.
It combines:
- Environment setup (with controlled parameters)
- Agent creation
- Episode execution
- Metric collection
- Result aggregation

All experiments are configured through ExperimentConfig and produce
ExperimentResult with complete statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from forbidden_sets.core.types import EpisodeOutcome
from forbidden_sets.environments.corridor import CorridorMDP
from forbidden_sets.environments.conflicting_graph import ConflictingGraphMDP
from forbidden_sets.environments.observation import ObservationMapping
from forbidden_sets.agents.forbidden import ForbiddenSetAgent
from forbidden_sets.metrics.constraint_size import ConstraintSizeTracker, GrowthFit
from forbidden_sets.metrics.false_elimination import FalseEliminationTracker
from forbidden_sets.metrics.feasibility import FeasibilityTracker, FeasibilityStats
from forbidden_sets.harness.rollout import DeterministicRollout, EpisodeResult


@dataclass
class ExperimentConfig:
    """
    Configuration for a controlled experiment.
    
    This captures all parameters needed to reproduce an experiment:
    - Environment parameters
    - Aliasing configuration
    - Agent configuration
    - Experiment duration
    """
    
    # Environment configuration
    env_type: str = "corridor"    # "corridor" or "conflicting_graph"
    diameter: int = 20            # Length of corridor / number of state pairs
    num_actions: int = 4          # Action space size
    
    # Aliasing configuration
    alias_factor: int = 1         # 1 = no aliasing, >1 = floor-division aliasing
    
    # Agent configuration
    agent_type: str = "forbidden_set"  # "forbidden_set" or "stateless" or "history"
    history_depth: int = 0        # 0 = stateless, >0 = use history
    
    # Experiment configuration
    num_episodes: int = 500       # Number of episodes to run
    max_steps_per_episode: int = 1000  # Max steps per episode
    
    # Seed (for documentation, not used in deterministic execution)
    seed: int = 42
    
    # Labels for grouping
    experiment_name: str = ""
    experiment_group: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "env_type": self.env_type,
            "diameter": self.diameter,
            "num_actions": self.num_actions,
            "alias_factor": self.alias_factor,
            "agent_type": self.agent_type,
            "history_depth": self.history_depth,
            "num_episodes": self.num_episodes,
            "seed": self.seed,
            "experiment_name": self.experiment_name,
        }


@dataclass
class ExperimentResult:
    """
    Complete results from an experiment run.
    
    Contains all metrics and raw data for analysis.
    """
    
    # Configuration used
    config: ExperimentConfig
    
    # Final metrics
    final_forbidden_set_size: int
    final_success_rate: float
    final_failure_rate: float
    final_infeasibility_rate: float
    
    # Growth tracking
    constraint_growth: ConstraintSizeTracker
    growth_fit: GrowthFit
    
    # False elimination tracking
    false_elimination_count: int
    false_elimination_rate: float
    
    # Feasibility tracking
    feasibility_stats: FeasibilityStats
    
    # Time to convergence
    episodes_to_convergence: int  # -1 if never converged
    
    # Raw episode results (optional, can be large)
    episode_results: Optional[List[EpisodeResult]] = None
    
    # Per-episode forbidden set sizes
    forbidden_size_trajectory: List[int] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"ExperimentResult({self.config.experiment_name}):\n"
            f"  Environment: {self.config.env_type}(D={self.config.diameter})\n"
            f"  Aliasing: factor={self.config.alias_factor}\n"
            f"  Agent: {self.config.agent_type}(depth={self.config.history_depth})\n"
            f"  ---\n"
            f"  Final |F|: {self.final_forbidden_set_size}\n"
            f"  Success rate: {self.final_success_rate:.2%}\n"
            f"  False eliminations: {self.false_elimination_count} ({self.false_elimination_rate:.2%})\n"
            f"  Growth: {self.growth_fit}\n"
            f"  Convergence: {self.episodes_to_convergence} episodes"
        )


class ExperimentRunner:
    """
    Run complete experiments with metric collection.
    
    This is the main entry point for running research experiments.
    It handles:
    - Environment construction based on config
    - Agent creation
    - Episode execution
    - Metric tracking
    - Result aggregation
    
    Example:
        >>> config = ExperimentConfig(diameter=20, alias_factor=4)
        >>> runner = ExperimentRunner()
        >>> result = runner.run(config)
        >>> print(result.summary())
    """
    
    def __init__(self, store_episodes: bool = False):
        """
        Initialize the experiment runner.
        
        Args:
            store_episodes: Whether to store raw episode data (uses more memory)
        """
        self._store_episodes = store_episodes
    
    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a complete experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            ExperimentResult with all metrics
        """
        # Create environment
        env = self._create_env(config)
        
        # Create observation mapping
        obs_mapping = self._create_obs_mapping(env, config)
        
        # Create agent
        agent = self._create_agent(env, config)
        
        # Create trackers
        size_tracker = ConstraintSizeTracker(
            environment_diameter=config.diameter,
            alias_factor=config.alias_factor,
            history_depth=config.history_depth,
        )
        false_elim_tracker = FalseEliminationTracker()
        feasibility_tracker = FeasibilityTracker(num_actions=config.num_actions)
        
        # Create rollout engine
        rollout = DeterministicRollout(max_steps=config.max_steps_per_episode)
        
        # Run episodes
        episode_results = []
        forbidden_sizes = []
        
        for episode in range(config.num_episodes):
            # Run episode
            result = rollout.run_episode(env, agent, obs_mapping, episode)
            
            if self._store_episodes:
                episode_results.append(result)
            
            # Track metrics
            current_size = agent.memory_size
            forbidden_sizes.append(current_size)
            size_tracker.log(episode, current_size)
            
            # Track false eliminations
            if result.constraint_added and result.added_pair:
                obs, action = result.added_pair
                # Get the true state from the trajectory
                if result.trajectory.steps:
                    last_step = result.trajectory.steps[-1]
                    from forbidden_sets.core.types import ForbiddenPair
                    forbid_record = ForbiddenPair(
                        observation=obs,
                        action=action,
                        episode_forbidden=episode,
                        true_state=last_step.state,
                    )
                    false_elim_tracker.check_and_record(forbid_record, env, obs_mapping)
            
            # Track feasibility
            forbidden_per_obs = {}
            if hasattr(agent, '_forbidden_by_key'):
                for key, actions in agent._forbidden_by_key.items():
                    if isinstance(key, int):  # Observation key
                        forbidden_per_obs[key] = len(actions)
            feasibility_tracker.record(episode, result.outcome, forbidden_per_obs)
        
        # Compute final metrics
        success_count = sum(1 for r in 
            [feasibility_tracker.outcomes[i][1] for i in range(len(feasibility_tracker.outcomes))]
            if r == EpisodeOutcome.SUCCESS
        )
        failure_count = sum(1 for r in
            [feasibility_tracker.outcomes[i][1] for i in range(len(feasibility_tracker.outcomes))]
            if r == EpisodeOutcome.FAILURE
        )
        infeasible_count = sum(1 for r in
            [feasibility_tracker.outcomes[i][1] for i in range(len(feasibility_tracker.outcomes))]
            if r == EpisodeOutcome.INFEASIBLE
        )
        
        total = config.num_episodes
        final_success_rate = success_count / total if total > 0 else 0.0
        final_failure_rate = failure_count / total if total > 0 else 0.0
        final_infeasibility_rate = infeasible_count / total if total > 0 else 0.0
        
        # Compute growth fit
        growth_fit = size_tracker.fit_polynomial()
        
        # Compute convergence (when success rate stabilizes)
        convergence = feasibility_tracker.time_to_feasibility()
        
        return ExperimentResult(
            config=config,
            final_forbidden_set_size=agent.memory_size,
            final_success_rate=final_success_rate,
            final_failure_rate=final_failure_rate,
            final_infeasibility_rate=final_infeasibility_rate,
            constraint_growth=size_tracker,
            growth_fit=growth_fit,
            false_elimination_count=false_elim_tracker.false_elimination_count,
            false_elimination_rate=false_elim_tracker.false_elimination_rate,
            feasibility_stats=feasibility_tracker.stats,
            episodes_to_convergence=convergence,
            episode_results=episode_results if self._store_episodes else None,
            forbidden_size_trajectory=forbidden_sizes,
        )
    
    def _create_env(self, config: ExperimentConfig):
        """Create environment from config."""
        if config.env_type == "corridor":
            return CorridorMDP(
                diameter=config.diameter,
                num_actions=config.num_actions,
            )
        elif config.env_type == "conflicting_graph":
            return ConflictingGraphMDP(
                num_pairs=config.diameter // 2,
                num_actions=config.num_actions,
            )
        else:
            raise ValueError(f"Unknown env_type: {config.env_type}")
    
    def _create_obs_mapping(self, env, config: ExperimentConfig):
        """Create observation mapping from config."""
        if config.alias_factor <= 1:
            # No aliasing: identity mapping
            return ObservationMapping.identity(env.states)
        else:
            # Floor-division aliasing
            return ObservationMapping.floor_division(
                num_states=env.num_states,
                alias_factor=config.alias_factor,
            )
    
    def _create_agent(self, env, config: ExperimentConfig):
        """Create agent from config."""
        return ForbiddenSetAgent(
            actions=env.actions,
            history_depth=config.history_depth,
        )
    
    def run_parameter_sweep(
        self,
        base_config: ExperimentConfig,
        parameter_name: str,
        parameter_values: List[Any]
    ) -> List[ExperimentResult]:
        """
        Run experiments sweeping over a parameter.
        
        Args:
            base_config: Base configuration
            parameter_name: Name of parameter to sweep
            parameter_values: Values to try
            
        Returns:
            List of ExperimentResults
        """
        results = []
        for value in parameter_values:
            # Create modified config
            config_dict = base_config.to_dict()
            config_dict[parameter_name] = value
            config_dict["experiment_name"] = f"{base_config.experiment_name}_{parameter_name}={value}"
            
            config = ExperimentConfig(**config_dict)
            result = self.run(config)
            results.append(result)
        
        return results

# forbidden_sets/metrics/feasibility.py
"""
Feasibility tracking: Measure success/failure rates and cascade depth.

Feasibility is the primary success metric for elimination-based learning:
- Can the agent reach the goal without hitting a forbidden state?
- Do eliminations enable or prevent future success?

This is distinct from reward-based metrics:
- We don't track cumulative reward
- We don't track regret
- We track binary success/failure and structural properties
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, FrozenSet, TYPE_CHECKING
from collections import defaultdict

from forbidden_sets.core.types import (
    Observation,
    Action,
    EpisodeOutcome,
)

if TYPE_CHECKING:
    from forbidden_sets.core.types import Trajectory


@dataclass
class FeasibilityStats:
    """Summary statistics for feasibility."""
    total_episodes: int
    successes: int
    failures: int
    infeasible: int  # Episodes where agent had no legal action
    
    @property
    def success_rate(self) -> float:
        """Fraction of episodes that succeeded."""
        if self.total_episodes == 0:
            return 0.0
        return self.successes / self.total_episodes
    
    @property
    def failure_rate(self) -> float:
        """Fraction of episodes that failed (hit terminal failure)."""
        if self.total_episodes == 0:
            return 0.0
        return self.failures / self.total_episodes
    
    @property
    def infeasibility_rate(self) -> float:
        """Fraction of episodes where no action was available."""
        if self.total_episodes == 0:
            return 0.0
        return self.infeasible / self.total_episodes
    
    @property
    def is_fully_feasible(self) -> bool:
        """True if all episodes succeeded (after learning)."""
        return self.failures == 0 and self.infeasible == 0
    
    def __str__(self) -> str:
        return (
            f"Feasibility(success={self.success_rate:.2%}, "
            f"failure={self.failure_rate:.2%}, "
            f"infeasible={self.infeasibility_rate:.2%})"
        )


@dataclass
class CascadeInfo:
    """Information about elimination cascades per observation."""
    observation: Observation
    total_forbidden: int
    max_actions: int  # Total number of actions
    is_fully_blocked: bool  # All actions forbidden?
    
    @property
    def blocking_fraction(self) -> float:
        """Fraction of actions forbidden for this observation."""
        if self.max_actions == 0:
            return 0.0
        return self.total_forbidden / self.max_actions


@dataclass
class FeasibilityTracker:
    """
    Track feasibility and related metrics during learning.
    
    Metrics tracked:
    - Success/failure/infeasibility rates over time
    - Cascade depth: how many actions get forbidden per observation
    - Convergence: when does feasibility stabilize?
    
    Example:
        >>> tracker = FeasibilityTracker()
        >>> for episode in range(100):
        ...     outcome = run_episode(agent, env)
        ...     tracker.record(episode, outcome, agent.forbidden_set)
        >>> print(tracker.stats)
    """
    
    # Episode outcomes in order
    outcomes: List[Tuple[int, EpisodeOutcome]] = field(default_factory=list)
    
    # Forbidden counts per observation over time
    # observation → list of (episode, count) pairs
    cascade_history: Dict[Observation, List[Tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    
    # Number of actions (needed for cascade analysis)
    num_actions: int = 4
    
    def record(
        self,
        episode: int,
        outcome: EpisodeOutcome,
        forbidden_per_obs: Dict[Observation, int] = None
    ) -> None:
        """
        Record the outcome of an episode.
        
        Args:
            episode: Episode number
            outcome: How episode ended
            forbidden_per_obs: Optional dict of observation → forbidden count
        """
        self.outcomes.append((episode, outcome))
        
        if forbidden_per_obs:
            for obs, count in forbidden_per_obs.items():
                self.cascade_history[obs].append((episode, count))
    
    @property
    def stats(self) -> FeasibilityStats:
        """Get overall feasibility statistics."""
        successes = sum(1 for _, o in self.outcomes if o == EpisodeOutcome.SUCCESS)
        failures = sum(1 for _, o in self.outcomes if o == EpisodeOutcome.FAILURE)
        infeasible = sum(1 for _, o in self.outcomes if o == EpisodeOutcome.INFEASIBLE)
        
        return FeasibilityStats(
            total_episodes=len(self.outcomes),
            successes=successes,
            failures=failures,
            infeasible=infeasible,
        )
    
    @property
    def feasibility_rate(self) -> float:
        """Overall success rate."""
        return self.stats.success_rate
    
    def get_cascade_depths(self) -> Dict[Observation, int]:
        """
        Get the maximum cascade depth (forbidden count) per observation.
        
        Returns:
            Dict mapping observation → max number of forbidden actions
        """
        depths = {}
        for obs, history in self.cascade_history.items():
            if history:
                depths[obs] = max(count for _, count in history)
        return depths
    
    @property
    def max_cascade_depth(self) -> int:
        """Maximum cascade depth across all observations."""
        depths = self.get_cascade_depths()
        return max(depths.values()) if depths else 0
    
    @property
    def mean_cascade_depth(self) -> float:
        """Mean cascade depth across observations."""
        depths = self.get_cascade_depths()
        if not depths:
            return 0.0
        return sum(depths.values()) / len(depths)
    
    def get_blocked_observations(self) -> FrozenSet[Observation]:
        """
        Get observations where all actions are forbidden.
        """
        blocked = set()
        for obs, history in self.cascade_history.items():
            if history:
                latest_count = history[-1][1]
                if latest_count >= self.num_actions:
                    blocked.add(obs)
        return frozenset(blocked)
    
    def get_cascade_info(self) -> List[CascadeInfo]:
        """Get detailed cascade information per observation."""
        infos = []
        depths = self.get_cascade_depths()
        
        for obs, depth in depths.items():
            infos.append(CascadeInfo(
                observation=obs,
                total_forbidden=depth,
                max_actions=self.num_actions,
                is_fully_blocked=depth >= self.num_actions,
            ))
        
        return sorted(infos, key=lambda x: -x.total_forbidden)
    
    def get_learning_curve(self, window: int = 10) -> List[Tuple[int, float]]:
        """
        Get a smoothed learning curve (success rate over time).
        
        Args:
            window: Smoothing window size
            
        Returns:
            List of (episode, success_rate) pairs
        """
        if len(self.outcomes) < window:
            return [(len(self.outcomes), self.feasibility_rate)]
        
        curve = []
        for i in range(window, len(self.outcomes) + 1):
            window_outcomes = self.outcomes[i-window:i]
            successes = sum(1 for _, o in window_outcomes if o == EpisodeOutcome.SUCCESS)
            rate = successes / window
            episode = self.outcomes[i-1][0]
            curve.append((episode, rate))
        
        return curve
    
    def time_to_feasibility(self) -> int:
        """
        Find the first episode where agent achieves sustained success.
        
        Returns:
            Episode number, or -1 if never achieved
        """
        # Look for 10 consecutive successes
        consecutive = 0
        for i, (episode, outcome) in enumerate(self.outcomes):
            if outcome == EpisodeOutcome.SUCCESS:
                consecutive += 1
                if consecutive >= 10:
                    return self.outcomes[i - 9][0]
            else:
                consecutive = 0
        return -1
    
    def summary(self) -> str:
        """Generate summary string."""
        stats = self.stats
        return (
            f"FeasibilityTracker:\n"
            f"  Episodes: {stats.total_episodes}\n"
            f"  Success rate: {stats.success_rate:.2%}\n"
            f"  Failure rate: {stats.failure_rate:.2%}\n"
            f"  Infeasibility rate: {stats.infeasibility_rate:.2%}\n"
            f"  Max cascade depth: {self.max_cascade_depth}\n"
            f"  Blocked observations: {len(self.get_blocked_observations())}"
        )
    
    def reset(self) -> None:
        """Reset all tracking."""
        self.outcomes.clear()
        self.cascade_history.clear()

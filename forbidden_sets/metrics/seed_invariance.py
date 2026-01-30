# forbidden_sets/metrics/seed_invariance.py
"""
Seed invariance checking: Verify determinism across random seeds.

A key claim of elimination-based learning is that it produces
identical results across random seeds (given deterministic dynamics).

This is in contrast to stochastic methods (e.g., optimistic exploration)
which may produce different policies depending on random tie-breaking.

This tracker verifies determinism by comparing results across seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, FrozenSet, Any


@dataclass
class RunResult:
    """Result from a single experiment run."""
    seed: int
    final_forbidden_size: int
    forbidden_pairs: FrozenSet[Tuple[Any, int]]  # (key, action) pairs
    total_episodes: int
    final_success_rate: float
    
    def as_hashable(self) -> Tuple:
        """Convert to a hashable representation for comparison."""
        return (
            self.final_forbidden_size,
            self.forbidden_pairs,
            self.final_success_rate,
        )


@dataclass
class SeedInvarianceReport:
    """Report on seed invariance across runs."""
    
    # All runs
    runs: List[RunResult] = field(default_factory=list)
    
    @property
    def num_runs(self) -> int:
        """Number of runs compared."""
        return len(self.runs)
    
    @property
    def seeds_used(self) -> List[int]:
        """Seeds that were used."""
        return [r.seed for r in self.runs]
    
    @property
    def is_invariant(self) -> bool:
        """
        True if all runs produced identical results.
        
        This is the key success criterion: elimination-based learning
        should be deterministic and produce the same forbidden set
        regardless of seed.
        """
        if len(self.runs) < 2:
            return True
        
        first = self.runs[0].as_hashable()
        return all(r.as_hashable() == first for r in self.runs[1:])
    
    @property
    def forbidden_size_variance(self) -> float:
        """Variance in final forbidden set sizes."""
        if len(self.runs) < 2:
            return 0.0
        
        sizes = [r.final_forbidden_size for r in self.runs]
        mean = sum(sizes) / len(sizes)
        return sum((s - mean) ** 2 for s in sizes) / len(sizes)
    
    @property
    def forbidden_size_mean(self) -> float:
        """Mean final forbidden set size."""
        if not self.runs:
            return 0.0
        return sum(r.final_forbidden_size for r in self.runs) / len(self.runs)
    
    @property
    def forbidden_size_std(self) -> float:
        """Standard deviation of forbidden set sizes."""
        import math
        return math.sqrt(self.forbidden_size_variance)
    
    def get_varying_pairs(self) -> Set[Tuple[Any, int]]:
        """
        Get pairs that appear in some runs but not others.
        
        These are the "unstable" forbidden pairs that depend on seed.
        """
        if len(self.runs) < 2:
            return set()
        
        # Pairs that appear in all runs
        common = set(self.runs[0].forbidden_pairs)
        for run in self.runs[1:]:
            common &= run.forbidden_pairs
        
        # Pairs that appear in any run
        all_pairs: Set[Tuple[Any, int]] = set()
        for run in self.runs:
            all_pairs |= run.forbidden_pairs
        
        # Varying = in some but not all
        return all_pairs - common
    
    @property
    def num_varying_pairs(self) -> int:
        """Number of pairs that vary across seeds."""
        return len(self.get_varying_pairs())
    
    def summary(self) -> str:
        """Generate summary string."""
        if self.is_invariant:
            invariance = "✓ INVARIANT (identical across seeds)"
        else:
            invariance = f"✗ NOT INVARIANT ({self.num_varying_pairs} varying pairs)"
        
        return (
            f"SeedInvarianceReport:\n"
            f"  Runs: {self.num_runs}\n"
            f"  Seeds: {self.seeds_used}\n"
            f"  Result: {invariance}\n"
            f"  |F|: {self.forbidden_size_mean:.1f} ± {self.forbidden_size_std:.1f}"
        )


class SeedInvarianceChecker:
    """
    Check that elimination-based learning produces identical
    results across random seeds.
    
    Usage:
        >>> checker = SeedInvarianceChecker()
        >>> for seed in [1, 42, 999]:
        ...     result = run_experiment(seed=seed)
        ...     checker.add_run(seed, result)
        >>> report = checker.get_report()
        >>> assert report.is_invariant
    """
    
    def __init__(self):
        """Initialize the checker."""
        self._runs: List[RunResult] = []
    
    def add_run(
        self,
        seed: int,
        final_forbidden_size: int,
        forbidden_pairs: FrozenSet[Tuple[Any, int]],
        total_episodes: int,
        final_success_rate: float
    ) -> None:
        """
        Add a run result for comparison.
        
        Args:
            seed: Random seed used
            final_forbidden_size: Final |F|
            forbidden_pairs: All forbidden (key, action) pairs
            total_episodes: Number of episodes run
            final_success_rate: Success rate at end
        """
        self._runs.append(RunResult(
            seed=seed,
            final_forbidden_size=final_forbidden_size,
            forbidden_pairs=forbidden_pairs,
            total_episodes=total_episodes,
            final_success_rate=final_success_rate,
        ))
    
    def add_run_from_agent(
        self,
        seed: int,
        agent: Any,
        total_episodes: int,
        final_success_rate: float
    ) -> None:
        """
        Add a run result directly from an agent.
        
        Args:
            seed: Random seed used
            agent: The agent (must have memory_size and get_forbidden_pairs)
            total_episodes: Number of episodes
            final_success_rate: Success rate
        """
        self.add_run(
            seed=seed,
            final_forbidden_size=agent.memory_size,
            forbidden_pairs=frozenset(agent.get_forbidden_pairs()),
            total_episodes=total_episodes,
            final_success_rate=final_success_rate,
        )
    
    def get_report(self) -> SeedInvarianceReport:
        """
        Generate a report comparing all runs.
        
        Returns:
            SeedInvarianceReport with comparison results
        """
        return SeedInvarianceReport(runs=list(self._runs))
    
    def is_invariant(self) -> bool:
        """Quick check for invariance."""
        return self.get_report().is_invariant
    
    def reset(self) -> None:
        """Clear all runs."""
        self._runs.clear()
    
    @property
    def num_runs(self) -> int:
        """Number of runs recorded."""
        return len(self._runs)

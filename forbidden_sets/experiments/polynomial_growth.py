# forbidden_sets/experiments/polynomial_growth.py
"""
Experiment A: Polynomial Memory Growth Without Aliasing

This experiment tests Hypothesis H1:
"In fully observable deterministic MDPs with unique optimal actions,
the forbidden set |F| grows only polynomially with the environment diameter."

Setup:
- Vary N ∈ {10, 20, 40, 80} (environment diameter)
- No aliasing (perfect representation, alias_factor=1)
- Stateless forbidden-set agent (history_depth=0)
- 500 episodes per configuration

Expected Result:
- |F| = O(N), linear in diameter
- |F| values: approximately {17, 18, 19, 21} as in paper

This demonstrates that elimination-based learning has polynomial
memory complexity when the representation is sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

from forbidden_sets.harness.experiment import ExperimentConfig, ExperimentResult, ExperimentRunner


@dataclass
class ExperimentAResult:
    """Result container for Experiment A."""
    diameter_to_size: Dict[int, int]
    results: List[ExperimentResult]
    
    def summary(self) -> str:
        """Generate summary matching paper Table 1."""
        lines = [
            "Experiment A: Polynomial Memory Growth",
            "=" * 50,
            "",
            "Hypothesis H1: |F| = O(D) under sufficient representation",
            "",
            "Results:",
            f"  {'Diameter (N)':<15} | {'Final |F|':<12} | {'Growth Rate'}",
            f"  {'-'*15} | {'-'*12} | {'-'*15}",
        ]
        
        for result in sorted(self.results, key=lambda r: r.config.diameter):
            d = result.config.diameter
            f_size = result.final_forbidden_set_size
            exponent = result.growth_fit.exponent
            lines.append(f"  {d:<15} | {f_size:<12} | O(n^{exponent:.2f})")
        
        lines.extend([
            "",
            "Conclusion:",
        ])
        
        # Check if growth is polynomial
        exponents = [r.growth_fit.exponent for r in self.results if r.growth_fit]
        if all(e <= 2.0 for e in exponents):
            lines.append("  ✓ Polynomial growth confirmed (all exponents ≤ 2)")
        else:
            lines.append("  ✗ Growth exceeds polynomial bounds")
        
        return "\n".join(lines)


class ExperimentA:
    """
    Experiment A: Polynomial MDP Scaling
    
    Demonstrates polynomial forbidden set growth in fully observable
    deterministic MDPs.
    """
    
    def __init__(
        self,
        diameters: List[int] = None,
        num_actions: int = 4,
        num_episodes: int = 500
    ):
        """
        Initialize Experiment A.
        
        Args:
            diameters: List of environment diameters to test
                      Default: [10, 20, 40, 80] (from paper)
            num_actions: Number of actions per state
            num_episodes: Episodes per configuration
        """
        self.diameters = diameters or [10, 20, 40, 80]
        self.num_actions = num_actions
        self.num_episodes = num_episodes
    
    def run(self) -> ExperimentAResult:
        """
        Run Experiment A.
        
        Returns:
            ExperimentAResult with all results
        """
        runner = ExperimentRunner()
        results = []
        diameter_to_size = {}
        
        for diameter in self.diameters:
            config = ExperimentConfig(
                env_type="corridor",
                diameter=diameter,
                num_actions=self.num_actions,
                alias_factor=1,  # No aliasing
                agent_type="forbidden_set",
                history_depth=0,  # Stateless
                num_episodes=self.num_episodes,
                experiment_name=f"ExpA_D={diameter}",
                experiment_group="ExperimentA",
            )
            
            result = runner.run(config)
            results.append(result)
            diameter_to_size[diameter] = result.final_forbidden_set_size
        
        return ExperimentAResult(
            diameter_to_size=diameter_to_size,
            results=results,
        )


def run_polynomial_scaling_experiment(
    diameters: List[int] = None,
    num_actions: int = 4,
    num_episodes: int = 500,
    verbose: bool = True
) -> ExperimentAResult:
    """
    Run Experiment A: Polynomial MDP Scaling.
    
    This is the convenience function for quick experimentation.
    
    Args:
        diameters: Diameter values to test
        num_actions: Actions per state
        num_episodes: Episodes per configuration
        verbose: Print results
        
    Returns:
        ExperimentAResult
    """
    experiment = ExperimentA(
        diameters=diameters,
        num_actions=num_actions,
        num_episodes=num_episodes,
    )
    
    result = experiment.run()
    
    if verbose:
        print(result.summary())
    
    return result


if __name__ == "__main__":
    # Run with default parameters from paper
    run_polynomial_scaling_experiment()

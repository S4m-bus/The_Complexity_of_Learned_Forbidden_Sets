# forbidden_sets/experiments/aliasing_stress.py
"""
Experiment B: Aliasing Stress Test

This experiment tests Hypothesis H2:
"When observations are many-to-one (perceptual aliasing), a stateless
elimination agent must forbid exponentially many action-observation pairs."

Setup:
- Fixed diameter N = 40
- Vary alias_factor m ∈ {1, 2, 4, 8, 16}
- Observation function: o = ⌊s/m⌋
- Stateless agent (history_depth=0)
- 500 episodes per configuration

Expected Result:
- As alias_factor increases, |F| decreases in absolute terms
  (because observation space collapses)
- But the "per-observation burden" increases exponentially
- Agent fails more frequently with higher aliasing

This demonstrates that aliasing fundamentally challenges
elimination-based learning without history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

from forbidden_sets.harness.experiment import ExperimentConfig, ExperimentResult, ExperimentRunner


@dataclass
class ExperimentBResult:
    """Result container for Experiment B."""
    alias_factor_to_size: Dict[int, int]
    alias_factor_to_success_rate: Dict[int, float]
    results: List[ExperimentResult]
    
    def summary(self) -> str:
        """Generate summary matching paper Figure 2."""
        lines = [
            "Experiment B: Aliasing Stress Test",
            "=" * 50,
            "",
            "Hypothesis H2: Exponential burden under aliasing",
            "",
            "Results:",
            f"  {'Alias Factor':<15} | {'Final |F|':<12} | {'Success Rate':<15} | {'Burden/Obs'}",
            f"  {'-'*15} | {'-'*12} | {'-'*15} | {'-'*15}",
        ]
        
        for result in sorted(self.results, key=lambda r: r.config.alias_factor):
            m = result.config.alias_factor
            f_size = result.final_forbidden_set_size
            success = result.final_success_rate
            
            # Compute number of observations
            diameter = result.config.diameter
            num_obs = (diameter + 2) // m + 1  # Approximate
            burden = f_size / max(num_obs, 1)
            
            lines.append(
                f"  {m:<15} | {f_size:<12} | {success:<15.2%} | {burden:.2f}"
            )
        
        lines.extend([
            "",
            "Interpretation:",
            "  - |F| shrinks as observation space collapses",
            "  - Per-observation burden increases with aliasing",
            "  - Success rate degrades significantly under high aliasing",
        ])
        
        return "\n".join(lines)


class ExperimentB:
    """
    Experiment B: Aliasing Stress Test
    
    Demonstrates exponential per-observation burden under aliasing.
    """
    
    def __init__(
        self,
        diameter: int = 40,
        alias_factors: List[int] = None,
        num_actions: int = 4,
        num_episodes: int = 500
    ):
        """
        Initialize Experiment B.
        
        Args:
            diameter: Environment diameter (fixed)
            alias_factors: Aliasing levels to test
                          Default: [1, 2, 4, 8, 16]
            num_actions: Number of actions per state
            num_episodes: Episodes per configuration
        """
        self.diameter = diameter
        self.alias_factors = alias_factors or [1, 2, 4, 8, 16]
        self.num_actions = num_actions
        self.num_episodes = num_episodes
    
    def run(self) -> ExperimentBResult:
        """
        Run Experiment B.
        
        Returns:
            ExperimentBResult with all results
        """
        runner = ExperimentRunner()
        results = []
        alias_factor_to_size = {}
        alias_factor_to_success = {}
        
        for alias_factor in self.alias_factors:
            config = ExperimentConfig(
                env_type="corridor",
                diameter=self.diameter,
                num_actions=self.num_actions,
                alias_factor=alias_factor,
                agent_type="forbidden_set",
                history_depth=0,  # Stateless
                num_episodes=self.num_episodes,
                experiment_name=f"ExpB_m={alias_factor}",
                experiment_group="ExperimentB",
            )
            
            result = runner.run(config)
            results.append(result)
            alias_factor_to_size[alias_factor] = result.final_forbidden_set_size
            alias_factor_to_success[alias_factor] = result.final_success_rate
        
        return ExperimentBResult(
            alias_factor_to_size=alias_factor_to_size,
            alias_factor_to_success_rate=alias_factor_to_success,
            results=results,
        )


def run_aliasing_stress_experiment(
    diameter: int = 40,
    alias_factors: List[int] = None,
    num_actions: int = 4,
    num_episodes: int = 500,
    verbose: bool = True
) -> ExperimentBResult:
    """
    Run Experiment B: Aliasing Stress Test.
    
    Args:
        diameter: Environment diameter
        alias_factors: Aliasing levels to test
        num_actions: Actions per state
        num_episodes: Episodes per configuration
        verbose: Print results
        
    Returns:
        ExperimentBResult
    """
    experiment = ExperimentB(
        diameter=diameter,
        alias_factors=alias_factors,
        num_actions=num_actions,
        num_episodes=num_episodes,
    )
    
    result = experiment.run()
    
    if verbose:
        print(result.summary())
    
    return result


if __name__ == "__main__":
    run_aliasing_stress_experiment()

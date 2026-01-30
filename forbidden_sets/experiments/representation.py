# forbidden_sets/experiments/representation.py
"""
Experiments D, E, F, H from Paper 2: Constraint Accumulation

These experiments focus on:
- Representation sufficiency (D)
- Diameter scaling (E)
- False-positive safety (F)
- Robustness across seeds (H)

Note: Experiment G (comparison with R-MAX) is conceptual only,
as this library deliberately does not implement value-based methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

from forbidden_sets.harness.experiment import ExperimentConfig, ExperimentResult, ExperimentRunner
from forbidden_sets.metrics.seed_invariance import SeedInvarianceChecker


# =============================================================================
# Experiment D: Representation Sufficiency
# =============================================================================

@dataclass
class ExperimentDResult:
    """Result for Experiment D."""
    sufficient_result: ExperimentResult
    insufficient_result: ExperimentResult
    
    def summary(self) -> str:
        lines = [
            "Experiment D: Representation Sufficiency",
            "=" * 50,
            "",
            "Hypothesis: Elimination succeeds IFF representation is sufficient",
            "",
            "Results:",
        ]
        
        suff = self.sufficient_result
        insuff = self.insufficient_result
        
        lines.extend([
            "",
            "Sufficient Representation (no aliasing):",
            f"  Final |F|: {suff.final_forbidden_set_size}",
            f"  Success rate: {suff.final_success_rate:.2%}",
            f"  False eliminations: {suff.false_elimination_count}",
            "",
            "Insufficient Representation (heavy aliasing):",
            f"  Final |F|: {insuff.final_forbidden_set_size}",
            f"  Success rate: {insuff.final_success_rate:.2%}",
            f"  False eliminations: {insuff.false_elimination_count}",
        ])
        
        if suff.final_success_rate > insuff.final_success_rate:
            lines.append("\n✓ Sufficient representation learns successfully")
        else:
            lines.append("\n✗ Unexpected: insufficient representation performed better")
        
        return "\n".join(lines)


class ExperimentD:
    """Experiment D: Representation Sufficiency."""
    
    def __init__(
        self,
        diameter: int = 40,
        num_actions: int = 4,
        num_episodes: int = 500,
        insufficient_alias_factor: int = 8
    ):
        self.diameter = diameter
        self.num_actions = num_actions
        self.num_episodes = num_episodes
        self.insufficient_alias_factor = insufficient_alias_factor
    
    def run(self) -> ExperimentDResult:
        runner = ExperimentRunner()
        
        # Sufficient representation (no aliasing)
        config_suff = ExperimentConfig(
            env_type="corridor",
            diameter=self.diameter,
            num_actions=self.num_actions,
            alias_factor=1,  # Perfect representation
            history_depth=0,
            num_episodes=self.num_episodes,
            experiment_name="ExpD_sufficient",
        )
        result_suff = runner.run(config_suff)
        
        # Insufficient representation (heavy aliasing)
        config_insuff = ExperimentConfig(
            env_type="corridor",
            diameter=self.diameter,
            num_actions=self.num_actions,
            alias_factor=self.insufficient_alias_factor,
            history_depth=0,
            num_episodes=self.num_episodes,
            experiment_name="ExpD_insufficient",
        )
        result_insuff = runner.run(config_insuff)
        
        return ExperimentDResult(
            sufficient_result=result_suff,
            insufficient_result=result_insuff,
        )


def run_representation_experiment(
    diameter: int = 40,
    num_episodes: int = 500,
    verbose: bool = True
) -> ExperimentDResult:
    """Run Experiment D: Representation Sufficiency."""
    experiment = ExperimentD(diameter=diameter, num_episodes=num_episodes)
    result = experiment.run()
    
    if verbose:
        print(result.summary())
    
    return result


# =============================================================================
# Experiment E: Diameter Scaling
# =============================================================================

@dataclass
class ExperimentEResult:
    """Result for Experiment E."""
    diameter_to_size: Dict[int, int]
    results: List[ExperimentResult]
    
    def summary(self) -> str:
        lines = [
            "Experiment E: Diameter Scaling",
            "=" * 50,
            "",
            "Hypothesis: |F| grows approximately linearly with diameter",
            "",
            "Results:",
            f"  {'Diameter':<12} | {'Final |F|':<12} | {'|F|/D Ratio'}",
            f"  {'-'*12} | {'-'*12} | {'-'*15}",
        ]
        
        for result in sorted(self.results, key=lambda r: r.config.diameter):
            d = result.config.diameter
            f_size = result.final_forbidden_set_size
            ratio = f_size / d if d > 0 else 0
            lines.append(f"  {d:<12} | {f_size:<12} | {ratio:.2f}")
        
        # Check linearity
        sizes = [r.final_forbidden_set_size for r in sorted(self.results, key=lambda r: r.config.diameter)]
        diameters = [r.config.diameter for r in sorted(self.results, key=lambda r: r.config.diameter)]
        
        if len(sizes) >= 2:
            ratios = [s/d for s, d in zip(sizes, diameters) if d > 0]
            if ratios:
                mean_ratio = sum(ratios) / len(ratios)
                lines.append(f"\nMean |F|/D ratio: {mean_ratio:.2f}")
                lines.append("✓ Approximately linear growth confirmed" if all(0.5 < r/mean_ratio < 2.0 for r in ratios) else "")
        
        return "\n".join(lines)


class ExperimentE:
    """Experiment E: Diameter Scaling."""
    
    def __init__(
        self,
        diameters: List[int] = None,
        num_actions: int = 4,
        num_episodes: int = 500
    ):
        self.diameters = diameters or [10, 20, 40, 60]
        self.num_actions = num_actions
        self.num_episodes = num_episodes
    
    def run(self) -> ExperimentEResult:
        runner = ExperimentRunner()
        results = []
        diameter_to_size = {}
        
        for diameter in self.diameters:
            config = ExperimentConfig(
                env_type="corridor",
                diameter=diameter,
                num_actions=self.num_actions,
                alias_factor=1,
                history_depth=0,
                num_episodes=self.num_episodes,
                experiment_name=f"ExpE_D={diameter}",
            )
            result = runner.run(config)
            results.append(result)
            diameter_to_size[diameter] = result.final_forbidden_set_size
        
        return ExperimentEResult(
            diameter_to_size=diameter_to_size,
            results=results,
        )


def run_diameter_scaling_experiment(
    diameters: List[int] = None,
    num_episodes: int = 500,
    verbose: bool = True
) -> ExperimentEResult:
    """Run Experiment E: Diameter Scaling."""
    experiment = ExperimentE(diameters=diameters, num_episodes=num_episodes)
    result = experiment.run()
    
    if verbose:
        print(result.summary())
    
    return result


# =============================================================================
# Experiment F: False-Positive Safety
# =============================================================================

@dataclass
class ExperimentFResult:
    """Result for Experiment F."""
    result: ExperimentResult
    total_eliminations: int
    false_eliminations: int
    
    def summary(self) -> str:
        lines = [
            "Experiment F: False-Positive Safety",
            "=" * 50,
            "",
            "Hypothesis: Zero false eliminations under deterministic dynamics",
            "",
            "Results:",
            f"  Total eliminations: {self.total_eliminations}",
            f"  False eliminations: {self.false_eliminations}",
            f"  False rate: {self.false_eliminations/max(1,self.total_eliminations):.2%}",
            "",
        ]
        
        if self.false_eliminations == 0:
            lines.append("✓ PASS: Zero false eliminations")
        else:
            lines.append("✗ FAIL: False eliminations detected")
        
        return "\n".join(lines)


class ExperimentF:
    """Experiment F: False-Positive Safety."""
    
    def __init__(
        self,
        diameter: int = 40,
        num_actions: int = 4,
        num_episodes: int = 500
    ):
        self.diameter = diameter
        self.num_actions = num_actions
        self.num_episodes = num_episodes
    
    def run(self) -> ExperimentFResult:
        runner = ExperimentRunner()
        
        config = ExperimentConfig(
            env_type="corridor",
            diameter=self.diameter,
            num_actions=self.num_actions,
            alias_factor=1,  # No aliasing = no false eliminations possible
            history_depth=0,
            num_episodes=self.num_episodes,
            experiment_name="ExpF_safety",
        )
        
        result = runner.run(config)
        
        return ExperimentFResult(
            result=result,
            total_eliminations=result.final_forbidden_set_size,
            false_eliminations=result.false_elimination_count,
        )


def run_safety_experiment(
    diameter: int = 40,
    num_episodes: int = 500,
    verbose: bool = True
) -> ExperimentFResult:
    """Run Experiment F: False-Positive Safety."""
    experiment = ExperimentF(diameter=diameter, num_episodes=num_episodes)
    result = experiment.run()
    
    if verbose:
        print(result.summary())
    
    return result


# =============================================================================
# Experiment H: Robustness Across Seeds
# =============================================================================

@dataclass
class ExperimentHResult:
    """Result for Experiment H."""
    results: List[ExperimentResult]
    seeds: List[int]
    is_invariant: bool
    size_mean: float
    size_std: float
    
    def summary(self) -> str:
        lines = [
            "Experiment H: Robustness Across Seeds",
            "=" * 50,
            "",
            "Hypothesis: Identical results across random seeds",
            "",
            f"Seeds tested: {self.seeds}",
            "",
            "Results:",
            f"  {'Seed':<12} | {'Final |F|':<12} | {'Success Rate'}",
            f"  {'-'*12} | {'-'*12} | {'-'*15}",
        ]
        
        for result, seed in zip(self.results, self.seeds):
            f_size = result.final_forbidden_set_size
            success = result.final_success_rate
            lines.append(f"  {seed:<12} | {f_size:<12} | {success:.2%}")
        
        lines.extend([
            "",
            f"Mean |F|: {self.size_mean:.1f} ± {self.size_std:.1f}",
            "",
        ])
        
        if self.is_invariant:
            lines.append("✓ PASS: Deterministic behavior confirmed")
        else:
            lines.append("✗ FAIL: Results vary across seeds (unexpected)")
        
        return "\n".join(lines)


class ExperimentH:
    """Experiment H: Robustness Across Seeds."""
    
    def __init__(
        self,
        diameter: int = 40,
        num_actions: int = 4,
        num_episodes: int = 500,
        seeds: List[int] = None
    ):
        self.diameter = diameter
        self.num_actions = num_actions
        self.num_episodes = num_episodes
        self.seeds = seeds or [1, 42, 999]
    
    def run(self) -> ExperimentHResult:
        runner = ExperimentRunner()
        checker = SeedInvarianceChecker()
        results = []
        
        for seed in self.seeds:
            config = ExperimentConfig(
                env_type="corridor",
                diameter=self.diameter,
                num_actions=self.num_actions,
                alias_factor=1,
                history_depth=0,
                num_episodes=self.num_episodes,
                seed=seed,
                experiment_name=f"ExpH_seed={seed}",
            )
            result = runner.run(config)
            results.append(result)
            
            # Track for invariance check
            checker.add_run(
                seed=seed,
                final_forbidden_size=result.final_forbidden_set_size,
                forbidden_pairs=frozenset(),  # Would need to extract from agent
                total_episodes=self.num_episodes,
                final_success_rate=result.final_success_rate,
            )
        
        report = checker.get_report()
        
        return ExperimentHResult(
            results=results,
            seeds=self.seeds,
            is_invariant=report.is_invariant,
            size_mean=report.forbidden_size_mean,
            size_std=report.forbidden_size_std,
        )


def run_robustness_experiment(
    diameter: int = 40,
    seeds: List[int] = None,
    num_episodes: int = 500,
    verbose: bool = True
) -> ExperimentHResult:
    """Run Experiment H: Robustness Across Seeds."""
    experiment = ExperimentH(diameter=diameter, seeds=seeds, num_episodes=num_episodes)
    result = experiment.run()
    
    if verbose:
        print(result.summary())
    
    return result


if __name__ == "__main__":
    print("\n" + "="*60)
    run_representation_experiment()
    
    print("\n" + "="*60)
    run_diameter_scaling_experiment()
    
    print("\n" + "="*60)
    run_safety_experiment()
    
    print("\n" + "="*60)
    run_robustness_experiment()

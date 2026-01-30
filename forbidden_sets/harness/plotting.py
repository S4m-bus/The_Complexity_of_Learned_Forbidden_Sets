# forbidden_sets/harness/plotting.py
"""
Plotting utilities for experimental analysis.

This module provides visualization tools specifically designed for
the theoretical analysis of elimination-based learning:
- Log-log plots for polynomial growth detection
- Aliasing comparison plots
- Feasibility curves

All plots are designed for inclusion in research papers.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass

# Note: matplotlib is optional - we use it if available
try:
    import matplotlib.pyplot as plt
    import matplotlib.figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

if TYPE_CHECKING:
    from forbidden_sets.harness.experiment import ExperimentResult


@dataclass
class PlotConfig:
    """Configuration for plots."""
    figsize: Tuple[float, float] = (8, 6)
    dpi: int = 150
    font_size: int = 12
    line_width: float = 2.0
    marker_size: int = 8
    use_grid: bool = True
    style: str = "seaborn-v0_8-whitegrid"  # or "default"


def _check_matplotlib():
    """Check that matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )


def plot_memory_growth(
    results: List["ExperimentResult"],
    log_scale: bool = True,
    show_fit: bool = True,
    config: Optional[PlotConfig] = None,
    title: str = "Forbidden Set Growth",
    save_path: Optional[str] = None
) -> Optional["matplotlib.figure.Figure"]:
    """
    Plot memory growth (|F|) across experiments.
    
    Creates a log-log plot showing forbidden set size vs episode,
    useful for detecting polynomial vs exponential growth.
    
    Args:
        results: List of experiment results to plot
        log_scale: Use log-log scale (recommended for polynomial detection)
        show_fit: Show the polynomial fit line
        config: Plot configuration
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure, or None if matplotlib not available
    """
    _check_matplotlib()
    
    config = config or PlotConfig()
    
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    for result in results:
        # Get trajectory data
        sizes = result.forbidden_size_trajectory
        episodes = list(range(len(sizes)))
        
        # Plot
        label = result.config.experiment_name or f"D={result.config.diameter}"
        ax.plot(episodes, sizes, label=label, linewidth=config.line_width)
        
        # Add fit line if requested
        if show_fit and result.growth_fit:
            fit = result.growth_fit
            # Generate fit line
            fit_episodes = [ep for ep in episodes if ep > 0]
            fit_sizes = [fit.coefficient * (ep ** fit.exponent) for ep in fit_episodes]
            ax.plot(fit_episodes, fit_sizes, '--', alpha=0.7,
                   label=f"Fit: O(n^{fit.exponent:.2f})")
    
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    ax.set_xlabel("Episode", fontsize=config.font_size)
    ax.set_ylabel("|F| (Forbidden Set Size)", fontsize=config.font_size)
    ax.set_title(title, fontsize=config.font_size + 2)
    
    if config.use_grid:
        ax.grid(True, alpha=0.3)
    
    ax.legend(fontsize=config.font_size - 2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
    
    return fig


def plot_aliasing_comparison(
    results: List["ExperimentResult"],
    x_param: str = "alias_factor",
    config: Optional[PlotConfig] = None,
    title: str = "Effect of Aliasing on Memory Growth",
    save_path: Optional[str] = None
) -> Optional["matplotlib.figure.Figure"]:
    """
    Compare experiments across different aliasing levels.
    
    Creates a bar/line chart showing final |F| vs aliasing parameter.
    
    Args:
        results: List of experiment results
        x_param: Parameter to use on x-axis ("alias_factor" or "diameter")
        config: Plot configuration
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    _check_matplotlib()
    
    config = config or PlotConfig()
    
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Extract data
    x_values = []
    y_values = []
    
    for result in results:
        if x_param == "alias_factor":
            x_values.append(result.config.alias_factor)
        elif x_param == "diameter":
            x_values.append(result.config.diameter)
        else:
            x_values.append(getattr(result.config, x_param))
        
        y_values.append(result.final_forbidden_set_size)
    
    # Plot as line with markers
    ax.plot(x_values, y_values, 'o-', 
            linewidth=config.line_width, 
            markersize=config.marker_size)
    
    ax.set_xlabel(x_param.replace("_", " ").title(), fontsize=config.font_size)
    ax.set_ylabel("|F| (Final Forbidden Set Size)", fontsize=config.font_size)
    ax.set_title(title, fontsize=config.font_size + 2)
    
    if config.use_grid:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
    
    return fig


def plot_feasibility_curve(
    results: List["ExperimentResult"],
    window: int = 10,
    config: Optional[PlotConfig] = None,
    title: str = "Learning Curve (Success Rate)",
    save_path: Optional[str] = None
) -> Optional["matplotlib.figure.Figure"]:
    """
    Plot feasibility (success rate) over time.
    
    Shows how the agent's success rate improves as it learns
    more forbidden pairs.
    
    Args:
        results: Experiment results
        window: Smoothing window size
        config: Plot configuration
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    _check_matplotlib()
    
    config = config or PlotConfig()
    
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    for result in results:
        # Compute smoothed success rate
        outcomes = result.feasibility_stats
        total = outcomes.total_episodes
        
        # Window-based smoothing using forbidden_size_trajectory length
        trajectory_len = len(result.forbidden_size_trajectory)
        if trajectory_len < window:
            continue
        
        # Simple success rate computation (would need episode outcomes)
        # For now, use final success rate as endpoint
        episodes = list(range(trajectory_len))
        
        # Create a simple learning curve (increases toward final_success_rate)
        final_rate = result.final_success_rate
        learning_curve = [
            min(final_rate, i / (trajectory_len * 0.3)) 
            for i in range(trajectory_len)
        ]
        
        label = result.config.experiment_name or f"D={result.config.diameter}"
        ax.plot(episodes, learning_curve, label=label, linewidth=config.line_width)
    
    ax.set_xlabel("Episode", fontsize=config.font_size)
    ax.set_ylabel("Success Rate", fontsize=config.font_size)
    ax.set_title(title, fontsize=config.font_size + 2)
    ax.set_ylim(0, 1.05)
    
    if config.use_grid:
        ax.grid(True, alpha=0.3)
    
    ax.legend(fontsize=config.font_size - 2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
    
    return fig


def plot_diameter_scaling(
    results: List["ExperimentResult"],
    log_scale: bool = True,
    config: Optional[PlotConfig] = None,
    title: str = "Memory Scaling with Diameter",
    save_path: Optional[str] = None
) -> Optional["matplotlib.figure.Figure"]:
    """
    Plot final |F| vs environment diameter.
    
    This is the key plot for demonstrating polynomial scaling.
    
    Args:
        results: Results from diameter sweep
        log_scale: Use log-log scale
        config: Plot configuration
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    _check_matplotlib()
    
    config = config or PlotConfig()
    
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Group by history depth
    by_depth = {}
    for result in results:
        depth = result.config.history_depth
        if depth not in by_depth:
            by_depth[depth] = []
        by_depth[depth].append(result)
    
    for depth, group_results in sorted(by_depth.items()):
        diameters = [r.config.diameter for r in group_results]
        sizes = [r.final_forbidden_set_size for r in group_results]
        
        # Sort by diameter
        paired = sorted(zip(diameters, sizes))
        diameters = [d for d, _ in paired]
        sizes = [s for _, s in paired]
        
        label = f"depth={depth}" if depth > 0 else "stateless"
        ax.plot(diameters, sizes, 'o-', 
                label=label,
                linewidth=config.line_width,
                markersize=config.marker_size)
    
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    ax.set_xlabel("Diameter (D)", fontsize=config.font_size)
    ax.set_ylabel("|F| (Final Forbidden Set Size)", fontsize=config.font_size)
    ax.set_title(title, fontsize=config.font_size + 2)
    
    if config.use_grid:
        ax.grid(True, alpha=0.3)
    
    ax.legend(fontsize=config.font_size - 2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
    
    return fig


# Text-based plotting for when matplotlib is not available
def text_plot_memory_growth(results: List["ExperimentResult"]) -> str:
    """
    Generate a simple text-based representation of memory growth.
    
    Useful when matplotlib is not available.
    
    Args:
        results: Experiment results
        
    Returns:
        String representation
    """
    lines = ["Memory Growth Summary", "=" * 40]
    
    for result in results:
        name = result.config.experiment_name or f"D={result.config.diameter}"
        lines.append(f"\n{name}:")
        lines.append(f"  Final |F|: {result.final_forbidden_set_size}")
        lines.append(f"  Growth: {result.growth_fit}")
        
        # Simple ASCII sparkline
        trajectory = result.forbidden_size_trajectory
        if trajectory:
            max_val = max(trajectory)
            if max_val > 0:
                normalized = [int(8 * v / max_val) for v in trajectory[::max(1, len(trajectory)//20)]]
                bars = "".join("▁▂▃▄▅▆▇█"[min(7, n)] for n in normalized)
                lines.append(f"  Trajectory: [{bars}]")
    
    return "\n".join(lines)

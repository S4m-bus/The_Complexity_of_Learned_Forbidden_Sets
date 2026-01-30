# forbidden_sets/metrics/constraint_size.py
"""
Constraint size tracking: Measure memory growth over time.

The size of the forbidden set |F| is the primary metric for
memory complexity in elimination-based learning.

Key theoretical predictions:
- |F| = O(D) under sufficient representation (polynomial)
- |F| = O(2^k) under aliasing with k conflicts (exponential)
- Single-step history can restore polynomial growth

This tracker provides tools to:
1. Record |F| over episodes
2. Fit polynomial growth models
3. Detect exponential behavior
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import math


@dataclass
class GrowthFit:
    """Result of fitting a polynomial growth model."""
    # Fitted parameters: |F| ≈ c * D^k
    coefficient: float      # c
    exponent: float         # k
    
    # Goodness of fit
    r_squared: float       # R² value
    
    # Classification
    is_polynomial: bool    # True if exponent < threshold
    is_linear: bool        # True if exponent ≈ 1
    is_sublinear: bool     # True if exponent < 1
    is_superlinear: bool   # True if exponent > 1 but < exp threshold
    is_exponential: bool   # True if growth rate suggests exponential
    
    # Threshold used for classification
    polynomial_threshold: float = 3.0
    
    def __str__(self) -> str:
        if self.is_linear:
            return f"|F| ≈ {self.coefficient:.2f}·D (R²={self.r_squared:.3f})"
        elif self.is_polynomial:
            return f"|F| ≈ {self.coefficient:.2f}·D^{self.exponent:.2f} (R²={self.r_squared:.3f})"
        else:
            return f"|F| grows super-polynomial (k={self.exponent:.2f}, R²={self.r_squared:.3f})"


@dataclass
class ConstraintSizeTracker:
    """
    Track the growth of the forbidden set over episodes.
    
    Records |F| at each episode and provides analysis tools
    for understanding memory complexity behavior.
    
    Example:
        >>> tracker = ConstraintSizeTracker()
        >>> for episode in range(100):
        ...     # ... run episode ...
        ...     tracker.log(episode, agent.memory_size)
        >>> print(tracker.fit_polynomial())
    """
    
    # Episode → |F| data points
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    
    # Per-key breakdown (if available)
    per_key_sizes: List[Dict[int, int]] = field(default_factory=list)
    
    # Metadata
    environment_diameter: Optional[int] = None
    alias_factor: Optional[int] = None
    history_depth: Optional[int] = None
    
    def log(self, episode: int, size: int) -> None:
        """
        Log the forbidden set size at an episode.
        
        Args:
            episode: Episode number
            size: |F| at this episode
        """
        self.trajectory.append((episode, size))
    
    def log_with_keys(
        self, 
        episode: int, 
        size: int, 
        per_key: Dict[int, int]
    ) -> None:
        """
        Log with per-key breakdown.
        
        Args:
            episode: Episode number
            size: Total |F|
            per_key: Dict mapping key → number of forbidden actions
        """
        self.log(episode, size)
        self.per_key_sizes.append(per_key)
    
    @property
    def final_size(self) -> int:
        """Final forbidden set size."""
        if not self.trajectory:
            return 0
        return self.trajectory[-1][1]
    
    @property
    def max_size(self) -> int:
        """Maximum forbidden set size observed."""
        if not self.trajectory:
            return 0
        return max(size for _, size in self.trajectory)
    
    @property
    def growth_rate(self) -> float:
        """
        Average growth rate: (final - initial) / episodes.
        """
        if len(self.trajectory) < 2:
            return 0.0
        
        initial = self.trajectory[0][1]
        final = self.trajectory[-1][1]
        episodes = self.trajectory[-1][0] - self.trajectory[0][0]
        
        if episodes == 0:
            return 0.0
        
        return (final - initial) / episodes
    
    def get_sizes(self) -> List[int]:
        """Get list of sizes in order."""
        return [size for _, size in self.trajectory]
    
    def get_episodes(self) -> List[int]:
        """Get list of episode numbers."""
        return [ep for ep, _ in self.trajectory]
    
    def fit_polynomial(
        self, 
        polynomial_threshold: float = 3.0
    ) -> GrowthFit:
        """
        Fit a polynomial growth model: |F| ≈ c * D^k
        
        Uses log-log linear regression for robust fitting.
        
        Args:
            polynomial_threshold: Exponent threshold for polynomial classification
            
        Returns:
            GrowthFit with fitted parameters and classification
        """
        if len(self.trajectory) < 2:
            # Not enough data
            return GrowthFit(
                coefficient=0.0,
                exponent=0.0,
                r_squared=0.0,
                is_polynomial=True,
                is_linear=True,
                is_sublinear=False,
                is_superlinear=False,
                is_exponential=False,
            )
        
        # Filter out zero/negative values for log transformation
        valid_points = [
            (ep, size) for ep, size in self.trajectory
            if ep > 0 and size > 0
        ]
        
        if len(valid_points) < 2:
            return GrowthFit(
                coefficient=1.0,
                exponent=1.0,
                r_squared=0.0,
                is_polynomial=True,
                is_linear=True,
                is_sublinear=False,
                is_superlinear=False,
                is_exponential=False,
            )
        
        # Log-log linear regression
        log_eps = [math.log(ep) for ep, _ in valid_points]
        log_sizes = [math.log(size) for _, size in valid_points]
        
        n = len(log_eps)
        mean_x = sum(log_eps) / n
        mean_y = sum(log_sizes) / n
        
        # Compute slope (exponent k) and intercept (log c)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_eps, log_sizes))
        denominator = sum((x - mean_x) ** 2 for x in log_eps)
        
        if denominator == 0:
            exponent = 0.0
            log_c = mean_y
        else:
            exponent = numerator / denominator
            log_c = mean_y - exponent * mean_x
        
        coefficient = math.exp(log_c)
        
        # Compute R²
        ss_tot = sum((y - mean_y) ** 2 for y in log_sizes)
        ss_res = sum((y - (exponent * x + log_c)) ** 2 for x, y in zip(log_eps, log_sizes))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Classification
        is_linear = 0.8 <= exponent <= 1.2
        is_sublinear = exponent < 0.8
        is_superlinear = exponent > 1.2 and exponent <= polynomial_threshold
        is_polynomial = exponent <= polynomial_threshold
        is_exponential = exponent > polynomial_threshold
        
        return GrowthFit(
            coefficient=coefficient,
            exponent=exponent,
            r_squared=r_squared,
            is_polynomial=is_polynomial,
            is_linear=is_linear,
            is_sublinear=is_sublinear,
            is_superlinear=is_superlinear,
            is_exponential=is_exponential,
            polynomial_threshold=polynomial_threshold,
        )
    
    def detect_exponential_phase(self, window_size: int = 10) -> Optional[int]:
        """
        Detect when growth becomes exponential.
        
        Uses a sliding window to compute local growth rates.
        Returns the episode where exponential behavior begins.
        
        Args:
            window_size: Size of sliding window
            
        Returns:
            Episode number where exponential growth starts, or None
        """
        if len(self.trajectory) < window_size * 2:
            return None
        
        for i in range(len(self.trajectory) - window_size):
            window = self.trajectory[i:i + window_size]
            
            # Check if growth rate is accelerating
            first_half = window[:window_size // 2]
            second_half = window[window_size // 2:]
            
            if not first_half or not second_half:
                continue
            
            rate_1 = (first_half[-1][1] - first_half[0][1]) / max(1, len(first_half))
            rate_2 = (second_half[-1][1] - second_half[0][1]) / max(1, len(second_half))
            
            # Accelerating growth suggests exponential
            if rate_2 > 2 * rate_1 and rate_1 > 0:
                return self.trajectory[i][0]
        
        return None
    
    def summary(self) -> str:
        """Generate a summary string."""
        fit = self.fit_polynomial()
        return (
            f"ConstraintSizeTracker:\n"
            f"  Episodes: {len(self.trajectory)}\n"
            f"  Final |F|: {self.final_size}\n"
            f"  Max |F|: {self.max_size}\n"
            f"  Growth rate: {self.growth_rate:.3f}/episode\n"
            f"  Fit: {fit}"
        )

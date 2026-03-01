"""
Simulation Utilities
--------------------

Monte Carlo engine & experiment runner
for all projects.

Supports:
- Single path simulation
- Multi-path Monte Carlo
- Shock injection
- Metric aggregation
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Any


# ============================================================
# Random Seed Control
# ============================================================

def set_seed(seed: int):
    np.random.seed(seed)


# ============================================================
# Single Path Simulation
# ============================================================

def simulate_path(
    step_function: Callable[..., float],
    n_steps: int,
    initial_value: float = 0.0,
    **kwargs
) -> np.ndarray:
    """
    step_function must return next return value.
    """

    values = [initial_value]

    for _ in range(n_steps):
        next_value = step_function(**kwargs)
        values.append(next_value)

    return np.array(values)


# ============================================================
# Monte Carlo Simulation
# ============================================================

def monte_carlo(
    simulate_fn: Callable[..., np.ndarray],
    n_paths: int = 100,
    **kwargs
) -> np.ndarray:
    """
    Run multiple independent simulations.
    """

    results = []

    for _ in range(n_paths):
        path = simulate_fn(**kwargs)
        results.append(path)

    return np.array(results)


# ============================================================
# Shock Injection
# ============================================================

def inject_shock(
    series: np.ndarray,
    shock_value: float,
    shock_time: int
) -> np.ndarray:
    """
    Inject external shock into series.
    """

    shocked = series.copy()

    if 0 <= shock_time < len(shocked):
        shocked[shock_time] += shock_value

    return shocked


# ============================================================
# Aggregate Metrics Across Paths
# ============================================================

def aggregate_metric(
    paths: np.ndarray,
    metric_fn: Callable[[np.ndarray], float]
) -> np.ndarray:
    """
    Apply metric function to each path.
    """

    return np.array([metric_fn(path) for path in paths])


# ============================================================
# Ruin Monte Carlo
# ============================================================

def monte_carlo_ruin_probability(
    simulate_fn: Callable[..., np.ndarray],
    threshold: float,
    n_paths: int = 1000,
    **kwargs
) -> float:
    """
    Estimate ruin probability across Monte Carlo runs.
    """

    ruin_count = 0

    for _ in range(n_paths):
        path = simulate_fn(**kwargs)
        cumulative = np.cumprod(1 + path)
        drawdown = (cumulative - np.maximum.accumulate(cumulative)) / np.maximum.accumulate(cumulative)

        if np.min(drawdown) <= threshold:
            ruin_count += 1

    return ruin_count / n_paths
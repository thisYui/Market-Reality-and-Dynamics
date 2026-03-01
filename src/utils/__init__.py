"""
Utils Package
-------------

Shared utilities across the framework.

Includes:

- Metrics
- Plotting
- Simulation tools
"""

# ============================================================
# Metrics
# ============================================================

from .metrics import (
    mean_return,
    volatility,
    sharpe_ratio,
    cumulative_returns,
    drawdown,
    max_drawdown,
    value_at_risk,
    expected_shortfall,
    ruin_probability,
    rolling_sharpe,
    rolling_volatility,
    volatility_clustering_score,
)

# ============================================================
# Plotting
# ============================================================

from .plotting import (
    set_style,
    plot_price_series,
    plot_return_series,
    plot_drawdown,
    plot_volatility_clustering,
    plot_regime_series,
    plot_rolling_correlation,
    plot_return_distribution,
)

# ============================================================
# Simulation
# ============================================================

from .simulation import (
    set_seed,
    simulate_path,
    monte_carlo,
    inject_shock,
    aggregate_metric,
    monte_carlo_ruin_probability,
)

# ============================================================
# Public API
# ============================================================

__all__ = [

    # Metrics
    "mean_return",
    "volatility",
    "sharpe_ratio",
    "cumulative_returns",
    "drawdown",
    "max_drawdown",
    "value_at_risk",
    "expected_shortfall",
    "ruin_probability",
    "rolling_sharpe",
    "rolling_volatility",
    "volatility_clustering_score",

    # Plotting
    "set_style",
    "plot_price_series",
    "plot_return_series",
    "plot_drawdown",
    "plot_volatility_clustering",
    "plot_regime_series",
    "plot_rolling_correlation",
    "plot_return_distribution",

    # Simulation
    "set_seed",
    "simulate_path",
    "monte_carlo",
    "inject_shock",
    "aggregate_metric",
    "monte_carlo_ruin_probability",
]
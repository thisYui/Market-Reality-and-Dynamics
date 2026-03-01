"""
Dynamics Package
----------------

Market dynamics models used in:

Project 1 — Market Reality & Dynamics
Project 2 — Risk, Uncertainty & Fat-Tail
Project 3 — Forecast Failure & Decision

Public API includes:

- Volatility models
- Regime switching models
- Momentum / Mean reversion dynamics
- Correlation models
"""

# ============================================================
# Volatility
# ============================================================

from .volatility import (
    rolling_volatility,
    ewma_volatility,
    GARCHLikeModel,
    VolatilityFeedbackModel,
)

# ============================================================
# Regime
# ============================================================

from .regime import (
    MarkovRegimeProcess,
    RegimeReturnModel,
    StructuralBreakModel,
)

# ============================================================
# Momentum
# ============================================================

from .momentum import (
    compute_momentum,
    MomentumRegimeModel,
)

# ============================================================
# Correlation
# ============================================================

from .correlation import (
    compute_correlation,
    rolling_correlation,
    RegimeCorrelationModel,
    CorrelationBreakdownModel,
)

# ============================================================
# Public API
# ============================================================

__all__ = [

    # Volatility
    "rolling_volatility",
    "ewma_volatility",
    "GARCHLikeModel",
    "VolatilityFeedbackModel",

    # Regime
    "MarkovRegimeProcess",
    "RegimeReturnModel",
    "StructuralBreakModel",

    # Momentum
    "compute_momentum",
    "MomentumRegimeModel",

    # Correlation
    "compute_correlation",
    "rolling_correlation",
    "RegimeCorrelationModel",
    "CorrelationBreakdownModel",
]
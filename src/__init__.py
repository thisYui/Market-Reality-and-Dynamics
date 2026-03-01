"""
Market Instability Research Framework
=====================================

Root package for:

Project 1 — Market Reality & Dynamics
Project 2 — Risk, Uncertainty & Fat-Tail
Project 3 — Forecast Failure & Decision

Subpackages:
    - behavior
    - dynamics
    - structure
    - utils
"""

# ============================================================
# Subpackages
# ============================================================

from . import behavior
from . import dynamics
from . import structure
from . import utils


# ============================================================
# Optional: High-Level Shortcuts
# ============================================================

# Frequently used models (optional convenience API)

from .behavior import (
    OverconfidenceModel,
    LossAversionModel,
    HerdingModel,
    PanicCascadeModel,
)

from .dynamics import (
    RegimeReturnModel,
    MomentumRegimeModel,
    GARCHLikeModel,
)

from .structure import (
    OrderBook,
    MarketMaker,
    LiquidationCascade,
)

from .utils import (
    sharpe_ratio,
    max_drawdown,
    value_at_risk,
)

__all__ = [

    # Subpackages
    "behavior",
    "dynamics",
    "structure",
    "utils",

    # High-level models
    "OverconfidenceModel",
    "LossAversionModel",
    "HerdingModel",
    "PanicCascadeModel",
    "RegimeReturnModel",
    "MomentumRegimeModel",
    "GARCHLikeModel",
    "OrderBook",
    "MarketMaker",
    "LiquidationCascade",

    # Metrics shortcuts
    "sharpe_ratio",
    "max_drawdown",
    "value_at_risk",
]
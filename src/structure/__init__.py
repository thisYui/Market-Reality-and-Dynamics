"""
Structure Package
-----------------

Microstructure & structural components of the market system.

Includes:

- Order Book
- Market Maker
- Liquidity Model
- Leverage & Margin
- Liquidation Cascade

Designed for:
Project 1 — Market Reality & Dynamics
"""

# ============================================================
# Order Book
# ============================================================

from .orderbook import OrderBook

# ============================================================
# Market Maker
# ============================================================

from .market_maker import MarketMaker

# ============================================================
# Liquidity
# ============================================================

from .liquidity import (
    LiquidityModel,
    LiquidityShockModel,
)

# ============================================================
# Leverage
# ============================================================

from .leverage import (
    LeveragedPosition,
    LiquidationEngine,
)

# ============================================================
# Liquidation Cascade
# ============================================================

from .liquidation import (
    LeveragedAgent,
    LiquidationCascade,
)

# ============================================================
# Public API
# ============================================================

__all__ = [

    # Order book
    "OrderBook",

    # Market maker
    "MarketMaker",

    # Liquidity
    "LiquidityModel",
    "LiquidityShockModel",

    # Leverage
    "LeveragedPosition",
    "LiquidationEngine",

    # Liquidation cascade
    "LeveragedAgent",
    "LiquidationCascade",
]
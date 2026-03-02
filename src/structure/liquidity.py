"""
Liquidity & Market Depth Dynamics
----------------------------------

Mô hình hóa:

- Market depth
- Bid-ask spread
- Slippage
- Liquidity shock
- Liquidity evaporation during crisis

Designed for:
Project 1 — Market Reality & Dynamics
"""

from __future__ import annotations
import numpy as np
from typing import Dict


# ============================================================
# Basic Liquidity Model
# ============================================================

class LiquidityModel:
    """
    Linear market depth model.

    Parameters
    ----------
    base_depth : float
        Tổng khối lượng có thể absorb mà không gây impact lớn.
    base_spread : float
        Bid-ask spread cơ bản.
    impact_coefficient : float
        Mức độ order size ảnh hưởng đến giá.
    """

    def __init__(
        self,
        base_depth: float = 1000.0,
        base_spread: float = 0.001,
        impact_coefficient: float = 0.01,
    ):
        self.base_depth = base_depth
        self.base_spread = base_spread
        self.impact_coefficient = impact_coefficient

        self.current_depth = base_depth

    # ------------------------------------------------------------
    # Spread
    # ------------------------------------------------------------

    def spread(self) -> float:
        """
        Spread tăng khi depth giảm.
        """
        if self.current_depth <= 0:
            return self.base_spread * 100  # crisis spread blowout

        liquidity_ratio = self.current_depth / self.base_depth
        return self.base_spread / liquidity_ratio

    # ------------------------------------------------------------
    # Price Impact
    # ------------------------------------------------------------

    def price_impact(self, order_size: float) -> float:
        """
        Impact phi tuyến theo depth hiện tại.
        """

        if self.current_depth <= 0:
            return -np.sign(order_size) * 0.1  # market dislocation

        relative_size = order_size / self.current_depth

        impact = self.impact_coefficient * relative_size

        return impact

    # ------------------------------------------------------------
    # Execute Trade
    # ------------------------------------------------------------

    def execute_trade(self, order_size: float) -> Dict:
        """
        Thực thi order và cập nhật depth.
        """

        available = self.current_depth

        executed = min(abs(order_size), available)

        impact = self.price_impact(executed)

        self.current_depth -= executed

        return {
            "impact": impact,
            "remaining_depth": self.current_depth,
            "spread": self.spread(),
        }

    # ------------------------------------------------------------
    # Liquidity Refill
    # ------------------------------------------------------------

    def refill(self, refill_rate: float = 0.1):
        """
        Liquidity được tái tạo theo thời gian.
        """
        self.current_depth += refill_rate * self.base_depth
        self.current_depth = min(self.current_depth, self.base_depth)


# ============================================================
# Liquidity Shock Model
# ============================================================

class LiquidityShockModel:
    """
    Liquidity evaporation during crisis.

    Depth giảm mạnh khi volatility tăng.
    """

    def __init__(
        self,
        base_depth: float = 1000.0,
        evaporation_sensitivity: float = 10.0,
    ):
        self.base_depth = base_depth
        self.evaporation_sensitivity = evaporation_sensitivity
        self.current_depth = base_depth

    def update(self, volatility: float):
        """
        Volatility tăng -> liquidity giảm phi tuyến.
        """
        reduction = self.evaporation_sensitivity * volatility

        self.current_depth = self.base_depth * np.exp(-reduction)

    def get_depth(self) -> float:
        return self.current_depth
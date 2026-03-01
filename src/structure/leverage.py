"""
Leverage & Margin Dynamics
--------------------------

Mô hình hóa:

- Leverage ratio
- Margin requirement
- Mark-to-market PnL
- Margin call
- Forced liquidation

Designed for:
Project 1 — Market Reality & Dynamics
"""

from __future__ import annotations
import numpy as np
from typing import Dict


# ============================================================
# Core Leverage Position
# ============================================================

class LeveragedPosition:
    """
    Một vị thế sử dụng đòn bẩy.

    Parameters
    ----------
    initial_capital : float
    leverage : float
        Ví dụ 5 = 5x leverage
    maintenance_margin : float
        Tỷ lệ vốn tối thiểu (ví dụ 0.25)
    """

    def __init__(
        self,
        initial_capital: float,
        leverage: float = 2.0,
        maintenance_margin: float = 0.25,
    ):
        if leverage <= 1:
            raise ValueError("Leverage must be > 1")

        self.initial_capital = initial_capital
        self.leverage = leverage
        self.maintenance_margin = maintenance_margin

        self.position_value = initial_capital * leverage
        self.equity = initial_capital
        self.debt = self.position_value - self.equity

        self.alive = True

    # ------------------------------------------------------------
    # Update Position
    # ------------------------------------------------------------

    def mark_to_market(self, return_t: float) -> Dict:
        """
        Cập nhật giá trị vị thế theo return.
        """

        if not self.alive:
            return {"status": "liquidated"}

        self.position_value *= (1 + return_t)
        self.equity = self.position_value - self.debt

        margin_ratio = self.equity / self.position_value

        if margin_ratio < self.maintenance_margin:
            self.alive = False
            return {
                "status": "margin_call",
                "equity": self.equity,
                "margin_ratio": margin_ratio,
            }

        return {
            "status": "active",
            "equity": self.equity,
            "margin_ratio": margin_ratio,
        }


# ============================================================
# Forced Liquidation Engine
# ============================================================

class LiquidationEngine:
    """
    Mô phỏng hệ thống nhiều vị thế leverage.

    Khi nhiều vị thế bị liquidate,
    tạo additional price impact.
    """

    def __init__(
        self,
        n_agents: int = 100,
        initial_capital: float = 100,
        leverage: float = 5.0,
        maintenance_margin: float = 0.25,
        impact_coefficient: float = 0.02,
    ):
        self.impact_coefficient = impact_coefficient

        self.positions = [
            LeveragedPosition(
                initial_capital,
                leverage,
                maintenance_margin,
            )
            for _ in range(n_agents)
        ]

    def step(self, market_return: float) -> Dict:
        """
        Một bước thị trường:
        - Cập nhật vị thế
        - Đếm liquidation
        - Tính additional price impact
        """

        liquidations = 0

        for position in self.positions:
            result = position.mark_to_market(market_return)

            if result["status"] == "margin_call":
                liquidations += 1

        liquidation_ratio = liquidations / len(self.positions)

        # Liquidation tạo áp lực bán
        additional_return = -self.impact_coefficient * liquidation_ratio

        return {
            "liquidations": liquidations,
            "liquidation_ratio": liquidation_ratio,
            "additional_return": additional_return,
        }

    def alive_ratio(self) -> float:
        """
        Tỷ lệ vị thế còn sống.
        """
        alive = sum(p.alive for p in self.positions)
        return alive / len(self.positions)
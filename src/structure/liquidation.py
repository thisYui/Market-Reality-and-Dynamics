"""
Systemic Liquidation Dynamics
-----------------------------

Mô hình hóa:

- Liquidation cascade nhiều vòng
- Price impact do forced selling
- Feedback loop giữa giá và margin call
- Endogenous crash formation

Designed for:
Project 1 — Market Reality & Dynamics
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List


# ============================================================
# Agent Position
# ============================================================

class LeveragedAgent:
    """
    Agent với leverage và liquidation threshold riêng.
    """

    def __init__(
        self,
        capital: float,
        leverage: float,
        maintenance_margin: float,
    ):
        self.capital = capital
        self.leverage = leverage
        self.maintenance_margin = maintenance_margin

        self.position_value = capital * leverage
        self.debt = self.position_value - capital
        self.equity = capital

        self.alive = True

    def update(self, market_return: float) -> bool:
        """
        Update theo return thị trường.
        Trả về True nếu bị liquidate.
        """

        if not self.alive:
            return False

        self.position_value *= (1 + market_return)
        self.equity = self.position_value - self.debt

        margin_ratio = self.equity / self.position_value

        if margin_ratio < self.maintenance_margin:
            self.alive = False
            return True

        return False


# ============================================================
# Liquidation Cascade Engine
# ============================================================

class LiquidationCascade:
    """
    Hệ thống liquidation nhiều vòng.

    Liquidation tạo sell pressure,
    sell pressure tạo return âm,
    return âm kích hoạt thêm liquidation.
    """

    def __init__(
        self,
        n_agents: int = 500,
        capital: float = 100,
        leverage_mean: float = 5.0,
        leverage_std: float = 1.0,
        maintenance_margin: float = 0.25,
        impact_coefficient: float = 0.05,
    ):
        self.impact_coefficient = impact_coefficient

        leverages = np.random.normal(
            leverage_mean,
            leverage_std,
            size=n_agents
        )

        leverages = np.clip(leverages, 1.1, None)

        self.agents: List[LeveragedAgent] = [
            LeveragedAgent(capital, lev, maintenance_margin)
            for lev in leverages
        ]

    # ------------------------------------------------------------
    # Single Cascade Step
    # ------------------------------------------------------------

    def step(self, market_return: float) -> Dict:
        """
        Một vòng cập nhật liquidation.
        """

        liquidated = 0

        for agent in self.agents:
            if agent.update(market_return):
                liquidated += 1

        liquidation_ratio = liquidated / len(self.agents)

        # Price impact do forced selling
        additional_return = -self.impact_coefficient * liquidation_ratio

        return {
            "liquidated": liquidated,
            "liquidation_ratio": liquidation_ratio,
            "additional_return": additional_return,
        }

    # ------------------------------------------------------------
    # Multi-round Cascade
    # ------------------------------------------------------------

    def simulate(
        self,
        initial_shock: float,
        max_rounds: int = 50,
    ) -> Dict:
        """
        Mô phỏng cascade nhiều vòng.
        """

        returns = [initial_shock]
        liquidation_ratios = []

        current_return = initial_shock

        for _ in range(max_rounds):

            result = self.step(current_return)
            liquidation_ratios.append(result["liquidation_ratio"])

            if result["liquidation_ratio"] == 0:
                break

            current_return = result["additional_return"]
            returns.append(current_return)

        return {
            "returns": np.array(returns),
            "liquidation_ratios": np.array(liquidation_ratios),
        }

    # ------------------------------------------------------------
    # System Health
    # ------------------------------------------------------------

    def alive_ratio(self) -> float:
        alive = sum(agent.alive for agent in self.agents)
        return alive / len(self.agents)
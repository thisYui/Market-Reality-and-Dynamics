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
    State-Dependent Liquidation Engine

    Features:
    - Price level evolution
    - Liquidity-dependent impact
    - Multi-round cascade
    - Endogenous feedback loop
    """

    def __init__(
        self,
        n_agents: int = 100,
        initial_capital: float = 100,
        leverage: float = 5.0,
        maintenance_margin: float = 0.25,
        liquidation_size: float = 10.0,
        liquidity_model=None,
        initial_price: float = 100.0,
    ):
        self.positions = [
            LeveragedPosition(
                initial_capital,
                leverage,
                maintenance_margin,
            )
            for _ in range(n_agents)
        ]

        self.liquidation_size = liquidation_size
        self.liquidity_model = liquidity_model
        self.price = initial_price

    # ============================================================
    # Single Step
    # ============================================================

    def step(self) -> dict:
        """
        Một vòng cascade:
        - Check liquidation
        - Tạo sell pressure
        - Impact qua liquidity model
        - Update price
        """

        liquidations = 0

        # 1. Check margin calls
        for position in self.positions:
            result = position.mark_to_market(
                return_t=0  # mark-to-market dùng price change riêng
            )

            if result["status"] == "margin_call":
                liquidations += 1

        # 2. Compute total forced selling volume
        total_sell_volume = liquidations * self.liquidation_size

        if total_sell_volume == 0:
            return {
                "liquidations": 0,
                "price": self.price,
                "regime": None,
            }

        # 3. Pass through liquidity model
        if self.liquidity_model is not None:
            liquidity_result = self.liquidity_model.execute_trade(
                order_size=total_sell_volume
            )

            impact = liquidity_result["impact"]
            regime = liquidity_result["regime"]

        else:
            # fallback linear impact
            impact = -0.001 * total_sell_volume
            regime = None

        # 4. Update price
        self.price *= (1 + impact)

        # 5. Update positions with realized return
        realized_return = impact

        for position in self.positions:
            if position.alive:
                position.mark_to_market(realized_return)

        return {
            "liquidations": liquidations,
            "price": self.price,
            "impact": impact,
            "regime": regime,
        }

    # ============================================================
    # Multi-Round Simulation
    # ============================================================

    def simulate(
        self,
        max_rounds: int = 50,
        initial_shock: float = -0.02,
    ) -> dict:
        """
        Simulate cascade starting with initial shock.
        """

        # Apply initial shock
        self.price *= (1 + initial_shock)

        for position in self.positions:
            if position.alive:
                position.mark_to_market(initial_shock)

        prices = [self.price]
        liquidation_counts = []
        regimes = []

        for _ in range(max_rounds):

            result = self.step()

            liquidation_counts.append(result["liquidations"])
            prices.append(result["price"])
            regimes.append(result["regime"])

            if result["liquidations"] == 0:
                break

        return {
            "prices": prices,
            "liquidations": liquidation_counts,
            "regimes": regimes,
        }

    # ============================================================
    # System Health
    # ============================================================

    def alive_ratio(self) -> float:
        alive = sum(p.alive for p in self.positions)
        return alive / len(self.positions)
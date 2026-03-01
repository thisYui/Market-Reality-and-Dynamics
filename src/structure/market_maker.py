"""
Market Maker Model
------------------

Mô hình hóa:

- Bid-ask quoting
- Inventory risk
- Spread adjustment
- Adverse selection
- Liquidity withdrawal

Designed for:
Project 1 — Market Reality & Dynamics
"""

from __future__ import annotations
import numpy as np
from typing import Dict


class MarketMaker:
    """
    Simple inventory-based market maker.

    Parameters
    ----------
    base_spread : float
    inventory_limit : float
    risk_aversion : float
    volatility_sensitivity : float
    """

    def __init__(
        self,
        base_spread: float = 0.001,
        inventory_limit: float = 1000,
        risk_aversion: float = 0.5,
        volatility_sensitivity: float = 5.0,
    ):
        self.base_spread = base_spread
        self.inventory_limit = inventory_limit
        self.risk_aversion = risk_aversion
        self.volatility_sensitivity = volatility_sensitivity

        self.inventory = 0.0
        self.cash = 0.0
        self.active = True

    # ------------------------------------------------------------
    # Spread Adjustment
    # ------------------------------------------------------------

    def compute_spread(self, volatility: float) -> float:
        """
        Spread tăng khi volatility tăng.
        """
        return self.base_spread * (
            1 + self.volatility_sensitivity * volatility
        )

    # ------------------------------------------------------------
    # Quote Prices
    # ------------------------------------------------------------

    def quote(self, mid_price: float, volatility: float) -> Dict:
        """
        Trả về bid và ask.
        """
        if not self.active:
            return {"bid": None, "ask": None}

        spread = self.compute_spread(volatility)

        # Inventory skew adjustment
        skew = self.risk_aversion * (self.inventory / self.inventory_limit)

        bid = mid_price * (1 - spread / 2 - skew)
        ask = mid_price * (1 + spread / 2 - skew)

        return {"bid": bid, "ask": ask}

    # ------------------------------------------------------------
    # Execute Trade
    # ------------------------------------------------------------

    def execute_trade(
        self,
        trade_price: float,
        quantity: float,
        side: str,
    ):
        """
        side:
            'buy'  -> client buys from MM (MM sells)
            'sell' -> client sells to MM (MM buys)
        """

        if not self.active:
            return

        if side == "buy":
            self.inventory -= quantity
            self.cash += trade_price * quantity
        elif side == "sell":
            self.inventory += quantity
            self.cash -= trade_price * quantity
        else:
            raise ValueError("side must be 'buy' or 'sell'")

        # Inventory risk control
        if abs(self.inventory) > self.inventory_limit:
            self.active = False  # withdraw liquidity

    # ------------------------------------------------------------
    # Mark-to-Market PnL
    # ------------------------------------------------------------

    def mark_to_market(self, mid_price: float) -> float:
        """
        Tính PnL hiện tại.
        """
        return self.cash + self.inventory * mid_price

    # ------------------------------------------------------------
    # Reset / Reactivate
    # ------------------------------------------------------------

    def reset_inventory(self):
        self.inventory = 0.0
        self.active = True
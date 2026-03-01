"""
Limit Order Book Model
----------------------

Mô hình hóa:

- Bid / Ask book nhiều level
- Limit order insertion
- Market order execution
- Slippage
- Price impact
- Liquidity depletion

Designed for:
Project 1 — Market Reality & Dynamics
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List


# ============================================================
# OrderBook
# ============================================================

class OrderBook:
    """
    Simplified limit order book.

    Attributes
    ----------
    bids : list of (price, quantity)
    asks : list of (price, quantity)
    """

    def __init__(
        self,
        mid_price: float = 100.0,
        spread: float = 0.001,
        depth_levels: int = 5,
        base_quantity: float = 100.0,
    ):
        self.mid_price = mid_price
        self.spread = spread
        self.depth_levels = depth_levels
        self.base_quantity = base_quantity

        self.bids: List[List[float]] = []
        self.asks: List[List[float]] = []

        self._initialize_book()

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------

    def _initialize_book(self):
        """
        Create symmetric book around mid.
        """
        self.bids.clear()
        self.asks.clear()

        for i in range(1, self.depth_levels + 1):
            price_bid = self.mid_price * (1 - self.spread * i)
            price_ask = self.mid_price * (1 + self.spread * i)

            self.bids.append([price_bid, self.base_quantity])
            self.asks.append([price_ask, self.base_quantity])

        self._sort_books()

    def _sort_books(self):
        self.bids.sort(key=lambda x: -x[0])
        self.asks.sort(key=lambda x: x[0])

    # ------------------------------------------------------------
    # Best Quotes
    # ------------------------------------------------------------

    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else None

    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else None

    def mid(self) -> float:
        if self.best_bid() and self.best_ask():
            return (self.best_bid() + self.best_ask()) / 2
        return self.mid_price

    # ------------------------------------------------------------
    # Limit Order
    # ------------------------------------------------------------

    def add_limit_order(self, side: str, price: float, quantity: float):
        """
        Add new limit order.
        """
        if side == "buy":
            self.bids.append([price, quantity])
        elif side == "sell":
            self.asks.append([price, quantity])
        else:
            raise ValueError("side must be 'buy' or 'sell'")

        self._sort_books()

    # ------------------------------------------------------------
    # Market Order Execution
    # ------------------------------------------------------------

    def execute_market_order(self, side: str, quantity: float) -> Dict:
        """
        Execute market order against opposite side.
        """

        book_side = self.asks if side == "buy" else self.bids

        remaining = quantity
        total_cost = 0.0

        while remaining > 0 and book_side:
            price, avail_qty = book_side[0]

            traded = min(remaining, avail_qty)

            total_cost += traded * price
            remaining -= traded
            book_side[0][1] -= traded

            if book_side[0][1] <= 0:
                book_side.pop(0)

        if quantity > 0:
            avg_price = total_cost / (quantity - remaining) if quantity != remaining else None
        else:
            avg_price = None

        self.mid_price = self.mid()

        return {
            "avg_price": avg_price,
            "filled": quantity - remaining,
            "remaining": remaining,
            "new_mid": self.mid_price,
        }

    # ------------------------------------------------------------
    # Liquidity Snapshot
    # ------------------------------------------------------------

    def depth_snapshot(self) -> Dict:
        return {
            "bids": self.bids.copy(),
            "asks": self.asks.copy(),
        }

    # ------------------------------------------------------------
    # Liquidity Shock
    # ------------------------------------------------------------

    def evaporate_liquidity(self, fraction: float):
        """
        Remove fraction of liquidity from book.
        """
        for level in self.bids:
            level[1] *= (1 - fraction)

        for level in self.asks:
            level[1] *= (1 - fraction)

    # ------------------------------------------------------------
    # Refill Book
    # ------------------------------------------------------------

    def refill(self):
        """
        Rebuild book around current mid.
        """
        self.mid_price = self.mid()
        self._initialize_book()
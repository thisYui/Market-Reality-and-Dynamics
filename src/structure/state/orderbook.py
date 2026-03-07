"""
OrderBook - Core limit order book data structure for market microstructure simulation.

Supports:
- Limit & market order submission
- Order cancellation / modification
- Best bid/ask, spread, mid-price
- Depth snapshots & imbalance metrics
- Trade tape & VWAP
- Order flow toxicity (OFI, Kyle's lambda proxy)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enums & Data-classes
# ---------------------------------------------------------------------------

class Side(Enum):
    BID = "bid"
    ASK = "ask"


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(Enum):
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class Order:
    side: Side
    price: float          # ignored for market orders
    quantity: float
    order_type: OrderType = OrderType.LIMIT
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    trader_id: Optional[str] = None

    # mutable state
    remaining: float = field(init=False)
    status: OrderStatus = field(init=False, default=OrderStatus.OPEN)

    def __post_init__(self):
        self.remaining = self.quantity

    @property
    def filled(self) -> float:
        return self.quantity - self.remaining

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)


@dataclass
class Trade:
    price: float
    quantity: float
    aggressor_side: Side          # side of the incoming (taker) order
    maker_order_id: str
    taker_order_id: str
    timestamp: float = field(default_factory=time.time)
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Price Level
# ---------------------------------------------------------------------------

class PriceLevel:
    """FIFO queue of orders at a single price."""

    def __init__(self, price: float):
        self.price = price
        self._orders: List[Order] = []

    # ---- mutation ----------------------------------------------------------

    def add(self, order: Order) -> None:
        self._orders.append(order)

    def remove(self, order_id: str) -> bool:
        before = len(self._orders)
        self._orders = [o for o in self._orders if o.order_id != order_id]
        return len(self._orders) < before

    # ---- properties --------------------------------------------------------

    @property
    def total_quantity(self) -> float:
        return sum(o.remaining for o in self._orders if o.is_active)

    @property
    def order_count(self) -> int:
        return sum(1 for o in self._orders if o.is_active)

    @property
    def orders(self) -> List[Order]:
        return [o for o in self._orders if o.is_active]

    def is_empty(self) -> bool:
        return self.total_quantity == 0

    def __repr__(self) -> str:
        return f"PriceLevel(price={self.price}, qty={self.total_quantity:.4f}, n={self.order_count})"


# ---------------------------------------------------------------------------
# OrderBook
# ---------------------------------------------------------------------------

class OrderBook:
    """
    Full limit order book with matching engine.

    Parameters
    ----------
    tick_size : float
        Minimum price increment (used for rounding & validation).
    lot_size : float
        Minimum quantity increment.
    symbol : str
        Instrument identifier for logging/display.
    """

    def __init__(
        self,
        tick_size: float = 0.01,
        lot_size: float = 1.0,
        symbol: str = "UNKNOWN",
    ):
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.symbol = symbol

        # price → PriceLevel  (sorted dicts maintained via sorted keys)
        self._bids: Dict[float, PriceLevel] = {}   # descending order preferred
        self._asks: Dict[float, PriceLevel] = {}   # ascending order preferred

        # fast order look-up
        self._orders: Dict[str, Order] = {}

        # trade tape
        self.trades: List[Trade] = []

        # statistics
        self._total_volume: float = 0.0
        self._buy_volume: float = 0.0
        self._sell_volume: float = 0.0

    # =========================================================================
    # Public API – Order management
    # =========================================================================

    def submit(self, order: Order) -> List[Trade]:
        """
        Submit a new order.  Returns list of trades generated (may be empty).
        Market orders are immediately matched; unmatched remainder is dropped.
        Limit orders are matched first, then any remainder is queued.
        """
        self._validate(order)
        self._orders[order.order_id] = order

        if order.order_type == OrderType.MARKET:
            trades = self._match_market(order)
        else:
            trades = self._match_limit(order)
            if order.is_active:          # queue the remainder
                self._queue(order)

        return trades

    def cancel(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if successfully cancelled."""
        order = self._orders.get(order_id)
        if order is None or not order.is_active:
            return False

        book = self._bids if order.side == Side.BID else self._asks
        level = book.get(order.price)
        if level:
            level.remove(order_id)
            if level.is_empty():
                del book[order.price]

        order.status = OrderStatus.CANCELLED
        return True

    def modify(
        self,
        order_id: str,
        new_quantity: Optional[float] = None,
        new_price: Optional[float] = None,
    ) -> bool:
        """
        Modify quantity / price of an open limit order.
        Price change loses queue priority (cancel + re-submit).
        Quantity *reduction* preserves queue priority.
        Returns True if modification was applied.
        """
        order = self._orders.get(order_id)
        if order is None or not order.is_active or order.order_type == OrderType.MARKET:
            return False

        if new_price is not None and new_price != order.price:
            # Cancel and resubmit (loses priority)
            self.cancel(order_id)
            order.price = new_price
            if new_quantity is not None:
                order.quantity = new_quantity
                order.remaining = new_quantity
            order.status = OrderStatus.OPEN
            order.timestamp = time.time()
            self._orders[order_id] = order
            self._match_limit(order)
            if order.is_active:
                self._queue(order)
        elif new_quantity is not None:
            if new_quantity < order.filled:
                raise ValueError("new_quantity cannot be less than already-filled amount.")
            delta = new_quantity - order.quantity
            order.quantity = new_quantity
            order.remaining = max(0.0, order.remaining + delta)
            if order.remaining == 0:
                order.status = OrderStatus.FILLED

        return True

    # =========================================================================
    # Public API – Quotes & Depth
    # =========================================================================

    @property
    def best_bid(self) -> Optional[float]:
        return max(self._bids) if self._bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return min(self._asks) if self._asks else None

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb

    @property
    def relative_spread(self) -> Optional[float]:
        mid = self.mid_price
        if mid is None or mid == 0:
            return None
        return self.spread / mid  # type: ignore[operator]

    def bid_depth(self, levels: int = 10) -> List[Tuple[float, float]]:
        """Returns [(price, qty), ...] for top `levels` bid levels (best first)."""
        sorted_prices = sorted(self._bids.keys(), reverse=True)[:levels]
        return [(p, self._bids[p].total_quantity) for p in sorted_prices]

    def ask_depth(self, levels: int = 10) -> List[Tuple[float, float]]:
        """Returns [(price, qty), ...] for top `levels` ask levels (best first)."""
        sorted_prices = sorted(self._asks.keys())[:levels]
        return [(p, self._asks[p].total_quantity) for p in sorted_prices]

    def depth_snapshot(self, levels: int = 10) -> Dict:
        """Full depth snapshot as a dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": time.time(),
            "bids": self.bid_depth(levels),
            "asks": self.ask_depth(levels),
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "spread": self.spread,
        }

    # =========================================================================
    # Public API – Microstructure metrics
    # =========================================================================

    def order_imbalance(self, levels: int = 5) -> Optional[float]:
        """
        Volume-weighted order imbalance in [-1, +1].
        +1 → all volume on bid side  |  -1 → all volume on ask side.
        """
        bid_vol = sum(q for _, q in self.bid_depth(levels))
        ask_vol = sum(q for _, q in self.ask_depth(levels))
        total = bid_vol + ask_vol
        if total == 0:
            return None
        return (bid_vol - ask_vol) / total

    def vwap(self, side: Optional[Side] = None) -> Optional[float]:
        """Volume-weighted average price of all trades (optionally filtered by side)."""
        trades = self.trades
        if side is not None:
            trades = [t for t in trades if t.aggressor_side == side]
        if not trades:
            return None
        total_qty = sum(t.quantity for t in trades)
        if total_qty == 0:
            return None
        return sum(t.price * t.quantity for t in trades) / total_qty

    def kyle_lambda(self, window: int = 50) -> Optional[float]:
        """
        Proxy for Kyle's lambda (price impact coefficient).
        Regresses mid-price change on signed order flow for last `window` trades.
        Returns slope coefficient (price impact per unit of signed volume).
        """
        if len(self.trades) < 10:
            return None

        recent = self.trades[-window:]
        signed_vols = []
        prices = []
        for t in recent:
            sign = 1.0 if t.aggressor_side == Side.BID else -1.0
            signed_vols.append(sign * t.quantity)
            prices.append(t.price)

        sv = np.array(signed_vols)
        dp = np.diff(prices)
        if len(dp) == 0 or np.std(sv[:-1]) == 0:
            return None

        # OLS: dp = lambda * sv[:-1]
        lam = np.cov(dp, sv[:-1])[0, 1] / np.var(sv[:-1])
        return float(lam)

    def order_flow_imbalance(self, window: int = 20) -> Optional[float]:
        """
        Order Flow Imbalance (OFI) proxy based on recent trades.
        Positive → net buying pressure; negative → net selling pressure.
        """
        recent = self.trades[-window:]
        if not recent:
            return None
        ofi = sum(
            t.quantity if t.aggressor_side == Side.BID else -t.quantity
            for t in recent
        )
        return float(ofi)

    # =========================================================================
    # Public API – Convenience
    # =========================================================================

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    @property
    def total_volume(self) -> float:
        return self._total_volume

    @property
    def buy_volume(self) -> float:
        return self._buy_volume

    @property
    def sell_volume(self) -> float:
        return self._sell_volume

    def reset(self) -> None:
        """Clear the book, trades, and all statistics."""
        self._bids.clear()
        self._asks.clear()
        self._orders.clear()
        self.trades.clear()
        self._total_volume = 0.0
        self._buy_volume = 0.0
        self._sell_volume = 0.0

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _validate(self, order: Order) -> None:
        if order.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {order.quantity}.")
        if order.order_type == OrderType.LIMIT and order.price <= 0:
            raise ValueError(f"Limit order price must be positive, got {order.price}.")

    def _queue(self, order: Order) -> None:
        """Add resting order to the appropriate side of the book."""
        book = self._bids if order.side == Side.BID else self._asks
        if order.price not in book:
            book[order.price] = PriceLevel(order.price)
        book[order.price].add(order)

    def _match_limit(self, order: Order) -> List[Trade]:
        """Match a limit order against the opposite side."""
        trades: List[Trade] = []

        if order.side == Side.BID:
            while order.is_active and self._asks:
                best_ask = min(self._asks)
                if order.price < best_ask:
                    break
                trades.extend(self._fill_against_level(order, self._asks, best_ask))
        else:
            while order.is_active and self._bids:
                best_bid = max(self._bids)
                if order.price > best_bid:
                    break
                trades.extend(self._fill_against_level(order, self._bids, best_bid))

        return trades

    def _match_market(self, order: Order) -> List[Trade]:
        """Match a market order; drops any unfilled remainder."""
        trades: List[Trade] = []

        if order.side == Side.BID:
            while order.is_active and self._asks:
                best_ask = min(self._asks)
                trades.extend(self._fill_against_level(order, self._asks, best_ask))
        else:
            while order.is_active and self._bids:
                best_bid = max(self._bids)
                trades.extend(self._fill_against_level(order, self._bids, best_bid))

        if order.is_active:
            order.status = OrderStatus.CANCELLED  # unfilled market order expires

        return trades

    def _fill_against_level(
        self,
        aggressor: Order,
        book_side: Dict[float, PriceLevel],
        price: float,
    ) -> List[Trade]:
        """FIFO fill of aggressor against all resting orders at `price`."""
        trades: List[Trade] = []
        level = book_side[price]

        for maker in level.orders:
            if not aggressor.is_active:
                break

            fill_qty = min(aggressor.remaining, maker.remaining)

            # record trade
            trade = Trade(
                price=maker.price,
                quantity=fill_qty,
                aggressor_side=aggressor.side,
                maker_order_id=maker.order_id,
                taker_order_id=aggressor.order_id,
            )
            trades.append(trade)
            self.trades.append(trade)

            # update volume stats
            self._total_volume += fill_qty
            if aggressor.side == Side.BID:
                self._buy_volume += fill_qty
            else:
                self._sell_volume += fill_qty

            # update order states
            aggressor.remaining -= fill_qty
            maker.remaining -= fill_qty

            aggressor.status = (
                OrderStatus.FILLED if aggressor.remaining == 0
                else OrderStatus.PARTIALLY_FILLED
            )
            maker.status = (
                OrderStatus.FILLED if maker.remaining == 0
                else OrderStatus.PARTIALLY_FILLED
            )

        # clean up exhausted level
        if level.is_empty():
            del book_side[price]

        return trades

    # =========================================================================
    # Display
    # =========================================================================

    def __repr__(self) -> str:
        bb = f"{self.best_bid:.4f}" if self.best_bid else "—"
        ba = f"{self.best_ask:.4f}" if self.best_ask else "—"
        sp = f"{self.spread:.4f}" if self.spread else "—"
        return (
            f"OrderBook({self.symbol}) | "
            f"bid={bb} ask={ba} spread={sp} | "
            f"trades={len(self.trades)} vol={self._total_volume:.2f}"
        )

    def display(self, levels: int = 5) -> str:
        """Human-readable depth table."""
        asks = list(reversed(self.ask_depth(levels)))
        bids = self.bid_depth(levels)
        lines = [f"\n{'='*42}", f" {self.symbol} Order Book", f"{'='*42}"]
        lines.append(f"  {'Price':>12}  {'Qty':>12}  {'Side':>6}")
        lines.append(f"  {'-'*12}  {'-'*12}  {'-'*6}")
        for p, q in asks:
            lines.append(f"  {p:>12.4f}  {q:>12.4f}  {'ASK':>6}")
        if self.spread is not None:
            lines.append(f"  {'--- spread: ' + f'{self.spread:.4f}':>32}")
        for p, q in bids:
            lines.append(f"  {p:>12.4f}  {q:>12.4f}  {'BID':>6}")
        lines.append(f"{'='*42}\n")
        return "\n".join(lines)
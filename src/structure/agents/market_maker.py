"""
market_maker.py  –  Market Maker Agent
=======================================

Covers the following concepts for 04_market_maker_inventory.ipynb:

1. Quote generation               – Avellaneda-Stoikov optimal quotes
2. Inventory management           – target inventory, skew, hard limits
3. Adverse selection avoidance    – quote withdrawal, spread widening
4. PnL attribution                – spread capture, inventory PnL, adverse selection cost
5. Risk limits                    – position limits, loss limits, drawdown circuit breaker
6. Market making strategies       – symmetric, inventory-skewed, volatility-adaptive
7. Performance metrics            – Sharpe, fill rate, spread earned, inventory turnover
8. Multi-level quoting            – layered limit orders at multiple depths

Design principle
----------------
- MarketMaker is self-contained: takes market state as input, outputs QuoteDecision
- No direct import of OrderBook, leverage.py, liquidity.py
- All state is explicit and serializable (suitable for simulation loop)
- Strategy is injectable (Strategy pattern) for easy comparison in notebook
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class MMStatus(Enum):
    ACTIVE    = "active"
    PAUSED    = "paused"      # temporarily withdrawn (e.g. news event)
    STOPPED   = "stopped"     # risk limit breached


@dataclass
class MarketState:
    """Snapshot of market conditions passed to market maker each tick."""
    timestamp:    float
    mid_price:    float
    best_bid:     float
    best_ask:     float
    bid_qty:      float
    ask_qty:      float
    volatility:   float        # short-term realized vol (e.g. 1-min)
    trade_sign:   int = 0      # last trade: +1 buy, -1 sell, 0 unknown
    trade_size:   float = 0.0
    spread:       float = 0.0  # current market spread

    def __post_init__(self):
        if self.spread == 0.0:
            self.spread = self.best_ask - self.best_bid


@dataclass
class QuoteDecision:
    """Output of market maker strategy: bid/ask quotes to post."""
    timestamp:  float
    bid_price:  float
    bid_size:   float
    ask_price:  float
    ask_size:   float
    reason:     str = ""
    # optional: multiple levels
    extra_bids: List[Tuple[float, float]] = field(default_factory=list)  # [(price, size)]
    extra_asks: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def quoted_spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def mid(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0

    def is_valid(self) -> bool:
        return (
            self.bid_price > 0
            and self.ask_price > self.bid_price
            and self.bid_size > 0
            and self.ask_size > 0
        )


@dataclass
class Fill:
    """A fill received by the market maker (passive execution)."""
    timestamp:  float
    side:       str       # "bid" | "ask"
    price:      float
    size:       float
    is_adverse: bool = False   # flagged if followed by adverse price move


@dataclass
class MMSnapshot:
    """Full state snapshot for logging / notebook replay."""
    timestamp:    float
    inventory:    float
    cash:         float
    mid_price:    float
    unrealized:   float
    realized:     float
    total_pnl:    float
    spread_pnl:   float
    inventory_pnl: float
    adverse_pnl:  float
    leverage:     float
    status:       str
    quote:        Optional[QuoteDecision] = None


# ---------------------------------------------------------------------------
# 1. Inventory Manager
# ---------------------------------------------------------------------------

class InventoryManager:
    """
    Tracks inventory and computes skew signals.

    Inventory risk
    --------------
    Long inventory → want to sell → skew ask down, bid down
    Short inventory → want to buy  → skew bid up, ask up

    Parameters
    ----------
    max_inventory   : float – hard position limit (absolute)
    target_inventory: float – desired inventory (typically 0)
    inventory_decay : float – mean-reversion speed of inventory target
    """

    def __init__(
        self,
        max_inventory:    float = 1000.0,
        target_inventory: float = 0.0,
        inventory_decay:  float = 0.01,
    ):
        self.max_inventory    = max_inventory
        self.target_inventory = target_inventory
        self.inventory_decay  = inventory_decay

        self.inventory: float = 0.0
        self._history:  List[float] = []

    def update(self, fill: Fill):
        delta = fill.size if fill.side == "bid" else -fill.size
        self.inventory += delta
        self._history.append(self.inventory)

    @property
    def utilization(self) -> float:
        """Inventory as fraction of max_inventory ∈ [-1, +1]."""
        return self.inventory / (self.max_inventory + 1e-10)

    @property
    def is_at_limit(self) -> bool:
        return abs(self.inventory) >= self.max_inventory

    @property
    def skew_signal(self) -> float:
        """
        Skew ∈ [-1, +1].
        Positive → long inventory → skew quotes downward (want to sell).
        Negative → short inventory → skew quotes upward (want to buy).
        """
        excess = self.inventory - self.target_inventory
        return np.tanh(excess / (self.max_inventory * 0.5 + 1e-10))

    def should_block_side(self, side: str) -> bool:
        """
        Block quoting if at hard limit.
        Long limit → stop posting bids | Short limit → stop posting asks.
        """
        if self.inventory >= self.max_inventory and side == "bid":
            return True
        if self.inventory <= -self.max_inventory and side == "ask":
            return True
        return False

    def unrealized_pnl(self, mark_price: float, avg_entry: float) -> float:
        return self.inventory * (mark_price - avg_entry)

    @property
    def inventory_series(self) -> np.ndarray:
        return np.array(self._history)


# ---------------------------------------------------------------------------
# 2. PnL Tracker
# ---------------------------------------------------------------------------

class PnLTracker:
    """
    Decomposes market maker PnL into:
    - Spread capture   : earned from bid-ask spread (passive fills)
    - Inventory PnL    : mark-to-market gain/loss on open inventory
    - Adverse selection: loss from informed order flow
    """

    def __init__(self):
        self.cash:          float = 0.0
        self.avg_entry:     float = 0.0
        self._inventory:    float = 0.0

        self.spread_pnl:    float = 0.0
        self.inventory_pnl: float = 0.0
        self.adverse_pnl:   float = 0.0
        self.realized_pnl:  float = 0.0

        self._fills: List[Fill] = []
        self._pnl_history: List[float] = []

    def record_fill(self, fill: Fill, mid_price: float):
        """Update cash and avg_entry on each fill."""
        self._fills.append(fill)

        if fill.side == "bid":
            # bought: pay cash, receive inventory
            half_spread = mid_price - fill.price
            self.spread_pnl += half_spread * fill.size
            self.cash -= fill.price * fill.size
            # update avg entry
            old_inv = self._inventory
            self._inventory += fill.size
            if self._inventory != 0:
                self.avg_entry = (
                    (old_inv * self.avg_entry + fill.size * fill.price)
                    / self._inventory
                ) if old_inv >= 0 else fill.price
        else:
            # sold: receive cash, reduce inventory
            half_spread = fill.price - mid_price
            self.spread_pnl += half_spread * fill.size
            self.cash += fill.price * fill.size
            old_inv = self._inventory
            self._inventory -= fill.size
            if old_inv > 0 and self._inventory >= 0:
                self.realized_pnl += (fill.price - self.avg_entry) * fill.size

        if fill.is_adverse:
            self.adverse_pnl -= abs(fill.size) * abs(fill.price - mid_price)

    def mark_to_market(self, mark_price: float) -> float:
        """Update inventory PnL. Returns total PnL."""
        if self._inventory != 0:
            self.inventory_pnl = self._inventory * (mark_price - self.avg_entry)
        else:
            self.inventory_pnl = 0.0

        total = self.cash + self._inventory * mark_price
        self._pnl_history.append(total)
        return total

    @property
    def total_pnl(self) -> float:
        return self.spread_pnl + self.inventory_pnl + self.realized_pnl + self.adverse_pnl

    @property
    def n_fills(self) -> int:
        return len(self._fills)

    @property
    def pnl_series(self) -> np.ndarray:
        return np.array(self._pnl_history)

    def attribution(self) -> Dict[str, float]:
        return {
            "spread_capture":    round(self.spread_pnl,    4),
            "inventory_pnl":     round(self.inventory_pnl, 4),
            "realized_pnl":      round(self.realized_pnl,  4),
            "adverse_selection": round(self.adverse_pnl,   4),
            "total":             round(self.total_pnl,      4),
        }


# ---------------------------------------------------------------------------
# 3. Risk Manager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Enforces risk limits and returns status signal to market maker.

    Limits
    ------
    - max_inventory     : absolute position limit
    - max_drawdown      : maximum PnL drawdown before stop
    - max_daily_loss    : maximum loss in one session
    - max_adverse_trades: number of consecutive adverse fills before pause
    - vol_threshold     : pause quoting above this volatility
    """

    def __init__(
        self,
        max_inventory:      float = 1000.0,
        max_drawdown:       float = 5000.0,
        max_daily_loss:     float = 3000.0,
        max_adverse_trades: int   = 5,
        vol_threshold:      float = 0.05,
        pause_duration:     int   = 10,   # ticks to pause after trigger
    ):
        self.max_inventory      = max_inventory
        self.max_drawdown       = max_drawdown
        self.max_daily_loss     = max_daily_loss
        self.max_adverse_trades = max_adverse_trades
        self.vol_threshold      = vol_threshold
        self.pause_duration     = pause_duration

        self._peak_pnl:          float = 0.0
        self._session_start_pnl: float = 0.0
        self._adverse_streak:    int   = 0
        self._pause_ticks_left:  int   = 0
        self.breach_log: List[Dict] = []

    def check(
        self,
        inventory:   float,
        total_pnl:   float,
        volatility:  float,
        last_fill:   Optional[Fill] = None,
        t:           float = 0.0,
    ) -> MMStatus:
        """
        Evaluate all risk limits. Returns recommended MMStatus.
        STOPPED > PAUSED > ACTIVE priority.
        """
        # update peak
        self._peak_pnl = max(self._peak_pnl, total_pnl)

        # 1. Hard stops
        drawdown = self._peak_pnl - total_pnl
        daily_loss = total_pnl - self._session_start_pnl

        if drawdown > self.max_drawdown:
            self._log("max_drawdown", t, drawdown=drawdown)
            return MMStatus.STOPPED

        if daily_loss < -self.max_daily_loss:
            self._log("max_daily_loss", t, daily_loss=daily_loss)
            return MMStatus.STOPPED

        # 2. Inventory hard limit
        if abs(inventory) > self.max_inventory * 1.05:   # 5% buffer
            self._log("inventory_limit", t, inventory=inventory)
            return MMStatus.STOPPED

        # 3. Pause conditions
        if self._pause_ticks_left > 0:
            self._pause_ticks_left -= 1
            return MMStatus.PAUSED

        if volatility > self.vol_threshold:
            self._pause_ticks_left = self.pause_duration
            self._log("high_volatility", t, volatility=volatility)
            return MMStatus.PAUSED

        if last_fill and last_fill.is_adverse:
            self._adverse_streak += 1
            if self._adverse_streak >= self.max_adverse_trades:
                self._adverse_streak = 0
                self._pause_ticks_left = self.pause_duration
                self._log("adverse_streak", t)
                return MMStatus.PAUSED
        elif last_fill:
            self._adverse_streak = 0

        return MMStatus.ACTIVE

    def reset_session(self, current_pnl: float):
        self._session_start_pnl = current_pnl
        self._peak_pnl = current_pnl

    def _log(self, reason: str, t: float, **kwargs):
        self.breach_log.append({"t": t, "reason": reason, **kwargs})


# ---------------------------------------------------------------------------
# 4. Quoting Strategies
# ---------------------------------------------------------------------------

class QuotingStrategy(ABC):
    """Abstract base for quoting strategies."""

    @abstractmethod
    def compute_quotes(
        self,
        state:     MarketState,
        inventory: float,
        skew:      float,
    ) -> QuoteDecision:
        ...


class SymmetricStrategy(QuotingStrategy):
    """
    Fixed symmetric spread around mid price.
    Simplest baseline strategy.

    Parameters
    ----------
    half_spread : float – half of quoted spread (in price units)
    order_size  : float – size to quote on each side
    """

    def __init__(self, half_spread: float = 0.05, order_size: float = 10.0):
        self.half_spread = half_spread
        self.order_size  = order_size

    def compute_quotes(self, state, inventory, skew) -> QuoteDecision:
        return QuoteDecision(
            timestamp=state.timestamp,
            bid_price=state.mid_price - self.half_spread,
            bid_size=self.order_size,
            ask_price=state.mid_price + self.half_spread,
            ask_size=self.order_size,
            reason="symmetric",
        )


class AvellanedaStoikovStrategy(QuotingStrategy):
    """
    Avellaneda-Stoikov (2008) optimal market making.

    Reservation price (risk-adjusted mid):
        r = s - q · γ · σ² · (T - t)

    Optimal spread:
        δ_bid + δ_ask = γ · σ² · (T-t) + (2/γ) · ln(1 + γ/κ)

    where:
        s = mid price
        q = inventory
        γ = risk aversion
        σ = volatility
        κ = order arrival intensity
        T-t = time remaining in session

    Parameters
    ----------
    gamma       : float – risk aversion (higher → wider spread, faster inventory reversion)
    kappa       : float – order arrival rate (higher → tighter spread)
    sigma       : float – overrides market volatility if > 0
    order_size  : float – base order size
    time_horizon: float – session length in same units as timestamp
    """

    def __init__(
        self,
        gamma:        float = 0.1,
        kappa:        float = 1.5,
        sigma:        float = 0.0,   # 0 = use market volatility
        order_size:   float = 10.0,
        time_horizon: float = 1.0,
    ):
        self.gamma        = gamma
        self.kappa        = kappa
        self.sigma        = sigma
        self.order_size   = order_size
        self.time_horizon = time_horizon
        self._t0:         Optional[float] = None

    def compute_quotes(self, state, inventory, skew) -> QuoteDecision:
        if self._t0 is None:
            self._t0 = state.timestamp

        time_remaining = max(
            0.01,
            self.time_horizon - (state.timestamp - self._t0) / self.time_horizon
        )
        sigma = self.sigma if self.sigma > 0 else state.volatility

        # reservation price
        reservation = state.mid_price - inventory * self.gamma * sigma**2 * time_remaining

        # optimal half-spread
        variance_term = self.gamma * sigma**2 * time_remaining
        log_term = (2.0 / self.gamma) * math.log(1.0 + self.gamma / (self.kappa + 1e-10))
        total_spread = variance_term + log_term
        half_spread  = total_spread / 2.0

        half_spread = max(half_spread, state.spread / 2.0)   # never quote inside market

        return QuoteDecision(
            timestamp=state.timestamp,
            bid_price=reservation - half_spread,
            bid_size=self.order_size,
            ask_price=reservation + half_spread,
            ask_size=self.order_size,
            reason="avellaneda_stoikov",
        )


class InventorySkewStrategy(QuotingStrategy):
    """
    Symmetric spread with dynamic skew based on inventory imbalance.

    Skew mechanism:
    - Long inventory  → lower bid AND ask (want to sell, discourage buying)
    - Short inventory → raise bid AND ask (want to buy, discourage selling)
    - Additionally narrows the side that reduces inventory (fill incentive)

    Parameters
    ----------
    base_half_spread : float – minimum half-spread
    skew_factor      : float – price shift per unit of skew signal
    spread_skew      : float – asymmetric spread adjustment
    order_size       : float – base order size per side
    size_skew        : float – size adjustment factor based on inventory
    """

    def __init__(
        self,
        base_half_spread: float = 0.05,
        skew_factor:      float = 0.03,
        spread_skew:      float = 0.02,
        order_size:       float = 10.0,
        size_skew:        float = 0.3,
    ):
        self.base_half_spread = base_half_spread
        self.skew_factor      = skew_factor
        self.spread_skew      = spread_skew
        self.order_size       = order_size
        self.size_skew        = size_skew

    def compute_quotes(self, state, inventory, skew) -> QuoteDecision:
        # shift mid against inventory
        mid_adj = state.mid_price - skew * self.skew_factor * state.mid_price

        # widen side that adds to inventory, narrow side that reduces it
        if skew > 0:   # long → want to sell → narrow ask, widen bid
            bid_spread = self.base_half_spread + abs(skew) * self.spread_skew
            ask_spread = max(state.spread / 4, self.base_half_spread - abs(skew) * self.spread_skew * 0.5)
            bid_size   = self.order_size * (1.0 - abs(skew) * self.size_skew)
            ask_size   = self.order_size * (1.0 + abs(skew) * self.size_skew)
        else:          # short → want to buy → narrow bid, widen ask
            ask_spread = self.base_half_spread + abs(skew) * self.spread_skew
            bid_spread = max(state.spread / 4, self.base_half_spread - abs(skew) * self.spread_skew * 0.5)
            ask_size   = self.order_size * (1.0 - abs(skew) * self.size_skew)
            bid_size   = self.order_size * (1.0 + abs(skew) * self.size_skew)

        return QuoteDecision(
            timestamp=state.timestamp,
            bid_price=mid_adj - bid_spread,
            bid_size=max(1.0, bid_size),
            ask_price=mid_adj + ask_spread,
            ask_size=max(1.0, ask_size),
            reason="inventory_skew",
        )


class VolatilityAdaptiveStrategy(QuotingStrategy):
    """
    Spread widens with realized volatility, narrows in calm periods.

    Spread = base_spread × (1 + vol_multiplier × σ / σ_target)

    Also withdraws quotes entirely if vol exceeds vol_ceiling.

    Parameters
    ----------
    base_half_spread : float – spread at target volatility
    vol_multiplier   : float – sensitivity to vol deviation
    sigma_target     : float – reference volatility
    vol_ceiling      : float – withdraw quotes above this vol
    order_size       : float
    """

    def __init__(
        self,
        base_half_spread: float = 0.05,
        vol_multiplier:   float = 2.0,
        sigma_target:     float = 0.01,
        vol_ceiling:      float = 0.05,
        order_size:       float = 10.0,
    ):
        self.base_half_spread = base_half_spread
        self.vol_multiplier   = vol_multiplier
        self.sigma_target     = sigma_target
        self.vol_ceiling      = vol_ceiling
        self.order_size       = order_size

    def compute_quotes(self, state, inventory, skew) -> QuoteDecision:
        sigma = state.volatility

        if sigma >= self.vol_ceiling:
            # withdraw: post far from market (effectively no quote)
            return QuoteDecision(
                timestamp=state.timestamp,
                bid_price=state.mid_price * 0.5,
                bid_size=0.0,
                ask_price=state.mid_price * 1.5,
                ask_size=0.0,
                reason="vol_withdrawal",
            )

        vol_ratio    = sigma / (self.sigma_target + 1e-10)
        half_spread  = self.base_half_spread * (1.0 + self.vol_multiplier * max(0, vol_ratio - 1.0))

        # also skew mid slightly for inventory
        mid_adj = state.mid_price - skew * self.base_half_spread * 0.5

        return QuoteDecision(
            timestamp=state.timestamp,
            bid_price=mid_adj - half_spread,
            bid_size=self.order_size,
            ask_price=mid_adj + half_spread,
            ask_size=self.order_size,
            reason="vol_adaptive",
        )


class MultiLevelStrategy(QuotingStrategy):
    """
    Post multiple layers of quotes at increasing distances from mid.

    Level k:
        spread_k = base_half_spread × level_multiplier^k
        size_k   = base_size × size_multiplier^k

    Outer levels act as inventory buffer during large moves.

    Parameters
    ----------
    n_levels        : int   – number of quote levels per side
    base_half_spread: float
    level_multiplier: float – spread multiplier per level (e.g. 1.5)
    base_size       : float – size at innermost level
    size_multiplier : float – size multiplier per level (e.g. 2.0)
    """

    def __init__(
        self,
        n_levels:         int   = 3,
        base_half_spread: float = 0.05,
        level_multiplier: float = 1.5,
        base_size:        float = 5.0,
        size_multiplier:  float = 2.0,
    ):
        self.n_levels         = n_levels
        self.base_half_spread = base_half_spread
        self.level_multiplier = level_multiplier
        self.base_size        = base_size
        self.size_multiplier  = size_multiplier

    def compute_quotes(self, state, inventory, skew) -> QuoteDecision:
        mid = state.mid_price - skew * self.base_half_spread

        levels_bid = []
        levels_ask = []

        for k in range(self.n_levels):
            half_spread = self.base_half_spread * (self.level_multiplier ** k)
            size        = self.base_size * (self.size_multiplier ** k)
            levels_bid.append((mid - half_spread, size))
            levels_ask.append((mid + half_spread, size))

        best_bid_price, best_bid_size = levels_bid[0]
        best_ask_price, best_ask_size = levels_ask[0]

        return QuoteDecision(
            timestamp=state.timestamp,
            bid_price=best_bid_price,
            bid_size=best_bid_size,
            ask_price=best_ask_price,
            ask_size=best_ask_size,
            reason="multi_level",
            extra_bids=levels_bid[1:],
            extra_asks=levels_ask[1:],
        )


# ---------------------------------------------------------------------------
# 5. Adverse Selection Detector
# ---------------------------------------------------------------------------

class AdverseSelectionDetector:
    """
    Flags fills that are likely from informed traders.

    Detection heuristics
    --------------------
    1. Post-fill price move : if mid moves against MM within n_ticks → adverse
    2. Trade size anomaly   : trade size > size_threshold × avg_size → likely informed
    3. Consecutive same-side fills : rapid one-sided flow → informed

    Parameters
    ----------
    lookback_ticks    : int   – how many ticks to check post-fill for adverse move
    adverse_threshold : float – fraction of spread considered adverse (e.g. 0.5)
    size_threshold    : float – size multiple to flag large trades
    """

    def __init__(
        self,
        lookback_ticks:    int   = 5,
        adverse_threshold: float = 0.5,
        size_threshold:    float = 3.0,
    ):
        self.lookback_ticks    = lookback_ticks
        self.adverse_threshold = adverse_threshold
        self.size_threshold    = size_threshold

        self._recent_fills: List[Fill]  = []
        self._mid_history:  List[float] = []
        self._avg_size:     float       = 10.0

    def update_mid(self, mid_price: float):
        self._mid_history.append(mid_price)

    def evaluate(self, fill: Fill, current_mid: float) -> bool:
        """
        Evaluate if a fill was adverse. Returns True if adverse.
        Updates avg_size estimate.
        """
        self._avg_size = 0.9 * self._avg_size + 0.1 * fill.size

        # size anomaly
        if fill.size > self.size_threshold * self._avg_size:
            return True

        # check prior mid moves
        n = min(self.lookback_ticks, len(self._mid_history))
        if n < 2:
            return False

        mid_at_fill = self._mid_history[-n]
        move        = current_mid - mid_at_fill

        # if bought from us (ask fill) and price went up → adverse
        if fill.side == "ask" and move > self.adverse_threshold * (current_mid * 0.001):
            return True
        # if sold to us (bid fill) and price went down → adverse
        if fill.side == "bid" and move < -self.adverse_threshold * (current_mid * 0.001):
            return True

        return False

    @property
    def adverse_fill_rate(self) -> float:
        if not self._recent_fills:
            return 0.0
        n_adverse = sum(1 for f in self._recent_fills if f.is_adverse)
        return n_adverse / len(self._recent_fills)


# ---------------------------------------------------------------------------
# 6. Performance Monitor
# ---------------------------------------------------------------------------

class PerformanceMonitor:
    """
    Computes market maker performance metrics for notebook analysis.
    """

    def __init__(self):
        self._pnl_series:    List[float] = []
        self._spread_earned: List[float] = []
        self._fill_sides:    List[str]   = []
        self._timestamps:    List[float] = []

    def record(self, t: float, pnl: float, spread_earned: float, fill_side: Optional[str] = None):
        self._pnl_series.append(pnl)
        self._spread_earned.append(spread_earned)
        self._timestamps.append(t)
        if fill_side:
            self._fill_sides.append(fill_side)

    def metrics(self) -> Dict:
        pnl = np.array(self._pnl_series)
        if len(pnl) < 2:
            return {}

        returns  = np.diff(pnl)
        mu       = np.mean(returns)
        sigma    = np.std(returns)
        sharpe   = (mu / sigma * np.sqrt(252)) if sigma > 0 else 0.0

        drawdown = self._max_drawdown(pnl)

        buy_fills  = self._fill_sides.count("bid")
        sell_fills = self._fill_sides.count("ask")
        total_fills = buy_fills + sell_fills

        return {
            "total_pnl":         float(pnl[-1]),
            "sharpe_ratio":      round(sharpe, 4),
            "max_drawdown":      round(drawdown, 4),
            "avg_spread_earned": round(np.mean(self._spread_earned), 6),
            "total_fills":       total_fills,
            "buy_fills":         buy_fills,
            "sell_fills":        sell_fills,
            "fill_imbalance":    (buy_fills - sell_fills) / (total_fills + 1e-10),
            "pnl_per_fill":      float(pnl[-1]) / (total_fills + 1e-10),
        }

    @staticmethod
    def _max_drawdown(pnl: np.ndarray) -> float:
        peak = np.maximum.accumulate(pnl)
        dd   = peak - pnl
        return float(np.max(dd)) if len(dd) > 0 else 0.0

    @property
    def pnl_series(self) -> np.ndarray:
        return np.array(self._pnl_series)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array(self._timestamps)


# ---------------------------------------------------------------------------
# 7. Market Maker Agent
# ---------------------------------------------------------------------------

class MarketMaker:
    """
    Full market maker agent.

    Wires together:
    - QuotingStrategy    → generates bid/ask quotes
    - InventoryManager   → tracks and limits inventory
    - PnLTracker         → decomposes P&L
    - RiskManager        → enforces limits, controls status
    - AdverseSelectionDetector → flags toxic flow
    - PerformanceMonitor → records metrics

    Usage (simulation loop)
    -----------------------
    mm = MarketMaker(strategy=AvellanedaStoikovStrategy())

    for state in market_states:
        quote = mm.on_market_update(state)
        if quote:
            submit_to_book(quote)

    if fill received:
        mm.on_fill(fill, state)

    Parameters
    ----------
    strategy        : QuotingStrategy instance
    max_inventory   : float
    max_drawdown    : float
    max_daily_loss  : float
    vol_threshold   : float
    tick_size       : float
    """

    def __init__(
        self,
        strategy:       Optional[QuotingStrategy] = None,
        max_inventory:  float = 1000.0,
        max_drawdown:   float = 5000.0,
        max_daily_loss: float = 3000.0,
        vol_threshold:  float = 0.05,
        tick_size:      float = 0.01,
    ):
        self.strategy  = strategy or AvellanedaStoikovStrategy()
        self.tick_size = tick_size

        self.inventory_mgr = InventoryManager(max_inventory=max_inventory)
        self.pnl_tracker   = PnLTracker()
        self.risk_mgr      = RiskManager(
            max_inventory=max_inventory,
            max_drawdown=max_drawdown,
            max_daily_loss=max_daily_loss,
            vol_threshold=vol_threshold,
        )
        self.adverse_detector = AdverseSelectionDetector()
        self.perf_monitor     = PerformanceMonitor()

        self.status:       MMStatus = MMStatus.ACTIVE
        self._last_quote:  Optional[QuoteDecision] = None
        self._last_state:  Optional[MarketState]   = None
        self._last_fill:   Optional[Fill]           = None
        self.snapshot_log: List[MMSnapshot]         = []

    # ---- main event handlers -----------------------------------------------

    def on_market_update(self, state: MarketState) -> Optional[QuoteDecision]:
        """
        Called each tick with new market state.
        Returns QuoteDecision to post, or None if quoting is suspended.
        """
        self._last_state = state
        self.adverse_detector.update_mid(state.mid_price)
        self.pnl_tracker.mark_to_market(state.mid_price)

        # risk check
        self.status = self.risk_mgr.check(
            inventory=self.inventory_mgr.inventory,
            total_pnl=self.pnl_tracker.total_pnl,
            volatility=state.volatility,
            last_fill=self._last_fill,
            t=state.timestamp,
        )

        if self.status != MMStatus.ACTIVE:
            self._record_snapshot(state, quote=None)
            return None

        # generate quotes
        skew  = self.inventory_mgr.skew_signal
        quote = self.strategy.compute_quotes(state, self.inventory_mgr.inventory, skew)

        # block sides at hard inventory limit
        if self.inventory_mgr.should_block_side("bid"):
            quote.bid_size = 0.0
        if self.inventory_mgr.should_block_side("ask"):
            quote.ask_size = 0.0

        # enforce minimum spread of 1 tick
        if quote.ask_price - quote.bid_price < self.tick_size:
            mid  = (quote.bid_price + quote.ask_price) / 2.0
            quote.bid_price = mid - self.tick_size / 2.0
            quote.ask_price = mid + self.tick_size / 2.0

        self._last_quote = quote
        self.perf_monitor.record(
            t=state.timestamp,
            pnl=self.pnl_tracker.total_pnl,
            spread_earned=self.pnl_tracker.spread_pnl,
        )
        self._record_snapshot(state, quote)
        return quote

    def on_fill(self, fill: Fill, state: MarketState):
        """Called when one of our resting orders is executed."""
        # check adverse selection
        fill.is_adverse = self.adverse_detector.evaluate(fill, state.mid_price)

        self.inventory_mgr.update(fill)
        self.pnl_tracker.record_fill(fill, state.mid_price)
        self._last_fill = fill

        self.perf_monitor.record(
            t=state.timestamp,
            pnl=self.pnl_tracker.total_pnl,
            spread_earned=self.pnl_tracker.spread_pnl,
            fill_side=fill.side,
        )

    # ---- convenience --------------------------------------------------------

    def reset_session(self):
        """Call at start of each trading session."""
        self.risk_mgr.reset_session(self.pnl_tracker.total_pnl)
        self.status = MMStatus.ACTIVE

    def summary(self) -> Dict:
        perf  = self.perf_monitor.metrics()
        attrib = self.pnl_tracker.attribution()
        return {
            "status":       self.status.value,
            "inventory":    self.inventory_mgr.inventory,
            "pnl":          attrib,
            "performance":  perf,
            "margin_calls": len(self.risk_mgr.breach_log),
            "risk_breaches": self.risk_mgr.breach_log[-3:],
        }

    def _record_snapshot(self, state: MarketState, quote: Optional[QuoteDecision]):
        inv = self.inventory_mgr.inventory
        self.snapshot_log.append(MMSnapshot(
            timestamp=state.timestamp,
            inventory=inv,
            cash=self.pnl_tracker.cash,
            mid_price=state.mid_price,
            unrealized=self.pnl_tracker.inventory_pnl,
            realized=self.pnl_tracker.realized_pnl,
            total_pnl=self.pnl_tracker.total_pnl,
            spread_pnl=self.pnl_tracker.spread_pnl,
            inventory_pnl=self.pnl_tracker.inventory_pnl,
            adverse_pnl=self.pnl_tracker.adverse_pnl,
            leverage=abs(inv) / (self.inventory_mgr.max_inventory + 1e-10),
            status=self.status.value,
            quote=quote,
        ))

    def snapshot_arrays(self) -> Dict[str, np.ndarray]:
        """Export full history as numpy arrays for notebook plotting."""
        if not self.snapshot_log:
            return {}
        keys = [
            "timestamp", "inventory", "cash", "mid_price",
            "unrealized", "realized", "total_pnl",
            "spread_pnl", "inventory_pnl", "adverse_pnl", "leverage",
        ]
        return {
            k: np.array([getattr(s, k) for s in self.snapshot_log])
            for k in keys
        }
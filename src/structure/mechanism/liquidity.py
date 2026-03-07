"""
liquidity.py  –  Market Liquidity Mechanism
============================================

Covers the following concepts for 02_liquidity_and_depth.ipynb:

1. Liquidity dimensions        – tightness, depth, resiliency, immediacy
2. Spread decomposition        – adverse selection, inventory, order processing costs
3. Depth & market impact       – Amihud illiquidity, Kyle's lambda, sqrt impact
4. Resilience                  – how fast the book recovers after a large trade
5. Effective & realized spread – execution quality metrics
6. Liquidity supply curve      – cost-to-execute as function of order size
7. Inventory-adjusted quotes   – dealer adjusts spread based on inventory risk

Design principle
----------------
- Consumes OrderBook snapshots (dicts) — no direct import of OrderBook class
- Stateless functions + stateful tracker classes
- All heavy math via numpy only (no scipy dependency)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Quote:
    """Best bid/ask at a single point in time."""
    timestamp:  float
    bid_price:  float
    bid_qty:    float
    ask_price:  float
    ask_qty:    float

    @property
    def mid(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def relative_spread(self) -> float:
        return self.spread / self.mid if self.mid > 0 else 0.0

    @property
    def imbalance(self) -> float:
        """Queue imbalance at best level ∈ [-1, +1]."""
        total = self.bid_qty + self.ask_qty
        return (self.bid_qty - self.ask_qty) / total if total > 0 else 0.0


@dataclass
class DepthSnapshot:
    """
    Full order book depth at a point in time.
    bids / asks: list of (price, qty) sorted best-first.
    """
    timestamp: float
    bids: List[Tuple[float, float]]   # [(price, qty), ...] descending
    asks: List[Tuple[float, float]]   # [(price, qty), ...] ascending
    mid_price: float

    # ---- depth metrics -------------------------------------------------------

    def cumulative_bid_depth(self, price_pct: float = 0.01) -> float:
        """Total bid volume within price_pct of mid."""
        threshold = self.mid_price * (1.0 - price_pct)
        return sum(q for p, q in self.bids if p >= threshold)

    def cumulative_ask_depth(self, price_pct: float = 0.01) -> float:
        """Total ask volume within price_pct of mid."""
        threshold = self.mid_price * (1.0 + price_pct)
        return sum(q for p, q in self.asks if p <= threshold)

    def depth_imbalance(self, price_pct: float = 0.01) -> Optional[float]:
        """Depth imbalance ∈ [-1, +1] within band."""
        b = self.cumulative_bid_depth(price_pct)
        a = self.cumulative_ask_depth(price_pct)
        total = b + a
        return (b - a) / total if total > 0 else None

    def liquidity_supply_curve(self, side: str = "ask") -> Tuple[np.ndarray, np.ndarray]:
        """
        Cost-to-execute curve: for each cumulative quantity, what is the VWAP?
        Returns (cumulative_qty, vwap_price).
        side: "ask" for buy orders, "bid" for sell orders.
        """
        levels = self.asks if side == "ask" else list(reversed(self.bids))
        if not levels:
            return np.array([]), np.array([])

        cum_qty = np.cumsum([q for _, q in levels])
        prices  = np.array([p for p, _ in levels])
        qtys    = np.array([q for _, q in levels])

        vwaps = np.cumsum(prices * qtys) / cum_qty
        return cum_qty, vwaps

    def market_impact_estimate(self, order_size: float, side: str = "ask") -> Optional[float]:
        """
        Estimate average execution price (VWAP) for a market order of given size.
        Returns None if insufficient liquidity.
        """
        levels = self.asks if side == "ask" else list(reversed(self.bids))
        remaining = order_size
        cost = 0.0

        for price, qty in levels:
            fill = min(remaining, qty)
            cost += fill * price
            remaining -= fill
            if remaining <= 0:
                break

        if remaining > 0:
            return None
        return cost / order_size

    def slippage(self, order_size: float, side: str = "ask") -> Optional[float]:
        """
        Slippage = execution VWAP - mid_price (buy) or mid_price - VWAP (sell).
        Positive slippage = cost to taker.
        """
        vwap = self.market_impact_estimate(order_size, side)
        if vwap is None:
            return None
        if side == "ask":
            return vwap - self.mid_price
        else:
            return self.mid_price - vwap


# ---------------------------------------------------------------------------
# 1. Spread Decomposition
# ---------------------------------------------------------------------------

class SpreadDecomposition:
    """
    Glosten-Harris (1988) / Huang-Stoll (1997) decomposition:

        Quoted spread = Adverse Selection + Inventory + Order Processing

    Estimated from time-series of quotes and trade signs.

    Parameters
    ----------
    rho : float – first-order serial correlation of trade signs (inventory proxy)
    """

    def __init__(self, rho: float = 0.0):
        self.rho = rho

    def decompose(
        self,
        spreads:    np.ndarray,
        trade_signs: np.ndarray,   # +1 buy, -1 sell
        price_changes: np.ndarray,
    ) -> Dict[str, float]:
        """
        Returns fraction of spread attributable to each component.
        Uses Huang-Stoll (1997) GMM-style OLS proxy.
        """
        s = np.mean(spreads)
        if s <= 0:
            return {"adverse_selection": 0.0, "inventory": 0.0, "order_processing": 0.0}

        # Adverse selection: covariance of consecutive price changes with trade sign
        if len(price_changes) > 1 and len(trade_signs) > 1:
            n = min(len(price_changes), len(trade_signs)) - 1
            cov_as = np.cov(price_changes[:n], trade_signs[:n])[0, 1]
        else:
            cov_as = 0.0

        alpha = max(0.0, min(1.0, abs(cov_as) / (s / 2 + 1e-10)))
        beta  = max(0.0, min(1.0 - alpha, abs(self.rho) * (1.0 - alpha)))
        gamma = max(0.0, 1.0 - alpha - beta)

        return {
            "adverse_selection": round(alpha, 4),
            "inventory":         round(beta,  4),
            "order_processing":  round(gamma, 4),
        }

    @staticmethod
    def effective_spread(trade_price: float, mid_price: float) -> float:
        """
        Effective spread = 2 · |trade_price - mid_price|.
        Measures actual execution cost vs quoted spread.
        """
        return 2.0 * abs(trade_price - mid_price)

    @staticmethod
    def realized_spread(
        trade_price: float,
        mid_price_t: float,
        mid_price_t5: float,
        sign: int,
    ) -> float:
        """
        Realized spread = effective spread - price impact.
        sign: +1 buy, -1 sell.
        Measures dealer profit after adverse price movement.
        """
        effective = 2.0 * sign * (trade_price - mid_price_t)
        impact    = 2.0 * sign * (mid_price_t5 - mid_price_t)
        return effective - impact

    @staticmethod
    def price_impact_component(
        mid_price_t:  float,
        mid_price_t5: float,
        sign: int,
    ) -> float:
        """
        Price impact = mid price change in direction of trade.
        Proxy for adverse selection cost.
        """
        return 2.0 * sign * (mid_price_t5 - mid_price_t)


# ---------------------------------------------------------------------------
# 2. Amihud Illiquidity
# ---------------------------------------------------------------------------

class AmihudIlliquidity:
    """
    Amihud (2002) illiquidity ratio:
        ILLIQ_t = |r_t| / Volume_t

    Daily average:
        ILLIQ = (1/T) · Σ |r_t| / Vol_t

    Higher ILLIQ → price moves more per unit volume → less liquid.

    Parameters
    ----------
    window : int – rolling window for smoothed estimate
    """

    def __init__(self, window: int = 20):
        self.window = window
        self._series: List[float] = []

    def update(self, price_return: float, volume: float) -> Optional[float]:
        if volume <= 0:
            return None
        ratio = abs(price_return) / volume
        self._series.append(ratio)

        if len(self._series) >= self.window:
            return float(np.mean(self._series[-self.window:]))
        return None

    @staticmethod
    def from_series(returns: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Vectorized Amihud ratio series."""
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(volumes > 0, np.abs(returns) / volumes, np.nan)
        return ratios

    @staticmethod
    def rolling(returns: np.ndarray, volumes: np.ndarray, window: int = 20) -> np.ndarray:
        """Rolling mean of Amihud ratio."""
        ratios = AmihudIlliquidity.from_series(returns, volumes)
        result = np.full_like(ratios, np.nan)
        for i in range(window - 1, len(ratios)):
            window_vals = ratios[i - window + 1: i + 1]
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) > 0:
                result[i] = np.mean(valid)
        return result


# ---------------------------------------------------------------------------
# 3. Kyle's Lambda (Price Impact)
# ---------------------------------------------------------------------------

class KyleLambda:
    """
    Kyle (1985) lambda – permanent price impact per unit of signed order flow.

        ΔP = λ · Q + ε

    Estimated via OLS regression of price changes on signed volume.

    Higher λ → market is less liquid (price moves more per trade).
    """

    def __init__(self, window: int = 50):
        self.window = window
        self._price_changes:  List[float] = []
        self._signed_volumes: List[float] = []
        self.lambda_: Optional[float] = None

    def update(self, price_change: float, signed_volume: float) -> Optional[float]:
        self._price_changes.append(price_change)
        self._signed_volumes.append(signed_volume)

        if len(self._price_changes) >= self.window:
            dp = np.array(self._price_changes[-self.window:])
            sv = np.array(self._signed_volumes[-self.window:])
            self.lambda_ = self._ols(dp, sv)

        return self.lambda_

    @staticmethod
    def _ols(y: np.ndarray, x: np.ndarray) -> float:
        var_x = np.var(x)
        if var_x < 1e-12:
            return 0.0
        return float(np.cov(y, x)[0, 1] / var_x)

    @staticmethod
    def estimate(
        price_changes:  np.ndarray,
        signed_volumes: np.ndarray,
    ) -> float:
        """One-shot OLS estimate."""
        return KyleLambda._ols(price_changes, signed_volumes)

    @staticmethod
    def sqrt_impact(
        quantity:   float,
        volatility: float,
        adv:        float,
        eta:        float = 0.1,
    ) -> float:
        """
        Square-root market impact model (Almgren et al. 2005):
            ΔP / σ = η · √(Q / ADV)

        Parameters
        ----------
        quantity   : order size
        volatility : daily return volatility
        adv        : average daily volume
        eta        : market impact constant (default 0.1 for liquid markets)
        """
        if adv <= 0:
            return 0.0
        return eta * volatility * np.sqrt(quantity / adv)


# ---------------------------------------------------------------------------
# 4. Liquidity Resilience
# ---------------------------------------------------------------------------

class LiquidityResilience:
    """
    Measures how fast the order book recovers depth after a large trade.

    Method: track best bid/ask depth (or spread) over time after a shock.
    Fit exponential recovery: depth(t) = D_∞ · (1 - exp(-κ·t))

    High κ → fast recovery → resilient market.
    """

    def __init__(self):
        self._pre_shock_depth:  Optional[float] = None
        self._shock_time:       Optional[float] = None
        self._recovery_series:  List[Tuple[float, float]] = []  # (dt, depth)

    def register_shock(self, depth_before: float, t: float):
        """Call just before a large trade executes."""
        self._pre_shock_depth = depth_before
        self._shock_time      = t
        self._recovery_series.clear()

    def observe(self, depth: float, t: float):
        """Call after the shock to track recovery."""
        if self._shock_time is None:
            return
        dt = t - self._shock_time
        self._recovery_series.append((dt, depth))

    def recovery_rate(self) -> Optional[float]:
        """
        Estimate κ via log-linear OLS on depth recovery series.
        Returns None if insufficient data.
        """
        if len(self._recovery_series) < 3 or self._pre_shock_depth is None:
            return None

        dts    = np.array([x[0] for x in self._recovery_series])
        depths = np.array([x[1] for x in self._recovery_series])

        D_inf = self._pre_shock_depth
        ratio = np.clip(1.0 - depths / D_inf, 1e-10, 1.0 - 1e-10)
        log_ratio = np.log(ratio)

        # log(1 - d/D∞) = -κ·t → OLS slope = -κ
        valid = np.isfinite(log_ratio)
        if valid.sum() < 2:
            return None

        slope = KyleLambda._ols(log_ratio[valid], dts[valid])
        return float(-slope)

    def half_life(self) -> Optional[float]:
        """
        Half-life of depth recovery (seconds) = ln(2) / κ.
        Lower half-life → more resilient.
        """
        kappa = self.recovery_rate()
        if kappa is None or kappa <= 0:
            return None
        return float(np.log(2) / kappa)


# ---------------------------------------------------------------------------
# 5. Inventory-Adjusted Quotes
# ---------------------------------------------------------------------------

class InventoryAdjustedQuotes:
    """
    Avellaneda-Stoikov (2008) dealer quote model.

    Dealer adjusts mid-quote based on inventory risk:
        r(s, q) = s - q · γ · σ² · (T - t)          [reservation price]
        δ_bid   = 1/γ · ln(1 + γ/κ) + (2q+1)/2 · √(σ²·γ/κ)·(T-t)
        δ_ask   = 1/γ · ln(1 + γ/κ) - (2q-1)/2 · √(σ²·γ/κ)·(T-t)

    Simplified linear version used here for tractability:
        mid_adj = mid - q · skew_factor
        spread  = base_spread + |q| · inventory_penalty

    Parameters
    ----------
    gamma           : float – risk aversion coefficient
    base_spread     : float – minimum quoted spread (e.g. 1 tick)
    inventory_penalty: float – additional spread per unit of inventory
    skew_factor     : float – mid-price skew per unit of inventory
    max_inventory   : float – absolute inventory limit
    """

    def __init__(
        self,
        gamma:             float = 0.1,
        base_spread:       float = 0.02,
        inventory_penalty: float = 0.005,
        skew_factor:       float = 0.003,
        max_inventory:     float = 1000.0,
    ):
        self.gamma             = gamma
        self.base_spread       = base_spread
        self.inventory_penalty = inventory_penalty
        self.skew_factor       = skew_factor
        self.max_inventory     = max_inventory

    def quotes(
        self,
        mid_price: float,
        inventory: float,
        volatility: float = 0.01,
        time_remaining: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Returns (bid_price, ask_price) adjusted for inventory.

        Parameters
        ----------
        mid_price     : current market mid
        inventory     : dealer's current signed inventory (+ long, - short)
        volatility    : short-term price volatility
        time_remaining: fraction of trading session remaining [0, 1]
        """
        inv_ratio = inventory / (self.max_inventory + 1e-10)

        # Reservation price: shift mid against inventory
        reservation = mid_price - inv_ratio * self.skew_factor * mid_price

        # Spread widens with inventory and volatility
        half_spread = (
            self.base_spread / 2.0
            + abs(inventory) * self.inventory_penalty
            + self.gamma * volatility ** 2 * time_remaining
        )

        bid = reservation - half_spread
        ask = reservation + half_spread
        return round(bid, 6), round(ask, 6)

    def optimal_inventory(self, mid_price: float, volatility: float) -> float:
        """
        Target inventory that minimizes expected PnL variance.
        In A-S model this is always 0; in practice bounded by risk limits.
        """
        return 0.0


# ---------------------------------------------------------------------------
# 6. Liquidity Tracker (rolling aggregator)
# ---------------------------------------------------------------------------

class LiquidityTracker:
    """
    Ingests a stream of Quote objects and maintains rolling liquidity metrics.

    Metrics tracked
    ---------------
    - quoted spread (mean, std, percentiles)
    - relative spread
    - depth at best (bid & ask)
    - depth imbalance
    - Amihud illiquidity
    - Kyle's lambda
    """

    def __init__(self, window: int = 100):
        self.window = window

        self._quotes:         List[Quote]  = []
        self._trade_prices:   List[float]  = []
        self._trade_signs:    List[int]    = []
        self._trade_volumes:  List[float]  = []

        self.amihud  = AmihudIlliquidity(window=window)
        self.kyle    = KyleLambda(window=window)

    def update_quote(self, quote: Quote) -> None:
        self._quotes.append(quote)
        if len(self._quotes) > self.window * 2:
            self._quotes = self._quotes[-self.window:]

    def update_trade(self, price: float, volume: float, sign: int) -> None:
        self._trade_prices.append(price)
        self._trade_signs.append(sign)
        self._trade_volumes.append(volume)

        if len(self._trade_prices) > 1:
            dp = self._trade_prices[-1] - self._trade_prices[-2]
            sv = sign * volume
            self.kyle.update(dp, sv)

        if len(self._trade_prices) > 1:
            ret = (self._trade_prices[-1] - self._trade_prices[-2]) / (self._trade_prices[-2] + 1e-10)
            self.amihud.update(ret, volume)

    def snapshot_metrics(self) -> Dict:
        """Return current rolling liquidity metrics as a flat dict."""
        recent = self._quotes[-self.window:] if self._quotes else []

        spreads    = np.array([q.spread          for q in recent])
        rel_spread = np.array([q.relative_spread  for q in recent])
        bid_depth  = np.array([q.bid_qty          for q in recent])
        ask_depth  = np.array([q.ask_qty          for q in recent])
        imbalance  = np.array([q.imbalance        for q in recent])

        def _safe(arr, fn):
            return float(fn(arr)) if len(arr) > 0 else None

        return {
            "n_quotes":         len(recent),
            "spread_mean":      _safe(spreads,    np.mean),
            "spread_std":       _safe(spreads,    np.std),
            "spread_p95":       _safe(spreads,    lambda x: np.percentile(x, 95)),
            "rel_spread_mean":  _safe(rel_spread, np.mean),
            "bid_depth_mean":   _safe(bid_depth,  np.mean),
            "ask_depth_mean":   _safe(ask_depth,  np.mean),
            "imbalance_mean":   _safe(imbalance,  np.mean),
            "kyle_lambda":      self.kyle.lambda_,
            "amihud_illiq":     float(np.mean(self.amihud._series[-self.window:])) if self.amihud._series else None,
        }
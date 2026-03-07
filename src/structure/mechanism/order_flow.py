"""
order_flow.py  –  Order Flow Mechanism
=======================================

Covers the following microstructure concepts for 01_order_flow.ipynb:

1. Order arrival process       – Poisson arrivals, rate asymmetry
2. Order classification        – tick rule, Lee-Ready algorithm
3. Order Flow Imbalance (OFI)  – Cont, Kukanov & Stoikov (2014)
4. Trade sign & toxicity       – PIN proxy, adverse selection cost
5. Informed vs noise traders   – two-type flow decomposition
6. Price impact                – temporary vs permanent impact
7. VPIN                        – volume-synchronized PIN (Easley et al.)

Design principle
----------------
- Pure generation + classification logic lives here (mechanism layer)
- Metrics (OFI, PIN, lambda...) are computed by microstructure.py
- OrderBook state is consumed but NOT imported (passed as argument)
  to keep the dependency graph acyclic
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TradeSign(Enum):
    BUY  = 1
    SELL = -1
    UNKNOWN = 0


class TraderType(Enum):
    NOISE    = "noise"       # uninformed, random direction
    INFORMED = "informed"    # trades on private signal
    MARKET_MAKER = "mm"      # liquidity provider (passive)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OrderEvent:
    """
    Atomic order flow event – input to OrderBook.submit() or used standalone
    for flow analysis without a live book.
    """
    timestamp:   float
    side:        str          # "bid" | "ask"
    order_type:  str          # "limit" | "market"
    price:       float        # 0.0 for market orders
    quantity:    float
    trader_type: TraderType = TraderType.NOISE
    trader_id:   Optional[str] = None
    # filled in post-hoc by classifier
    sign:        TradeSign = TradeSign.UNKNOWN


@dataclass
class FlowSnapshot:
    """Aggregated order flow over a time bucket."""
    t_start:      float
    t_end:        float
    buy_volume:   float = 0.0
    sell_volume:  float = 0.0
    buy_orders:   int   = 0
    sell_orders:  int   = 0
    limit_orders: int   = 0
    market_orders: int  = 0
    cancel_orders: int  = 0

    @property
    def net_flow(self) -> float:
        return self.buy_volume - self.sell_volume

    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume

    @property
    def imbalance(self) -> Optional[float]:
        """OFI in [-1, +1]."""
        total = self.total_volume
        return self.net_flow / total if total > 0 else None

    @property
    def order_imbalance_ratio(self) -> Optional[float]:
        total = self.buy_orders + self.sell_orders
        return (self.buy_orders - self.sell_orders) / total if total > 0 else None


# ---------------------------------------------------------------------------
# 1. Order Arrival Process
# ---------------------------------------------------------------------------

class PoissonArrivalProcess:
    """
    Homogeneous Poisson process for order arrivals.

    Parameters
    ----------
    lambda_buy  : float  – arrival rate of buy orders (orders / unit time)
    lambda_sell : float  – arrival rate of sell orders
    dt          : float  – simulation time step
    rng         : np.random.Generator
    """

    def __init__(
        self,
        lambda_buy:  float = 10.0,
        lambda_sell: float = 10.0,
        dt:          float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.lambda_buy  = lambda_buy
        self.lambda_sell = lambda_sell
        self.dt          = dt
        self.rng = rng or np.random.default_rng()

    def sample(self) -> Tuple[int, int]:
        """Return (n_buy, n_sell) arrivals in one dt step."""
        n_buy  = self.rng.poisson(self.lambda_buy  * self.dt)
        n_sell = self.rng.poisson(self.lambda_sell * self.dt)
        return int(n_buy), int(n_sell)

    def arrival_rate_asymmetry(self) -> float:
        """
        Rate asymmetry ∈ [-1, +1].
        Positive → more buy pressure; negative → more sell pressure.
        """
        total = self.lambda_buy + self.lambda_sell
        if total == 0:
            return 0.0
        return (self.lambda_buy - self.lambda_sell) / total


class HawkesArrivalProcess:
    """
    Self-exciting (Hawkes) arrival process – captures order flow clustering.

    λ(t) = μ + Σ α·exp(-β·(t - tᵢ))   for tᵢ < t

    Parameters
    ----------
    mu    : float  – baseline arrival rate
    alpha : float  – self-excitation magnitude
    beta  : float  – decay rate of excitation kernel
    """

    def __init__(
        self,
        mu:    float = 5.0,
        alpha: float = 0.6,
        beta:  float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.mu    = mu
        self.alpha = alpha
        self.beta  = beta
        self.rng   = rng or np.random.default_rng()
        self._history: List[float] = []   # event timestamps

    @property
    def is_stationary(self) -> bool:
        """Stationarity condition: alpha < beta."""
        return self.alpha < self.beta

    def intensity(self, t: float) -> float:
        """Current arrival intensity λ(t)."""
        kernel = sum(
            self.alpha * np.exp(-self.beta * (t - ti))
            for ti in self._history if ti < t
        )
        return self.mu + kernel

    def simulate(self, T: float) -> List[float]:
        """
        Simulate event times in [0, T] via Ogata thinning algorithm.
        Returns list of event timestamps.
        """
        self._history = []
        t = 0.0

        while t < T:
            lam_bar = self.intensity(t) + self.alpha   # upper bound
            dt = self.rng.exponential(1.0 / lam_bar)
            t += dt
            if t >= T:
                break
            u = self.rng.uniform()
            if u <= self.intensity(t) / lam_bar:
                self._history.append(t)

        return list(self._history)

    def reset(self):
        self._history = []


# ---------------------------------------------------------------------------
# 2. Order Size Distribution
# ---------------------------------------------------------------------------

class OrderSizeDistribution:
    """
    Empirically, order sizes follow a log-normal or power-law distribution.

    Parameters
    ----------
    distribution : str   – "lognormal" | "powerlaw" | "uniform"
    mu_log       : float – log-mean  (lognormal)
    sigma_log    : float – log-std   (lognormal)
    alpha_pl     : float – tail exponent (powerlaw, typically 1.5–3.0)
    size_min     : float – minimum order size
    size_max     : float – maximum order size (clip)
    """

    def __init__(
        self,
        distribution: str  = "lognormal",
        mu_log:       float = 2.0,
        sigma_log:    float = 1.0,
        alpha_pl:     float = 2.0,
        size_min:     float = 1.0,
        size_max:     float = 1000.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.distribution = distribution
        self.mu_log    = mu_log
        self.sigma_log = sigma_log
        self.alpha_pl  = alpha_pl
        self.size_min  = size_min
        self.size_max  = size_max
        self.rng = rng or np.random.default_rng()

    def sample(self, n: int = 1) -> np.ndarray:
        if self.distribution == "lognormal":
            sizes = self.rng.lognormal(self.mu_log, self.sigma_log, size=n)
        elif self.distribution == "powerlaw":
            # Pareto: x_min * (1 - u)^(-1/(alpha-1))
            u = self.rng.uniform(size=n)
            sizes = self.size_min * (1.0 - u) ** (-1.0 / (self.alpha_pl - 1.0))
        else:  # uniform
            sizes = self.rng.uniform(self.size_min, self.size_max, size=n)

        return np.clip(sizes, self.size_min, self.size_max)


# ---------------------------------------------------------------------------
# 3. Informed vs Noise Trader Flow Generator
# ---------------------------------------------------------------------------

class OrderFlowGenerator:
    """
    Two-type flow model (Glosten-Milgrom inspired).

    - Noise traders  : direction is iid random, size from size_dist
    - Informed traders: direction aligned with private signal, larger sizes

    Parameters
    ----------
    prob_informed    : float  – probability an arriving order is informed
    signal_strength  : float  – probability informed trader trades in signal direction
    arrival_process  : PoissonArrivalProcess | HawkesArrivalProcess
    size_dist        : OrderSizeDistribution
    limit_ratio      : float  – fraction of orders that are limit (vs market)
    price_offset_ticks: int   – how far from mid a limit order is placed
    """

    def __init__(
        self,
        prob_informed:      float = 0.15,
        signal_strength:    float = 0.80,
        arrival_process=    None,
        size_dist=          None,
        limit_ratio:        float = 0.70,
        price_offset_ticks: int   = 2,
        tick_size:          float = 0.01,
        rng: Optional[np.random.Generator] = None,
    ):
        self.prob_informed      = prob_informed
        self.signal_strength    = signal_strength
        self.limit_ratio        = limit_ratio
        self.price_offset_ticks = price_offset_ticks
        self.tick_size          = tick_size
        self.rng = rng or np.random.default_rng()

        self.arrival  = arrival_process or PoissonArrivalProcess(rng=self.rng)
        self.size_dist = size_dist or OrderSizeDistribution(rng=self.rng)

        # private signal: +1 (buy) or -1 (sell), updated each period
        self._signal: int = 1

    def update_signal(self, signal: Optional[int] = None):
        """Refresh private signal. If None, randomize."""
        if signal is not None:
            self._signal = signal
        else:
            self._signal = 1 if self.rng.random() > 0.5 else -1

    def generate(
        self,
        n_steps:   int,
        mid_price: float,
        dt:        float = 1.0,
    ) -> List[OrderEvent]:
        """
        Generate a stream of OrderEvents over n_steps time steps.

        Parameters
        ----------
        n_steps   : number of time steps
        mid_price : reference price (updated internally each step)
        dt        : duration of each time step
        """
        events: List[OrderEvent] = []
        t = 0.0

        for _ in range(n_steps):
            n_buy, n_sell = self.arrival.sample() if hasattr(self.arrival, 'sample') \
                else (1, 1)

            for side_char, n in [("bid", n_buy), ("ask", n_sell)]:
                for _ in range(n):
                    event = self._make_event(t, side_char, mid_price)
                    events.append(event)

            t += dt

        return events

    def _make_event(self, t: float, side: str, mid: float) -> OrderEvent:
        is_informed = self.rng.random() < self.prob_informed
        trader_type = TraderType.INFORMED if is_informed else TraderType.NOISE

        # Informed trader may flip direction based on signal
        if is_informed:
            trades_with_signal = self.rng.random() < self.signal_strength
            if trades_with_signal:
                side = "bid" if self._signal == 1 else "ask"

        is_limit = self.rng.random() < self.limit_ratio
        order_type = "limit" if is_limit else "market"

        qty = float(self.size_dist.sample(1)[0])

        # Informed traders tend to use larger sizes
        if is_informed:
            qty *= 1.5 + self.rng.exponential(0.5)

        if is_limit:
            offset = self.price_offset_ticks * self.tick_size
            price = mid - offset if side == "bid" else mid + offset
        else:
            price = 0.0

        return OrderEvent(
            timestamp=t,
            side=side,
            order_type=order_type,
            price=price,
            quantity=qty,
            trader_type=trader_type,
        )


# ---------------------------------------------------------------------------
# 4. Trade Sign Classification
# ---------------------------------------------------------------------------

class TradeClassifier:
    """
    Classifies trades as buyer- or seller-initiated.

    Methods
    -------
    tick_rule      : sign follows last price change direction
    lee_ready      : compare trade price to prevailing quote midpoint
    bulk_classify  : apply to a list of (price, bid, ask) tuples
    """

    @staticmethod
    def tick_rule(prices: List[float]) -> List[TradeSign]:
        """
        Tick rule: BUY if price ↑, SELL if price ↓, carry forward if unchanged.
        O(n), simple but noisy.
        """
        signs = [TradeSign.UNKNOWN]
        for i in range(1, len(prices)):
            dp = prices[i] - prices[i - 1]
            if dp > 0:
                signs.append(TradeSign.BUY)
            elif dp < 0:
                signs.append(TradeSign.SELL)
            else:
                signs.append(signs[-1])   # carry forward
        return signs

    @staticmethod
    def lee_ready(
        trade_prices: List[float],
        bid_prices:   List[float],
        ask_prices:   List[float],
    ) -> List[TradeSign]:
        """
        Lee & Ready (1991):
        - price > mid  → BUY
        - price < mid  → SELL
        - price == mid → tick rule fallback
        """
        n = len(trade_prices)
        signs: List[TradeSign] = []
        last_sign = TradeSign.UNKNOWN

        for i in range(n):
            mid = (bid_prices[i] + ask_prices[i]) / 2.0
            tp  = trade_prices[i]

            if tp > mid:
                s = TradeSign.BUY
            elif tp < mid:
                s = TradeSign.SELL
            else:
                # tick rule fallback
                if i > 0 and trade_prices[i] != trade_prices[i - 1]:
                    s = TradeSign.BUY if tp > trade_prices[i - 1] else TradeSign.SELL
                else:
                    s = last_sign

            signs.append(s)
            last_sign = s

        return signs

    @staticmethod
    def bulk_volume_classify(
        opens:  np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bulk Volume Classification (Easley et al. 2012) – bar-level proxy.
        buy_vol  = V · Z((C - O) / σ)
        sell_vol = V - buy_vol

        Returns (buy_volumes, sell_volumes).
        """
        returns = closes - opens
        sigma   = np.std(returns[returns != 0]) or 1e-8

        z = returns / sigma
        # Φ(z) via logistic approximation (fast, no scipy dependency)
        phi = 1.0 / (1.0 + np.exp(-1.7 * z))

        buy_vols  = volumes * phi
        sell_vols = volumes * (1.0 - phi)
        return buy_vols, sell_vols


# ---------------------------------------------------------------------------
# 5. Order Flow Imbalance (OFI)
# ---------------------------------------------------------------------------

class OFICalculator:
    """
    Order Flow Imbalance per Cont, Kukanov & Stoikov (2014).

    OFI captures changes in *queue* depth at best bid/ask:
        e_bid(t) = qB(t)·1[PB(t)≥PB(t-1)] - qB(t-1)·1[PB(t)≤PB(t-1)]
        e_ask(t) = qA(t-1)·1[PA(t)≥PA(t-1)] - qA(t)·1[PA(t)≤PA(t-1)]
        OFI(t) = e_bid(t) - e_ask(t)

    Positive OFI → net buying pressure → price tends to rise.
    """

    def __init__(self):
        self._prev_bid_price: Optional[float] = None
        self._prev_bid_qty:   Optional[float] = None
        self._prev_ask_price: Optional[float] = None
        self._prev_ask_qty:   Optional[float] = None
        self.ofi_series: List[float] = []

    def update(
        self,
        bid_price: float, bid_qty: float,
        ask_price: float, ask_qty: float,
    ) -> Optional[float]:
        """
        Feed one tick of best bid/ask. Returns OFI for this tick.
        Returns None on first call (no previous state).
        """
        if self._prev_bid_price is None:
            self._prev_bid_price = bid_price
            self._prev_bid_qty   = bid_qty
            self._prev_ask_price = ask_price
            self._prev_ask_qty   = ask_qty
            return None

        # bid side contribution
        e_bid = 0.0
        if bid_price >= self._prev_bid_price:
            e_bid += bid_qty
        if bid_price <= self._prev_bid_price:
            e_bid -= self._prev_bid_qty  # type: ignore[operator]

        # ask side contribution
        e_ask = 0.0
        if ask_price >= self._prev_ask_price:
            e_ask -= ask_qty
        if ask_price <= self._prev_ask_price:
            e_ask += self._prev_ask_qty  # type: ignore[operator]

        ofi = e_bid - e_ask

        self.ofi_series.append(ofi)
        self._prev_bid_price = bid_price
        self._prev_bid_qty   = bid_qty
        self._prev_ask_price = ask_price
        self._prev_ask_qty   = ask_qty

        return ofi

    def reset(self):
        self._prev_bid_price = self._prev_bid_qty = None
        self._prev_ask_price = self._prev_ask_qty = None
        self.ofi_series.clear()

    def cumulative_ofi(self) -> np.ndarray:
        return np.cumsum(self.ofi_series)


# ---------------------------------------------------------------------------
# 6. VPIN – Volume-Synchronized PIN
# ---------------------------------------------------------------------------

class VPINCalculator:
    """
    VPIN (Easley, López de Prado & O'Hara, 2012).

    Partitions volume into equal-size buckets, classifies buy/sell volume
    per bucket via BVC, then computes:
        VPIN = (1/n) · Σ |V_buy - V_sell| / V_bucket

    High VPIN → high probability of informed trading → toxic flow.

    Parameters
    ----------
    bucket_size : float  – target volume per bucket
    window      : int    – rolling window of buckets for VPIN estimate
    """

    def __init__(self, bucket_size: float = 1000.0, window: int = 50):
        self.bucket_size = bucket_size
        self.window      = window
        self._buckets: List[Tuple[float, float]] = []   # (buy_vol, sell_vol)
        self._buffer_vol:  float = 0.0
        self._buffer_buy:  float = 0.0
        self._buffer_sell: float = 0.0

    def update(self, buy_vol: float, sell_vol: float) -> Optional[float]:
        """
        Feed volume for one bar. Returns current VPIN estimate when
        at least `window` buckets are available, else None.
        """
        total = buy_vol + sell_vol
        self._buffer_buy  += buy_vol
        self._buffer_sell += sell_vol
        self._buffer_vol  += total

        vpin = None

        while self._buffer_vol >= self.bucket_size:
            frac = self.bucket_size / self._buffer_vol
            b = self._buffer_buy  * frac
            s = self._buffer_sell * frac
            self._buckets.append((b, s))
            self._buffer_buy  -= b
            self._buffer_sell -= s
            self._buffer_vol  -= self.bucket_size

            if len(self._buckets) >= self.window:
                recent = self._buckets[-self.window:]
                vpin = sum(abs(b - s) for b, s in recent) / (self.window * self.bucket_size)

        return vpin

    def current_vpin(self) -> Optional[float]:
        if len(self._buckets) < self.window:
            return None
        recent = self._buckets[-self.window:]
        return sum(abs(b - s) for b, s in recent) / (self.window * self.bucket_size)

    def reset(self):
        self._buckets.clear()
        self._buffer_vol = self._buffer_buy = self._buffer_sell = 0.0


# ---------------------------------------------------------------------------
# 7. Flow Bucketing – aggregate raw events into FlowSnapshot
# ---------------------------------------------------------------------------

def bucket_flow(
    events: List[OrderEvent],
    bucket_duration: float = 60.0,
) -> List[FlowSnapshot]:
    """
    Aggregate OrderEvents into fixed-duration time buckets.

    Parameters
    ----------
    events          : list of OrderEvent (must be time-sorted)
    bucket_duration : seconds per bucket

    Returns
    -------
    List of FlowSnapshot, one per bucket.
    """
    if not events:
        return []

    t0 = events[0].timestamp
    buckets: List[FlowSnapshot] = []
    current = FlowSnapshot(t_start=t0, t_end=t0 + bucket_duration)

    for ev in events:
        # roll to next bucket if needed
        while ev.timestamp >= current.t_end:
            buckets.append(current)
            current = FlowSnapshot(
                t_start=current.t_end,
                t_end=current.t_end + bucket_duration,
            )

        if ev.order_type == "market":
            current.market_orders += 1
            if ev.side == "bid":
                current.buy_volume  += ev.quantity
                current.buy_orders  += 1
            else:
                current.sell_volume += ev.quantity
                current.sell_orders += 1
        elif ev.order_type == "limit":
            current.limit_orders += 1
        else:
            current.cancel_orders += 1

    buckets.append(current)
    return buckets


# ---------------------------------------------------------------------------
# 8. Price Impact Model
# ---------------------------------------------------------------------------

class PriceImpactModel:
    """
    Decompose price impact into temporary and permanent components.

    Almgren-Chriss linear model:
        ΔP_permanent  = λ · Q        (information-driven, persists)
        ΔP_temporary  = η · Q        (liquidity cost, reverts)
        ΔP_total      = (λ + η) · Q

    Parameters
    ----------
    lambda_ : float  – permanent impact coefficient (Kyle's lambda proxy)
    eta     : float  – temporary impact coefficient
    """

    def __init__(self, lambda_: float = 0.001, eta: float = 0.002):
        self.lambda_ = lambda_
        self.eta     = eta

    def total_impact(self, quantity: float, sign: int = 1) -> float:
        """sign: +1 buy, -1 sell."""
        return sign * (self.lambda_ + self.eta) * quantity

    def permanent_impact(self, quantity: float, sign: int = 1) -> float:
        return sign * self.lambda_ * quantity

    def temporary_impact(self, quantity: float, sign: int = 1) -> float:
        return sign * self.eta * quantity

    def execution_shortfall(
        self,
        quantities: np.ndarray,
        signs:      np.ndarray,
        arrival_price: float,
    ) -> float:
        """
        Implementation shortfall: total cost vs arrival price benchmark.
        """
        impacts = np.array([
            self.total_impact(q, s) for q, s in zip(quantities, signs)
        ])
        prices = arrival_price + np.cumsum(impacts)
        costs  = quantities * (prices - arrival_price) * signs
        return float(np.sum(costs))

    def calibrate(
        self,
        signed_volumes: np.ndarray,
        price_changes:  np.ndarray,
    ) -> None:
        """
        OLS calibration: ΔP = lambda · signed_vol + ε.
        Updates self.lambda_ in place.
        """
        x = signed_volumes
        y = price_changes
        if np.std(x) < 1e-10:
            return
        self.lambda_ = float(np.cov(y, x)[0, 1] / np.var(x))
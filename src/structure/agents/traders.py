"""
traders.py  –  Trader Agents
=============================

Agent roster for market simulation:

1. BaseTrader          – abstract interface all agents implement
2. NoiseTrader         – random direction, Poisson arrivals, size from lognormal
3. InformedTrader      – trades on private signal, size scales with conviction
4. MomentumTrader      – trend-following, reacts to recent price moves
5. MeanReversionTrader – fades extreme moves, mean-reverts to fundamental
6. LiquiditySeeker     – large institution, breaks orders into slices (TWAP/VWAP)
7. StopLossTrader      – holds position until stop level, then exits aggressively

All agents share the same interface:
    agent.on_market_update(state) → List[OrderIntent]
    agent.on_fill(fill)
    agent.reset()

OrderIntent is a lightweight struct passed to the simulation engine,
which converts it to actual OrderBook calls. This keeps agents
decoupled from the OrderBook implementation.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

class OrderSide(Enum):
    BID = "bid"
    ASK = "ask"


class OrderType(Enum):
    LIMIT  = "limit"
    MARKET = "market"


@dataclass
class MarketState:
    """
    Minimal market snapshot passed to every agent each tick.
    Mirrors market_maker.MarketState for interface consistency.
    """
    timestamp:       float
    mid_price:       float
    best_bid:        float
    best_ask:        float
    bid_qty:         float
    ask_qty:         float
    volatility:      float
    fundamental:     float = 0.0   # private-value / fundamental price (set by simulation)
    last_trade_sign: int   = 0     # +1 buy, -1 sell, 0 none
    last_trade_size: float = 0.0
    spread:          float = 0.0

    def __post_init__(self):
        if self.spread == 0.0:
            self.spread = self.best_ask - self.best_bid


@dataclass
class OrderIntent:
    """
    Lightweight order request produced by an agent.
    Simulation engine converts this to an actual Order.
    """
    agent_id:   str
    agent_type: str
    side:       OrderSide
    order_type: OrderType
    price:      float        # 0.0 for market orders
    quantity:   float
    timestamp:  float
    tag:        str = ""     # optional label e.g. "stop_loss", "twap_slice"

    @property
    def is_valid(self) -> bool:
        return self.quantity > 0 and (
            self.order_type == OrderType.MARKET or self.price > 0
        )


@dataclass
class FillNotice:
    """Execution confirmation sent back to the agent."""
    agent_id:  str
    side:      OrderSide
    price:     float
    quantity:  float
    timestamp: float
    tag:       str = ""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseTrader(ABC):
    """
    Abstract base for all trader agents.

    Subclasses must implement:
        on_market_update(state) → List[OrderIntent]

    Optionally override:
        on_fill(fill)   – react to own executions
        reset()         – clear internal state
    """

    def __init__(
        self,
        agent_id:   Optional[str] = None,
        rng:        Optional[np.random.Generator] = None,
    ):
        self.agent_id   = agent_id or str(uuid.uuid4())[:8]
        self.rng        = rng or np.random.default_rng()
        self.fills:     List[FillNotice] = []
        self.inventory: float = 0.0
        self.cash:      float = 0.0
        self._active:   bool  = True

    @property
    def agent_type(self) -> str:
        return self.__class__.__name__

    @property
    def is_active(self) -> bool:
        return self._active

    @abstractmethod
    def on_market_update(self, state: MarketState) -> List[OrderIntent]:
        """React to market state. Return list of order intents (may be empty)."""
        ...

    def on_fill(self, fill: FillNotice):
        """Update inventory and cash on execution."""
        self.fills.append(fill)
        if fill.side == OrderSide.BID:
            self.inventory += fill.quantity
            self.cash      -= fill.price * fill.quantity
        else:
            self.inventory -= fill.quantity
            self.cash      += fill.price * fill.quantity

    def reset(self):
        self.fills.clear()
        self.inventory = 0.0
        self.cash      = 0.0
        self._active   = True

    def pnl(self, mark_price: float) -> float:
        return self.cash + self.inventory * mark_price

    def _make_intent(
        self,
        side:       OrderSide,
        order_type: OrderType,
        price:      float,
        quantity:   float,
        timestamp:  float,
        tag:        str = "",
    ) -> OrderIntent:
        return OrderIntent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            side=side,
            order_type=order_type,
            price=price,
            quantity=max(quantity, 1e-8),
            timestamp=timestamp,
            tag=tag,
        )


# ---------------------------------------------------------------------------
# 1. Noise Trader
# ---------------------------------------------------------------------------

class NoiseTrader(BaseTrader):
    """
    Uninformed trader with random direction and Poisson arrivals.

    Behavioral model
    ----------------
    - Each tick arrives with probability p_arrive (Poisson approximation)
    - Direction is iid uniform
    - Size drawn from lognormal distribution
    - Mix of limit and market orders (controlled by limit_ratio)
    - Limit orders placed within price_offset_ticks of best quote

    Parameters
    ----------
    p_arrive         : float – probability of submitting an order per tick
    mu_size          : float – log-mean of order size distribution
    sigma_size       : float – log-std of order size distribution
    limit_ratio      : float – fraction of orders that are limit (vs market)
    price_offset_ticks: int  – how many ticks from best quote for limit orders
    tick_size        : float
    """

    def __init__(
        self,
        p_arrive:           float = 0.3,
        mu_size:            float = 2.0,
        sigma_size:         float = 0.8,
        limit_ratio:        float = 0.6,
        price_offset_ticks: int   = 2,
        tick_size:          float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p_arrive           = p_arrive
        self.mu_size            = mu_size
        self.sigma_size         = sigma_size
        self.limit_ratio        = limit_ratio
        self.price_offset_ticks = price_offset_ticks
        self.tick_size          = tick_size

    def on_market_update(self, state: MarketState) -> List[OrderIntent]:
        if not self._active:
            return []
        if self.rng.random() > self.p_arrive:
            return []

        side = OrderSide.BID if self.rng.random() < 0.5 else OrderSide.ASK
        size = float(np.clip(
            self.rng.lognormal(self.mu_size, self.sigma_size), 1.0, 1e6
        ))
        is_limit = self.rng.random() < self.limit_ratio

        if is_limit:
            offset = self.price_offset_ticks * self.tick_size
            price  = (state.best_bid - offset) if side == OrderSide.BID \
                     else (state.best_ask + offset)
            return [self._make_intent(side, OrderType.LIMIT, price, size, state.timestamp)]
        else:
            return [self._make_intent(side, OrderType.MARKET, 0.0, size, state.timestamp)]


# ---------------------------------------------------------------------------
# 2. Informed Trader
# ---------------------------------------------------------------------------

class InformedTrader(BaseTrader):
    """
    Trader with a private signal about the fundamental value.

    Behavioral model (Glosten-Milgrom)
    ------------------------------------
    - Observes fundamental price v (passed via state.fundamental)
    - Trades when |v - mid| > min_edge  (profitable to trade)
    - Buys if v > mid + min_edge, sells if v < mid - min_edge
    - Size scales with conviction: size = base_size × (|v - mid| / sigma)^conviction
    - Uses market orders to ensure execution (informed traders are impatient)
    - Camouflages by splitting large orders (stealth parameter)

    Parameters
    ----------
    min_edge    : float – minimum profit threshold to trade
    base_size   : float – order size at minimum edge
    conviction  : float – size scaling exponent (higher → more aggressive sizing)
    stealth     : float – probability of splitting into smaller slices [0, 1]
    max_position: float – maximum absolute inventory
    """

    def __init__(
        self,
        min_edge:     float = 0.05,
        base_size:    float = 20.0,
        conviction:   float = 1.5,
        stealth:      float = 0.4,
        max_position: float = 500.0,
        sigma_ref:    float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_edge     = min_edge
        self.base_size    = base_size
        self.conviction   = conviction
        self.stealth      = stealth
        self.max_position = max_position
        self.sigma_ref    = sigma_ref

    def on_market_update(self, state: MarketState) -> List[OrderIntent]:
        if not self._active:
            return []

        v   = state.fundamental if state.fundamental > 0 else state.mid_price
        mid = state.mid_price
        edge = v - mid

        if abs(edge) < self.min_edge:
            return []

        side = OrderSide.BID if edge > 0 else OrderSide.ASK

        # inventory check – don't exceed max position
        if side == OrderSide.BID  and self.inventory >= self.max_position:
            return []
        if side == OrderSide.ASK  and self.inventory <= -self.max_position:
            return []

        # size proportional to conviction
        sigma = max(state.volatility, self.sigma_ref)
        raw_size = self.base_size * (abs(edge) / sigma) ** self.conviction
        size = float(np.clip(raw_size, 1.0, self.max_position / 2))

        # stealth splitting
        if self.rng.random() < self.stealth:
            n_slices = self.rng.integers(2, 5)
            slice_size = size / n_slices
            return [
                self._make_intent(side, OrderType.MARKET, 0.0, slice_size,
                                  state.timestamp, tag=f"informed_slice_{i}")
                for i in range(n_slices)
            ]

        return [self._make_intent(side, OrderType.MARKET, 0.0, size,
                                  state.timestamp, tag="informed")]


# ---------------------------------------------------------------------------
# 3. Momentum Trader
# ---------------------------------------------------------------------------

class MomentumTrader(BaseTrader):
    """
    Trend-following agent: buys after up-moves, sells after down-moves.

    Signal: exponential moving average crossover or return-over-window.

    Parameters
    ----------
    lookback      : int   – number of ticks for momentum signal
    threshold     : float – minimum return to trigger trade
    base_size     : float – order size
    max_position  : float – position limit
    fast_ema      : float – fast EMA decay (α in [0,1])
    slow_ema      : float – slow EMA decay
    use_crossover : bool  – use EMA crossover instead of raw return
    """

    def __init__(
        self,
        lookback:     int   = 10,
        threshold:    float = 0.001,
        base_size:    float = 15.0,
        max_position: float = 300.0,
        fast_ema:     float = 0.3,
        slow_ema:     float = 0.1,
        use_crossover: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lookback      = lookback
        self.threshold     = threshold
        self.base_size     = base_size
        self.max_position  = max_position
        self.fast_ema      = fast_ema
        self.slow_ema      = slow_ema
        self.use_crossover = use_crossover

        self._price_history: List[float] = []
        self._fast:  float = 0.0
        self._slow:  float = 0.0
        self._initialized: bool = False

    def on_market_update(self, state: MarketState) -> List[OrderIntent]:
        if not self._active:
            return []

        mid = state.mid_price
        self._price_history.append(mid)

        # update EMAs
        if not self._initialized:
            self._fast = mid
            self._slow = mid
            self._initialized = True
        else:
            self._fast = self.fast_ema * mid + (1 - self.fast_ema) * self._fast
            self._slow = self.slow_ema * mid + (1 - self.slow_ema) * self._slow

        if len(self._price_history) < self.lookback:
            return []

        # compute signal
        if self.use_crossover:
            signal = (self._fast - self._slow) / (self._slow + 1e-10)
        else:
            past_price = self._price_history[-self.lookback]
            signal = (mid - past_price) / (past_price + 1e-10)

        if abs(signal) < self.threshold:
            return []

        side = OrderSide.BID if signal > 0 else OrderSide.ASK

        if side == OrderSide.BID  and self.inventory >= self.max_position:
            return []
        if side == OrderSide.ASK  and self.inventory <= -self.max_position:
            return []

        # size scales with signal strength
        size = self.base_size * min(abs(signal) / self.threshold, 3.0)

        return [self._make_intent(side, OrderType.MARKET, 0.0, size,
                                  state.timestamp, tag="momentum")]

    def reset(self):
        super().reset()
        self._price_history.clear()
        self._fast = self._slow = 0.0
        self._initialized = False


# ---------------------------------------------------------------------------
# 4. Mean Reversion Trader
# ---------------------------------------------------------------------------

class MeanReversionTrader(BaseTrader):
    """
    Fades extreme price moves, expects reversion to fundamental or moving average.

    Signal: deviation of mid from rolling mean (z-score).

    Parameters
    ----------
    window        : int   – lookback for rolling mean / std
    z_threshold   : float – z-score to trigger trade (e.g. 2.0 = 2σ)
    base_size     : float
    max_position  : float
    use_fundamental: bool – use state.fundamental instead of rolling mean
    """

    def __init__(
        self,
        window:          int   = 20,
        z_threshold:     float = 1.5,
        base_size:       float = 12.0,
        max_position:    float = 300.0,
        use_fundamental: bool  = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window          = window
        self.z_threshold     = z_threshold
        self.base_size       = base_size
        self.max_position    = max_position
        self.use_fundamental = use_fundamental

        self._price_history: List[float] = []

    def on_market_update(self, state: MarketState) -> List[OrderIntent]:
        if not self._active:
            return []

        self._price_history.append(state.mid_price)
        if len(self._price_history) < self.window:
            return []

        window_prices = np.array(self._price_history[-self.window:])

        if self.use_fundamental and state.fundamental > 0:
            mean  = state.fundamental
            sigma = np.std(window_prices) + 1e-10
        else:
            mean  = np.mean(window_prices)
            sigma = np.std(window_prices) + 1e-10

        z = (state.mid_price - mean) / sigma

        if abs(z) < self.z_threshold:
            return []

        # fade the move: sell if overshot up, buy if overshot down
        side = OrderSide.ASK if z > 0 else OrderSide.BID

        if side == OrderSide.BID  and self.inventory >= self.max_position:
            return []
        if side == OrderSide.ASK  and self.inventory <= -self.max_position:
            return []

        size = self.base_size * min(abs(z) / self.z_threshold, 3.0)

        return [self._make_intent(side, OrderType.MARKET, 0.0, size,
                                  state.timestamp, tag="mean_rev")]

    def reset(self):
        super().reset()
        self._price_history.clear()


# ---------------------------------------------------------------------------
# 5. Liquidity Seeker (Institutional TWAP)
# ---------------------------------------------------------------------------

class LiquiditySeeker(BaseTrader):
    """
    Large institutional trader that executes a target quantity over time
    using TWAP (time-weighted average price) slicing.

    Mimics real-world institutional order execution:
    - Breaks large order into equal time slices
    - Randomizes slice timing slightly to avoid detection
    - Uses limit orders near touch to minimize market impact
    - Switches to market if time runs out (urgency escalation)

    Parameters
    ----------
    target_qty      : float – total quantity to execute (signed: + buy, - sell)
    total_ticks     : int   – total ticks to complete execution
    use_limit       : bool  – prefer limit orders (False → always market)
    urgency_ticks   : int   – switch to market orders in last N ticks
    randomize       : float – fraction of slice size to randomize [0, 1]
    """

    def __init__(
        self,
        target_qty:    float = 500.0,
        total_ticks:   int   = 50,
        use_limit:     bool  = True,
        urgency_ticks: int   = 5,
        randomize:     float = 0.2,
        tick_size:     float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_qty    = target_qty
        self.total_ticks   = total_ticks
        self.use_limit     = use_limit
        self.urgency_ticks = urgency_ticks
        self.randomize     = randomize
        self.tick_size     = tick_size

        self._remaining:   float = abs(target_qty)
        self._side:        OrderSide = OrderSide.BID if target_qty > 0 else OrderSide.ASK
        self._ticks_left:  int   = total_ticks
        self._slice_size:  float = abs(target_qty) / max(total_ticks, 1)

    def on_market_update(self, state: MarketState) -> List[OrderIntent]:
        if not self._active or self._remaining <= 0:
            self._active = False
            return []

        self._ticks_left -= 1

        # skip tick randomly (spread execution over time)
        if self._ticks_left > self.urgency_ticks:
            if self.rng.random() > (1.0 / max(self._ticks_left / self.total_ticks, 0.1)):
                return []

        # compute slice size with randomization
        jitter = 1.0 + self.rng.uniform(-self.randomize, self.randomize)
        slice_qty = min(self._slice_size * jitter, self._remaining)

        # urgency: switch to market
        in_urgency = self._ticks_left <= self.urgency_ticks
        use_market = (not self.use_limit) or in_urgency

        if use_market:
            intent = self._make_intent(
                self._side, OrderType.MARKET, 0.0, slice_qty,
                state.timestamp, tag="twap_market"
            )
        else:
            # aggressive limit: 1 tick inside spread
            if self._side == OrderSide.BID:
                price = state.best_ask - self.tick_size
            else:
                price = state.best_bid + self.tick_size
            intent = self._make_intent(
                self._side, OrderType.LIMIT, price, slice_qty,
                state.timestamp, tag="twap_limit"
            )

        self._remaining -= slice_qty
        if self._remaining <= 1e-8:
            self._active = False

        return [intent]

    def on_fill(self, fill: FillNotice):
        super().on_fill(fill)

    def reset(self):
        super().reset()
        self._remaining  = abs(self.target_qty)
        self._ticks_left = self.total_ticks
        self._active     = True


# ---------------------------------------------------------------------------
# 6. Stop-Loss Trader
# ---------------------------------------------------------------------------

class StopLossTrader(BaseTrader):
    """
    Holds an existing position and liquidates when price crosses stop level.

    Important for cascade simulation:
    - Stop-loss triggers create sudden market orders
    - Clustering of stops at common price levels amplifies moves
    - Contributes to liquidation cascade in 06_liquidation_cascade.ipynb

    Parameters
    ----------
    initial_qty   : float – signed initial position (+ long, - short)
    entry_price   : float – average entry price
    stop_price    : float – trigger level
    take_profit   : float – optional take-profit level (0 = no TP)
    exit_type     : str   – "market" | "limit" (limit = at stop price)
    """

    def __init__(
        self,
        initial_qty:  float = 100.0,
        entry_price:  float = 100.0,
        stop_price:   float = 98.0,
        take_profit:  float = 0.0,
        exit_type:    str   = "market",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.initial_qty  = initial_qty
        self.entry_price  = entry_price
        self.stop_price   = stop_price
        self.take_profit  = take_profit
        self.exit_type    = exit_type

        # initialize inventory to reflect pre-existing position
        self.inventory = initial_qty
        self.cash      = -initial_qty * entry_price   # paid to open
        self._exited   = False

    @property
    def side(self) -> str:
        return "long" if self.initial_qty > 0 else "short"

    def on_market_update(self, state: MarketState) -> List[OrderIntent]:
        if not self._active or self._exited:
            return []
        if self.inventory == 0:
            self._exited = True
            return []

        mid = state.mid_price
        exit_side = OrderSide.ASK if self.inventory > 0 else OrderSide.BID
        qty = abs(self.inventory)

        # check stop
        stop_hit = (
            (self.inventory > 0 and mid <= self.stop_price) or
            (self.inventory < 0 and mid >= self.stop_price)
        )

        # check take-profit
        tp_hit = self.take_profit > 0 and (
            (self.inventory > 0 and mid >= self.take_profit) or
            (self.inventory < 0 and mid <= self.take_profit)
        )

        if stop_hit or tp_hit:
            tag = "stop_loss" if stop_hit else "take_profit"
            self._exited = True

            if self.exit_type == "limit":
                price = self.stop_price if stop_hit else self.take_profit
                return [self._make_intent(exit_side, OrderType.LIMIT, price, qty,
                                          state.timestamp, tag=tag)]
            else:
                return [self._make_intent(exit_side, OrderType.MARKET, 0.0, qty,
                                          state.timestamp, tag=tag)]

        return []

    def reset(self):
        super().reset()
        self.inventory = self.initial_qty
        self.cash      = -self.initial_qty * self.entry_price
        self._exited   = False


# ---------------------------------------------------------------------------
# Trader factory
# ---------------------------------------------------------------------------

def build_trader_population(
    n_noise:        int   = 50,
    n_informed:     int   = 5,
    n_momentum:     int   = 10,
    n_mean_rev:     int   = 10,
    noise_p_arrive: float = 0.3,
    informed_edge:  float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> List[BaseTrader]:
    """
    Convenience factory: build a diverse population for simulation.

    Parameters
    ----------
    n_noise     : number of noise traders
    n_informed  : number of informed traders
    n_momentum  : number of momentum traders
    n_mean_rev  : number of mean reversion traders
    rng         : shared RNG (will be split per agent if None)
    """
    rng = rng or np.random.default_rng()
    traders: List[BaseTrader] = []

    # noise traders – heterogeneous arrival rates
    for i in range(n_noise):
        p = float(np.clip(rng.normal(noise_p_arrive, 0.1), 0.05, 0.8))
        traders.append(NoiseTrader(
            p_arrive=p,
            mu_size=rng.uniform(1.5, 2.5),
            limit_ratio=rng.uniform(0.4, 0.8),
            agent_id=f"noise_{i:03d}",
            rng=np.random.default_rng(rng.integers(1e9)),
        ))

    # informed traders – varying conviction
    for i in range(n_informed):
        traders.append(InformedTrader(
            min_edge=float(rng.uniform(informed_edge * 0.5, informed_edge * 1.5)),
            conviction=float(rng.uniform(1.0, 2.5)),
            stealth=float(rng.uniform(0.2, 0.7)),
            agent_id=f"informed_{i:03d}",
            rng=np.random.default_rng(rng.integers(1e9)),
        ))

    # momentum traders
    for i in range(n_momentum):
        traders.append(MomentumTrader(
            lookback=int(rng.integers(5, 20)),
            threshold=float(rng.uniform(0.0005, 0.003)),
            agent_id=f"momentum_{i:03d}",
            rng=np.random.default_rng(rng.integers(1e9)),
        ))

    # mean reversion traders
    for i in range(n_mean_rev):
        traders.append(MeanReversionTrader(
            window=int(rng.integers(10, 30)),
            z_threshold=float(rng.uniform(1.0, 2.5)),
            agent_id=f"meanrev_{i:03d}",
            rng=np.random.default_rng(rng.integers(1e9)),
        ))

    return traders
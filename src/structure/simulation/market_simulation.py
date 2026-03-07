"""
market_simulation.py  –  Market Simulation Engine
===================================================

Integrates all components into a runnable simulation:

1. FundamentalProcess     – GBM, jump-diffusion, Ornstein-Uhlenbeck price paths
2. SimulationClock        – discrete tick management, intraday scheduling
3. MarketSimulation       – main engine: price process → agents → order book → metrics
4. SimulationResult       – structured output with full history arrays
5. ScenarioBuilder        – convenience presets for notebook experiments

Notebooks served
----------------
00_stylized_facts.ipynb       – run long simulation, extract return distribution
01_order_flow.ipynb           – OFI, trade sign, VPIN series
02_liquidity_and_depth.ipynb  – spread, depth, Amihud, Kyle lambda over time
03_price_formation.ipynb      – price discovery, informed vs noise contribution
04_market_maker_inventory.ipynb – MM PnL attribution, inventory path
05_leverage_balance_sheet.ipynb – leverage cycle, margin calls
06_liquidation_cascade.ipynb  – cascade events, fire-sale externality
07_structural_fragility.ipynb – regime detection, systemic risk metrics

Design
------
- Simulation loop is a pure Python for-loop (easy to debug in notebook)
- All randomness seeded via np.random.Generator
- Results are plain numpy arrays + dicts (no dependencies beyond numpy)
- Components injected via constructor (swap any piece for experiments)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# 1. Fundamental Price Processes
# ============================================================================

class FundamentalProcess:
    """
    Generates the latent fundamental (true) price path.

    Supported models
    ----------------
    gbm          : Geometric Brownian Motion  dS = μS dt + σS dW
    jump_diffusion: GBM + Poisson jumps  (Merton 1976)
    ou           : Ornstein-Uhlenbeck mean-reverting process
    ou_gbm       : OU around a GBM drift (regime-switching friendly)

    Parameters
    ----------
    model    : str   – "gbm" | "jump_diffusion" | "ou" | "ou_gbm"
    S0       : float – initial price
    mu       : float – drift (annualized for gbm)
    sigma    : float – volatility (annualized for gbm)
    dt       : float – time step size (fraction of year, e.g. 1/252/390)
    kappa    : float – mean reversion speed (ou)
    theta    : float – long-run mean (ou)
    lambda_j : float – jump arrival rate (jump_diffusion, jumps per year)
    mu_j     : float – mean log jump size
    sigma_j  : float – std of log jump size
    rng      : np.random.Generator
    """

    def __init__(
        self,
        model:    str   = "gbm",
        S0:       float = 100.0,
        mu:       float = 0.0,
        sigma:    float = 0.20,
        dt:       float = 1 / (252 * 390),   # 1 minute in trading year
        kappa:    float = 5.0,
        theta:    float = 100.0,
        lambda_j: float = 5.0,
        mu_j:     float = -0.02,
        sigma_j:  float = 0.03,
        rng:      Optional[np.random.Generator] = None,
    ):
        self.model    = model
        self.S        = S0
        self.S0       = S0
        self.mu       = mu
        self.sigma    = sigma
        self.dt       = dt
        self.kappa    = kappa
        self.theta    = theta
        self.lambda_j = lambda_j
        self.mu_j     = mu_j
        self.sigma_j  = sigma_j
        self.rng      = rng or np.random.default_rng()
        self._path:   List[float] = [S0]

    def step(self) -> float:
        """Advance one dt. Returns new fundamental price."""
        if self.model == "gbm":
            self.S = self._gbm_step(self.S)
        elif self.model == "jump_diffusion":
            self.S = self._jump_step(self.S)
        elif self.model == "ou":
            self.S = self._ou_step(self.S)
        elif self.model == "ou_gbm":
            drift  = self._gbm_step(self.theta)
            self.theta = drift
            self.S = self._ou_step(self.S)
        else:
            raise ValueError(f"Unknown model: {self.model}")

        self.S = max(self.S, 1e-4)
        self._path.append(self.S)
        return self.S

    def simulate(self, n_steps: int) -> np.ndarray:
        """Simulate full path of n_steps. Resets state."""
        self.reset()
        for _ in range(n_steps):
            self.step()
        return np.array(self._path)

    def reset(self):
        self.S = self.S0
        self._path = [self.S0]

    @property
    def path(self) -> np.ndarray:
        return np.array(self._path)

    # ---- private step methods -----------------------------------------------

    def _gbm_step(self, S: float) -> float:
        dW = self.rng.standard_normal() * np.sqrt(self.dt)
        return S * np.exp((self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * dW)

    def _jump_step(self, S: float) -> float:
        S_cont = self._gbm_step(S)
        n_jumps = self.rng.poisson(self.lambda_j * self.dt)
        if n_jumps > 0:
            log_jumps = self.rng.normal(self.mu_j, self.sigma_j, size=n_jumps)
            S_cont *= np.exp(np.sum(log_jumps))
        return S_cont

    def _ou_step(self, S: float) -> float:
        dW = self.rng.standard_normal() * np.sqrt(self.dt)
        mean_rev = self.kappa * (self.theta - S) * self.dt
        diffusion = self.sigma * S * dW
        return S + mean_rev + diffusion


# ============================================================================
# 2. Simulation Clock
# ============================================================================

class SimulationClock:
    """
    Manages discrete simulation time.

    Parameters
    ----------
    n_ticks    : int   – total simulation ticks
    dt         : float – duration of each tick (arbitrary units)
    burn_in    : int   – warm-up ticks excluded from results
    """

    def __init__(self, n_ticks: int = 1000, dt: float = 1.0, burn_in: int = 50):
        self.n_ticks  = n_ticks
        self.dt       = dt
        self.burn_in  = burn_in
        self._tick    = 0

    @property
    def t(self) -> float:
        return self._tick * self.dt

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def is_burn_in(self) -> bool:
        return self._tick < self.burn_in

    @property
    def is_done(self) -> bool:
        return self._tick >= self.n_ticks

    def advance(self):
        self._tick += 1

    def reset(self):
        self._tick = 0

    def progress(self) -> float:
        return self._tick / self.n_ticks


# ============================================================================
# 3. Order Book Adapter
# ============================================================================

class OrderBookAdapter:
    """
    Thin adapter between OrderIntent (from agents) and the aggregated OrderBook.

    Wraps the dict-based OrderBook from state/orderbook.py.
    Converts OrderIntent → add_bid / add_ask / execute_market_buy/sell.

    If no external OrderBook is provided, uses a minimal internal stub
    so simulation can run standalone for testing.
    """

    def __init__(self, orderbook=None, tick_size: float = 0.01):
        self.tick_size = tick_size
        if orderbook is not None:
            self.book = orderbook
        else:
            self.book = self._make_stub()
        self._last_trade_price: float = 0.0
        self._last_trade_sign:  int   = 0
        self._last_trade_size:  float = 0.0

    def _make_stub(self):
        """Minimal stand-alone order book for testing without imports."""
        from collections import defaultdict

        class _Stub:
            def __init__(self):
                self.bids   = defaultdict(float)
                self.asks   = defaultdict(float)
                self.trades = []

            def add_bid(self, price, size):
                self.bids[round(price, 6)] += size

            def add_ask(self, price, size):
                self.asks[round(price, 6)] += size

            def best_bid(self):
                return max(self.bids) if self.bids else None

            def best_ask(self):
                return min(self.asks) if self.asks else None

            def mid_price(self):
                bb, ba = self.best_bid(), self.best_ask()
                return (bb + ba) / 2 if bb and ba else None

            def execute_market_buy(self, size):
                remaining, cost = size, 0.0
                for price in sorted(self.asks):
                    avail  = self.asks[price]
                    traded = min(remaining, avail)
                    cost  += traded * price
                    remaining -= traded
                    self.asks[price] -= traded
                    if self.asks[price] <= 0:
                        del self.asks[price]
                    self.trades.append({"side": "buy", "price": price, "size": traded})
                    if remaining <= 0:
                        break
                return cost / size if size > 0 else 0.0

            def execute_market_sell(self, size):
                remaining, revenue = size, 0.0
                for price in sorted(self.bids, reverse=True):
                    avail  = self.bids[price]
                    traded = min(remaining, avail)
                    revenue += traded * price
                    remaining -= traded
                    self.bids[price] -= traded
                    if self.bids[price] <= 0:
                        del self.bids[price]
                    self.trades.append({"side": "sell", "price": price, "size": traded})
                    if remaining <= 0:
                        break
                return revenue / size if size > 0 else 0.0

            def snapshot(self):
                bb, ba = self.best_bid(), self.best_ask()
                mid = (bb + ba) / 2 if bb and ba else None
                return {"bids": dict(self.bids), "asks": dict(self.asks),
                        "best_bid": bb, "best_ask": ba, "mid_price": mid}

            def total_bid_liquidity(self):
                return sum(self.bids.values())

            def total_ask_liquidity(self):
                return sum(self.asks.values())

            def clear(self):
                self.bids.clear()
                self.asks.clear()

        return _Stub()

    def seed_book(self, mid_price: float, n_levels: int = 10, depth_per_level: float = 100.0):
        """Populate book with symmetric initial depth around mid_price."""
        self.book.clear()
        for i in range(1, n_levels + 1):
            bid_price = round(mid_price - i * self.tick_size, 6)
            ask_price = round(mid_price + i * self.tick_size, 6)
            size = depth_per_level * np.exp(-0.2 * i)   # exponentially declining depth
            self.book.add_bid(bid_price, size)
            self.book.add_ask(ask_price, size)

    def process_intent(self, intent) -> Optional[Dict]:
        """
        Execute an OrderIntent. Returns trade dict or None.
        """
        from collections import namedtuple
        # lazy import avoids hard dependency on traders.py
        BID_STR = "bid"
        side_str = intent.side.value if hasattr(intent.side, "value") else intent.side

        try:
            if hasattr(intent, 'order_type'):
                otype = intent.order_type.value if hasattr(intent.order_type, 'value') \
                        else intent.order_type
            else:
                otype = "market"

            if otype == "limit":
                if side_str == BID_STR:
                    self.book.add_bid(intent.price, intent.quantity)
                else:
                    self.book.add_ask(intent.price, intent.quantity)
                return None   # limit orders rest in book, no immediate fill

            else:  # market order
                if side_str == BID_STR:
                    avg_price = self.book.execute_market_buy(intent.quantity)
                    sign = 1
                else:
                    avg_price = self.book.execute_market_sell(intent.quantity)
                    sign = -1

                self._last_trade_price = avg_price
                self._last_trade_sign  = sign
                self._last_trade_size  = intent.quantity

                return {
                    "agent_id":  intent.agent_id,
                    "agent_type": intent.agent_type,
                    "side":      side_str,
                    "price":     avg_price,
                    "quantity":  intent.quantity,
                    "sign":      sign,
                    "tag":       getattr(intent, "tag", ""),
                }
        except (ValueError, KeyError):
            return None   # insufficient liquidity – silently drop

    def market_state_snapshot(self, t: float, volatility: float, fundamental: float) -> Dict:
        """Build a MarketState-compatible dict from current book."""
        snap = self.book.snapshot()
        bb   = snap.get("best_bid") or 0.0
        ba   = snap.get("best_ask") or 0.0
        mid  = snap.get("mid_price") or (bb + ba) / 2 if bb and ba else fundamental

        bid_qty = sum(v for v in self.book.bids.values()) if hasattr(self.book, 'bids') else 0.0
        ask_qty = sum(v for v in self.book.asks.values()) if hasattr(self.book, 'asks') else 0.0

        return dict(
            timestamp=t,
            mid_price=mid or fundamental,
            best_bid=bb or (mid - self.tick_size),
            best_ask=ba or (mid + self.tick_size),
            bid_qty=bid_qty,
            ask_qty=ask_qty,
            volatility=volatility,
            fundamental=fundamental,
            last_trade_sign=self._last_trade_sign,
            last_trade_size=self._last_trade_size,
            spread=max((ba - bb), self.tick_size) if bb and ba else self.tick_size,
        )


# ============================================================================
# 4. Simulation Result
# ============================================================================

@dataclass
class SimulationResult:
    """
    Structured container for all simulation output.
    All fields are numpy arrays aligned on the tick axis.
    """
    # price & fundamentals
    timestamps:    np.ndarray
    mid_prices:    np.ndarray
    fundamentals:  np.ndarray
    best_bids:     np.ndarray
    best_asks:     np.ndarray
    spreads:       np.ndarray

    # volume & flow
    buy_volumes:   np.ndarray
    sell_volumes:  np.ndarray
    trade_counts:  np.ndarray
    ofi:           np.ndarray   # order flow imbalance

    # depth
    bid_depths:    np.ndarray
    ask_depths:    np.ndarray

    # volatility
    realized_vol:  np.ndarray

    # agent-specific (optional, may be empty)
    mm_inventory:  np.ndarray = field(default_factory=lambda: np.array([]))
    mm_pnl:        np.ndarray = field(default_factory=lambda: np.array([]))
    mm_spread_pnl: np.ndarray = field(default_factory=lambda: np.array([]))

    # cascade events
    cascade_events: List[Dict] = field(default_factory=list)

    # raw trade tape
    trades:        List[Dict] = field(default_factory=list)

    # metadata
    config:        Dict = field(default_factory=dict)

    # ---- derived series -----------------------------------------------

    def returns(self, log: bool = True) -> np.ndarray:
        prices = self.mid_prices
        if log:
            return np.diff(np.log(prices + 1e-10))
        return np.diff(prices) / (prices[:-1] + 1e-10)

    def price_impact_series(self) -> np.ndarray:
        """Amihud illiquidity proxy per tick."""
        rets   = np.abs(self.returns(log=True))
        volume = self.buy_volumes[1:] + self.sell_volumes[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(volume > 0, rets / volume, np.nan)

    def signed_volume(self) -> np.ndarray:
        return self.buy_volumes - self.sell_volumes

    def realized_variance(self, window: int = 20) -> np.ndarray:
        r = self.returns()
        result = np.full(len(r), np.nan)
        for i in range(window - 1, len(r)):
            result[i] = np.var(r[i - window + 1: i + 1])
        return result

    def kyle_lambda_series(self, window: int = 30) -> np.ndarray:
        """Rolling OLS estimate of Kyle's lambda."""
        sv = self.signed_volume()
        dp = np.diff(self.mid_prices)
        n  = min(len(sv), len(dp))
        sv, dp = sv[:n], dp[:n]

        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            x = sv[i - window + 1: i + 1]
            y = dp[i - window + 1: i + 1]
            if np.std(x) > 1e-10:
                result[i] = np.cov(y, x)[0, 1] / np.var(x)
        return result

    def vpin_series(self, bucket_size: float = None, window: int = 50) -> np.ndarray:
        """Volume-synchronized PIN (Easley et al. 2012)."""
        bv = self.buy_volumes
        sv = self.sell_volumes

        if bucket_size is None:
            total_vol = (bv + sv).sum()
            bucket_size = max(total_vol / (len(bv) * 0.5), 1.0)

        vpin_out = np.full(len(bv), np.nan)
        buckets: List[Tuple[float, float]] = []
        buf_buy = buf_sell = buf_vol = 0.0

        for i in range(len(bv)):
            buf_buy  += bv[i]
            buf_sell += sv[i]
            buf_vol  += bv[i] + sv[i]

            while buf_vol >= bucket_size:
                frac = bucket_size / buf_vol
                buckets.append((buf_buy * frac, buf_sell * frac))
                buf_buy  -= buf_buy  * frac
                buf_sell -= buf_sell * frac
                buf_vol  -= bucket_size

            if len(buckets) >= window:
                recent = buckets[-window:]
                vpin_out[i] = sum(abs(b - s) for b, s in recent) / (window * bucket_size)

        return vpin_out

    def summary_stats(self) -> Dict:
        r = self.returns()
        valid_r = r[np.isfinite(r)]
        prices  = self.mid_prices

        return {
            "n_ticks":         len(self.timestamps),
            "price_start":     float(prices[0]),
            "price_end":       float(prices[-1]),
            "total_return":    float((prices[-1] - prices[0]) / prices[0]),
            "annualized_vol":  float(np.std(valid_r) * np.sqrt(252 * 390)) if len(valid_r) > 1 else np.nan,
            "mean_spread":     float(np.nanmean(self.spreads)),
            "mean_bid_depth":  float(np.nanmean(self.bid_depths)),
            "mean_ask_depth":  float(np.nanmean(self.ask_depths)),
            "total_buy_vol":   float(self.buy_volumes.sum()),
            "total_sell_vol":  float(self.sell_volumes.sum()),
            "n_trades":        int(self.trade_counts.sum()),
            "n_cascade_events": len(self.cascade_events),
            "kurtosis":        float(self._kurtosis(valid_r)),
            "skewness":        float(self._skewness(valid_r)),
        }

    @staticmethod
    def _kurtosis(r: np.ndarray) -> float:
        if len(r) < 4:
            return np.nan
        m2 = np.mean((r - r.mean())**2)
        m4 = np.mean((r - r.mean())**4)
        return m4 / (m2**2) - 3 if m2 > 0 else np.nan

    @staticmethod
    def _skewness(r: np.ndarray) -> float:
        if len(r) < 3:
            return np.nan
        m2 = np.mean((r - r.mean())**2)
        m3 = np.mean((r - r.mean())**3)
        return m3 / (m2**1.5) if m2 > 0 else np.nan


# ============================================================================
# 5. Volatility Estimator (rolling, for agent state)
# ============================================================================

class RollingVolatility:
    """Welford online algorithm for rolling realized volatility."""

    def __init__(self, window: int = 20):
        self.window  = window
        self._prices: List[float] = []

    def update(self, price: float) -> float:
        self._prices.append(price)
        if len(self._prices) < 3:
            return 0.001   # default seed vol
        window_prices = self._prices[-self.window - 1:]
        log_rets = np.diff(np.log(np.array(window_prices) + 1e-10))
        return float(np.std(log_rets)) if len(log_rets) > 0 else 0.001


# ============================================================================
# 6. Market Simulation Engine
# ============================================================================

class MarketSimulation:
    """
    Main simulation engine.

    Wires together:
    - FundamentalProcess  → latent true price
    - Traders             → generate order intents
    - MarketMaker         → provide liquidity
    - OrderBookAdapter    → execute orders, maintain book state
    - SimulationClock     → discrete time management

    Loop per tick
    -------------
    1. Fundamental price steps
    2. Replenish book depth around new fundamental
    3. Build MarketState snapshot
    4. Query each trader → collect OrderIntents
    5. Shuffle + execute intents → fills, trade tape
    6. Update MarketMaker (quotes)
    7. Record metrics

    Parameters
    ----------
    fundamental    : FundamentalProcess
    traders        : list of BaseTrader agents
    market_maker   : MarketMaker instance (optional)
    orderbook      : aggregated OrderBook instance (optional, uses stub if None)
    clock          : SimulationClock
    tick_size      : float
    depth_refresh  : int  – replenish book every N ticks
    depth_levels   : int  – number of price levels to seed
    depth_qty      : float – quantity per level
    shuffle_orders : bool  – randomize execution order within tick
    verbose        : bool  – print progress
    """

    def __init__(
        self,
        fundamental:   FundamentalProcess,
        traders:       List = None,
        market_maker=  None,
        orderbook=     None,
        clock:         Optional[SimulationClock] = None,
        tick_size:     float = 0.01,
        depth_refresh: int   = 5,
        depth_levels:  int   = 10,
        depth_qty:     float = 100.0,
        shuffle_orders: bool = True,
        verbose:       bool  = False,
        rng:           Optional[np.random.Generator] = None,
    ):
        self.fundamental    = fundamental
        self.traders        = traders or []
        self.market_maker   = market_maker
        self.clock          = clock or SimulationClock()
        self.tick_size      = tick_size
        self.depth_refresh  = depth_refresh
        self.depth_levels   = depth_levels
        self.depth_qty      = depth_qty
        self.shuffle_orders = shuffle_orders
        self.verbose        = verbose
        self.rng            = rng or np.random.default_rng()

        self.adapter = OrderBookAdapter(orderbook, tick_size=tick_size)
        self.vol_est = RollingVolatility(window=20)

        # result buffers
        self._timestamps:   List[float] = []
        self._mid_prices:   List[float] = []
        self._fundamentals: List[float] = []
        self._best_bids:    List[float] = []
        self._best_asks:    List[float] = []
        self._spreads:      List[float] = []
        self._buy_vols:     List[float] = []
        self._sell_vols:    List[float] = []
        self._trade_counts: List[int]   = []
        self._bid_depths:   List[float] = []
        self._ask_depths:   List[float] = []
        self._ofi:          List[float] = []
        self._realized_vol: List[float] = []
        self._trades:       List[Dict]  = []
        self._cascade_events: List[Dict] = []

        # OFI state
        self._prev_bid_price: Optional[float] = None
        self._prev_bid_qty:   Optional[float] = None
        self._prev_ask_price: Optional[float] = None
        self._prev_ask_qty:   Optional[float] = None

    # -------------------------------------------------------------------------
    # Main run
    # -------------------------------------------------------------------------

    def run(self) -> SimulationResult:
        """Execute full simulation. Returns SimulationResult."""
        self.clock.reset()
        self.fundamental.reset()

        # seed book at initial price
        S0 = self.fundamental.S0
        self.adapter.seed_book(S0, self.depth_levels, self.depth_qty)

        if self.verbose:
            print(f"Starting simulation: {self.clock.n_ticks} ticks, "
                  f"{len(self.traders)} traders, MM={'yes' if self.market_maker else 'no'}")

        while not self.clock.is_done:
            self._tick()
            self.clock.advance()

            if self.verbose and self.clock.tick % 100 == 0:
                pct = self.clock.progress() * 100
                mid = self._mid_prices[-1] if self._mid_prices else S0
                print(f"  {pct:5.1f}%  tick={self.clock.tick:5d}  mid={mid:.4f}")

        return self._build_result()

    # -------------------------------------------------------------------------
    # Single tick
    # -------------------------------------------------------------------------

    def _tick(self):
        t   = self.clock.t
        tok = self.clock.tick

        # 1. Step fundamental price
        fundamental = self.fundamental.step()

        # 2. Periodically replenish passive depth
        if tok % self.depth_refresh == 0:
            self.adapter.seed_book(fundamental, self.depth_levels, self.depth_qty)

        # 3. Current volatility estimate
        vol = self.vol_est.update(fundamental)

        # 4. Build market state snapshot
        state_dict = self.adapter.market_state_snapshot(t, vol, fundamental)
        state = self._dict_to_state(state_dict)

        # 5. Collect order intents from all traders
        intents = []
        for trader in self.traders:
            if trader.is_active:
                try:
                    new_intents = trader.on_market_update(state)
                    intents.extend(new_intents)
                except Exception:
                    pass  # isolate agent failures

        # 6. Market maker quotes (add as limit orders)
        if self.market_maker is not None:
            try:
                quote = self.market_maker.on_market_update(state)
                if quote and quote.is_valid():
                    if quote.bid_size > 0:
                        self.adapter.book.add_bid(quote.bid_price, quote.bid_size)
                    if quote.ask_size > 0:
                        self.adapter.book.add_ask(quote.ask_price, quote.ask_size)
                    for bp, bs in quote.extra_bids:
                        self.adapter.book.add_bid(bp, bs)
                    for ap, as_ in quote.extra_asks:
                        self.adapter.book.add_ask(ap, as_)
            except Exception:
                pass

        # 7. Execute intents
        if self.shuffle_orders:
            self.rng.shuffle(intents)

        tick_buy_vol  = 0.0
        tick_sell_vol = 0.0
        tick_trades   = 0

        for intent in intents:
            if not intent.is_valid:
                continue
            trade = self.adapter.process_intent(intent)
            if trade is not None:
                self._trades.append({**trade, "t": t})
                if trade["sign"] > 0:
                    tick_buy_vol  += trade["quantity"]
                else:
                    tick_sell_vol += trade["quantity"]
                tick_trades += 1

                # notify trader of fill
                self._notify_fill(intent, trade)

                # notify market maker of fill
                if self.market_maker is not None:
                    self._notify_mm_fill(trade, state)

        # 8. Record metrics (skip burn-in)
        if not self.clock.is_burn_in:
            snap      = self.adapter.market_state_snapshot(t, vol, fundamental)
            mid       = snap["mid_price"]
            best_bid  = snap["best_bid"]
            best_ask  = snap["best_ask"]
            spread    = snap["spread"]
            bid_depth = self.adapter.book.total_bid_liquidity() \
                        if hasattr(self.adapter.book, 'total_bid_liquidity') else 0.0
            ask_depth = self.adapter.book.total_ask_liquidity() \
                        if hasattr(self.adapter.book, 'total_ask_liquidity') else 0.0

            ofi = self._compute_ofi(best_bid, bid_depth, best_ask, ask_depth)

            self._timestamps.append(t)
            self._mid_prices.append(mid)
            self._fundamentals.append(fundamental)
            self._best_bids.append(best_bid)
            self._best_asks.append(best_ask)
            self._spreads.append(spread)
            self._buy_vols.append(tick_buy_vol)
            self._sell_vols.append(tick_sell_vol)
            self._trade_counts.append(tick_trades)
            self._bid_depths.append(bid_depth)
            self._ask_depths.append(ask_depth)
            self._ofi.append(ofi if ofi is not None else 0.0)
            self._realized_vol.append(vol)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _dict_to_state(self, d: Dict):
        """Convert snapshot dict to a duck-typed state object."""
        class _State:
            pass
        s = _State()
        for k, v in d.items():
            setattr(s, k, v)
        return s

    def _compute_ofi(self, bid_p, bid_q, ask_p, ask_q) -> Optional[float]:
        """Cont-Kukanov-Stoikov OFI from best queue changes."""
        if self._prev_bid_price is None:
            self._prev_bid_price = bid_p
            self._prev_bid_qty   = bid_q
            self._prev_ask_price = ask_p
            self._prev_ask_qty   = ask_q
            return None

        e_bid = 0.0
        if bid_p >= self._prev_bid_price:
            e_bid += bid_q
        if bid_p <= self._prev_bid_price:
            e_bid -= self._prev_bid_qty

        e_ask = 0.0
        if ask_p >= self._prev_ask_price:
            e_ask -= ask_q
        if ask_p <= self._prev_ask_price:
            e_ask += self._prev_ask_qty

        self._prev_bid_price = bid_p
        self._prev_bid_qty   = bid_q
        self._prev_ask_price = ask_p
        self._prev_ask_qty   = ask_q

        return e_bid - e_ask

    def _notify_fill(self, intent, trade: Dict):
        """Send fill notice back to the originating trader."""
        for trader in self.traders:
            if trader.agent_id == intent.agent_id:
                try:
                    from src.structure import FillNotice, OrderSide
                    fill = FillNotice(
                        agent_id=intent.agent_id,
                        side=intent.side,
                        price=trade["price"],
                        quantity=trade["quantity"],
                        timestamp=trade.get("t", 0.0),
                        tag=trade.get("tag", ""),
                    )
                    trader.on_fill(fill)
                except Exception:
                    pass
                break

    def _notify_mm_fill(self, trade: Dict, state):
        """Notify market maker of a fill against its quotes."""
        if self.market_maker is None:
            return
        try:
            from src.structure import Fill
            from src.structure import OrderSide
            side_str = trade.get("side", "bid")
            fill = Fill(
                timestamp=trade.get("t", 0.0),
                side=side_str,
                price=trade["price"],
                size=trade["quantity"],
            )
            self.market_maker.on_fill(fill, state)
        except Exception:
            pass

    def _build_result(self) -> SimulationResult:
        """Package all buffers into SimulationResult."""
        to_arr = np.array

        mm_inv = mm_pnl = mm_sp = np.array([])
        if self.market_maker is not None:
            try:
                arrs   = self.market_maker.snapshot_arrays()
                mm_inv = arrs.get("inventory", np.array([]))
                mm_pnl = arrs.get("total_pnl",  np.array([]))
                mm_sp  = arrs.get("spread_pnl",  np.array([]))
            except Exception:
                pass

        return SimulationResult(
            timestamps=to_arr(self._timestamps),
            mid_prices=to_arr(self._mid_prices),
            fundamentals=to_arr(self._fundamentals),
            best_bids=to_arr(self._best_bids),
            best_asks=to_arr(self._best_asks),
            spreads=to_arr(self._spreads),
            buy_volumes=to_arr(self._buy_vols),
            sell_volumes=to_arr(self._sell_vols),
            trade_counts=to_arr(self._trade_counts),
            ofi=to_arr(self._ofi),
            bid_depths=to_arr(self._bid_depths),
            ask_depths=to_arr(self._ask_depths),
            realized_vol=to_arr(self._realized_vol),
            mm_inventory=mm_inv,
            mm_pnl=mm_pnl,
            mm_spread_pnl=mm_sp,
            cascade_events=self._cascade_events,
            trades=self._trades,
            config=self._build_config(),
        )

    def _build_config(self) -> Dict:
        return {
            "n_ticks":       self.clock.n_ticks,
            "burn_in":       self.clock.burn_in,
            "dt":            self.clock.dt,
            "tick_size":     self.tick_size,
            "n_traders":     len(self.traders),
            "has_mm":        self.market_maker is not None,
            "fundamental_model": self.fundamental.model,
            "S0":            self.fundamental.S0,
            "sigma":         self.fundamental.sigma,
        }


# ============================================================================
# 7. Scenario Builder
# ============================================================================

class ScenarioBuilder:
    """
    Convenience factory for common simulation setups used in notebooks.

    Usage
    -----
    sim = ScenarioBuilder.calm_market(n_ticks=2000)
    result = sim.run()

    sim = ScenarioBuilder.flash_crash(n_ticks=1000)
    result = sim.run()
    """

    @staticmethod
    def calm_market(
        n_ticks:      int   = 2000,
        n_noise:      int   = 40,
        n_informed:   int   = 5,
        S0:           float = 100.0,
        sigma:        float = 0.15,
        with_mm:      bool  = True,
        seed:         int   = 42,
    ) -> MarketSimulation:
        """Low-volatility baseline with market maker providing liquidity."""
        from src.structure import build_trader_population
        from src.structure import MarketMaker, AvellanedaStoikovStrategy

        rng = np.random.default_rng(seed)

        fundamental = FundamentalProcess(
            model="gbm", S0=S0, mu=0.0, sigma=sigma,
            dt=1 / (252 * 390), rng=np.random.default_rng(seed)
        )
        traders = build_trader_population(
            n_noise=n_noise, n_informed=n_informed,
            n_momentum=5, n_mean_rev=5, rng=rng
        )
        mm = MarketMaker(
            strategy=AvellanedaStoikovStrategy(gamma=0.1, kappa=1.5),
            max_inventory=500.0,
        ) if with_mm else None

        return MarketSimulation(
            fundamental=fundamental,
            traders=traders,
            market_maker=mm,
            clock=SimulationClock(n_ticks=n_ticks, burn_in=50),
            tick_size=0.01,
            depth_qty=150.0,
            rng=rng,
        )

    @staticmethod
    def informed_flow(
        n_ticks:    int   = 1500,
        n_informed: int   = 15,
        min_edge:   float = 0.03,
        S0:         float = 100.0,
        seed:       int   = 7,
    ) -> MarketSimulation:
        """High informed trader fraction – tests adverse selection and price discovery."""
        from src.structure import build_trader_population, InformedTrader
        from src.structure import MarketMaker, InventorySkewStrategy

        rng = np.random.default_rng(seed)

        fundamental = FundamentalProcess(
            model="ou_gbm", S0=S0, mu=0.05, sigma=0.25,
            kappa=3.0, theta=S0, dt=1 / (252 * 390),
            rng=np.random.default_rng(seed)
        )
        traders = build_trader_population(
            n_noise=20, n_informed=n_informed,
            n_momentum=3, n_mean_rev=3, rng=rng,
            informed_edge=min_edge,
        )
        mm = MarketMaker(
            strategy=InventorySkewStrategy(base_half_spread=0.03, skew_factor=0.05),
            max_inventory=300.0,
        )

        return MarketSimulation(
            fundamental=fundamental,
            traders=traders,
            market_maker=mm,
            clock=SimulationClock(n_ticks=n_ticks, burn_in=50),
            tick_size=0.01,
            depth_qty=80.0,
            rng=rng,
        )

    @staticmethod
    def flash_crash(
        n_ticks:       int   = 1000,
        crash_tick:    int   = 500,
        crash_size:    float = 2000.0,
        S0:            float = 100.0,
        seed:          int   = 13,
    ) -> MarketSimulation:
        """
        Large institutional sell order hits illiquid book mid-simulation,
        triggering stop-losses and potential cascade.
        """
        from traders import (build_trader_population, LiquiditySeeker,
                             StopLossTrader)
        from market_maker import MarketMaker, VolatilityAdaptiveStrategy

        rng = np.random.default_rng(seed)

        fundamental = FundamentalProcess(
            model="jump_diffusion", S0=S0, mu=0.0, sigma=0.20,
            lambda_j=2.0, mu_j=-0.015, sigma_j=0.02,
            dt=1 / (252 * 390), rng=np.random.default_rng(seed)
        )

        # background traders
        traders = build_trader_population(
            n_noise=30, n_informed=3, n_momentum=8, n_mean_rev=5, rng=rng
        )

        # institutional seller arrives at crash_tick
        seller = LiquiditySeeker(
            target_qty=-crash_size,
            total_ticks=n_ticks - crash_tick,
            use_limit=False,
            agent_id="crash_seller",
            rng=np.random.default_rng(seed + 1),
        )

        # stop-loss holders clustered around S0 * 0.97
        for i in range(10):
            stop_px  = S0 * (0.95 + rng.uniform(0, 0.04))
            entry_px = S0 * (1.0  + rng.uniform(0, 0.02))
            qty      = float(rng.uniform(50, 200))
            traders.append(StopLossTrader(
                initial_qty=qty,
                entry_price=entry_px,
                stop_price=stop_px,
                agent_id=f"stop_{i:02d}",
                rng=np.random.default_rng(seed + 100 + i),
            ))

        # The seller only becomes active after crash_tick
        class _DelayedSeller:
            """Wrapper that activates LiquiditySeeker after delay."""
            def __init__(self, inner, start_tick: int, clock: SimulationClock):
                self._inner = inner
                self._start = start_tick
                self._clock = clock
                self.agent_id   = inner.agent_id
                self.agent_type = inner.agent_type

            @property
            def is_active(self):
                return self._clock.tick >= self._start and self._inner.is_active

            def on_market_update(self, state):
                if self._clock.tick < self._start:
                    return []
                return self._inner.on_market_update(state)

            def on_fill(self, fill):
                self._inner.on_fill(fill)

        clock = SimulationClock(n_ticks=n_ticks, burn_in=50)
        traders.append(_DelayedSeller(seller, crash_tick, clock))

        mm = MarketMaker(
            strategy=VolatilityAdaptiveStrategy(
                base_half_spread=0.05,
                vol_ceiling=0.04,
            ),
            max_inventory=200.0,
            max_drawdown=3000.0,
        )

        return MarketSimulation(
            fundamental=fundamental,
            traders=traders,
            market_maker=mm,
            clock=clock,
            tick_size=0.01,
            depth_qty=60.0,
            rng=rng,
            verbose=False,
        )

    @staticmethod
    def leverage_cycle(
        n_ticks:      int   = 2000,
        n_leveraged:  int   = 20,
        S0:           float = 100.0,
        sigma:        float = 0.25,
        seed:         int   = 99,
    ) -> MarketSimulation:
        """
        Population of leveraged momentum traders.
        Demonstrates procyclical leverage and potential cascade.
        """
        from src.structure import build_trader_population, MomentumTrader
        from src.structure import MarketMaker, InventorySkewStrategy

        rng = np.random.default_rng(seed)

        fundamental = FundamentalProcess(
            model="gbm", S0=S0, mu=0.02, sigma=sigma,
            dt=1 / (252 * 390), rng=np.random.default_rng(seed)
        )
        # heavy momentum trader population
        traders = build_trader_population(
            n_noise=20, n_informed=3,
            n_momentum=n_leveraged, n_mean_rev=2, rng=rng,
        )
        mm = MarketMaker(
            strategy=InventorySkewStrategy(),
            max_inventory=400.0,
        )

        return MarketSimulation(
            fundamental=fundamental,
            traders=traders,
            market_maker=mm,
            clock=SimulationClock(n_ticks=n_ticks, burn_in=50),
            tick_size=0.01,
            depth_qty=100.0,
            rng=rng,
        )
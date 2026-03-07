"""
leverage.py  –  Leverage & Margin Mechanism
============================================

Covers the following concepts for 05_leverage_balance_sheet.ipynb:

1. Position & margin accounting   – initial, maintenance, variation margin
2. Leverage ratio & constraints   – gross/net leverage, regulatory limits
3. Mark-to-market PnL             – unrealized, realized, funding cost
4. Margin call mechanics          – trigger, cure period, forced reduction
5. Balance sheet dynamics         – equity, assets, liabilities, NAV
6. Value-at-Risk margin           – VaR-based initial margin (SPAN proxy)
7. Funding liquidity risk         – haircut, repo margin, spiral model
8. Leverage cycle                 – Adrian-Shin procyclical leverage model

Design principle
----------------
- No direct import of OrderBook or liquidity.py
- Liquidity cost injected via callback / parameter (execution_cost_fn)
- PositionBook aggregates multiple Position objects
- MarginAccount wraps PositionBook with margin logic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Single instrument position."""
    symbol:        str
    quantity:      float          # signed: + long, - short
    avg_entry:     float          # average entry price
    mark_price:    float          # current mark-to-market price
    margin_rate:   float = 0.10   # initial margin fraction (10% = 10x max leverage)

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.mark_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.mark_price - self.avg_entry)

    @property
    def side(self) -> str:
        return "long" if self.quantity >= 0 else "short"

    @property
    def initial_margin_required(self) -> float:
        return self.notional * self.margin_rate

    def update_mark(self, new_price: float) -> float:
        """Update mark price. Returns PnL delta."""
        delta = self.quantity * (new_price - self.mark_price)
        self.mark_price = new_price
        return delta

    def reduce(self, qty: float) -> Tuple[float, float]:
        """
        Reduce position by qty (absolute). Returns (realized_pnl, margin_released).
        qty must be <= abs(self.quantity).
        """
        qty = min(abs(qty), abs(self.quantity))
        sign = 1 if self.quantity > 0 else -1
        realized = sign * qty * (self.mark_price - self.avg_entry)
        margin_released = qty * self.mark_price * self.margin_rate
        self.quantity -= sign * qty
        return realized, margin_released


@dataclass
class Trade:
    """Executed trade record."""
    symbol:    str
    quantity:  float       # signed
    price:     float
    timestamp: float
    margin_used: float = 0.0
    fee:         float = 0.0


# ---------------------------------------------------------------------------
# 1. Position Book
# ---------------------------------------------------------------------------

class PositionBook:
    """
    Aggregates positions across symbols.
    Tracks realized PnL, margin usage, and funding costs.
    """

    def __init__(self, funding_rate: float = 0.0001):
        """
        Parameters
        ----------
        funding_rate : float – periodic funding cost per unit of notional
                               (e.g. 0.0001 = 1 bps per period)
        """
        self.funding_rate = funding_rate
        self.positions:    Dict[str, Position] = {}
        self.realized_pnl: float = 0.0
        self.funding_paid: float = 0.0
        self.trade_log:    List[Trade] = []

    # ---- position mutation --------------------------------------------------

    def open_or_add(
        self,
        symbol:      str,
        quantity:    float,   # signed
        price:       float,
        margin_rate: float = 0.10,
        fee:         float = 0.0,
        timestamp:   float = 0.0,
    ) -> Trade:
        """Open new position or add to existing (FIFO avg cost)."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            old_qty = pos.quantity
            new_qty = old_qty + quantity
            if new_qty == 0:
                # full close
                self.realized_pnl += pos.unrealized_pnl
                del self.positions[symbol]
            elif (old_qty > 0) == (new_qty > 0):
                # same direction: update avg entry
                pos.avg_entry = (
                    (abs(old_qty) * pos.avg_entry + abs(quantity) * price)
                    / abs(new_qty)
                )
                pos.quantity = new_qty
            else:
                # flip: close existing, open new
                self.realized_pnl += pos.unrealized_pnl
                remaining = abs(new_qty)
                sign_new  = 1 if new_qty > 0 else -1
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=sign_new * remaining,
                    avg_entry=price,
                    mark_price=price,
                    margin_rate=margin_rate,
                )
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_entry=price,
                mark_price=price,
                margin_rate=margin_rate,
            )

        trade = Trade(
            symbol=symbol,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            margin_used=abs(quantity) * price * margin_rate,
            fee=fee,
        )
        self.trade_log.append(trade)
        self.realized_pnl -= fee
        return trade

    def mark_to_market(self, prices: Dict[str, float]) -> float:
        """Update all mark prices. Returns total unrealized PnL delta."""
        total_delta = 0.0
        for symbol, price in prices.items():
            if symbol in self.positions:
                total_delta += self.positions[symbol].update_mark(price)
        return total_delta

    def charge_funding(self) -> float:
        """Charge funding on all open notional. Returns total funding paid."""
        cost = sum(pos.notional * self.funding_rate for pos in self.positions.values())
        self.funding_paid += cost
        self.realized_pnl -= cost
        return cost

    # ---- aggregates ---------------------------------------------------------

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_notional(self) -> float:
        return sum(p.notional for p in self.positions.values())

    @property
    def gross_notional(self) -> float:
        return self.total_notional

    @property
    def net_notional(self) -> float:
        return sum(p.quantity * p.mark_price for p in self.positions.values())

    @property
    def total_initial_margin(self) -> float:
        return sum(p.initial_margin_required for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.total_unrealized_pnl


# ---------------------------------------------------------------------------
# 2. Balance Sheet
# ---------------------------------------------------------------------------

@dataclass
class BalanceSheet:
    """
    Stylized trader / fund balance sheet.

        Assets     = Cash + Mark-to-Market Position Value
        Liabilities= Borrowed Capital (repo / margin loan)
        Equity     = Assets - Liabilities  (NAV)
        Leverage   = Assets / Equity

    Parameters
    ----------
    equity        : float – initial equity (own capital)
    borrowed      : float – initial borrowed capital
    haircut       : float – repo haircut (fraction of collateral not lendable)
    """

    equity:    float
    borrowed:  float = 0.0
    haircut:   float = 0.05    # 5% haircut → max leverage = 1/(haircut) = 20x

    # internal state
    _cash:          float = field(init=False)
    _position_value: float = field(init=False, default=0.0)
    _history:       List[Dict] = field(init=False, default_factory=list)

    def __post_init__(self):
        self._cash = self.equity + self.borrowed

    # ---- core properties ----------------------------------------------------

    @property
    def assets(self) -> float:
        return self._cash + self._position_value

    @property
    def liabilities(self) -> float:
        return self.borrowed

    @property
    def nav(self) -> float:
        return self.assets - self.liabilities

    @property
    def leverage_ratio(self) -> float:
        """Gross leverage = Assets / Equity."""
        return self.assets / self.nav if self.nav > 0 else np.inf

    @property
    def net_leverage(self) -> float:
        """Net leverage = Net Position / Equity."""
        return self._position_value / self.nav if self.nav > 0 else np.inf

    @property
    def max_borrowable(self) -> float:
        """Maximum additional borrowing given haircut constraint."""
        collateral = self._position_value
        max_loan   = collateral * (1.0 - self.haircut)
        return max(0.0, max_loan - self.borrowed)

    # ---- mutation -----------------------------------------------------------

    def deploy_capital(self, amount: float):
        """Move cash into position (buy assets)."""
        self._cash -= amount
        self._position_value += amount

    def unwind_position(self, amount: float, pnl: float = 0.0):
        """Reduce position, receive cash + PnL."""
        amount = min(amount, self._position_value)
        self._position_value -= amount
        self._cash += amount + pnl

    def mark_position(self, new_value: float):
        """Update position value (mark-to-market). Equity absorbs gain/loss."""
        delta = new_value - self._position_value
        self._position_value = new_value
        self.equity += delta

    def repay_debt(self, amount: float):
        amount = min(amount, self.borrowed)
        self._cash -= amount
        self.borrowed -= amount

    def borrow(self, amount: float):
        self._cash += amount
        self.borrowed += amount

    def snapshot(self, t: float = 0.0) -> Dict:
        snap = {
            "t":              t,
            "assets":         self.assets,
            "liabilities":    self.liabilities,
            "equity":         self.nav,
            "leverage":       self.leverage_ratio,
            "position_value": self._position_value,
            "cash":           self._cash,
        }
        self._history.append(snap)
        return snap

    def history_array(self) -> Dict[str, np.ndarray]:
        if not self._history:
            return {}
        keys = self._history[0].keys()
        return {k: np.array([h[k] for h in self._history]) for k in keys}


# ---------------------------------------------------------------------------
# 3. Margin Account
# ---------------------------------------------------------------------------

class MarginAccount:
    """
    Full margin account: wraps PositionBook with margin call logic.

    Margin call triggers when:
        equity_ratio = (equity + unrealized_pnl) / gross_notional < maintenance_rate

    Cure: deposit more cash OR reduce positions until above initial_rate.

    Parameters
    ----------
    initial_equity      : float – starting cash / equity
    initial_margin_rate : float – fraction of notional required to open (e.g. 0.10)
    maintenance_rate    : float – minimum equity ratio before margin call (e.g. 0.05)
    """

    def __init__(
        self,
        initial_equity:      float = 100_000.0,
        initial_margin_rate: float = 0.10,
        maintenance_rate:    float = 0.05,
    ):
        self.initial_margin_rate = initial_margin_rate
        self.maintenance_rate    = maintenance_rate

        self.cash:     float = initial_equity
        self.book:     PositionBook = PositionBook()
        self.margin_calls: List[Dict] = []
        self._t:       float = 0.0

    # ---- core metrics -------------------------------------------------------

    @property
    def equity(self) -> float:
        return self.cash + self.book.total_unrealized_pnl + self.book.realized_pnl

    @property
    def gross_notional(self) -> float:
        return self.book.gross_notional

    @property
    def equity_ratio(self) -> float:
        if self.gross_notional == 0:
            return np.inf
        return self.equity / self.gross_notional

    @property
    def leverage(self) -> float:
        if self.equity <= 0:
            return np.inf
        return self.gross_notional / self.equity

    @property
    def free_margin(self) -> float:
        """Cash not tied up as margin."""
        return self.cash - self.book.total_initial_margin

    @property
    def margin_utilization(self) -> float:
        """Fraction of equity used as margin [0, 1+]."""
        if self.equity <= 0:
            return np.inf
        return self.book.total_initial_margin / self.equity

    def available_to_trade(self) -> float:
        """Maximum additional notional tradeable at current margin rate."""
        if self.initial_margin_rate <= 0:
            return np.inf
        return max(0.0, self.free_margin / self.initial_margin_rate)

    # ---- trade & MTM --------------------------------------------------------

    def trade(
        self,
        symbol:   str,
        quantity: float,
        price:    float,
        fee:      float = 0.0,
    ) -> bool:
        """
        Submit a trade. Returns True if accepted, False if margin insufficient.
        """
        margin_needed = abs(quantity) * price * self.initial_margin_rate
        if margin_needed > self.free_margin + 1e-6:
            return False

        self.cash -= fee
        self.book.open_or_add(
            symbol, quantity, price,
            margin_rate=self.initial_margin_rate,
            fee=fee,
            timestamp=self._t,
        )
        return True

    def mark_to_market(self, prices: Dict[str, float], t: float = 0.0):
        """Update marks and charge funding. Returns margin call dict if triggered."""
        self._t = t
        self.book.mark_to_market(prices)
        self.book.charge_funding()
        return self.check_margin_call(t)

    # ---- margin call --------------------------------------------------------

    def check_margin_call(self, t: float = 0.0) -> Optional[Dict]:
        """
        Check if margin call is triggered.
        Returns margin call dict if triggered, else None.
        """
        if self.gross_notional == 0:
            return None

        if self.equity_ratio < self.maintenance_rate:
            deficit = (
                self.initial_margin_rate * self.gross_notional - self.equity
            )
            call = {
                "t":              t,
                "equity":         self.equity,
                "equity_ratio":   self.equity_ratio,
                "gross_notional": self.gross_notional,
                "deficit":        deficit,
                "leverage":       self.leverage,
            }
            self.margin_calls.append(call)
            return call
        return None

    def deposit(self, amount: float):
        """Cure margin call by depositing cash."""
        self.cash += amount

    def force_reduce(
        self,
        target_equity_ratio: Optional[float] = None,
        execution_cost_fn: Optional[Callable[[str, float], float]] = None,
    ) -> float:
        """
        Force-reduce positions to restore equity ratio to initial_margin_rate.
        execution_cost_fn(symbol, qty) → slippage cost (positive = cost).
        Returns total notional liquidated.
        """
        target = target_equity_ratio or self.initial_margin_rate
        total_liquidated = 0.0

        for symbol, pos in list(self.book.positions.items()):
            if self.equity_ratio >= target:
                break

            # Compute how much to reduce
            deficit = target * self.gross_notional - self.equity
            qty_to_reduce = min(
                abs(pos.quantity),
                deficit / (pos.mark_price * (target - self.maintenance_rate) + 1e-10)
            )
            qty_to_reduce = max(0.0, qty_to_reduce)

            slippage = 0.0
            if execution_cost_fn is not None:
                slippage = execution_cost_fn(symbol, qty_to_reduce)

            realized, margin_released = pos.reduce(qty_to_reduce)
            self.book.realized_pnl += realized
            self.cash += margin_released - slippage
            total_liquidated += qty_to_reduce * pos.mark_price

            if pos.quantity == 0:
                del self.book.positions[symbol]

        return total_liquidated


# ---------------------------------------------------------------------------
# 4. VaR-Based Margin (SPAN Proxy)
# ---------------------------------------------------------------------------

class VaRMargin:
    """
    Value-at-Risk initial margin – simplified SPAN-style.

    IM = max(VaR_99, stress_loss) · position_notional

    Methods
    -------
    historical_var  : parametric VaR from return series
    stressed_var    : VaR under fat-tail scaling
    portfolio_var   : correlated multi-asset VaR
    """

    def __init__(self, confidence: float = 0.99, horizon_days: int = 1):
        self.confidence   = confidence
        self.horizon_days = horizon_days

    def parametric_var(
        self,
        notional:      float,
        return_series: np.ndarray,
    ) -> float:
        """
        Parametric (Gaussian) VaR.
        VaR = notional · σ · z_α · √horizon
        """
        sigma = float(np.std(return_series))
        z     = self._z_score(self.confidence)
        return notional * sigma * z * np.sqrt(self.horizon_days)

    def historical_var(
        self,
        notional:      float,
        return_series: np.ndarray,
    ) -> float:
        """
        Historical simulation VaR – no distributional assumption.
        """
        pnl = notional * return_series
        return float(-np.percentile(pnl, (1 - self.confidence) * 100))

    def stressed_var(
        self,
        notional:      float,
        return_series: np.ndarray,
        stress_window: int = 20,
    ) -> float:
        """
        Stressed VaR: use highest-volatility sub-window in history.
        """
        if len(return_series) < stress_window:
            return self.parametric_var(notional, return_series)

        max_vol = max(
            np.std(return_series[i: i + stress_window])
            for i in range(len(return_series) - stress_window + 1)
        )
        z = self._z_score(self.confidence)
        return notional * max_vol * z * np.sqrt(self.horizon_days)

    def portfolio_var(
        self,
        notionals:    np.ndarray,
        return_matrix: np.ndarray,   # shape (T, n_assets)
        signs:        np.ndarray,    # +1 long, -1 short
    ) -> float:
        """
        Correlated portfolio VaR.
        VaR_port = √(w' Σ w) · z_α · √horizon
        where w = signed_notionals.
        """
        w      = signs * notionals
        cov    = np.cov(return_matrix.T)
        port_var = float(np.sqrt(w @ cov @ w))
        z      = self._z_score(self.confidence)
        return port_var * z * np.sqrt(self.horizon_days)

    @staticmethod
    def _z_score(confidence: float) -> float:
        """Inverse normal CDF via rational approximation (Abramowitz & Stegun)."""
        p = confidence
        t = np.sqrt(-2.0 * np.log(1.0 - p))
        c = [2.515517, 0.802853, 0.010328]
        d = [1.432788, 0.189269, 0.001308]
        return t - (c[0] + c[1]*t + c[2]*t**2) / (1 + d[0]*t + d[1]*t**2 + d[2]*t**3)


# ---------------------------------------------------------------------------
# 5. Funding Liquidity & Haircut Spiral
# ---------------------------------------------------------------------------

class FundingLiquidityModel:
    """
    Adrian-Shin (2010, 2014) procyclical leverage model.

    In good times:  volatility ↓ → VaR ↓ → haircut ↓ → borrowing capacity ↑
                    → leverage ↑ (procyclical)
    In bad times:   volatility ↑ → VaR ↑ → haircut ↑ → margin calls
                    → forced sales → price ↓ → volatility ↑ (spiral)

    Haircut(σ) = h_min + (h_max - h_min) · (σ / σ_max)^κ

    Parameters
    ----------
    h_min   : float – minimum haircut in calm markets (e.g. 0.02)
    h_max   : float – maximum haircut in stressed markets (e.g. 0.50)
    sigma_max: float – volatility level at which haircut reaches h_max
    kappa   : float – convexity of haircut function (1 = linear)
    """

    def __init__(
        self,
        h_min:     float = 0.02,
        h_max:     float = 0.50,
        sigma_max: float = 0.10,
        kappa:     float = 2.0,
    ):
        self.h_min     = h_min
        self.h_max     = h_max
        self.sigma_max = sigma_max
        self.kappa     = kappa

    def haircut(self, sigma: float) -> float:
        """Current repo haircut given volatility σ."""
        ratio = min(sigma / self.sigma_max, 1.0)
        return self.h_min + (self.h_max - self.h_min) * ratio ** self.kappa

    def max_leverage(self, sigma: float) -> float:
        """Maximum leverage = 1 / haircut(σ)."""
        h = self.haircut(sigma)
        return 1.0 / h if h > 0 else np.inf

    def leverage_target(self, equity: float, sigma: float) -> float:
        """Target assets = equity × max_leverage."""
        return equity * self.max_leverage(sigma)

    def simulate_spiral(
        self,
        initial_equity:   float,
        initial_assets:   float,
        sigma_path:       np.ndarray,
        price_impact:     float = 0.001,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate leverage spiral over a volatility path.

        At each step:
        1. Compute new max leverage from σ(t)
        2. If current leverage > max → forced sale
        3. Forced sale depresses price → amplifies loss
        4. Update equity

        Parameters
        ----------
        initial_equity  : starting equity
        initial_assets  : starting asset value (equity + debt)
        sigma_path      : array of σ values over time
        price_impact    : fraction of price drop per unit of forced sale / assets

        Returns dict of time series: equity, assets, leverage, haircut, forced_sales.
        """
        T       = len(sigma_path)
        equity  = np.zeros(T)
        assets  = np.zeros(T)
        leverage = np.zeros(T)
        haircuts = np.zeros(T)
        forced_sales = np.zeros(T)

        eq = initial_equity
        as_ = initial_assets

        for t, sigma in enumerate(sigma_path):
            h   = self.haircut(sigma)
            lev_max = 1.0 / h
            lev_cur = as_ / eq if eq > 0 else np.inf

            sale = 0.0
            if lev_cur > lev_max and eq > 0:
                # must reduce assets to: eq * lev_max
                target_assets = eq * lev_max
                sale = as_ - target_assets
                sale = max(0.0, sale)

                # price impact of sale
                price_drop = price_impact * sale / (as_ + 1e-10)
                loss       = as_ * price_drop
                as_       -= sale + loss
                eq        -= loss

            equity[t]       = eq
            assets[t]       = as_
            leverage[t]     = as_ / eq if eq > 0 else np.inf
            haircuts[t]     = h
            forced_sales[t] = sale

        return {
            "equity":       equity,
            "assets":       assets,
            "leverage":     leverage,
            "haircut":      haircuts,
            "forced_sales": forced_sales,
        }


# ---------------------------------------------------------------------------
# 6. Leverage Cycle Tracker
# ---------------------------------------------------------------------------

class LeverageCycleTracker:
    """
    Tracks leverage ratio, margin utilization, and distance-to-margin-call
    over time for a MarginAccount.

    Useful for plotting the leverage cycle in the notebook.
    """

    def __init__(self, account: MarginAccount):
        self.account = account
        self.history: List[Dict] = []

    def record(self, t: float, extra: Optional[Dict] = None) -> Dict:
        acc = self.account
        snap = {
            "t":                  t,
            "equity":             acc.equity,
            "gross_notional":     acc.gross_notional,
            "leverage":           acc.leverage,
            "equity_ratio":       acc.equity_ratio,
            "free_margin":        acc.free_margin,
            "margin_utilization": acc.margin_utilization,
            "realized_pnl":       acc.book.realized_pnl,
            "unrealized_pnl":     acc.book.total_unrealized_pnl,
            "n_margin_calls":     len(acc.margin_calls),
            "distance_to_call":   acc.equity_ratio - acc.maintenance_rate,
        }
        if extra:
            snap.update(extra)
        self.history.append(snap)
        return snap

    def to_arrays(self) -> Dict[str, np.ndarray]:
        if not self.history:
            return {}
        keys = self.history[0].keys()
        return {k: np.array([h[k] for h in self.history]) for k in keys}

    def margin_call_times(self) -> np.ndarray:
        return np.array([h["t"] for h in self.history if h["distance_to_call"] < 0])
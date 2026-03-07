"""
liquidation.py  –  Liquidation & Cascade Mechanism
====================================================

Covers the following concepts for 06_liquidation_cascade.ipynb:

1. Liquidation trigger            – margin breach, stop-loss, regulatory
2. Liquidation execution          – partial, full, TWAP/aggressive
3. Cascade dynamics               – price impact → cross-margin breach → chain reaction
4. Contagion across accounts      – common asset exposure, correlation channel
5. Fire-sale externality          – social cost vs private cost of liquidation
6. Systemic liquidation pressure  – aggregate forced-sell volume at each price level
7. Liquidation heatmap            – where liquidations cluster in price space
8. Recovery & shortfall           – proceeds vs debt, insurance fund mechanics

Design principle
----------------
- MarginAccount from leverage.py consumed via dependency injection
- OrderBook consumed as dict snapshots (no direct import)
- LiquidationEngine orchestrates: detects → queues → executes → propagates
- All cascade state is explicit (no hidden mutation across objects)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class LiquidationReason(Enum):
    MARGIN_BREACH    = "margin_breach"      # equity_ratio < maintenance_rate
    STOP_LOSS        = "stop_loss"          # price hits stop level
    REGULATORY       = "regulatory"         # leverage cap exceeded
    INSURANCE_FUND   = "insurance_fund"     # taken over by exchange


class LiquidationStatus(Enum):
    PENDING    = "pending"
    PARTIAL    = "partial"
    COMPLETED  = "completed"
    FAILED     = "failed"       # insufficient liquidity


@dataclass
class LiquidationOrder:
    """
    A single liquidation instruction for one account + symbol.
    """
    account_id:  str
    symbol:      str
    quantity:    float           # absolute size to liquidate
    side:        str             # "sell" (long liq) | "buy" (short liq)
    reason:      LiquidationReason
    trigger_price: float         # mark price at trigger
    timestamp:   float

    # filled in during execution
    executed_qty:   float = 0.0
    avg_price:      float = 0.0
    slippage:       float = 0.0
    shortfall:      float = 0.0  # debt not recovered
    status: LiquidationStatus = LiquidationStatus.PENDING

    @property
    def remaining_qty(self) -> float:
        return self.quantity - self.executed_qty

    @property
    def is_complete(self) -> bool:
        return self.status in (LiquidationStatus.COMPLETED, LiquidationStatus.FAILED)


@dataclass
class CascadeEvent:
    """One hop in the liquidation cascade chain."""
    step:          int
    timestamp:     float
    account_id:    str
    symbol:        str
    qty_liquidated: float
    price_before:  float
    price_after:   float
    new_accounts_triggered: List[str] = field(default_factory=list)

    @property
    def price_impact(self) -> float:
        return abs(self.price_after - self.price_before) / (self.price_before + 1e-10)


# ---------------------------------------------------------------------------
# 1. Liquidation Trigger
# ---------------------------------------------------------------------------

class LiquidationTrigger:
    """
    Evaluates whether an account should be liquidated.

    Supports three trigger types:
    - Margin breach : equity_ratio < maintenance_rate
    - Stop-loss     : mark price crosses stop level
    - Regulatory    : leverage ratio exceeds cap
    """

    def __init__(
        self,
        maintenance_rate:  float = 0.05,
        max_leverage:      float = 20.0,
    ):
        self.maintenance_rate = maintenance_rate
        self.max_leverage     = max_leverage
        self._stop_levels: Dict[Tuple[str, str], float] = {}  # (account_id, symbol) → stop_price

    def set_stop_loss(self, account_id: str, symbol: str, stop_price: float):
        self._stop_levels[(account_id, symbol)] = stop_price

    def check(
        self,
        account_id:    str,
        equity_ratio:  float,
        leverage:      float,
        positions:     Dict[str, Tuple[float, float, float]],
        # positions: {symbol: (quantity, mark_price, avg_entry)}
    ) -> List[Tuple[str, LiquidationReason]]:
        """
        Returns list of (symbol, reason) that should be liquidated.
        Empty list → no liquidation needed.
        """
        triggers = []

        # 1. Margin breach → liquidate all positions
        if equity_ratio < self.maintenance_rate:
            for symbol in positions:
                triggers.append((symbol, LiquidationReason.MARGIN_BREACH))
            return triggers

        # 2. Regulatory leverage cap → reduce proportionally
        if leverage > self.max_leverage:
            for symbol in positions:
                triggers.append((symbol, LiquidationReason.REGULATORY))
            return triggers

        # 3. Stop-loss → per symbol
        for symbol, (qty, mark_price, _) in positions.items():
            key = (account_id, symbol)
            if key in self._stop_levels:
                stop = self._stop_levels[key]
                hit = (qty > 0 and mark_price <= stop) or \
                      (qty < 0 and mark_price >= stop)
                if hit:
                    triggers.append((symbol, LiquidationReason.STOP_LOSS))

        return triggers


# ---------------------------------------------------------------------------
# 2. Liquidation Executor
# ---------------------------------------------------------------------------

class LiquidationExecutor:
    """
    Executes a LiquidationOrder against available liquidity.

    Supports three execution strategies:
    - aggressive : immediate market order at any cost
    - twap       : spread over n_slices to reduce impact
    - partial    : liquidate up to available_liquidity, leave remainder

    Parameters
    ----------
    price_impact_fn : Callable[[str, float], float]
        price_impact_fn(symbol, qty) → new_mark_price after execution
    slippage_fn : Callable[[str, float], float]
        slippage_fn(symbol, qty) → slippage_cost (positive = cost to liquidator)
    """

    def __init__(
        self,
        price_impact_fn: Optional[Callable[[str, float], float]] = None,
        slippage_fn:     Optional[Callable[[str, float], float]] = None,
        strategy:        str   = "aggressive",  # "aggressive" | "twap" | "partial"
        n_slices:        int   = 5,
        available_liquidity_fn: Optional[Callable[[str], float]] = None,
    ):
        self.price_impact_fn  = price_impact_fn  or (lambda s, q: 0.0)
        self.slippage_fn      = slippage_fn       or (lambda s, q: 0.0)
        self.strategy         = strategy
        self.n_slices         = n_slices
        self.available_liquidity_fn = available_liquidity_fn or (lambda s: np.inf)

    def execute(
        self,
        order:      LiquidationOrder,
        mark_price: float,
    ) -> Tuple[LiquidationOrder, float]:
        """
        Execute liquidation order.
        Returns (updated_order, new_mark_price).
        """
        if self.strategy == "twap":
            return self._execute_twap(order, mark_price)
        elif self.strategy == "partial":
            return self._execute_partial(order, mark_price)
        else:
            return self._execute_aggressive(order, mark_price)

    def _execute_aggressive(
        self,
        order: LiquidationOrder,
        mark_price: float,
    ) -> Tuple[LiquidationOrder, float]:
        qty      = order.remaining_qty
        slippage = self.slippage_fn(order.symbol, qty)
        new_price = self.price_impact_fn(order.symbol, qty)

        exec_price = mark_price - slippage if order.side == "sell" else mark_price + slippage
        order.executed_qty = qty
        order.avg_price    = exec_price
        order.slippage     = slippage
        order.status       = LiquidationStatus.COMPLETED
        return order, new_price

    def _execute_twap(
        self,
        order: LiquidationOrder,
        mark_price: float,
    ) -> Tuple[LiquidationOrder, float]:
        slice_qty   = order.remaining_qty / self.n_slices
        total_cost  = 0.0
        current_price = mark_price

        for _ in range(self.n_slices):
            slippage  = self.slippage_fn(order.symbol, slice_qty)
            exec_price = current_price - slippage if order.side == "sell" \
                         else current_price + slippage
            total_cost    += slice_qty * exec_price
            current_price  = self.price_impact_fn(order.symbol, slice_qty)
            order.executed_qty += slice_qty

        order.avg_price = total_cost / order.quantity
        order.slippage  = abs(mark_price - order.avg_price)
        order.status    = LiquidationStatus.COMPLETED
        return order, current_price

    def _execute_partial(
        self,
        order: LiquidationOrder,
        mark_price: float,
    ) -> Tuple[LiquidationOrder, float]:
        available = self.available_liquidity_fn(order.symbol)
        fill_qty  = min(order.remaining_qty, available)

        if fill_qty <= 0:
            order.status = LiquidationStatus.FAILED
            return order, mark_price

        slippage  = self.slippage_fn(order.symbol, fill_qty)
        exec_price = mark_price - slippage if order.side == "sell" \
                     else mark_price + slippage
        new_price = self.price_impact_fn(order.symbol, fill_qty)

        order.executed_qty += fill_qty
        order.avg_price     = exec_price
        order.slippage      = slippage

        if order.remaining_qty <= 1e-8:
            order.status = LiquidationStatus.COMPLETED
        else:
            order.status = LiquidationStatus.PARTIAL

        return order, new_price


# ---------------------------------------------------------------------------
# 3. Insurance Fund
# ---------------------------------------------------------------------------

class InsuranceFund:
    """
    Exchange insurance fund – absorbs shortfall when liquidation proceeds
    are insufficient to cover debt.

    Funded by a fraction of trading fees.
    If fund is depleted → auto-deleveraging (ADL) kicks in.

    Parameters
    ----------
    initial_balance : float – starting fund balance
    fee_contribution: float – fraction of each liquidation fee added to fund
    """

    def __init__(
        self,
        initial_balance:  float = 1_000_000.0,
        fee_contribution: float = 0.10,
    ):
        self.balance          = initial_balance
        self.fee_contribution = fee_contribution
        self.total_shortfall_absorbed: float = 0.0
        self.adl_events: List[Dict] = []
        self._history: List[float] = [initial_balance]

    def add_fee(self, fee: float):
        contribution = fee * self.fee_contribution
        self.balance += contribution

    def absorb_shortfall(self, shortfall: float, t: float = 0.0) -> float:
        """
        Absorb liquidation shortfall. Returns unabsorbed amount (triggers ADL).
        """
        absorbed = min(shortfall, self.balance)
        self.balance -= absorbed
        self.total_shortfall_absorbed += absorbed
        self._history.append(self.balance)

        unabsorbed = shortfall - absorbed
        if unabsorbed > 0:
            self.adl_events.append({
                "t":          t,
                "shortfall":  shortfall,
                "absorbed":   absorbed,
                "adl_amount": unabsorbed,
            })

        return unabsorbed

    def is_depleted(self) -> bool:
        return self.balance <= 0

    def coverage_ratio(self, total_open_interest: float) -> float:
        """Fund balance as fraction of total open interest."""
        return self.balance / total_open_interest if total_open_interest > 0 else np.inf

    @property
    def history(self) -> np.ndarray:
        return np.array(self._history)


# ---------------------------------------------------------------------------
# 4. Cascade Engine
# ---------------------------------------------------------------------------

class LiquidationCascadeEngine:
    """
    Orchestrates multi-account liquidation cascades.

    Cascade mechanism
    -----------------
    Step 0: Detect accounts with margin breach
    Step 1: Execute their liquidation orders → price moves
    Step 2: New price → re-check all other accounts
    Step 3: Newly breached accounts join queue → repeat until stable

    Parameters
    ----------
    accounts : dict of account_id → account_state dict with keys:
        equity_ratio, leverage, positions {symbol: (qty, mark_price, avg_entry)},
        debt {symbol: float}
    prices   : dict of symbol → current mark price
    trigger  : LiquidationTrigger
    executor : LiquidationExecutor
    insurance_fund : InsuranceFund
    max_cascade_steps : int – safety limit to prevent infinite loops
    """

    def __init__(
        self,
        trigger:        LiquidationTrigger,
        executor:       LiquidationExecutor,
        insurance_fund: Optional[InsuranceFund] = None,
        max_cascade_steps: int = 50,
    ):
        self.trigger    = trigger
        self.executor   = executor
        self.insurance  = insurance_fund or InsuranceFund()
        self.max_steps  = max_cascade_steps

        self.cascade_log:      List[CascadeEvent] = []
        self.all_liquidations: List[LiquidationOrder] = []

    def run(
        self,
        accounts: Dict[str, Dict],
        prices:   Dict[str, float],
        t:        float = 0.0,
    ) -> Dict:
        """
        Run full cascade simulation starting from current account states.

        accounts[account_id] = {
            "equity_ratio": float,
            "leverage":     float,
            "positions":    {symbol: (qty, mark_price, avg_entry)},
            "debt":         {symbol: float},
        }

        Returns summary dict with cascade stats.
        """
        current_prices = dict(prices)
        pending_ids    = set(accounts.keys())
        processed_ids: set = set()
        step = 0

        while pending_ids and step < self.max_steps:
            newly_triggered: List[str] = []

            for account_id in list(pending_ids):
                acc = accounts[account_id]
                triggers = self.trigger.check(
                    account_id=account_id,
                    equity_ratio=acc["equity_ratio"],
                    leverage=acc["leverage"],
                    positions=acc["positions"],
                )

                if not triggers:
                    pending_ids.discard(account_id)
                    continue

                for symbol, reason in triggers:
                    if symbol not in acc["positions"]:
                        continue

                    qty, mark_price, avg_entry = acc["positions"][symbol]
                    liq_qty  = abs(qty)
                    liq_side = "sell" if qty > 0 else "buy"
                    price_before = current_prices.get(symbol, mark_price)

                    order = LiquidationOrder(
                        account_id=account_id,
                        symbol=symbol,
                        quantity=liq_qty,
                        side=liq_side,
                        reason=reason,
                        trigger_price=price_before,
                        timestamp=t,
                    )

                    order, new_price = self.executor.execute(order, price_before)
                    self.all_liquidations.append(order)

                    # compute shortfall
                    debt = acc["debt"].get(symbol, 0.0)
                    proceeds = order.executed_qty * order.avg_price
                    shortfall = max(0.0, debt - proceeds)
                    order.shortfall = shortfall

                    if shortfall > 0:
                        adl = self.insurance.absorb_shortfall(shortfall, t)
                        # adl > 0 means fund depleted – record for notebook

                    # update price
                    current_prices[symbol] = new_price

                    # mark account as liquidated
                    acc["positions"].pop(symbol, None)
                    acc["equity_ratio"] = 0.0

                    # find newly triggered accounts due to price move
                    newly = self._propagate(
                        accounts, processed_ids | {account_id},
                        symbol, new_price
                    )
                    newly_triggered.extend(newly)

                    # log cascade event
                    self.cascade_log.append(CascadeEvent(
                        step=step,
                        timestamp=t,
                        account_id=account_id,
                        symbol=symbol,
                        qty_liquidated=order.executed_qty,
                        price_before=price_before,
                        price_after=new_price,
                        new_accounts_triggered=newly,
                    ))

                processed_ids.add(account_id)
                pending_ids.discard(account_id)

            # add newly triggered accounts for next iteration
            for aid in newly_triggered:
                if aid not in processed_ids:
                    pending_ids.add(aid)

            step += 1

        return self._summary(t, step, current_prices)

    def _propagate(
        self,
        accounts:     Dict[str, Dict],
        exclude_ids:  set,
        symbol:       str,
        new_price:    float,
    ) -> List[str]:
        """
        After price moves to new_price, update mark prices and equity ratios
        for all remaining accounts exposed to symbol.
        Returns list of account_ids newly breached.
        """
        newly_breached = []

        for aid, acc in accounts.items():
            if aid in exclude_ids:
                continue
            if symbol not in acc["positions"]:
                continue

            qty, old_price, avg_entry = acc["positions"][symbol]
            acc["positions"][symbol] = (qty, new_price, avg_entry)

            # approximate new equity_ratio
            pnl_delta = qty * (new_price - old_price)
            notional  = abs(qty) * new_price
            old_equity = acc["equity_ratio"] * notional + pnl_delta
            acc["equity_ratio"] = old_equity / (notional + 1e-10) if notional > 0 else 0.0

            if acc["equity_ratio"] < self.trigger.maintenance_rate:
                newly_breached.append(aid)

        return newly_breached

    def _summary(self, t: float, steps: int, final_prices: Dict) -> Dict:
        total_liq_qty = sum(o.executed_qty  for o in self.all_liquidations)
        total_shortfall = sum(o.shortfall   for o in self.all_liquidations)
        n_accounts    = len({o.account_id   for o in self.all_liquidations})
        n_completed   = sum(1 for o in self.all_liquidations
                            if o.status == LiquidationStatus.COMPLETED)

        return {
            "t":                  t,
            "cascade_steps":      steps,
            "accounts_liquidated": n_accounts,
            "orders_total":       len(self.all_liquidations),
            "orders_completed":   n_completed,
            "total_qty_liquidated": total_liq_qty,
            "total_shortfall":    total_shortfall,
            "insurance_balance":  self.insurance.balance,
            "adl_events":         len(self.insurance.adl_events),
            "final_prices":       final_prices,
        }


# ---------------------------------------------------------------------------
# 5. Liquidation Pressure Map
# ---------------------------------------------------------------------------

class LiquidationPressureMap:
    """
    Estimates where liquidations will cluster across price levels.

    Given a set of positions with known entry prices and margin rates,
    compute the aggregate forced-sell volume that would be triggered
    at each price level.

    Useful for:
    - Identifying price levels with high liquidation density
    - Estimating cascade amplification at key support/resistance levels
    - Constructing the "liquidation heatmap" visualization
    """

    def __init__(self, tick_size: float = 1.0):
        self.tick_size = tick_size
        # price_level → {"long_liq": float, "short_liq": float}
        self._map: Dict[float, Dict[str, float]] = {}

    def add_position(
        self,
        quantity:      float,     # signed
        entry_price:   float,
        margin_rate:   float,
        leverage:      float = 1.0,
    ):
        """
        Register a position. Computes the liquidation price and adds
        the position size to the appropriate price bucket.

        Liquidation price (long) ≈ entry × (1 - 1/leverage + maintenance_rate)
        Liquidation price (short) ≈ entry × (1 + 1/leverage - maintenance_rate)
        """
        maintenance = margin_rate * 0.5  # proxy: maint = 50% of initial
        lev = max(leverage, 1.0 / (margin_rate + 1e-10))

        if quantity > 0:   # long position
            liq_price = entry_price * (1.0 - 1.0/lev + maintenance)
        else:              # short position
            liq_price = entry_price * (1.0 + 1.0/lev - maintenance)

        # round to tick
        liq_price = round(liq_price / self.tick_size) * self.tick_size

        if liq_price not in self._map:
            self._map[liq_price] = {"long_liq": 0.0, "short_liq": 0.0}

        if quantity > 0:
            self._map[liq_price]["long_liq"]  += abs(quantity)
        else:
            self._map[liq_price]["short_liq"] += abs(quantity)

    def to_arrays(
        self,
        price_range: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Returns arrays suitable for heatmap plotting.

        Returns dict with keys: prices, long_liq, short_liq, net_pressure.
        net_pressure > 0 → more long liquidations (selling pressure)
        net_pressure < 0 → more short liquidations (buying pressure)
        """
        prices = sorted(self._map.keys())
        if price_range:
            prices = [p for p in prices if price_range[0] <= p <= price_range[1]]

        if not prices:
            return {"prices": np.array([]), "long_liq": np.array([]),
                    "short_liq": np.array([]), "net_pressure": np.array([])}

        long_liq  = np.array([self._map[p]["long_liq"]  for p in prices])
        short_liq = np.array([self._map[p]["short_liq"] for p in prices])

        return {
            "prices":       np.array(prices),
            "long_liq":     long_liq,
            "short_liq":    short_liq,
            "net_pressure": long_liq - short_liq,
        }

    def dominant_liq_price(self) -> Optional[float]:
        """Price level with the highest total liquidation volume."""
        if not self._map:
            return None
        return max(
            self._map,
            key=lambda p: self._map[p]["long_liq"] + self._map[p]["short_liq"]
        )

    def reset(self):
        self._map.clear()


# ---------------------------------------------------------------------------
# 6. Fire-Sale Externality
# ---------------------------------------------------------------------------

class FireSaleExternality:
    """
    Quantifies the social cost of liquidation beyond the private cost.

    Private cost  = slippage borne by the liquidated account
    Social cost   = price impact on ALL other accounts holding the same asset

    Externality   = Social cost - Private cost
                  = (total_notional_affected - own_notional) × |ΔP| / P

    Parameters
    ----------
    total_market_notional : float – total market exposure to the asset
    """

    def __init__(self, total_market_notional: float):
        self.total_market_notional = total_market_notional

    def externality(
        self,
        own_notional:   float,
        price_before:   float,
        price_after:    float,
    ) -> Dict[str, float]:
        """
        Compute private vs social loss from a single liquidation event.
        """
        dp = abs(price_after - price_before)
        pct_move = dp / (price_before + 1e-10)

        private_loss = own_notional * pct_move
        social_loss  = self.total_market_notional * pct_move
        externality  = social_loss - private_loss

        return {
            "private_loss":       private_loss,
            "social_loss":        social_loss,
            "externality":        externality,
            "externality_ratio":  externality / (private_loss + 1e-10),
            "price_impact_pct":   pct_move,
        }

    def systemic_amplification(
        self,
        liquidation_sizes: np.ndarray,
        price_impacts:     np.ndarray,
    ) -> float:
        """
        Total externality across a cascade sequence.
        amplification = Σ social_cost / Σ private_cost
        """
        own_losses    = liquidation_sizes * price_impacts
        social_losses = self.total_market_notional * price_impacts
        total_own    = own_losses.sum()
        total_social = social_losses.sum()
        return total_social / (total_own + 1e-10)
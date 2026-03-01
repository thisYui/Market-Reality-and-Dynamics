"""
Momentum & Mean Reversion Dynamics
----------------------------------

Mô hình hóa:

- Price momentum
- Mean reversion
- Regime switching
- Feedback amplification

Designed for:
Project 1 — Market Reality & Dynamics
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


# ============================================================
# Momentum Signal
# ============================================================

def compute_momentum(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Simple momentum:
    M_t = P_t - P_{t-window}
    """
    if window <= 0:
        raise ValueError("window must be positive")

    momentum = np.zeros_like(prices)

    for t in range(window, len(prices)):
        momentum[t] = prices[t] - prices[t - window]

    return momentum


# ============================================================
# Regime Switching Model
# ============================================================

class MomentumRegimeModel:
    """
    Mô hình giá với 2 regime:

    - Momentum regime
    - Mean-reversion regime
    """

    def __init__(
        self,
        momentum_strength: float = 0.1,
        reversion_strength: float = 0.1,
        regime_switch_prob: float = 0.05,
        noise_vol: float = 0.01,
    ):
        self.momentum_strength = momentum_strength
        self.reversion_strength = reversion_strength
        self.regime_switch_prob = regime_switch_prob
        self.noise_vol = noise_vol

        self.current_regime = "momentum"

    # ------------------------------------------------------------
    # Regime switching
    # ------------------------------------------------------------

    def maybe_switch_regime(self):
        if np.random.rand() < self.regime_switch_prob:
            self.current_regime = (
                "mean_reversion"
                if self.current_regime == "momentum"
                else "momentum"
            )

    # ------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------

    def step(self, prev_price: float, prev_return: float, fundamental: float) -> Tuple[float, float]:
        """
        prev_price  : giá trước đó
        prev_return : return trước đó
        fundamental : giá trị nội tại (mean reversion anchor)
        """

        self.maybe_switch_regime()

        noise = np.random.normal(0, self.noise_vol)

        if self.current_regime == "momentum":
            # trend continuation
            drift = self.momentum_strength * prev_return
        else:
            # pull back toward fundamental
            drift = self.reversion_strength * (fundamental - prev_price)

        new_return = drift + noise
        new_price = prev_price * (1 + new_return)

        return new_price, new_return

    # ------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------

    def simulate(
        self,
        n_steps: int = 1000,
        initial_price: float = 100,
        fundamental: float = 100,
    ):
        prices = [initial_price]
        returns = [0.0]

        for _ in range(n_steps):
            price, ret = self.step(
                prev_price=prices[-1],
                prev_return=returns[-1],
                fundamental=fundamental,
            )
            prices.append(price)
            returns.append(ret)

        return np.array(prices), np.array(returns)
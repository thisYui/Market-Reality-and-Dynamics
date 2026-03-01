"""
Regime Switching Dynamics
-------------------------

Mô hình hóa:

- Markov regime switching
- Volatility regimes
- Drift regimes
- Structural break

Designed for:
Project 1 — Market Reality & Dynamics
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


# ============================================================
# Markov Regime Process
# ============================================================

class MarkovRegimeProcess:
    """
    Two-state Markov switching process.

    State 0: Normal
    State 1: Crisis
    """

    def __init__(
        self,
        p00: float = 0.95,  # stay in normal
        p11: float = 0.90   # stay in crisis
    ):
        if not (0 <= p00 <= 1 and 0 <= p11 <= 1):
            raise ValueError("Transition probabilities must be in [0,1]")

        self.p00 = p00
        self.p11 = p11
        self.state = 0  # start in normal

    def step(self) -> int:
        """
        Transition to next state.
        """
        if self.state == 0:
            if np.random.rand() > self.p00:
                self.state = 1
        else:
            if np.random.rand() > self.p11:
                self.state = 0

        return self.state


# ============================================================
# Regime-Based Return Generator
# ============================================================

class RegimeReturnModel:
    """
    Return model với regime-dependent parameters.

    Normal regime:
        low vol, small drift

    Crisis regime:
        high vol, negative drift
    """

    def __init__(
        self,
        normal_drift: float = 0.0005,
        normal_vol: float = 0.01,
        crisis_drift: float = -0.002,
        crisis_vol: float = 0.05,
        p00: float = 0.97,
        p11: float = 0.90,
    ):
        self.regime = MarkovRegimeProcess(p00, p11)

        self.normal_drift = normal_drift
        self.normal_vol = normal_vol
        self.crisis_drift = crisis_drift
        self.crisis_vol = crisis_vol

    def step(self) -> Tuple[float, int]:
        """
        Generate one return observation.
        """
        state = self.regime.step()

        if state == 0:
            r = np.random.normal(self.normal_drift, self.normal_vol)
        else:
            r = np.random.normal(self.crisis_drift, self.crisis_vol)

        return r, state

    def simulate(self, n_steps: int = 2000):
        """
        Simulate time series.
        """
        returns = []
        states = []

        for _ in range(n_steps):
            r, s = self.step()
            returns.append(r)
            states.append(s)

        return np.array(returns), np.array(states)


# ============================================================
# Structural Break Model
# ============================================================

class StructuralBreakModel:
    """
    One-time structural break.

    Parameter shift sau thời điểm t_break.
    """

    def __init__(
        self,
        t_break: int = 1000,
        pre_drift: float = 0.001,
        pre_vol: float = 0.01,
        post_drift: float = -0.001,
        post_vol: float = 0.03,
    ):
        self.t_break = t_break
        self.pre_drift = pre_drift
        self.pre_vol = pre_vol
        self.post_drift = post_drift
        self.post_vol = post_vol

    def simulate(self, n_steps: int = 2000):
        returns = []

        for t in range(n_steps):
            if t < self.t_break:
                r = np.random.normal(self.pre_drift, self.pre_vol)
            else:
                r = np.random.normal(self.post_drift, self.post_vol)

            returns.append(r)

        return np.array(returns)
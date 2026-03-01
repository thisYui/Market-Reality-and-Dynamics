"""
Volatility Dynamics Module
--------------------------

Mô hình hóa:

- Rolling volatility
- EWMA volatility
- GARCH-like clustering
- Endogenous volatility feedback

Designed for:
Project 1 — Market Reality & Dynamics
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


# ============================================================
# Basic Volatility Measures
# ============================================================

def rolling_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Rolling standard deviation.
    """
    if window <= 1:
        raise ValueError("window must be > 1")

    vols = []
    for i in range(window, len(returns) + 1):
        vols.append(np.std(returns[i - window:i]))

    return np.array(vols)


def ewma_volatility(
    returns: np.ndarray,
    lambda_: float = 0.94
) -> np.ndarray:
    """
    Exponentially weighted moving volatility.
    RiskMetrics style.
    """
    if not (0 < lambda_ < 1):
        raise ValueError("lambda_ must be between 0 and 1")

    var = returns[0] ** 2
    vols = [np.sqrt(var)]

    for r in returns[1:]:
        var = lambda_ * var + (1 - lambda_) * r**2
        vols.append(np.sqrt(var))

    return np.array(vols)


# ============================================================
# GARCH-like Volatility Clustering
# ============================================================

class GARCHLikeModel:
    """
    Simple GARCH(1,1)-style volatility process:

    σ_t^2 = ω + α r_{t-1}^2 + β σ_{t-1}^2

    Không dùng thư viện statsmodels,
    chỉ mô phỏng dynamics.
    """

    def __init__(
        self,
        omega: float = 0.000001,
        alpha: float = 0.1,
        beta: float = 0.85,
    ):
        if alpha + beta >= 1:
            raise ValueError("alpha + beta must be < 1 for stability")

        self.omega = omega
        self.alpha = alpha
        self.beta = beta

    def simulate(self, n_steps: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate returns with volatility clustering.
        """
        returns = []
        variances = []

        var = self.omega / (1 - self.alpha - self.beta)

        for _ in range(n_steps):
            shock = np.random.normal(0, np.sqrt(var))
            returns.append(shock)
            variances.append(var)

            var = (
                self.omega
                + self.alpha * shock**2
                + self.beta * var
            )

        return np.array(returns), np.array(variances)


# ============================================================
# Endogenous Volatility Feedback Model
# ============================================================

class VolatilityFeedbackModel:
    """
    Khi return lớn (đặc biệt là âm),
    volatility tăng phi tuyến.

    Mô phỏng leverage effect.
    """

    def __init__(
        self,
        base_vol: float = 0.01,
        sensitivity: float = 5.0,
    ):
        self.base_vol = base_vol
        self.sensitivity = sensitivity

    def step(self, previous_return: float) -> float:
        """
        Volatility phụ thuộc magnitude return trước đó.
        """
        vol = self.base_vol * (
            1 + self.sensitivity * abs(previous_return)
        )

        return np.random.normal(0, vol)

    def simulate(self, n_steps: int = 2000) -> np.ndarray:
        returns = []
        prev = 0.0

        for _ in range(n_steps):
            r = self.step(prev)
            returns.append(r)
            prev = r

        return np.array(returns)
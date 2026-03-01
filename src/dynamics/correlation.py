"""
Correlation Dynamics Module
---------------------------

Mô hình hóa:

- Static correlation
- Rolling correlation
- Regime-dependent correlation
- Correlation breakdown
- Contagion effect

Designed for:
Project 1 — Market Reality & Dynamics
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


# ============================================================
# Utility Functions
# ============================================================

def compute_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson correlation coefficient.
    """
    if len(x) != len(y):
        raise ValueError("Input series must have same length.")
    return float(np.corrcoef(x, y)[0, 1])


def rolling_correlation(
    x: np.ndarray,
    y: np.ndarray,
    window: int
) -> np.ndarray:
    """
    Rolling correlation series.
    """
    if window <= 1:
        raise ValueError("Window must be > 1.")

    corrs = []
    for i in range(window, len(x) + 1):
        corrs.append(
            compute_correlation(x[i - window:i], y[i - window:i])
        )

    return np.array(corrs)


# ============================================================
# Regime-Based Correlation
# ============================================================

class RegimeCorrelationModel:
    """
    Correlation thay đổi theo regime.

    Ví dụ:
        Low-vol regime  -> corr = 0.2
        Crisis regime   -> corr = 0.9
    """

    def __init__(
        self,
        low_corr: float = 0.2,
        high_corr: float = 0.9,
        crisis_probability: float = 0.1,
    ):
        self.low_corr = low_corr
        self.high_corr = high_corr
        self.crisis_probability = crisis_probability

    def sample_regime(self) -> float:
        """
        Randomly choose correlation regime.
        """
        if np.random.rand() < self.crisis_probability:
            return self.high_corr
        return self.low_corr

    def generate_returns(
        self,
        n_steps: int = 1000,
        vol: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sinh hai chuỗi return với correlation thay đổi theo regime.
        """

        returns_1 = []
        returns_2 = []

        for _ in range(n_steps):
            rho = self.sample_regime()

            cov_matrix = [
                [vol**2, rho * vol**2],
                [rho * vol**2, vol**2],
            ]

            r = np.random.multivariate_normal(
                mean=[0, 0],
                cov=cov_matrix
            )

            returns_1.append(r[0])
            returns_2.append(r[1])

        return np.array(returns_1), np.array(returns_2)


# ============================================================
# Correlation Breakdown Model
# ============================================================

class CorrelationBreakdownModel:
    """
    Mô hình hóa hiện tượng:

    - Correlation tăng mạnh khi thị trường giảm sâu
    - Diversification thất bại trong khủng hoảng
    """

    def __init__(
        self,
        normal_corr: float = 0.2,
        crisis_corr: float = 0.95,
        crisis_threshold: float = -0.03,
        vol: float = 0.01,
    ):
        self.normal_corr = normal_corr
        self.crisis_corr = crisis_corr
        self.crisis_threshold = crisis_threshold
        self.vol = vol

    def step(self) -> Tuple[float, float]:
        """
        Một bước return với correlation phụ thuộc shock.
        """

        # Shock hệ thống
        systemic_shock = np.random.normal(0, self.vol)

        if systemic_shock < self.crisis_threshold:
            rho = self.crisis_corr
        else:
            rho = self.normal_corr

        cov_matrix = [
            [self.vol**2, rho * self.vol**2],
            [rho * self.vol**2, self.vol**2],
        ]

        r = np.random.multivariate_normal(
            mean=[0, 0],
            cov=cov_matrix
        )

        return r[0], r[1]

    def simulate(self, n_steps: int = 1000):
        """
        Simulate time series with correlation breakdown.
        """
        x = []
        y = []

        for _ in range(n_steps):
            r1, r2 = self.step()
            x.append(r1)
            y.append(r2)

        return np.array(x), np.array(y)
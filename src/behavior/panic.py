"""
Panic Cascade Model
-------------------

Mô hình hóa hiện tượng panic selling và chain reaction.

Cơ chế:
- Khi return âm vượt threshold
- Xác suất bán tăng phi tuyến
- Bán tạo thêm price impact
- Tạo feedback loop

Designed for:
Project 1 — Market Reality & Dynamics
"""

from __future__ import annotations
import numpy as np


class PanicCascadeModel:
    """
    Panic selling dynamics.

    Parameters
    ----------
    n_agents : int
        Số lượng agent trong hệ thống.
    panic_threshold : float
        Ngưỡng return kích hoạt panic (ví dụ -0.03).
    base_panic_prob : float
        Xác suất panic cơ bản.
    amplification : float
        Hệ số khuếch đại khi vượt threshold.
    impact_coefficient : float
        Price impact do panic selling.
    """

    def __init__(
        self,
        n_agents: int = 100,
        panic_threshold: float = -0.03,
        base_panic_prob: float = 0.05,
        amplification: float = 5.0,
        impact_coefficient: float = 0.02,
    ):
        self.n_agents = n_agents
        self.panic_threshold = panic_threshold
        self.base_panic_prob = base_panic_prob
        self.amplification = amplification
        self.impact_coefficient = impact_coefficient

    # ============================================================
    # Panic Probability
    # ============================================================

    def panic_probability(self, market_return: float) -> float:
        """
        Xác suất panic phụ thuộc vào độ sâu drawdown.
        """

        if market_return >= self.panic_threshold:
            return self.base_panic_prob

        excess = abs(market_return - self.panic_threshold)

        # Nonlinear amplification
        amplified = self.base_panic_prob + self.amplification * excess

        # Giới hạn tối đa 1
        return min(1.0, amplified)

    # ============================================================
    # Panic Step
    # ============================================================

    def step(self, market_return: float) -> dict:
        """
        Một bước panic:

        1. Tính xác suất panic
        2. Sinh quyết định bán
        3. Tính additional impact
        """

        prob = self.panic_probability(market_return)

        panic_decisions = np.random.rand(self.n_agents) < prob

        panic_intensity = np.sum(panic_decisions) / self.n_agents

        additional_return = -self.impact_coefficient * panic_intensity

        return {
            "panic_probability": prob,
            "panic_intensity": panic_intensity,
            "additional_return": additional_return,
        }

    # ============================================================
    # Cascade Simulation
    # ============================================================

    def simulate(
        self,
        initial_return: float,
        n_steps: int = 20,
    ) -> dict:
        """
        Mô phỏng cascade nhiều bước.
        """

        returns = [initial_return]
        intensities = []

        current_return = initial_return

        for _ in range(n_steps):
            result = self.step(current_return)

            intensities.append(result["panic_intensity"])

            # feedback loop
            current_return = result["additional_return"]
            returns.append(current_return)

            # Nếu không còn selling pressure thì dừng
            if abs(current_return) < 1e-6:
                break

        return {
            "returns": np.array(returns),
            "panic_intensity": np.array(intensities),
        }
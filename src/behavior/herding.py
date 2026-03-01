"""
Herding Model
-------------

Mô hình hóa hành vi bầy đàn (herding) trong thị trường.

Agent có:
- Private signal
- Xác suất theo majority
- Price impact từ aggregate order flow

Designed for:
Project 1 — Market Reality & Dynamics
"""

from __future__ import annotations
import numpy as np


class HerdingModel:
    """
    Agent-based herding dynamics.

    Parameters
    ----------
    n_agents : int
        Số lượng agent.
    herding_strength : float ∈ [0,1]
        Xác suất agent bỏ private signal để theo majority.
    impact_coefficient : float
        Mức độ order imbalance ảnh hưởng lên price return.
    """

    def __init__(
        self,
        n_agents: int = 100,
        herding_strength: float = 0.6,
        impact_coefficient: float = 0.01,
    ):
        if not (0 <= herding_strength <= 1):
            raise ValueError("herding_strength must be between 0 and 1")

        self.n_agents = n_agents
        self.herding_strength = herding_strength
        self.impact_coefficient = impact_coefficient

    # ============================================================
    # Core Mechanics
    # ============================================================

    def generate_private_signals(self) -> np.ndarray:
        """
        Mỗi agent có tín hiệu riêng:
        -1 (sell) hoặc +1 (buy)
        """
        return np.random.choice([-1, 1], size=self.n_agents)

    def apply_herding(self, private_signals: np.ndarray) -> np.ndarray:
        """
        Một phần agent sẽ bỏ private signal và theo majority.
        """
        majority = np.sign(np.sum(private_signals))

        if majority == 0:
            return private_signals  # không có majority

        follow_mask = np.random.rand(self.n_agents) < self.herding_strength

        decisions = private_signals.copy()
        decisions[follow_mask] = majority

        return decisions

    def compute_order_imbalance(self, decisions: np.ndarray) -> float:
        """
        Order imbalance = tổng buy - sell
        Chuẩn hóa về [-1, 1]
        """
        imbalance = np.sum(decisions) / self.n_agents
        return imbalance

    def price_impact(self, imbalance: float) -> float:
        """
        Price return được quyết định bởi order imbalance.
        """
        return self.impact_coefficient * imbalance

    # ============================================================
    # Simulation Step
    # ============================================================

    def step(self) -> dict:
        """
        Một bước thị trường:
        1. Sinh private signals
        2. Áp dụng herding
        3. Tính order imbalance
        4. Tính return
        """

        private_signals = self.generate_private_signals()
        decisions = self.apply_herding(private_signals)
        imbalance = self.compute_order_imbalance(decisions)
        market_return = self.price_impact(imbalance)

        return {
            "private_signals": private_signals,
            "decisions": decisions,
            "imbalance": imbalance,
            "return": market_return,
        }

    # ============================================================
    # Multi-step Simulation
    # ============================================================

    def simulate(self, n_steps: int = 100) -> dict:
        """
        Chạy nhiều bước để quan sát volatility clustering.
        """

        returns = []
        imbalances = []

        for _ in range(n_steps):
            result = self.step()
            returns.append(result["return"])
            imbalances.append(result["imbalance"])

        return {
            "returns": np.array(returns),
            "imbalances": np.array(imbalances),
        }
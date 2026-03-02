"""
State-Dependent Liquidity System
--------------------------------

Nâng cấp từ linear depth model thành:

- Endogenous liquidity evaporation
- Regime transition (Normal → Stress → Crisis)
- Nonlinear price impact
- Gradual recovery

Designed for:
Project 1 — Market Reality & Dynamics
"""

from __future__ import annotations
import numpy as np
from typing import Dict


class LiquidityModel:
    """
    State-dependent liquidity model.

    Core Ideas:
    - Depth phụ thuộc volatility
    - Spread phụ thuộc depth
    - Impact phi tuyến khi depth thấp
    - Có regime structure
    """

    def __init__(
        self,
        base_depth: float = 1000.0,
        base_spread: float = 0.001,
        impact_coefficient: float = 0.01,
        evaporation_sensitivity: float = 8.0,
        recovery_rate: float = 0.05,
    ):
        self.base_depth = base_depth
        self.base_spread = base_spread
        self.impact_coefficient = impact_coefficient
        self.evaporation_sensitivity = evaporation_sensitivity
        self.recovery_rate = recovery_rate

        self.current_depth = base_depth
        self.regime = "normal"

    # ============================================================
    # Regime Update
    # ============================================================

    def update_regime(self):
        depth_ratio = self.current_depth / self.base_depth

        if depth_ratio > 0.6:
            self.regime = "normal"
        elif depth_ratio > 0.25:
            self.regime = "stress"
        else:
            self.regime = "crisis"

    # ============================================================
    # Volatility Shock → Depth Evaporation
    # ============================================================

    def apply_volatility(self, volatility: float):
        """
        Volatility làm depth giảm phi tuyến.
        """
        reduction = np.exp(-self.evaporation_sensitivity * volatility)
        self.current_depth = self.base_depth * reduction
        self.update_regime()

    # ============================================================
    # Spread
    # ============================================================

    def spread(self) -> float:
        """
        Spread blowout khi depth thấp.
        """
        depth_ratio = max(self.current_depth / self.base_depth, 1e-6)
        return self.base_spread / depth_ratio

    # ============================================================
    # Nonlinear Price Impact
    # ============================================================

    def price_impact(self, order_size: float) -> float:
        """
        Impact tăng mạnh khi depth thấp.
        """
        if self.current_depth <= 0:
            return -np.sign(order_size) * 0.2

        relative_size = abs(order_size) / self.current_depth

        # Nonlinear amplification
        amplification = 1 + 5 * (1 - self.current_depth / self.base_depth)

        impact = self.impact_coefficient * relative_size * amplification

        return -np.sign(order_size) * impact

    # ============================================================
    # Execute Trade
    # ============================================================

    def execute_trade(self, order_size: float) -> Dict:
        """
        Trade làm depth giảm.
        """
        executed = min(abs(order_size), self.current_depth)

        impact = self.price_impact(order_size)

        self.current_depth -= executed
        self.update_regime()

        return {
            "impact": impact,
            "remaining_depth": self.current_depth,
            "spread": self.spread(),
            "regime": self.regime,
        }

    # ============================================================
    # Recovery
    # ============================================================

    def recover(self):
        """
        Depth hồi phục dần theo thời gian.
        """
        gap = self.base_depth - self.current_depth
        self.current_depth += self.recovery_rate * gap
        self.update_regime()

    # ============================================================
    # Crisis Collapse
    # ============================================================

    def collapse(self, severity: float = 0.5):
        """
        External shock collapse depth.
        """
        self.current_depth *= (1 - severity)
        self.update_regime()

# ============================================================
# Liquidity Shock Model
# ============================================================

class LiquidityShockModel:
    """
    Liquidity evaporation during crisis.

    Depth giảm mạnh khi volatility tăng.
    """

    def __init__(
        self,
        base_depth: float = 1000.0,
        evaporation_sensitivity: float = 10.0,
    ):
        self.base_depth = base_depth
        self.evaporation_sensitivity = evaporation_sensitivity
        self.current_depth = base_depth

    def update(self, volatility: float):
        """
        Volatility tăng -> liquidity giảm phi tuyến.
        """
        reduction = self.evaporation_sensitivity * volatility

        self.current_depth = self.base_depth * np.exp(-reduction)

    def get_depth(self) -> float:
        return self.current_depth
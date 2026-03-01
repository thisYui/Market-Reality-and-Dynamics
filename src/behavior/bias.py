"""
Behavioral Bias Models
----------------------

Module này mô hình hóa các sai lệch hành vi cơ bản
ảnh hưởng đến:

- Position sizing
- Risk perception
- Utility evaluation
- Trading decision

Designed for:
Project 1 — Market Reality & Dynamics
"""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


# ============================================================
# Base Class
# ============================================================

class BehavioralBias(ABC):
    """
    Abstract base class cho mọi behavioral bias.
    """

    @abstractmethod
    def apply(self, value: np.ndarray | float) -> np.ndarray | float:
        """
        Apply bias lên input.
        """
        pass


# ============================================================
# Overconfidence Bias
# ============================================================

class OverconfidenceModel(BehavioralBias):
    """
    Overconfidence làm:

    - Tăng position size
    - Giảm perceived volatility
    - Tăng leverage

    confidence_multiplier > 1 => mức độ tự tin cao
    """

    def __init__(self, confidence_multiplier: float = 1.5):
        if confidence_multiplier <= 1:
            raise ValueError("confidence_multiplier must be > 1")
        self.confidence_multiplier = confidence_multiplier

    def apply(self, position_size: float) -> float:
        """
        Tăng position size do tự tin quá mức.
        """
        return position_size * self.confidence_multiplier

    def perceived_volatility(self, true_volatility: float) -> float:
        """
        Volatility bị đánh giá thấp hơn thực tế.
        """
        return true_volatility / self.confidence_multiplier


# ============================================================
# Loss Aversion Bias
# ============================================================

class LossAversionModel(BehavioralBias):
    """
    Loss aversion:

    Lỗ có trọng số lớn hơn lời.

    loss_weight > 1:
    - 2.0 nghĩa là lỗ đau gấp 2 lần lời cùng magnitude
    """

    def __init__(self, loss_weight: float = 2.0):
        if loss_weight <= 1:
            raise ValueError("loss_weight must be > 1")
        self.loss_weight = loss_weight

    def apply(self, returns: np.ndarray | float) -> np.ndarray | float:
        """
        Trả về utility bất đối xứng.
        """
        returns = np.asarray(returns)

        utility = np.where(
            returns >= 0,
            returns,
            returns * self.loss_weight,
        )

        return utility


# ============================================================
# Confirmation Bias
# ============================================================

class ConfirmationBiasModel(BehavioralBias):
    """
    Confirmation bias:

    Nhà đầu tư:
    - Tăng trọng số thông tin ủng hộ quan điểm
    - Giảm trọng số thông tin trái chiều
    """

    def __init__(self, confirmation_strength: float = 1.5):
        if confirmation_strength <= 1:
            raise ValueError("confirmation_strength must be > 1")
        self.confirmation_strength = confirmation_strength

    def apply(
        self,
        signals: np.ndarray,
        prior_belief: float
    ) -> np.ndarray:
        """
        signals: array tín hiệu (-1, 1)
        prior_belief: -1 hoặc 1

        Tín hiệu cùng chiều được khuếch đại.
        """
        signals = np.asarray(signals)

        aligned = signals == np.sign(prior_belief)

        adjusted = np.where(
            aligned,
            signals * self.confirmation_strength,
            signals / self.confirmation_strength,
        )

        return adjusted


# ============================================================
# Utility Wrapper (Optional)
# ============================================================

def apply_bias_pipeline(
    value: np.ndarray | float,
    biases: list[BehavioralBias]
) -> np.ndarray | float:
    """
    Cho phép áp dụng nhiều bias liên tiếp.

    Ví dụ:
        value -> overconfidence -> loss_aversion
    """
    result = value
    for bias in biases:
        result = bias.apply(result)
    return result
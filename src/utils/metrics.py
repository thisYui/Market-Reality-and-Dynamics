"""
Metrics Module
--------------

Risk & performance metrics used across:

Project 1 — Market Dynamics
Project 2 — Risk & Fat-Tail
Project 3 — Decision Under Uncertainty
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


# ============================================================
# Basic Statistics
# ============================================================

def mean_return(returns: np.ndarray) -> float:
    return float(np.mean(returns))


def volatility(returns: np.ndarray) -> float:
    return float(np.std(returns))


def sharpe_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    annualization: float = 1.0
) -> float:
    excess = returns - risk_free
    vol = np.std(excess)
    if vol == 0:
        return 0.0
    return float(np.mean(excess) / vol * np.sqrt(annualization))


# ============================================================
# Drawdown
# ============================================================

def cumulative_returns(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1 + returns)


def drawdown(returns: np.ndarray) -> np.ndarray:
    cum = cumulative_returns(returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return dd


def max_drawdown(returns: np.ndarray) -> float:
    return float(np.min(drawdown(returns)))


# ============================================================
# Tail Risk
# ============================================================

def value_at_risk(
    returns: np.ndarray,
    alpha: float = 0.05
) -> float:
    """
    Historical VaR.
    """
    return float(np.quantile(returns, alpha))


def expected_shortfall(
    returns: np.ndarray,
    alpha: float = 0.05
) -> float:
    """
    Conditional VaR (CVaR).
    """
    var = value_at_risk(returns, alpha)
    tail_losses = returns[returns <= var]
    if len(tail_losses) == 0:
        return var
    return float(np.mean(tail_losses))


# ============================================================
# Ruin Probability
# ============================================================

def ruin_probability(
    returns: np.ndarray,
    threshold: float = -0.5
) -> float:
    """
    Xác suất drawdown vượt threshold.
    threshold = -0.5 => mất 50% vốn
    """
    dd = drawdown(returns)
    ruin_events = dd <= threshold
    return float(np.mean(ruin_events))


# ============================================================
# Rolling Metrics
# ============================================================

def rolling_sharpe(
    returns: np.ndarray,
    window: int = 50
) -> np.ndarray:
    sharpe_vals = []

    for i in range(window, len(returns) + 1):
        window_slice = returns[i - window:i]
        sharpe_vals.append(sharpe_ratio(window_slice))

    return np.array(sharpe_vals)


def rolling_volatility(
    returns: np.ndarray,
    window: int = 50
) -> np.ndarray:
    vols = []

    for i in range(window, len(returns) + 1):
        vols.append(np.std(returns[i - window:i]))

    return np.array(vols)


# ============================================================
# Volatility Clustering Measure
# ============================================================

def volatility_clustering_score(returns: np.ndarray) -> float:
    """
    Đo mức độ autocorrelation của squared returns.
    """
    squared = returns**2
    if len(squared) < 2:
        return 0.0

    return float(np.corrcoef(squared[:-1], squared[1:])[0, 1])
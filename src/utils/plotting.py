"""
Plotting Utilities
------------------

Visualization tools for:

Project 1 — Market Dynamics
Project 2 — Risk & Fat-Tail
Project 3 — Decision & Forecast Failure
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Global Style
# ============================================================

def set_style():
    plt.style.use("ggplot")
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["axes.grid"] = True


# ============================================================
# Price & Return Plots
# ============================================================

def plot_price_series(prices: np.ndarray, title: str = "Price Series"):
    set_style()
    plt.plot(prices)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.show()


def plot_return_series(returns: np.ndarray, title: str = "Return Series"):
    set_style()
    plt.plot(returns)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.show()


# ============================================================
# Drawdown Plot
# ============================================================

def plot_drawdown(returns: np.ndarray):
    from .metrics import drawdown

    set_style()
    dd = drawdown(returns)

    plt.plot(dd)
    plt.title("Drawdown")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.show()


# ============================================================
# Volatility Clustering
# ============================================================

def plot_volatility_clustering(
    returns: np.ndarray,
    window: int = 50
):
    from .metrics import rolling_volatility

    set_style()
    vol = rolling_volatility(returns, window)

    plt.plot(vol)
    plt.title(f"Rolling Volatility (window={window})")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.show()


# ============================================================
# Regime Highlight
# ============================================================

def plot_regime_series(
    returns: np.ndarray,
    states: np.ndarray
):
    """
    Plot returns with crisis regime highlighted.
    state = 0 (normal), 1 (crisis)
    """
    set_style()

    plt.plot(returns, label="Returns")

    crisis_idx = np.where(states == 1)[0]
    plt.scatter(crisis_idx, returns[crisis_idx], marker="o")

    plt.title("Returns with Regime Highlight")
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.legend()
    plt.show()


# ============================================================
# Correlation Plot
# ============================================================

def plot_rolling_correlation(
    x: np.ndarray,
    y: np.ndarray,
    window: int = 50
):
    from src.dynamics import rolling_correlation

    set_style()
    corr = rolling_correlation(x, y, window)

    plt.plot(corr)
    plt.title(f"Rolling Correlation (window={window})")
    plt.xlabel("Time")
    plt.ylabel("Correlation")
    plt.show()


# ============================================================
# Histogram & Tail
# ============================================================

def plot_return_distribution(
    returns: np.ndarray,
    bins: int = 50
):
    set_style()
    plt.hist(returns, bins=bins, density=True)
    plt.title("Return Distribution")
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.show()
import numpy as np


# -------------------------------------------------
# return series
# -------------------------------------------------

def returns(prices):
    """
    Simple returns.

    r_t = (P_t - P_{t-1}) / P_{t-1}
    """

    prices = np.asarray(prices)

    if len(prices) < 2:
        return np.array([])

    return np.diff(prices) / prices[:-1]


def log_returns(prices):
    """
    Log returns.

    r_t = log(P_t / P_{t-1})
    """

    prices = np.asarray(prices)

    if len(prices) < 2:
        return np.array([])

    return np.diff(np.log(prices))


# -------------------------------------------------
# volatility
# -------------------------------------------------

def realized_volatility(prices):
    """
    Realized volatility.

    sqrt( Var(r) )
    """

    r = returns(prices)

    if len(r) == 0:
        return 0.0

    return np.std(r)


def variance(prices):
    """
    Return variance.
    """

    r = returns(prices)

    if len(r) == 0:
        return 0.0

    return np.var(r)


# -------------------------------------------------
# rolling volatility
# -------------------------------------------------

def rolling_volatility(prices, window=50):
    """
    Rolling volatility time series.
    """

    r = returns(prices)

    if len(r) < window:
        return np.array([])

    vols = []

    for i in range(window, len(r) + 1):

        segment = r[i - window:i]

        vols.append(np.std(segment))

    return np.array(vols)


# -------------------------------------------------
# volatility clustering
# -------------------------------------------------

def volatility_clustering(prices, lag=1):
    """
    Autocorrelation of absolute returns.

    Used to detect volatility clustering.
    """

    r = np.abs(returns(prices))

    if len(r) <= lag:
        return 0.0

    r1 = r[:-lag]
    r2 = r[lag:]

    if np.std(r1) == 0 or np.std(r2) == 0:
        return 0.0

    return np.corrcoef(r1, r2)[0, 1]


# -------------------------------------------------
# drawdown metrics
# -------------------------------------------------

def drawdown(prices):
    """
    Drawdown series.

    DD_t = (P_t - max(P_0..P_t)) / max(P_0..P_t)
    """

    prices = np.asarray(prices)

    peak = prices[0]

    dd = []

    for p in prices:

        peak = max(peak, p)

        dd.append((p - peak) / peak)

    return np.array(dd)


def max_drawdown(prices):
    """
    Maximum drawdown.
    """

    dd = drawdown(prices)

    if len(dd) == 0:
        return 0.0

    return np.min(dd)


# -------------------------------------------------
# volatility regime indicator
# -------------------------------------------------

def volatility_regime(prices, window=100):
    """
    Estimate volatility regime.

    Returns rolling volatility and mean volatility.
    """

    vols = rolling_volatility(prices, window)

    if len(vols) == 0:
        return {
            "volatility": None,
            "mean_volatility": None
        }

    return {
        "volatility": vols,
        "mean_volatility": np.mean(vols)
    }
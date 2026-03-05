import numpy as np


# -------------------------------------------------
# return series
# -------------------------------------------------

def returns(prices):

    prices = np.asarray(prices)

    if len(prices) < 2:
        return np.array([])

    return np.diff(prices) / prices[:-1]


# -------------------------------------------------
# fat tails
# -------------------------------------------------

def kurtosis(returns):
    """
    Excess kurtosis of returns.

    Gaussian → 3
    Financial markets → usually > 3
    """

    r = np.asarray(returns)

    if len(r) == 0:
        return 0.0

    mean = np.mean(r)
    std = np.std(r)

    if std == 0:
        return 0.0

    return np.mean(((r - mean) / std) ** 4)


def tail_ratio(returns, q=0.95):
    """
    Ratio of extreme quantiles.
    """

    r = np.asarray(returns)

    if len(r) == 0:
        return 0.0

    upper = np.quantile(r, q)
    lower = np.quantile(r, 1 - q)

    if lower == 0:
        return 0.0

    return abs(upper / lower)


# -------------------------------------------------
# absence of linear autocorrelation
# -------------------------------------------------

def return_autocorrelation(returns, lag=1):

    r = np.asarray(returns)

    if len(r) <= lag:
        return 0.0

    x = r[:-lag]
    y = r[lag:]

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    return np.corrcoef(x, y)[0, 1]


# -------------------------------------------------
# volatility clustering
# -------------------------------------------------

def volatility_clustering(returns, lag=1):
    """
    Autocorrelation of absolute returns.
    """

    r = np.abs(np.asarray(returns))

    if len(r) <= lag:
        return 0.0

    x = r[:-lag]
    y = r[lag:]

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    return np.corrcoef(x, y)[0, 1]


# -------------------------------------------------
# long memory in order flow
# -------------------------------------------------

def order_flow_signs(trades):

    signs = []

    for t in trades:

        if t["side"] == "buy":
            signs.append(1)
        else:
            signs.append(-1)

    return np.array(signs)


def order_flow_autocorrelation(trades, lag=1):

    signs = order_flow_signs(trades)

    if len(signs) <= lag:
        return 0.0

    x = signs[:-lag]
    y = signs[lag:]

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    return np.corrcoef(x, y)[0, 1]


# -------------------------------------------------
# square-root impact law
# -------------------------------------------------

def impact_scaling(trades, mid_prices):
    """
    Measure relation between trade size and price impact.
    """

    if not trades or not mid_prices:
        return None

    impacts = []
    sizes = []

    for t, mid in zip(trades, mid_prices):

        price = t["price"]

        impacts.append(abs(price - mid))
        sizes.append(t["size"])

    impacts = np.array(impacts)
    sizes = np.array(sizes)

    if len(sizes) < 2:
        return None

    log_size = np.log(sizes)
    log_impact = np.log(impacts + 1e-12)

    slope = np.polyfit(log_size, log_impact, 1)[0]

    return slope
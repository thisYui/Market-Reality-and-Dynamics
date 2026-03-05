import numpy as np


# -------------------------------------------------
# inventory statistics
# -------------------------------------------------

def average_inventory(inventory_series):
    """
    Mean inventory over time.
    """

    inv = np.asarray(inventory_series)

    if len(inv) == 0:
        return 0.0

    return np.mean(inv)


def inventory_volatility(inventory_series):
    """
    Standard deviation of inventory.
    Measures inventory risk.
    """

    inv = np.asarray(inventory_series)

    if len(inv) == 0:
        return 0.0

    return np.std(inv)


def max_inventory(inventory_series):
    """
    Maximum absolute inventory reached.
    """

    inv = np.asarray(inventory_series)

    if len(inv) == 0:
        return 0.0

    return np.max(np.abs(inv))


# -------------------------------------------------
# inventory risk metrics
# -------------------------------------------------

def inventory_pressure(inventory_series, limit):
    """
    Inventory pressure relative to limit.

    pressure = |inventory| / limit
    """

    inv = np.asarray(inventory_series)

    if len(inv) == 0 or limit == 0:
        return 0.0

    return np.mean(np.abs(inv) / limit)


def inventory_skew(inventory_series):
    """
    Inventory bias.

    Positive → long bias
    Negative → short bias
    """

    inv = np.asarray(inventory_series)

    if len(inv) == 0:
        return 0.0

    return np.mean(inv)


# -------------------------------------------------
# inventory turnover
# -------------------------------------------------

def inventory_turnover(inventory_series):
    """
    Measures how frequently inventory changes.

    High turnover → active inventory management.
    """

    inv = np.asarray(inventory_series)

    if len(inv) < 2:
        return 0.0

    changes = np.abs(np.diff(inv))

    return np.mean(changes)


# -------------------------------------------------
# inventory mean reversion
# -------------------------------------------------

def inventory_mean_reversion(inventory_series):
    """
    Autocorrelation of inventory changes.

    Negative correlation indicates
    inventory control behavior.
    """

    inv = np.asarray(inventory_series)

    if len(inv) < 3:
        return 0.0

    changes = np.diff(inv)

    x = changes[:-1]
    y = changes[1:]

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    return np.corrcoef(x, y)[0, 1]


# -------------------------------------------------
# inventory risk score
# -------------------------------------------------

def inventory_risk_score(inventory_series, limit):
    """
    Overall inventory risk indicator.
    """

    pressure = inventory_pressure(inventory_series, limit)

    vol = inventory_volatility(inventory_series)

    return pressure * vol
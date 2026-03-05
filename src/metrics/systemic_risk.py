import numpy as np


# -------------------------------------------------
# leverage metrics
# -------------------------------------------------

def average_leverage(accounts, price):
    """
    Average leverage across accounts.
    """

    if not accounts:
        return 0.0

    leverages = [a.leverage(price) for a in accounts]

    return np.mean(leverages)


def leverage_distribution(accounts, price):
    """
    Return leverage values for each account.
    Useful for histograms.
    """

    return [a.leverage(price) for a in accounts]


def leverage_concentration(accounts, price):
    """
    Measure leverage concentration using Herfindahl index.
    """

    leverages = np.array([a.leverage(price) for a in accounts])

    if len(leverages) == 0:
        return 0.0

    total = np.sum(leverages)

    if total == 0:
        return 0.0

    weights = leverages / total

    return np.sum(weights ** 2)


# -------------------------------------------------
# margin stress
# -------------------------------------------------

def margin_stress(accounts, price):
    """
    Fraction of accounts close to margin call.
    """

    if not accounts:
        return 0.0

    stressed = 0

    for acc in accounts:

        if acc.margin_ratio(price) < acc.maintenance_margin * 1.5:
            stressed += 1

    return stressed / len(accounts)


def margin_call_count(accounts, price):
    """
    Number of accounts violating maintenance margin.
    """

    count = 0

    for acc in accounts:

        if acc.margin_call(price):
            count += 1

    return count


# -------------------------------------------------
# liquidation metrics
# -------------------------------------------------

def liquidation_volume(liquidations):
    """
    Total liquidation volume.
    """

    if not liquidations:
        return 0.0

    return sum(e["size"] for e in liquidations)


def liquidation_intensity(liquidations):
    """
    Number of liquidation events.
    """

    if not liquidations:
        return 0

    return len(liquidations)


def liquidation_imbalance(liquidations):
    """
    Net directional pressure from liquidations.
    """

    if not liquidations:
        return 0.0

    buy = 0
    sell = 0

    for e in liquidations:

        if e["side"] == "buy":
            buy += e["size"]
        else:
            sell += e["size"]

    total = buy + sell

    if total == 0:
        return 0.0

    return (buy - sell) / total


# -------------------------------------------------
# cascade detection
# -------------------------------------------------

def cascade_size(liquidations):
    """
    Total number of liquidations in cascade.
    """

    if not liquidations:
        return 0

    return len(liquidations)


def cascade_volume(liquidations):
    """
    Total volume liquidated in cascade.
    """

    if not liquidations:
        return 0.0

    return sum(e["size"] for e in liquidations)


# -------------------------------------------------
# market fragility
# -------------------------------------------------

def fragility_index(accounts, price):
    """
    Estimate structural fragility of the market.

    Combines:
    - leverage
    - margin stress
    """

    if not accounts:
        return 0.0

    lev = average_leverage(accounts, price)

    stress = margin_stress(accounts, price)

    return lev * stress


def systemic_stress(accounts, price, liquidations=None):
    """
    Overall systemic stress indicator.
    """

    fragility = fragility_index(accounts, price)

    liquidation_pressure = liquidation_volume(liquidations) if liquidations else 0

    return {
        "fragility": fragility,
        "liquidation_pressure": liquidation_pressure,
    }
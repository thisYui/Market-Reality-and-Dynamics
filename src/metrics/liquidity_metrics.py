import numpy as np


# -------------------------------------------------
# Amihud illiquidity
# -------------------------------------------------

def amihud_illiquidity(returns, volume):
    """
    Amihud illiquidity measure.

    ILLIQ = mean( |r_t| / volume_t )

    Higher value → less liquid market
    """

    returns = np.asarray(returns)
    volume = np.asarray(volume)

    if len(returns) == 0 or len(volume) == 0:
        return 0.0

    n = min(len(returns), len(volume))

    r = np.abs(returns[:n])
    v = volume[:n]

    v[v == 0] = np.nan

    return np.nanmean(r / v)


# -------------------------------------------------
# Kyle lambda (price impact coefficient)
# -------------------------------------------------

def kyle_lambda(returns, signed_volume):
    """
    Estimate Kyle's lambda.

    r_t = λ * q_t

    λ = Cov(r, q) / Var(q)
    """

    r = np.asarray(returns)
    q = np.asarray(signed_volume)

    if len(r) == 0 or len(q) == 0:
        return 0.0

    n = min(len(r), len(q))

    r = r[:n]
    q = q[:n]

    var_q = np.var(q)

    if var_q == 0:
        return 0.0

    cov = np.cov(r, q)[0, 1]

    return cov / var_q


# -------------------------------------------------
# orderbook depth
# -------------------------------------------------

def orderbook_depth(orderbook, levels=5):
    """
    Total liquidity near mid price.
    """

    bids = sorted(orderbook.bids.items(), reverse=True)[:levels]
    asks = sorted(orderbook.asks.items())[:levels]

    bid_depth = sum(v for _, v in bids)
    ask_depth = sum(v for _, v in asks)

    return {
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "total_depth": bid_depth + ask_depth,
    }


# -------------------------------------------------
# depth imbalance
# -------------------------------------------------

def depth_imbalance(orderbook, levels=5):
    """
    Bid vs ask depth imbalance.
    """

    depth = orderbook_depth(orderbook, levels)

    bid = depth["bid_depth"]
    ask = depth["ask_depth"]

    if bid + ask == 0:
        return 0.0

    return (bid - ask) / (bid + ask)


# -------------------------------------------------
# slippage
# -------------------------------------------------

def slippage(mid_price, execution_price):
    """
    Trading slippage relative to mid price.
    """

    if mid_price == 0:
        return 0.0

    return execution_price - mid_price


# -------------------------------------------------
# average slippage from trade list
# -------------------------------------------------

def average_slippage(trades, mid_prices):
    """
    Mean slippage across trades.
    """

    if not trades or not mid_prices:
        return 0.0

    values = []

    for t, mid in zip(trades, mid_prices):

        price = t["price"]

        values.append(price - mid)

    return np.mean(values)


# -------------------------------------------------
# depth weighted price
# -------------------------------------------------

def depth_weighted_price(orderbook, side="ask", levels=5):
    """
    Volume weighted price using orderbook depth.
    """

    if side == "ask":
        levels_data = sorted(orderbook.asks.items())[:levels]
    else:
        levels_data = sorted(orderbook.bids.items(), reverse=True)[:levels]

    total_volume = 0
    weighted_price = 0

    for price, volume in levels_data:

        weighted_price += price * volume
        total_volume += volume

    if total_volume == 0:
        return None

    return weighted_price / total_volume


# -------------------------------------------------
# impact curve
# -------------------------------------------------

def impact_curve(orderbook, sizes):
    """
    Estimate impact curve for various trade sizes.
    """

    mid = orderbook.mid_price()

    impacts = []

    for s in sizes:

        price = orderbook.estimate_market_buy_cost(s)

        if price is None or mid is None:
            impacts.append(None)
        else:
            impacts.append(price - mid)

    return impacts


# -------------------------------------------------
# liquidity resilience
# -------------------------------------------------

def liquidity_resilience(depth_series):
    """
    Measures how quickly liquidity recovers.

    depth_series: list of depth measurements over time
    """

    depth = np.asarray(depth_series)

    if len(depth) < 2:
        return 0.0

    changes = np.diff(depth)

    recovery = changes[changes > 0]

    if len(recovery) == 0:
        return 0.0

    return np.mean(recovery)
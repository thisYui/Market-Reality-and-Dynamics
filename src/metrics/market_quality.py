import numpy as np


# -------------------------------------------------
# spread metrics
# -------------------------------------------------

def effective_spread(trades, mid_prices):
    """
    Effective spread measures trading cost.

    ES = 2 * |trade_price - mid_price|

    trades: list of trades with {"price": ...}
    mid_prices: mid price at time of trade
    """

    if not trades or not mid_prices:
        return 0.0

    spreads = []

    for t, mid in zip(trades, mid_prices):

        price = t["price"]

        spreads.append(2 * abs(price - mid))

    return np.mean(spreads)


def quoted_spread(orderbook):
    """
    Current quoted spread from orderbook.
    """

    return orderbook.spread()


# -------------------------------------------------
# realized spread
# -------------------------------------------------

def realized_spread(trades, future_mid_prices, lag=1):
    """
    Realized spread measures market maker revenue.

    RS = 2 * (trade_price - mid_future)

    lag: number of steps ahead
    """

    if not trades or not future_mid_prices:
        return 0.0

    spreads = []

    for i in range(len(trades) - lag):

        trade_price = trades[i]["price"]

        future_mid = future_mid_prices[i + lag]

        spreads.append(2 * (trade_price - future_mid))

    return np.mean(spreads)


# -------------------------------------------------
# price impact
# -------------------------------------------------

def average_price_impact(trades, mid_prices):
    """
    Average price impact of trades.

    impact = trade_price - mid_price
    """

    if not trades or not mid_prices:
        return 0.0

    impacts = []

    for t, mid in zip(trades, mid_prices):

        impacts.append(t["price"] - mid)

    return np.mean(np.abs(impacts))


# -------------------------------------------------
# price efficiency
# -------------------------------------------------

def price_efficiency(prices):
    """
    Measures how close price process is to random walk.

    Using variance ratio idea.
    """

    prices = np.asarray(prices)

    if len(prices) < 3:
        return 0.0

    returns = np.diff(prices)

    var_1 = np.var(returns)

    k = 5

    if len(returns) < k:
        return 0.0

    agg_returns = prices[k:] - prices[:-k]

    var_k = np.var(agg_returns)

    if var_1 == 0:
        return 0.0

    return var_k / (k * var_1)


# -------------------------------------------------
# trade flow imbalance
# -------------------------------------------------

def trade_flow_imbalance(trades):
    """
    Order flow imbalance.

    OFI = (buy - sell) / total
    """

    if not trades:
        return 0.0

    buy = 0
    sell = 0

    for t in trades:

        size = t["size"]

        if t["side"] == "buy":
            buy += size
        else:
            sell += size

    total = buy + sell

    if total == 0:
        return 0.0

    return (buy - sell) / total


# -------------------------------------------------
# liquidity score
# -------------------------------------------------

def liquidity_score(orderbook):
    """
    Simple liquidity score combining spread and depth.
    """

    spread = orderbook.spread()

    bid_liq = orderbook.total_bid_liquidity()
    ask_liq = orderbook.total_ask_liquidity()

    depth = bid_liq + ask_liq

    if depth == 0 or spread is None:
        return 0.0

    return depth / spread
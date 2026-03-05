import numpy as np


# -------------------------------------------------
# spread metrics
# -------------------------------------------------

def quoted_spread(orderbook):
    """
    Best ask - best bid.
    """

    return orderbook.spread()


def relative_spread(orderbook):
    """
    Spread relative to mid price.
    """

    mid = orderbook.mid_price()

    if mid is None or mid == 0:
        return 0.0

    spread = orderbook.spread()

    if spread is None:
        return 0.0

    return spread / mid


# -------------------------------------------------
# orderbook imbalance
# -------------------------------------------------

def orderbook_imbalance(orderbook, levels=5):
    """
    Volume imbalance near the mid price.
    """

    bids = sorted(orderbook.bids.items(), reverse=True)[:levels]
    asks = sorted(orderbook.asks.items())[:levels]

    bid_vol = sum(v for _, v in bids)
    ask_vol = sum(v for _, v in asks)

    if bid_vol + ask_vol == 0:
        return 0.0

    return (bid_vol - ask_vol) / (bid_vol + ask_vol)


# -------------------------------------------------
# depth metrics
# -------------------------------------------------

def bid_depth(orderbook, levels=5):

    bids = sorted(orderbook.bids.items(), reverse=True)[:levels]

    return sum(v for _, v in bids)


def ask_depth(orderbook, levels=5):

    asks = sorted(orderbook.asks.items())[:levels]

    return sum(v for _, v in asks)


def total_depth(orderbook, levels=5):

    return bid_depth(orderbook, levels) + ask_depth(orderbook, levels)


# -------------------------------------------------
# depth slope
# -------------------------------------------------

def depth_slope(orderbook, side="ask", levels=5):
    """
    Measures how liquidity grows away from mid price.
    """

    if side == "ask":
        levels_data = sorted(orderbook.asks.items())[:levels]
    else:
        levels_data = sorted(orderbook.bids.items(), reverse=True)[:levels]

    prices = []
    volumes = []

    cumulative = 0

    for price, volume in levels_data:

        cumulative += volume

        prices.append(price)
        volumes.append(cumulative)

    if len(prices) < 2:
        return 0.0

    prices = np.array(prices)
    volumes = np.array(volumes)

    slope = np.polyfit(prices, volumes, 1)[0]

    return slope


# -------------------------------------------------
# book pressure
# -------------------------------------------------

def book_pressure(orderbook, levels=5):
    """
    Pressure exerted by orderbook imbalance.
    """

    imbalance = orderbook_imbalance(orderbook, levels)

    spread = orderbook.spread()

    if spread is None or spread == 0:
        return 0.0

    return imbalance / spread


# -------------------------------------------------
# trade metrics
# -------------------------------------------------

def trade_intensity(trades, window=50):
    """
    Number of trades per window.
    """

    if not trades:
        return 0

    return len(trades[-window:])


def average_trade_size(trades):

    if not trades:
        return 0.0

    sizes = [t["size"] for t in trades]

    return np.mean(sizes)


# -------------------------------------------------
# order flow imbalance
# -------------------------------------------------

def order_flow_imbalance(trades):
    """
    Signed volume imbalance.
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
# trade sign autocorrelation
# -------------------------------------------------

def trade_sign_autocorrelation(trades, lag=1):
    """
    Measures persistence of order flow.
    """

    if len(trades) <= lag:
        return 0.0

    signs = []

    for t in trades:

        if t["side"] == "buy":
            signs.append(1)
        else:
            signs.append(-1)

    signs = np.array(signs)

    x = signs[:-lag]
    y = signs[lag:]

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    return np.corrcoef(x, y)[0, 1]


# -------------------------------------------------
# micro price
# -------------------------------------------------

def micro_price(orderbook):
    """
    Microprice weighted by bid/ask liquidity.
    """

    bid = orderbook.best_bid()
    ask = orderbook.best_ask()

    if bid is None or ask is None:
        return None

    bid_vol = orderbook.bids[bid]
    ask_vol = orderbook.asks[ask]

    denom = bid_vol + ask_vol

    if denom == 0:
        return (bid + ask) / 2

    return (ask * bid_vol + bid * ask_vol) / denom
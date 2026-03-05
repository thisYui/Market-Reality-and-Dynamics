from collections import defaultdict


class OrderBook:
    """
    Aggregated limit order book.

    Responsibilities:
    - maintain bid / ask price levels
    - execute market orders
    - estimate execution cost
    - expose raw state for metrics
    """

    def __init__(self, tick_size=0.01):

        self.tick_size = tick_size

        self.bids = defaultdict(float)
        self.asks = defaultdict(float)

        self.trades = []

    # -------------------------------------------------
    # utilities
    # -------------------------------------------------

    def _round_price(self, price):

        return round(price / self.tick_size) * self.tick_size


    def spread(self):

        bid = self.best_bid()
        ask = self.best_ask()

        if bid is None or ask is None:
            return None

        return ask - bid

    # -------------------------------------------------
    # order placement
    # -------------------------------------------------

    def add_bid(self, price, size):

        price = self._round_price(price)

        self.bids[price] += size

    def add_ask(self, price, size):

        price = self._round_price(price)

        self.asks[price] += size

    def remove_bid(self, price, size):

        if price not in self.bids:
            return

        self.bids[price] -= size

        if self.bids[price] <= 0:
            del self.bids[price]

    def remove_ask(self, price, size):

        if price not in self.asks:
            return

        self.asks[price] -= size

        if self.asks[price] <= 0:
            del self.asks[price]

    # -------------------------------------------------
    # best prices
    # -------------------------------------------------

    def best_bid(self):

        if not self.bids:
            return None

        return max(self.bids)

    def best_ask(self):

        if not self.asks:
            return None

        return min(self.asks)

    def mid_price(self):

        bid = self.best_bid()
        ask = self.best_ask()

        if bid is None or ask is None:
            return None

        return (bid + ask) / 2

    # -------------------------------------------------
    # orderbook state
    # -------------------------------------------------

    def bid_levels(self):

        return sorted(self.bids.items(), reverse=True)

    def ask_levels(self):

        return sorted(self.asks.items())

    def snapshot(self):

        return {
            "bids": dict(self.bids),
            "asks": dict(self.asks),
            "best_bid": self.best_bid(),
            "best_ask": self.best_ask(),
            "mid_price": self.mid_price(),
        }

    # -------------------------------------------------
    # market order execution
    # -------------------------------------------------

    def execute_market_buy(self, size):

        remaining = size
        cost = 0

        for price in sorted(self.asks):

            available = self.asks[price]

            traded = min(remaining, available)

            cost += traded * price

            remaining -= traded

            self.remove_ask(price, traded)

            self.trades.append({
                "side": "buy",
                "price": price,
                "size": traded
            })

            if remaining <= 0:
                break

        if remaining > 0:
            raise ValueError("Not enough ask liquidity")

        return cost / size

    def execute_market_sell(self, size):

        remaining = size
        revenue = 0

        for price in sorted(self.bids, reverse=True):

            available = self.bids[price]

            traded = min(remaining, available)

            revenue += traded * price

            remaining -= traded

            self.remove_bid(price, traded)

            self.trades.append({
                "side": "sell",
                "price": price,
                "size": traded
            })

            if remaining <= 0:
                break

        if remaining > 0:
            raise ValueError("Not enough bid liquidity")

        return revenue / size

    # -------------------------------------------------
    # execution estimation
    # -------------------------------------------------

    def estimate_market_buy_cost(self, size):

        remaining = size
        cost = 0

        for price in sorted(self.asks):

            available = self.asks[price]

            traded = min(remaining, available)

            cost += traded * price

            remaining -= traded

            if remaining <= 0:
                break

        if remaining > 0:
            return None

        return cost / size

    def estimate_market_sell_revenue(self, size):

        remaining = size
        revenue = 0

        for price in sorted(self.bids, reverse=True):

            available = self.bids[price]

            traded = min(remaining, available)

            revenue += traded * price

            remaining -= traded

            if remaining <= 0:
                break

        if remaining > 0:
            return None

        return revenue / size

    # -------------------------------------------------
    # book reset
    # -------------------------------------------------

    def clear(self):

        self.bids.clear()
        self.asks.clear()

    def clear_trades(self):

        self.trades.clear()

    # -------------------------------------------------
    # liquidity totals
    # -------------------------------------------------

    def total_bid_liquidity(self):

        return sum(self.bids.values())

    def total_ask_liquidity(self):

        return sum(self.asks.values())

    def total_liquidity(self):

        return self.total_bid_liquidity() + self.total_ask_liquidity()
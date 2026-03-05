class MarketMaker:
    """
    Inventory-based market maker.

    Quotes around mid price while managing inventory risk.
    """

    def __init__(
        self,
        base_spread=1.0,
        inventory_limit=500,
        inventory_aversion=0.002,
        order_size=10,
    ):

        self.base_spread = base_spread
        self.inventory_limit = inventory_limit
        self.inventory_aversion = inventory_aversion
        self.order_size = order_size

        self.inventory = 0.0
        self.cash = 0.0

        self.last_bid = None
        self.last_ask = None

        self.trade_history = []

    # -------------------------------------------------
    # pricing
    # -------------------------------------------------

    def reservation_price(self, mid):
        """
        Inventory-adjusted reservation price.
        """

        return mid - self.inventory_aversion * self.inventory

    def optimal_spread(self):
        """
        Spread increases with inventory risk.
        """

        penalty = self.inventory_aversion * abs(self.inventory)

        return self.base_spread + penalty

    def quote(self, orderbook):

        mid = orderbook.mid_price()

        if mid is None:
            return None

        reservation = self.reservation_price(mid)

        spread = self.optimal_spread()

        bid = reservation - spread / 2
        ask = reservation + spread / 2

        return bid, ask

    # -------------------------------------------------
    # liquidity provision
    # -------------------------------------------------

    def place_quotes(self, orderbook):

        quotes = self.quote(orderbook)

        if quotes is None:
            return

        bid, ask = quotes

        # cancel previous quotes
        if self.last_bid is not None:
            orderbook.remove_bid(self.last_bid, self.order_size)

        if self.last_ask is not None:
            orderbook.remove_ask(self.last_ask, self.order_size)

        # place new quotes
        orderbook.add_bid(bid, self.order_size)
        orderbook.add_ask(ask, self.order_size)

        self.last_bid = bid
        self.last_ask = ask

    # -------------------------------------------------
    # trade processing
    # -------------------------------------------------

    def process_trade(self, trade):

        side = trade["side"]
        size = trade["size"]
        price = trade["price"]

        if side == "buy":
            # market buy → MM sold
            self.inventory -= size
            self.cash += price * size

        else:
            # market sell → MM bought
            self.inventory += size
            self.cash -= price * size

        self.trade_history.append(trade)

    # -------------------------------------------------
    # pnl
    # -------------------------------------------------

    def mark_to_market(self, price):

        return self.cash + self.inventory * price

    # -------------------------------------------------
    # inventory risk
    # -------------------------------------------------

    def inventory_ratio(self):

        if self.inventory_limit == 0:
            return 0

        return self.inventory / self.inventory_limit

    def inventory_pressure(self):

        return abs(self.inventory_ratio())

    # -------------------------------------------------
    # risk control
    # -------------------------------------------------

    def reduce_inventory(self, orderbook):

        if abs(self.inventory) < self.inventory_limit:
            return None

        size = min(abs(self.inventory), self.order_size)

        if self.inventory > 0:

            price = orderbook.execute_market_sell(size)

            self.inventory -= size
            self.cash += price * size

            side = "sell"

        else:

            price = orderbook.execute_market_buy(size)

            self.inventory += size
            self.cash -= price * size

            side = "buy"

        trade = {
            "type": "inventory_reduction",
            "side": side,
            "size": size,
            "price": price
        }

        self.trade_history.append(trade)

        return trade

    # -------------------------------------------------
    # snapshot
    # -------------------------------------------------

    def snapshot(self, price):

        return {
            "inventory": self.inventory,
            "cash": self.cash,
            "pnl": self.mark_to_market(price),
            "inventory_ratio": self.inventory_ratio(),
        }

    # -------------------------------------------------
    # reset
    # -------------------------------------------------

    def reset(self):

        self.inventory = 0
        self.cash = 0
        self.trade_history.clear()
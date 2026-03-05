import random
import numpy as np


class MarketSimulation:
    """
    Main market simulation engine.

    Orchestrates:
    - order flow
    - market maker
    - liquidity model
    - leverage accounts
    - liquidation cascades
    """

    def __init__(
        self,
        orderbook,
        liquidity_model,
        market_maker,
        order_flow,
        accounts=None,
        liquidation_engine=None,
        initial_price=100,
        seed=42,
    ):

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.orderbook = orderbook
        self.liquidity = liquidity_model
        self.market_maker = market_maker
        self.order_flow = order_flow

        self.accounts = accounts if accounts else []
        self.liquidation_engine = liquidation_engine

        self.price = initial_price
        self.time = 0

        self.price_history = [initial_price]
        self.mid_history = []
        self.trade_history = []
        self.events = []

        self.seed_orderbook(initial_price)

    # -------------------------------------------------
    # initial liquidity
    # -------------------------------------------------

    def seed_orderbook(self, mid_price, levels=5, size=100):
        """
        Seed initial book liquidity.
        """

        for i in range(1, levels + 1):

            bid = mid_price - i
            ask = mid_price + i

            self.orderbook.add_bid(bid, size)
            self.orderbook.add_ask(ask, size)

    # -------------------------------------------------
    # market step
    # -------------------------------------------------

    def step(self):

        step_events = []

        # -------------------------------------------------
        # 1. market maker quotes
        # -------------------------------------------------

        self.market_maker.place_quotes(self.orderbook)

        # -------------------------------------------------
        # 2. generate order flow
        # -------------------------------------------------

        event = self.order_flow.generate()

        side = event["side"]
        size = event["size"]

        # -------------------------------------------------
        # 3. execute trade
        # -------------------------------------------------

        if side == "buy":

            result = self.liquidity.execute_market_buy(
                self.orderbook, size
            )

        else:

            result = self.liquidity.execute_market_sell(
                self.orderbook, size
            )

        trade = {
            "type": "trade",
            "side": side,
            "size": size,
            "price": result["avg_price"],
            "signed_size": event["signed_size"],
        }

        self.trade_history.append(trade)
        step_events.append(trade)

        # -------------------------------------------------
        # 4. update market maker
        # -------------------------------------------------

        self.market_maker.process_trade(trade)

        # -------------------------------------------------
        # 5. update price
        # -------------------------------------------------

        new_price = self.orderbook.mid_price()

        if new_price is not None:
            self.price = new_price
            self.mid_history.append(new_price)

        # -------------------------------------------------
        # 6. mark-to-market accounts
        # -------------------------------------------------

        for acc in self.accounts:
            acc.mark_to_market(self.price)

        # -------------------------------------------------
        # 7. liquidation cascade
        # -------------------------------------------------

        if self.liquidation_engine:

            liquidation_events = self.liquidation_engine.cascade_step(
                self.accounts,
                self.orderbook,
            )

            if liquidation_events:

                self.trade_history.extend(liquidation_events)
                step_events.extend(liquidation_events)

        # -------------------------------------------------
        # 8. inventory risk control
        # -------------------------------------------------

        inv_trade = self.market_maker.reduce_inventory(
            self.orderbook
        )

        if inv_trade:

            self.trade_history.append(inv_trade)
            step_events.append(inv_trade)

        # -------------------------------------------------
        # 9. record price
        # -------------------------------------------------

        self.price_history.append(self.price)

        self.events.append(step_events)

        self.time += 1

    # -------------------------------------------------
    # run simulation
    # -------------------------------------------------

    def run(self, steps=1000):

        for _ in range(steps):
            self.step()

        return {
            "prices": self.price_history,
            "mid_prices": self.mid_history,
            "trades": self.trade_history,
            "events": self.events,
        }

    # -------------------------------------------------
    # exogenous shock
    # -------------------------------------------------

    def price_shock(self, magnitude):
        """
        Inject exogenous price shock.
        """

        self.price += magnitude
        self.price_history.append(self.price)

    # -------------------------------------------------
    # state snapshot
    # -------------------------------------------------

    def snapshot(self):

        return {
            "time": self.time,
            "price": self.price,
            "spread": self.orderbook.spread()
            if hasattr(self.orderbook, "spread")
            else None,
            "inventory": self.market_maker.inventory,
        }
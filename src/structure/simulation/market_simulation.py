import random
import numpy as np


class MarketSimulation:
    """
    Main market simulation engine.

    Coordinates:
    - order flow generation
    - market maker liquidity provision
    - order book execution
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

        self.time = 0
        self.price = initial_price

        # histories
        self.price_history = [initial_price]
        self.mid_history = [initial_price]
        self.trade_history = []
        self.events = []

        self.seed_orderbook(initial_price)

    # -------------------------------------------------
    # initial liquidity
    # -------------------------------------------------

    def seed_orderbook(self, mid_price, levels=5, size=20, tick=0.5):
        """
        Initialize liquidity around mid price.
        """

        for i in range(1, levels + 1):

            bid = mid_price - i * tick
            ask = mid_price + i * tick

            self.orderbook.add_bid(bid, size * i)
            self.orderbook.add_ask(ask, size * i)

    # -------------------------------------------------
    # single market step
    # -------------------------------------------------

    def step(self):

        step_events = []

        # -----------------------------------------
        # 1. market maker quotes
        # -----------------------------------------

        self.market_maker.place_quotes(self.orderbook)

        # -----------------------------------------
        # 2. generate order flow
        # -----------------------------------------

        event = self.order_flow.generate()

        side = event["side"]
        size = event["size"]

        # -----------------------------------------
        # 3. execute trade
        # -----------------------------------------

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
            "time": self.time
        }

        self.trade_history.append(trade)
        step_events.append(trade)

        # -----------------------------------------
        # 4. update market maker inventory
        # -----------------------------------------

        self.market_maker.process_trade(trade)

        # -----------------------------------------
        # 5. update mid price
        # -----------------------------------------

        mid = self.orderbook.mid_price()

        if mid is not None:
            self.price = mid

        self.mid_history.append(self.price)

        # -----------------------------------------
        # 6. mark-to-market accounts
        # -----------------------------------------

        for acc in self.accounts:
            acc.mark_to_market(self.price)

        # -----------------------------------------
        # 7. liquidation cascade
        # -----------------------------------------

        if self.liquidation_engine:

            liquidation_events = self.liquidation_engine.cascade_step(
                self.accounts,
                self.orderbook,
            )

            if liquidation_events:

                for e in liquidation_events:
                    e["time"] = self.time

                self.trade_history.extend(liquidation_events)
                step_events.extend(liquidation_events)

        # -----------------------------------------
        # 8. inventory risk control
        # -----------------------------------------

        inv_trade = self.market_maker.reduce_inventory(
            self.orderbook
        )

        if inv_trade:

            inv_trade["time"] = self.time

            self.trade_history.append(inv_trade)
            step_events.append(inv_trade)

        # -----------------------------------------
        # 9. record price
        # -----------------------------------------

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
        Inject external price shock.
        """

        self.price += magnitude
        self.price_history.append(self.price)
        self.mid_history.append(self.price)

    # -------------------------------------------------
    # state snapshot
    # -------------------------------------------------

    def snapshot(self):

        spread = None
        if hasattr(self.orderbook, "spread"):
            spread = self.orderbook.spread()

        return {
            "time": self.time,
            "price": self.price,
            "spread": spread,
            "inventory": self.market_maker.inventory,
        }
class LiquidationEngine:
    """
    Handles forced liquidation when margin constraints are violated.

    Responsibilities:
    - detect margin violations
    - execute forced liquidations
    - propagate liquidation cascades
    """

    def __init__(self, liquidation_fraction=1.0):

        # fraction of position liquidated per event
        self.liquidation_fraction = liquidation_fraction

    # -------------------------------------------------
    # margin checks
    # -------------------------------------------------

    def needs_liquidation(self, account, price):

        return account.margin_call(price)

    # -------------------------------------------------
    # liquidation size
    # -------------------------------------------------

    def liquidation_size(self, account):

        if account.position == 0:
            return 0

        return abs(account.position) * self.liquidation_fraction

    # -------------------------------------------------
    # execute liquidation
    # -------------------------------------------------

    def liquidate(self, account, orderbook):

        if account.position == 0:
            return None

        size = self.liquidation_size(account)

        if size == 0:
            return None

        side = "sell" if account.position > 0 else "buy"

        # execution
        if side == "sell":
            price = orderbook.execute_market_sell(size)
        else:
            price = orderbook.execute_market_buy(size)

        # update position
        if account.position > 0:
            account.position -= size
        else:
            account.position += size

        # realized PnL
        pnl = size * (price - account.entry_price)

        if side == "sell":
            pnl = size * (price - account.entry_price)
        else:
            pnl = size * (account.entry_price - price)

        account.equity += pnl

        if account.position == 0:
            account.entry_price = None

        return {
            "type": "liquidation",
            "side": side,
            "size": size,
            "price": price,
            "pnl": pnl,
        }

    # -------------------------------------------------
    # cascade step
    # -------------------------------------------------

    def cascade_step(self, accounts, orderbook):
        """
        Execute one cascade iteration across all accounts.
        """

        events = []

        price = orderbook.mid_price()

        if price is None:
            return events

        for account in accounts:

            if self.needs_liquidation(account, price):

                result = self.liquidate(account, orderbook)

                if result:
                    events.append(result)

        return events

    # -------------------------------------------------
    # full cascade
    # -------------------------------------------------

    def run_full_cascade(self, accounts, orderbook, max_rounds=10):
        """
        Run liquidation cascade until no accounts violate margin.
        """

        all_events = []

        for _ in range(max_rounds):

            events = self.cascade_step(accounts, orderbook)

            if not events:
                break

            all_events.extend(events)

        return all_events
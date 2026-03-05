class LeverageAccount:
    """
    Balance sheet model for leveraged traders.

    Tracks:
    - equity
    - position
    - leverage
    - margin ratio

    Designed for simulation + systemic risk metrics.
    """

    def __init__(
        self,
        equity,
        leverage_limit=5.0,
        maintenance_margin=0.25,
    ):

        self.initial_equity = equity
        self.equity = equity

        self.leverage_limit = leverage_limit
        self.maintenance_margin = maintenance_margin

        self.position = 0.0
        self.entry_price = None

    # -------------------------------------------------
    # position utilities
    # -------------------------------------------------

    def position_value(self, price):

        return self.position * price

    def exposure(self, price):

        """
        Absolute market exposure.
        """

        return abs(self.position_value(price))

    # -------------------------------------------------
    # balance sheet
    # -------------------------------------------------

    def assets(self, price):

        return self.exposure(price)

    def liabilities(self, price):

        assets = self.assets(price)

        return max(assets - self.equity, 0)

    def leverage(self, price):

        if self.equity <= 0:
            return float("inf")

        return self.assets(price) / self.equity

    def margin_ratio(self, price):

        assets = self.assets(price)

        if assets == 0:
            return float("inf")

        return self.equity / assets

    # -------------------------------------------------
    # trading
    # -------------------------------------------------

    def open_position(self, size, price):
        """
        Open or increase position.

        size > 0 → long
        size < 0 → short
        """

        if self.entry_price is None:
            self.entry_price = price
        else:
            # average entry price
            total = self.position + size

            if total != 0:

                self.entry_price = (
                    self.entry_price * self.position
                    + price * size
                ) / total

        self.position += size

    def close_position(self, price):

        if self.position == 0 or self.entry_price is None:
            return 0.0

        pnl = self.position * (price - self.entry_price)

        self.equity += pnl

        self.position = 0
        self.entry_price = None

        return pnl

    # -------------------------------------------------
    # mark-to-market
    # -------------------------------------------------

    def unrealized_pnl(self, price):

        if self.position == 0 or self.entry_price is None:
            return 0.0

        return self.position * (price - self.entry_price)

    def mark_to_market(self, price):

        """
        Update equity using unrealized PnL.
        """

        pnl = self.unrealized_pnl(price)

        self.equity = self.initial_equity + pnl

        return pnl

    # -------------------------------------------------
    # margin checks
    # -------------------------------------------------

    def margin_call(self, price):

        return self.margin_ratio(price) < self.maintenance_margin

    def max_position(self, price):

        """
        Maximum allowed position under leverage constraint.
        """

        max_assets = self.equity * self.leverage_limit

        if price == 0:
            return 0

        return max_assets / price

    # -------------------------------------------------
    # systemic risk utilities
    # -------------------------------------------------

    def liquidation_size(self, price):

        """
        Position size to liquidate if margin violated.
        """

        return abs(self.position)

    # -------------------------------------------------
    # snapshot
    # -------------------------------------------------

    def snapshot(self, price):

        return {
            "equity": self.equity,
            "position": self.position,
            "exposure": self.exposure(price),
            "leverage": self.leverage(price),
            "margin_ratio": self.margin_ratio(price),
            "unrealized_pnl": self.unrealized_pnl(price),
        }

    def reset(self):

        self.equity = self.initial_equity
        self.position = 0
        self.entry_price = None
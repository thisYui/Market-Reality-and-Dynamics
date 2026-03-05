import numpy as np


class LiquidityModel:
    """
    Liquidity and price impact model.

    Responsibilities:
    - execute market orders
    - estimate slippage
    - compute theoretical price impact
    """

    def __init__(self, impact_coefficient=1e-4):

        self.impact_coefficient = impact_coefficient

    # -------------------------------------------------
    # theoretical impact models
    # -------------------------------------------------

    def linear_impact(self, size):
        """
        Kyle-style linear impact.

        ΔP = λ Q
        """

        return self.impact_coefficient * size

    def square_root_impact(self, size, daily_volume=1e6):
        """
        Empirical square-root impact law.

        ΔP ≈ σ √(Q / V)
        """

        if daily_volume <= 0:
            return 0.0

        return np.sqrt(size / daily_volume)

    # -------------------------------------------------
    # order execution
    # -------------------------------------------------

    def execute_market_buy(self, orderbook, size):

        mid_before = orderbook.mid_price()

        price = orderbook.execute_market_buy(size)

        mid_after = orderbook.mid_price()

        return {
            "side": "buy",
            "size": size,
            "avg_price": price,
            "mid_before": mid_before,
            "mid_after": mid_after,
            "impact": None if mid_before is None else price - mid_before,
        }

    def execute_market_sell(self, orderbook, size):

        mid_before = orderbook.mid_price()

        price = orderbook.execute_market_sell(size)

        mid_after = orderbook.mid_price()

        return {
            "side": "sell",
            "size": size,
            "avg_price": price,
            "mid_before": mid_before,
            "mid_after": mid_after,
            "impact": None if mid_before is None else mid_before - price,
        }

    # -------------------------------------------------
    # slippage estimation
    # -------------------------------------------------

    def estimate_buy_slippage(self, orderbook, size):

        mid = orderbook.mid_price()

        if mid is None:
            return None

        price = orderbook.estimate_market_buy_cost(size)

        if price is None:
            return None

        return price - mid

    def estimate_sell_slippage(self, orderbook, size):

        mid = orderbook.mid_price()

        if mid is None:
            return None

        price = orderbook.estimate_market_sell_revenue(size)

        if price is None:
            return None

        return mid - price

    # -------------------------------------------------
    # execution simulation
    # -------------------------------------------------

    def simulate_execution(self, orderbook, side, size):
        """
        Simulate execution without modifying book.
        """

        mid = orderbook.mid_price()

        if side == "buy":

            price = orderbook.estimate_market_buy_cost(size)

            if price is None:
                return None

            return {
                "side": "buy",
                "size": size,
                "avg_price": price,
                "impact": None if mid is None else price - mid
            }

        else:

            price = orderbook.estimate_market_sell_revenue(size)

            if price is None:
                return None

            return {
                "side": "sell",
                "size": size,
                "avg_price": price,
                "impact": None if mid is None else mid - price
            }
import random


class OrderFlow:
    """
    Base class for order flow generators.

    Generates market order events used by the simulation engine.
    """

    def generate(self):
        raise NotImplementedError

    def _event(self, side, size):

        signed_size = size if side == "buy" else -size

        return {
            "type": "market",
            "side": side,
            "size": size,
            "signed_size": signed_size
        }


class RandomOrderFlow(OrderFlow):

    """
    IID order flow.

    Order signs and sizes are independent.
    """

    def __init__(
        self,
        buy_probability=0.5,
        min_size=1,
        max_size=20,
    ):

        self.buy_probability = buy_probability
        self.min_size = min_size
        self.max_size = max_size

    def generate(self):

        side = "buy" if random.random() < self.buy_probability else "sell"

        size = random.randint(self.min_size, self.max_size)

        return self._event(side, size)


class PersistentOrderFlow(OrderFlow):

    """
    Generates order flow with sign autocorrelation.

    Captures long-memory in order signs.
    """

    def __init__(
        self,
        persistence=0.8,
        min_size=1,
        max_size=20,
    ):

        self.persistence = persistence
        self.min_size = min_size
        self.max_size = max_size

        self.last_side = random.choice(["buy", "sell"])

    def generate(self):

        if random.random() < self.persistence:
            side = self.last_side
        else:
            side = "buy" if self.last_side == "sell" else "sell"

        size = random.randint(self.min_size, self.max_size)

        self.last_side = side

        return self._event(side, size)


class MetaOrderFlow(OrderFlow):

    """
    Simulates meta-orders split into many smaller trades.

    Produces bursts of same-side trading.
    """

    def __init__(
        self,
        meta_order_probability=0.05,
        meta_order_size=200,
        child_order_size=10,
    ):

        self.meta_order_probability = meta_order_probability
        self.meta_order_size = meta_order_size
        self.child_order_size = child_order_size

        self.remaining_meta = 0
        self.meta_side = None

    def generate(self):

        # continue executing current meta order
        if self.remaining_meta > 0:

            size = min(self.child_order_size, self.remaining_meta)

            self.remaining_meta -= size

            return self._event(self.meta_side, size)

        # start new meta order
        if random.random() < self.meta_order_probability:

            self.meta_side = random.choice(["buy", "sell"])
            self.remaining_meta = self.meta_order_size

            size = min(self.child_order_size, self.remaining_meta)

            self.remaining_meta -= size

            return self._event(self.meta_side, size)

        # normal random trade
        side = random.choice(["buy", "sell"])
        size = random.randint(1, self.child_order_size)

        return self._event(side, size)
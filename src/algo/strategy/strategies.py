import warnings
from collections import deque
from itertools import tee
import numpy as np


class SimpleStrategyEMA:
    """Simple strategy based on monitoring of EMA"""

    def __init__(self, analytic_provider):
        self.analytic_provider = analytic_provider
        # use a dictionary with {assetid: weight} format for more sophisticated strategies?
        self.buy = set()
        self.sell = set()

    def score(self, assetid):
        """Return list of tuples with coins to long"""
        # for assetid in self.analytic_provider.expavg_long:
        diff = self.analytic_provider.expavg_long[assetid] - self.analytic_provider.expavg_short[assetid]
        # if difference becomes negative and was previously positive, buy coin, and vice vera
        if diff[-1] < 0:
            self.sell.discard(assetid)
            self.buy.add(assetid)
        if diff[-1] > 0:
            self.buy.discard(assetid)
            self.sell.add(assetid)


class StrategyArbitrage:
    """An arbitrage strategy looking for profitable loops in the exchange"""

    def __init__(self, pool_graph, gain_threshold=1.02):
        self.pool_graph = pool_graph
        self.swap_loops = list()
        self.gain_threshold = gain_threshold

    def score(self):
        self.pool_graph.update_candidates()
        self.swap_loops = list()
        self.max_gains = list()
        for i in reversed(self.pool_graph.candidate_indices):
            if self.pool_graph.cycles_gain[i] < self.gain_threshold:
                break
            c = self.pool_graph.cycles[i]
            # rotate the array until algo (0) is at the end
            if c[-1] != 0:
                if 0 not in c:
                    warnings.warn(f"Warning: Ignoring cycle {c} because it has no ALGO swap")
                    continue
                cd = deque(c)
                while cd[-1] != 0:
                    cd.rotate(1)
                c = np.array(cd)
            # add the last element at the start to close the loop
            c = np.append(c[-1], c)
            # create a list of tuples for the swaps
            # [a1, a2, a3, a1] -> [(a1, a2), (a2, a3), (a3, a1)]
            a, b = tee(c)
            next(b, None)
            c = list(zip(a, b))
            # add the current list of swaps to the full list
            self.swap_loops.append(c)
            self.max_gains.append(self.pool_graph.cycles_gain[i])

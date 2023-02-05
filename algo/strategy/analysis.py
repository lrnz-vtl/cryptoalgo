from algo.strategy.features import exp_average
from algo.stream.marketstream import MultiPoolStream
from itertools import tee
import networkx as nx
import numpy as np
import warnings
from collections import deque


class AnalyticProvider:
    def __init__(self, datastore, time_scale_long=10000, time_scale_short=1000):
        """Take a datastore and compute two EMA to be used in strategy"""
        self.datastore = datastore
        self.time_scale_long = time_scale_long
        self.time_scale_short = time_scale_short
        self.expavg_long = {}
        self.expavg_short = {}
        self.update()

    def update(self):
        """Using current datastore state, compute relevant metrics"""
        for assetid in self.datastore:
            price=self.datastore[assetid].slow.price_history['price']
            #time=self.datastore[assetid].slow.price_history.index.view(np.int64).astype(float)/10**9
            self.expavg_long[assetid]=exp_average(price,self.time_scale_long)
            self.expavg_short[assetid]=exp_average(price,self.time_scale_short)


class PoolGraph:
    def __init__(self, assetPairs, client, num_trades, logger=None, sample_interval=1, log_interval=5):
        self.client = client
        self.num_trades = num_trades
        self.update_graph(assetPairs)

        print(f'Started PoolGraph with {len(self.useful_pairs)} pool streams')
        self.mps = MultiPoolStream(assetPairs=self.useful_pairs, client=client, sample_interval=sample_interval,
                                   log_interval=log_interval, logger=logger)

    def update_graph(self, assetPairs):
        
        g = nx.Graph()
        for assetPair in assetPairs:
            if self.client.fetch_pool(assetPair[0], assetPair[1]).exists:
                g.add_edge(assetPair[0], assetPair[1])
            else:
                g.add_edge(assetPair[0], assetPair[1])
        
        cycles = nx.cycle_basis(g)
        self.useful_pairs = list()
        for c in cycles:
            if c[-1] != 0:
                if 0 not in c:
                    warnings.warn(f"Ignoring cycle {c} because it has no ALGO swap")
                    continue
                cd = deque(c)
                while cd[-1] != 0:
                    cd.rotate(1)
                c = np.list(cd)
            c = [c[-1]] + c
            a, b = tee(c)
            next(b, None)
            c = list(zip(a, b))

            # test that the pool is reasonable (for 0.1 output algo, we input at least 0.001 and no more than 2)
            val = 1_000_000
            for p in reversed(c):
                pool = self.client.fetch_pool(p[0],p[1])
                r1 = pool.asset1_reserves
                r2 = pool.asset2_reserves
                if (pool.asset1.id==p[1]):
                    r1, r2 = r2, r1
                val = 0 if r2==0 else val*0.997 * r1/r2 *  (1 - val / r2)
            if val > 500_000 and val < 2_000_000:
                self.useful_pairs+=c

        # redo the graph with only useful nodes
        self.graph = nx.Graph()
        for assetPair in self.useful_pairs:
            self.graph.add_edge(assetPair[0],assetPair[1])
        # find cycles of interest within the graph
        self.cycles = np.array(nx.cycle_basis(self.graph),dtype=object)
        self.candidate_indices = list()
        
    def update_candidates(self):
        self.cycles_gain = list()
        for cycle in self.cycles:
            price=1.0
            for i in range(len(cycle)):
                asset1 = cycle[i]
                asset2 = cycle[(i+1)%len(cycle)]
                if 'price' not in self.graph[asset1][asset2]:
                    price=np.nan
                    continue
                if (self.graph[asset1][asset2]['asset2']==asset1):
                    price=price/self.graph[asset1][asset2]['price']
                else:
                    price=price*self.graph[asset1][asset2]['price']
            self.cycles_gain.append(price)
        self.cycles_gain = np.array(self.cycles_gain)
        ind=np.argpartition(np.nan_to_num(self.cycles_gain), -self.num_trades)[-self.num_trades:]
        self.candidate_indices = ind[np.argsort(self.cycles_gain[ind])]
        
    async def run(self):
        async for row in self.mps.run():
            if row:
                self.graph[row.asset1][row.asset2]['price']=row.price
                self.graph[row.asset1][row.asset2]['asset1_reserves']=row.asset1_reserves
                self.graph[row.asset1][row.asset2]['asset2_reserves']=row.asset2_reserves
                self.graph[row.asset1][row.asset2]['asset1']=row.asset1
                self.graph[row.asset1][row.asset2]['asset2']=row.asset2
            print(row)

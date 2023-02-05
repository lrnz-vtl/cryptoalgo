from abc import ABC, abstractmethod
import warnings
from algo.universe import assets
import numpy as np
import asyncio


class TradingEngineBase(ABC):
    def __init__(self, strategy, portfolio, budget, swapper, trading_scale):
        self.portfolio = portfolio
        self.budget = budget
        self.strategy = strategy
        self.swapper = swapper
        self.trading_scale = trading_scale

    async def run_trading(self):
        """Run the main trading engine"""
        print('Running trading engine')
        while True:
            await asyncio.sleep(self.trading_scale)
            self.trading_step()

    @abstractmethod
    def trading_step(self):
        pass


class TradingEngineEMA(TradingEngineBase):
    def __init__(self, strategy, portfolio, budget, swapper, asset_ids=assets.assets, trading_scale=15):
        super().__init__(strategy, portfolio, budget, swapper, trading_scale)
        self.budget = self.budget / len(asset_ids)
        self.asset_ids = asset_ids

    def trading_step(self):
        """Evaluate the strategy and perform swaps"""
        print('Performing a trading step')
        for assetid in self.asset_ids:
            self.strategy.score(assetid)
            if assetid in self.strategy.buy:
                if self.portfolio[assetid] == 0:
                    print('Buying asset', assetid)
                    quantity = min(self.portfolio[0], self.budget)
                    self.swapper.swap(0, assetid, quantity, target_price=0.0)
            elif assetid in self.strategy.sell:
                if self.portfolio[assetid] > 0:
                    print('Selling asset', assetid)
                    quantity = self.portfolio[assetid]
                    self.swapper.swap(assetid, 0, quantity, target_price=0.0)


class TradingEngineArbitrage(TradingEngineBase):
    def __init__(self, strategy, portfolio, budget, swapper, f_threshold=0.90,
                 trading_scale=30, min_gain=20000):
        super().__init__(strategy, portfolio, budget, swapper, trading_scale)
        # set the threshold of total swapping fees
        self.f_threshold = f_threshold
        self.slippage = 0.005
        self.excess_min = 2000
        self.min_gain = min_gain
        self.skip_optin = False

    def trading_step(self):
        f = 1 - 0.003
        # find the candidates for swap loops
        self.strategy.score()
        # now go through them and perform the ones that are viable
        for i, (maxgain, swap_loop) in enumerate(zip(self.strategy.max_gains, self.strategy.swap_loops)):
            if np.isnan(maxgain):
                continue
            pools = list()
            q1max = self.budget
            input_amounts = list()
            max_prices = list()
            print('\nAttempting a trade for gain', maxgain, 'with loop:', swap_loop)
            for swap in swap_loop:
                pool = self.swapper.client.fetch_pool(int(swap[0]), int(swap[1]))
                pools.append(pool)
                r1 = pool.asset1_reserves
                r2 = pool.asset2_reserves
                if (pool.asset1.id == swap[1]):
                    r1, r2 = r2, r1
                max_prices.append(r2 / r1)
                # the quantity q2 we obtain for a quantity q1, to second order is given by
                #    q2 = q1 f r2/r1 (1 - q1 f/r1)
                # (while the quantity for q1 we obtain for a quantity q2 of output is:)
                #    q1 = q2 f r1/r2 (1 - q2 / r2)
                # where f=1-0.003 (to account for swap fees)
                # Therefore, the price is given by
                #    p = f r2/r1 (1 - q1 f/r1)
                # which converges to f r2/r1 when q1<<r1
                # the total "fee" multiplier we pay in the swap is thus given by
                #    f_tot = f - q1 f^2 / r1
                # If we fix f_tot to some threshold value, the maximum amount of
                # q1 that can be swapped is then given by
                #    q1max = r1/f^2 * (f - f_tot)
                # now set fmax, which has to be more than the fixed threshold
                # and of half the potential gain
                fmax = max(self.f_threshold, 2.0 / (1 + maxgain))
                q1max = min(r1 / f ** 2 * (f - fmax), q1max)
                # print('Considering swap',swap,' with price ',r2/r1,': a1(',q1max,') => a2(',
                #       q1max * f * r2/r1 * (1 - f*q1max/r1),')')
                q2max = q1max * f * r2 / r1 * (1 - f * q1max / r1)
                input_amounts.append(q1max)
                q2max_prev = q1max
                q1max = q2max
            # we need to back propagate to the start of the chain and set the budget accordingly
            for k in range(len(pools) - 2, -1, -1):
                # r1_prev = pools[i].asset1_reserves
                # r2_prev = pools[i].asset2_reserves
                # if (pools[i].asset1.id==swap_loop[i][1]):
                #     r1_prev, r2_prev = r2_prev, r1_prev
                # q1max_prev = f * q2max_prev * r1_prev/r2_prev * (1 - q2max_prev/r2_prev)
                # input_amounts[i] = q1max_prev
                asset2 = self.swapper.client.fetch_asset(int(swap_loop[k][1]))
                q1max_prev = pools[k].fetch_fixed_output_swap_quote(asset2(int(q2max_prev)),
                                                                    slippage=0.005).amount_in.amount
                input_amounts[k] = q1max_prev
                q2max_prev = q1max_prev
            input_amounts.append(q2max)
            print(f'Trading chain: {input_amounts} => {input_amounts[-1] - input_amounts[0]}')
            if (input_amounts[-1] - input_amounts[0] > self.min_gain) and input_amounts[0] > 0:
                # the input amount of algorand is larger than what we get out! Free money?!
                input_amount = input_amounts[0]
                for ip in range(len(pools)):
                    success = self.swapper.swap(swap_loop[ip][0], swap_loop[ip][1],
                                                input_amount, max_prices[ip] * self.f_threshold,
                                                slippage=self.slippage, excess_min=self.excess_min,
                                                skip_optin=self.skip_optin)
                    if not success:
                        # if the swap failed for any reason, we want to revert back the chain and exit
                        for j in range(ip - 1, -1, -1):
                            suc = self.swapper.swap(swap_loop[j][1], swap_loop[j][0],
                                                    input_amount, 0,
                                                    slippage=self.slippage, excess_min=self.excess_min,
                                                    skip_optin=True)
                            input_amount = suc
                        warnings.warn('Failed a swap loop, tried reverting and continuing')
                        break
                    input_amount = success
            # # now update all the graph edges for the pools that we already checked
            for p in pools:
                if not 'asset1' in self.strategy.pool_graph.graph[p.asset1.id][p.asset2.id]:
                    warnings.warn('Graph had no price data for a swap that was proposed')
                    continue
                if self.strategy.pool_graph.graph[p.asset1.id][p.asset2.id]['asset1'] == p.asset1.id:
                    price = p.asset2_reserves / p.asset1_reserves
                    self.strategy.pool_graph.graph[p.asset1.id][p.asset2.id]['price'] = price
                    self.strategy.pool_graph.graph[p.asset1.id][p.asset2.id]['asset1_reserves'] = p.asset1_reserves
                    self.strategy.pool_graph.graph[p.asset1.id][p.asset2.id]['asset2_reserves'] = p.asset2_reserves
                else:
                    price = p.asset1_reserves / p.asset2_reserves
                    self.strategy.pool_graph.graph[p.asset1.id][p.asset2.id]['price'] = price
                    self.strategy.pool_graph.graph[p.asset1.id][p.asset2.id]['asset1_reserves'] = p.asset2_reserves
                    self.strategy.pool_graph.graph[p.asset1.id][p.asset2.id]['asset2_reserves'] = p.asset1_reserves

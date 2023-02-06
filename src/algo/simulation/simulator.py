from __future__ import annotations

import copy
import datetime
import logging
from algo.trading.trades import TradeInfo
from algo.trading.impact import GlobalPositionAndImpactState, StateLog
from algo.optimizer.base import BaseOptimizer
from algo.trading.signalprovider import PriceSignalProvider
from algo.blockchain.stream import PoolState, PriceUpdate
from algo.universe.universe import SimpleUniverse
from typing import Callable, Generator, Any, Optional, Type
from algo.blockchain.utils import int_to_tzaware_utc_datetime
from algo.trading.swapper import SimulationSwapper
from algo.engine.base import BaseEngine, LAG_TRADE_LIMIT_SECONDS


class Simulator(BaseEngine):

    def __init__(self,
                 signal_providers: dict[int, PriceSignalProvider],
                 pos_impact_state: GlobalPositionAndImpactState,
                 universe: SimpleUniverse,
                 seed_time: datetime.timedelta,
                 price_stream: Generator[PriceUpdate, Any, Any],
                 simulation_step_seconds: int,
                 make_optimizer: Callable[[int], BaseOptimizer],
                 slippage: float = 0
                 ):

        self.asset_ids = [pool.asset1_id for pool in universe.pools]
        assert all(pool.asset2_id == 0 for pool in universe.pools)

        self.optimizers = {asset_id: make_optimizer(asset_id) for asset_id in self.asset_ids}

        self.logger = logging.getLogger(__name__)
        self.signal_providers = signal_providers

        self.initial_pos_impact_state = copy.deepcopy(pos_impact_state)
        self.pos_impact_state = pos_impact_state

        self.simulation_step = datetime.timedelta(seconds=simulation_step_seconds)
        self.price_stream = price_stream
        # The amount of time we spend seeding the prices and signals without trading
        self.seed_time = seed_time
        self.prices: dict[int, PoolState] = {}
        self.swapper = {aid: SimulationSwapper() for aid in self.asset_ids}
        self.last_update_times = {}

        self.slippage = slippage

    def current_time_prov(self) -> datetime.datetime:
        return self._sim_time

    def update_market_state(self, time, asset_id, price_update):
        self.prices[asset_id] = price_update
        asa_price_mualgo = price_update.asset2_reserves / price_update.asset1_reserves
        self.signal_providers[asset_id].update(time, asa_price_mualgo)

    def run(self, end_time: datetime.datetime,
            log_trade: Callable[[TradeInfo], None],
            log_state: Callable[[StateLog], None]):

        # Takes values on the times where we run the trading loop
        self._sim_time: Optional[datetime.datetime] = None

        initial_time: Optional[datetime.datetime] = None

        for x in self.price_stream:
            self.logger.debug(f'{x}')
            assert x.asset_ids[1] == 0
            asset_id, price_update = x.asset_ids[0], x.price_update
            # Time of the price update
            time = int_to_tzaware_utc_datetime(x.price_update.time)

            if initial_time is None:
                initial_time = time
                self._sim_time = time

            assert time+datetime.timedelta(seconds=LAG_TRADE_LIMIT_SECONDS) >= self._sim_time, f"{time}, {self._sim_time}"

            while self._sim_time + self.simulation_step < time + datetime.timedelta(seconds=LAG_TRADE_LIMIT_SECONDS):

                self._sim_time = self._sim_time + self.simulation_step
                self.last_market_state_update = self._sim_time - datetime.timedelta(seconds=LAG_TRADE_LIMIT_SECONDS)

                # Trade only if we are not seeding
                if self._sim_time - initial_time > self.seed_time:
                    self.logger.debug(f'Entering trading loop at sim time {self._sim_time}')
                    self.trade_loop(log_trade, log_state)
                else:
                    self.logger.debug(f'Still seeding at sim time {self._sim_time}')

            # End the simulation
            if time > end_time:
                break

            self.last_update_times[asset_id] = time
            self.update_market_state(time, asset_id, price_update)

from __future__ import annotations
import logging
from algo.trading.trades import TradeInfo
from algo.trading.impact import GlobalPositionAndImpactState, StateLog, PositionAndImpactState, ASAImpactState, \
    ASAPosition
from algo.optimizer.base import BaseOptimizer
from algo.trading.signalprovider import PriceSignalProvider
from algo.blockchain.stream import PoolState, PriceUpdate, StreamException
from algo.universe.universe import SimpleUniverse
from typing import Callable, Generator, Any, Type, Optional
from algo.trading.swapper import Swapper
from algo.engine.base import BaseEngine, lag_ms
import asyncio
import datetime
from algo.tools.wallets import get_account_data
import requests


class Engine(BaseEngine):

    def current_time_prov(self) -> datetime.datetime:
        return datetime.datetime.utcnow()

    def __init__(self,
                 universe: SimpleUniverse,
                 price_scraper: Callable[[], Generator[PriceUpdate, Any, Any]],
                 trading_step_seconds: int,
                 marketupdate_step_seconds: int,
                 syncpositions_step_seconds: int,
                 redeem_step_seconds: int,
                 make_optimizer: Callable[[int], BaseOptimizer],
                 make_swapper: Callable[[int], Swapper],
                 make_signal_provider: Callable[[int], PriceSignalProvider],
                 slippage: float,
                 address: str,
                 decay_impact_seconds: int
                 ):

        self.slippage = slippage
        self.asset_ids = [pool.asset1_id for pool in universe.pools]
        assert all(pool.asset2_id == 0 for pool in universe.pools)

        self.swapper = {aid: make_swapper(aid) for aid in self.asset_ids}

        self.optimizers: dict[int, BaseOptimizer] = {asset_id: make_optimizer(asset_id) for asset_id in self.asset_ids}
        self.logger = logging.getLogger(__name__)

        self.trading_step_seconds = trading_step_seconds
        self.marketupdate_step_seconds = marketupdate_step_seconds
        self.syncpositions_step_seconds = syncpositions_step_seconds

        self.price_scraper = price_scraper

        self.prices: dict[int, PoolState] = {}
        self.signal_providers = {aid: make_signal_provider(aid) for aid in self.asset_ids}
        self.last_update_times: dict[int, datetime.datetime] = {}

        self.last_market_state_update: datetime.datetime = None

        self.address = address
        self.pos_impact_state: Optional[GlobalPositionAndImpactState] = None
        self.decay_impact_seconds = decay_impact_seconds

        self.redeem_step_seconds = redeem_step_seconds

    def sync_market_state(self) -> None:
        start_time = datetime.datetime.utcnow()
        min_market_time = None
        max_market_time = None

        try:
            for x in self.price_scraper():
                assert x.asset_ids[1] == 0
                assert x.asset_ids[0] in self.asset_ids
                asset_id, price_update = x.asset_ids[0], x.price_update

                # Time of the price update
                # time = int_to_tzaware_utc_datetime(x.price_update.time)
                time = datetime.datetime.utcfromtimestamp(x.price_update.time)

                if min_market_time is None:
                    min_market_time = time
                else:
                    assert time >= max_market_time
                max_market_time = time

                if asset_id in self.last_update_times:
                    assert time >= self.last_update_times[asset_id]
                self.last_update_times[asset_id] = time
                self.prices[asset_id] = price_update
                self.signal_providers[asset_id].update(time,
                                                       price_update.asset2_reserves / price_update.asset1_reserves)

            end_time = datetime.datetime.utcnow()
            self.last_market_state_update = end_time

            dt_run = lag_ms(end_time - start_time)
            if max_market_time is not None:
                dt_market = lag_ms(max_market_time - min_market_time)
            else:
                dt_market = 0
            self.logger.debug(f'Scraped {dt_market} ms worth of market data in {dt_run} ms')

        except requests.exceptions.ConnectionError as e:
            self.logger.error(f'Price scraping in sync_market_state failed with ConnectionError: {e}')
        except StreamException as e:
            self.logger.error(f'Price scraping in sync_market_state failed with StreamException: {e}')

    def redeem_amounts(self) -> None:

        time = self.current_time_prov()
        self.logger.info(f'Entering redeem loop.')

        for aid in self.asset_ids:
            if aid not in self.prices:
                self.logger.warning(f'price for {aid} not present at time {time}. Skipping redeem amount')
            else:
                asa_price = self.prices[aid].asset2_reserves / self.prices[aid].asset1_reserves
                redeemed_amount = self.swapper[aid].fetch_excess_amounts(asa_price)
                if self.pos_impact_state is not None:
                    self.pos_impact_state.update_redeem(aid, redeemed_amount)

        dt = lag_ms(self.current_time_prov() - time)
        self.logger.info(f'Exiting redeem_amounts after {dt} ms')

    def sync_state(self) -> None:

        time = self.current_time_prov()
        self.logger.info(f'Entering sync_state loop.')

        check_diff = True
        if self.pos_impact_state is None:
            check_diff = False
            self.pos_impact_state = GlobalPositionAndImpactState(
                asa_states={aid: PositionAndImpactState(ASAImpactState(self.decay_impact_seconds), ASAPosition(0)) for
                            aid in self.asset_ids},
                mualgo_position=0
            )

        for aid, amount in get_account_data(self.address).items():
            if aid == 0:
                if check_diff and self.pos_impact_state.mualgo_position != amount:
                    self.logger.warning(
                        f"Actual and tracked Algo positions differ: {amount} vs {self.pos_impact_state.mualgo_position}")
                self.pos_impact_state.mualgo_position = amount
            elif aid in self.asset_ids:
                if check_diff and self.pos_impact_state.asa_states[aid].asa_position.value != amount:
                    self.logger.warning(
                        f"Actual and tracked ASA ({aid}) positions differ: {amount} vs {self.pos_impact_state.asa_states[aid].asa_position.value}")
                self.pos_impact_state.asa_states[aid].asa_position.value = amount
                assert self.pos_impact_state.asa_states[aid].asa_position.value == amount, f"{self.pos_impact_state.asa_states[aid].asa_position.value}, {amount}"
            else:
                self.logger.debug(f'aid {aid} in our portfolio but not in this universe')

        dt = lag_ms(self.current_time_prov() - time)
        self.logger.info(f'Exiting sync_state after {dt} ms')

    def run(self,
            log_trade: Callable[[TradeInfo], None],
            log_state: Callable[[StateLog], None]):

        self.sync_market_state()
        self.redeem_amounts()
        self.sync_state()

        async def trade():
            while True:
                self.trade_loop(log_trade, log_state)
                await asyncio.sleep(self.trading_step_seconds)

        async def market_update():
            while True:
                self.sync_market_state()
                await asyncio.sleep(self.marketupdate_step_seconds)

        async def sync_positions():
            while True:
                self.sync_state()
                await asyncio.sleep(self.syncpositions_step_seconds)

        async def redeem():
            while True:
                self.redeem_amounts()
                await asyncio.sleep(self.redeem_step_seconds)

        async def run():
            await asyncio.gather(market_update(), sync_positions(), trade(), redeem())

        asyncio.run(run())

import logging
import aiohttp
from algo.blockchain.algo_requests import QueryParams, get_current_round
from algo.blockchain.stream import PriceVolumeStream, PriceVolumeDataStore, DataStream
from algo.strategy.analytics import process_market_df
import time
import unittest
from algo.universe.universe import SimpleUniverse
from algo.blockchain.process_volumes import SwapScraper
from algo.blockchain.process_prices import PriceScraper
from algo.blockchain.utils import datetime_to_int
from tinyman.v1.client import TinymanMainnetClient
from tinyman_old.v1.client import TinymanMainnetClient as TinymanOldnetClient
import datetime
import asyncio


class TestData(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        logging.basicConfig(level=logging.NOTSET)
        self.logger = logging.getLogger("TestData")
        self.client = TinymanMainnetClient()

        self.date_min = datetime.datetime(year=2022, month=1, day=20)

        super().__init__(*args, **kwargs)

    def test_volumes(self, n_queries=10):
        asset1 = 0
        asset2 = 470842789

        sc = SwapScraper(self.client, asset1, asset2)

        async def main():
            async with aiohttp.ClientSession() as session:
                async for tx in sc.scrape(session, datetime_to_int(self.date_min), num_queries=n_queries,
                                          before_time=None):
                    print(tx)

        asyncio.run(main())

    def test_prices(self, n_queries=10):
        asset1 = 0
        asset2 = 470842789

        pool = self.client.fetch_pool(asset1, asset2)
        assert pool.exists

        ps = PriceScraper(self.client, asset1, asset2)

        async def main():
            async with aiohttp.ClientSession() as session:
                async for tx in ps.scrape(session, datetime_to_int(self.date_min),
                                          num_queries=n_queries,
                                          query_params=QueryParams()
                                          ):
                    print(tx)

        asyncio.run(main())

    def test_old_prices(self, n_queries=10):
        asset1 = 0
        # Yieldly
        asset2 = 226701642

        client = TinymanOldnetClient()

        ps = PriceScraper(client, asset1, asset2)

        async def main():
            async with aiohttp.ClientSession() as session:
                async for tx in ps.scrape(session, 0, num_queries=n_queries, query_params=QueryParams()):
                    print(tx)

        asyncio.run(main())


class TestStream(unittest.TestCase):

    def __init__(self, *args, **kwargs):

        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                            level=logging.DEBUG)

        self.logger = logging.getLogger(__name__)

        super().__init__(*args, **kwargs)

    def test_stream(self):
        universe = SimpleUniverse.from_cache('liquid_algo_pools_nousd_prehack')

        sleep_seconds = 1

        min_round = get_current_round() - 100

        query_params = QueryParams(min_block=min_round)

        ds = DataStream(universe, query_params)
        pvs = PriceVolumeDataStore(PriceVolumeStream(ds))
        ti = time.time()
        pvs.scrape()
        seconds = time.time() - ti

        prices = pvs.prices()
        volumes = pvs.volumes()

        market_data = process_market_df(prices, volumes)
        self.logger.info(market_data)

        time_max = max(prices['time'].max(), volumes['time'].max())
        time_min = min(prices['time'].min(), volumes['time'].min())

        self.logger.info(f'Scraped {time_max - time_min} seconds of data in {seconds} seconds.')
        for i in range(10):
            time.sleep(sleep_seconds)
            ti = time.time()
            pvs.scrape()
            self.logger.info(f'Scraped {sleep_seconds} seconds of data in {time.time() - ti} seconds.')

        prices = pvs.prices()
        volumes = pvs.volumes()

        # remove pools without algo
        prices = prices[prices['asset2'] == 0]
        volumes = volumes[volumes['asset2'] == 0]

        market_data = process_market_df(prices, volumes)
        self.logger.debug(market_data)

    def test_stream2(self):
        universe = SimpleUniverse.from_cache('liquid_algo_pools_nousd_prehack')

        query_params = QueryParams(min_block=19372771, max_block=19372866)

        ds = DataStream(universe, query_params)
        pvs = PriceVolumeStream(ds)
        ti = time.time()
        for tx in pvs.scrape():
            pass
        seconds = time.time() - ti

        self.logger.info(f'Seconds = {seconds}')

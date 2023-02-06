from __future__ import annotations

import unittest

from algo.blockchain.stream import DataStream, PriceVolumeStream, only_price
from algo.blockchain.process_prices import PriceScraper
from algo.blockchain.stream import PriceUpdate
import aiohttp
from tinyman.v1.client import TinymanClient, TinymanMainnetClient
import asyncio
import logging
from algo.blockchain.process_prices import PoolState
from algo.blockchain.algo_requests import QueryParams
from algo.universe.universe import SimpleUniverse
from algo.universe.pools import PoolId, PoolIdStore
from algo.dataloading.caching import make_filter_from_universe, load_algo_pools
import datetime
import uvloop
from dataclasses import asdict
import pandas as pd


class PriceStreamer:
    def __init__(self,
                 universe: SimpleUniverse,
                 client: TinymanClient,
                 date_min: datetime.datetime,
                 filter_tx_type: bool = True):
        self.client = client
        self.pools = [(x.asset1_id, x.asset2_id) for x in universe.pools]
        self.date_min = date_min

        self.filter_tx_type = filter_tx_type

        self.data: list[PriceUpdate] = []

        self.logger = logging.getLogger(__name__)

    def load(self) -> list[PriceUpdate]:
        async def main():
            async with aiohttp.ClientSession() as session:
                await asyncio.gather(*[self._load_pool(session, assets) for assets in self.pools])

        uvloop.install()
        asyncio.run(main())

        self.data.sort(key=lambda x: x.price_update.time)
        return self.data

    async def _load_pool(self, session, assets) -> None:
        assets = list(sorted(assets, reverse=True))

        scraper = PriceScraper(self.client, assets[0], assets[1], skip_same_time=False)

        if scraper is None:
            self.logger.error(f'Pool for assets {assets[0], assets[1]} does not exist')
            return

        async for pool_state in scraper.scrape(session=session,
                                               num_queries=None,
                                               timestamp_min=None,
                                               query_params=QueryParams(after_time=self.date_min),
                                               filter_tx_type=self.filter_tx_type):
            self.data.append(PriceUpdate(asset_ids=(max(assets), min(assets)), price_update=pool_state))


class TotalDataLoader:
    def __init__(self, cache_name: str, client: TinymanClient, ffilt):
        self.logger = logging.getLogger(__name__)
        self.cached_data = load_algo_pools(cache_name, 'prices', ffilt)

        self.min_dates = self.cached_data.groupby(['asset1', 'asset2']) \
                             .apply(lambda x: max(pd.to_datetime(x['time'], unit='s', utc=True).dt.date)) \
                         + datetime.timedelta(days=1)

        self.new_data = []

        self.assets = [(x['asset1'], x['asset2']) for _, x in
                       self.cached_data[['asset1', 'asset2']].drop_duplicates().iterrows()]

        self.client = client
        self.filter_tx_type = False

    async def main(self):
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*[self._load_pool(session, assets) for assets in self.assets])

    async def load(self) -> pd.DataFrame:
        async def main():
            async with aiohttp.ClientSession() as session:
                await asyncio.gather(*[self._load_pool(session, assets) for assets in self.assets])

        # uvloop.install()
        await main()

        self.new_data.sort(key=lambda x: x.price_update.time)

        data = [{**{'asset1': row.asset_ids[0], 'asset2': row.asset_ids[1]}, **asdict(row.price_update)}
                for row in self.new_data]

        new_df = pd.DataFrame(data)
        assert new_df['time'].min() - self.cached_data['time'].max()
        ret = pd.concat([self.cached_data, new_df]).sort_values(by=['time', 'asset1', 'asset2']).reset_index(drop=True)
        return ret

    async def _load_pool(self, session, assets) -> None:

        scraper = PriceScraper(self.client, int(assets[0]), int(assets[1]), skip_same_time=True)

        if scraper is None:
            self.logger.error(f'Pool for assets {assets[0], assets[1]} does not exist')
            return

        date_min = self.min_dates.loc[assets]

        async for pool_state in scraper.scrape(session=session,
                                               num_queries=None,
                                               timestamp_min=None,
                                               query_params=QueryParams(after_time=date_min),
                                               filter_tx_type=self.filter_tx_type):
            self.new_data.append(PriceUpdate(asset_ids=(max(assets), min(assets)), price_update=pool_state))


class TestDataLoader(unittest.TestCase):
    def test_loader(self):
        tdl = TotalDataLoader('20220209', TinymanMainnetClient())
        print(tdl.load())


class MixedPriceStreamer:
    def __init__(self, universe: SimpleUniverse, date_min: datetime.datetime, client: TinymanClient,
                 filter_tx_type: bool = True):

        self.universe = universe
        self.pvs = None
        self.date_min = date_min
        self.client = client
        self.filter_tx_type = filter_tx_type

    def scrape(self):
        if not self.pvs:
            max_block = -1
            ps = PriceStreamer(self.universe, self.client, date_min=self.date_min, filter_tx_type=self.filter_tx_type)
            for x in ps.load():
                yield x
                assert x.price_update.block >= max_block
                max_block = x.price_update.block
            query_params = QueryParams(min_block=max_block + 1)
            ds = DataStream(self.universe, query_params)
            self.pvs = PriceVolumeStream(ds)
        else:
            yield from only_price(self.pvs.scrape())

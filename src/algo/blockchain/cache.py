import logging
import aiohttp
from algo.blockchain.algo_requests import QueryParams
from algo.universe.pools import PoolIdStore
from tinyman.v1.client import TinymanClient
from algo.blockchain.utils import datetime_to_int, generator_to_df
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import os
import json
import pandas as pd
import datetime
from datetime import timezone
from typing import Optional, Iterable, AsyncGenerator
from abc import ABC, abstractmethod
import asyncio
import uvloop

pd.options.mode.chained_assignment = None  # default='warn'


def assert_date(date):
    if isinstance(date, datetime.datetime):
        assert date == datetime.datetime(year=date.year, month=date.month, day=date.day), \
            f"date {date} must be dates without hours, minutes etc."


class DateScheduler:
    def __init__(self,
                 date_min: datetime.datetime,
                 date_max: Optional[datetime.datetime]):

        for date in date_min, date_max:
            if date is not None:
                assert_date(date)

        if date_max is None:
            utcnow = datetime.datetime.utcnow()
            self.date_max = utcnow.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            self.date_max = date_max

        self.date_min = date_min

        assert self.date_max > self.date_min

    def get_dates_to_fetch(self, existing_dates: Iterable[datetime.date]) -> set[datetime.datetime]:

        needed_dates = set()
        date = self.date_min
        while date < self.date_max:
            if date.date() not in existing_dates:
                needed_dates.add(date)
            date = date + datetime.timedelta(days=1)

        return needed_dates


class DateValidator:
    def __init__(self, cache_basedir: str, cache_name: str, dest_cache: str, assets):
        self.assets = assets

        assets = list(sorted(assets, reverse=True))

        basedir = os.path.join(cache_basedir, cache_name)
        self.cache_dir = os.path.join(basedir, "_".join([str(x) for x in assets]))
        os.makedirs(self.cache_dir, exist_ok=True)

        if dest_cache is not None:
            destbasedir = os.path.join(cache_basedir, dest_cache)
            self.destcache_dir = os.path.join(destbasedir, "_".join([str(x) for x in assets]))
            os.makedirs(self.destcache_dir, exist_ok=True)

    def get_existing_dates(self):
        try:
            with open(f'{self.cache_dir}.json') as json_file:
                return eval(json.load(json_file))
        except FileNotFoundError:
            return set()

    def add_fetched_date(self, date):
        if isinstance(date, datetime.datetime):
            assert_date(date)
            date = date.date()
        existing_dates = self.get_existing_dates()
        with open(f'{self.destcache_dir}.json', 'w') as json_file:
            existing_dates.add(date)
            json.dump(existing_dates, json_file, default=str)

    def has_gaps(self) -> bool:
        existing_dates = self.get_existing_dates()
        try:
            date = min(existing_dates)
        except TypeError as e:
            print(existing_dates, self.assets)
            raise e
        date_max = max(existing_dates)
        while date < date_max:
            if date not in existing_dates:
                return True
            date = date + datetime.timedelta(days=1)
        return False


async def groupby_days(gen: AsyncGenerator):
    prev_date = None
    prev_data = []
    async for x in gen:
        date = datetime.datetime.fromtimestamp(x.time, timezone.utc).date()
        if prev_date is not None and date != prev_date and prev_data:
            yield prev_data
            prev_data = []
        prev_date = date
        prev_data.append(x)

    if prev_data:
        yield prev_data


class DataCacher(ABC):
    def __init__(self,
                 pool_id_store: PoolIdStore,
                 cache_basedir: str,
                 client: TinymanClient,
                 date_min: datetime.datetime,
                 date_max: Optional[datetime.datetime],
                 dry_run: bool):

        self.dry_run = dry_run

        self.client = client
        self.pools = [(x.asset1_id, x.asset2_id) for x in pool_id_store.pools]

        self.dateScheduler = DateScheduler(date_min, date_max)

        self.cache_basedir = cache_basedir
        self.logger = logging.getLogger('DataCacher')

    @abstractmethod
    def make_scraper(self, asset1_id: int, asset2_id: int):
        pass

    def cache(self, cache_name: str, dest_cache: str):
        basedir = os.path.join(self.cache_basedir, cache_name)
        dest_basedir = os.path.join(self.cache_basedir, dest_cache)

        os.makedirs(basedir, exist_ok=True)
        os.makedirs(dest_basedir, exist_ok=True)

        async def main():
            async with aiohttp.ClientSession() as session:
                await asyncio.gather(*[self._cache_pool(session, assets, cache_name, dest_cache) for assets in self.pools])

        uvloop.install()

        asyncio.run(main())

    async def _cache_pool(self, session, assets, cache_name, dest_cache):
        assets = list(sorted(assets, reverse=True))

        dv = DateValidator(self.cache_basedir, cache_name, dest_cache, assets)

        def file_name(date):
            return os.path.join(dv.destcache_dir, f'{date}.parquet')

        existing_dates = dv.get_existing_dates()

        dates_to_fetch = self.dateScheduler.get_dates_to_fetch(existing_dates)
        if len(dates_to_fetch) == 0:
            self.logger.info(f'Skipping assets {assets[0], assets[1]} because all data is present in the cache')
            return

        # Not optimal if there are non-contiguous day windows to fetch, should rather separate the single query into
        # multiple queries for each contiguous window. However, if we just update the cache forward it does not matter.
        date_min = min(dates_to_fetch)
        date_max = max(dates_to_fetch) + datetime.timedelta(days=1)

        self.logger.info(
            f'Found min,max dates to scrape for assets {assets[0], assets[1]} = {date_min}, {date_max}')

        scraper = self.make_scraper(assets[0], assets[1])
        if scraper is None:
            self.logger.warning(f'Pool for assets {assets[0], assets[1]} does not exist')
            return

        def cache_day_df(daydf: pd.DataFrame, date):
            fname = file_name(date)
            table = pa.Table.from_pandas(daydf)
            if not self.dry_run:
                pq.write_table(table, fname)

        async for data in groupby_days(scraper.scrape(session=session,
                                                      num_queries=None,
                                                      timestamp_min=None,
                                                      query_params=QueryParams(after_time=date_min, before_time=date_max))
                                       ):
            df = generator_to_df(data)
            dates = df['time'].dt.date.unique()
            df['time'] = df['time'].view(dtype=np.int64) // 1000000000
            assert len(dates) == 1, f"{dates}"
            date = dates[0]
            cache_day_df(df, date)
            dv.add_fetched_date(date)
            self.logger.info(f'Cached date {date} for assets {assets}')

        # Assume previous operation was successful, fill the missing dates even if there is no data
        existing_dates = dv.get_existing_dates()
        for date in dates_to_fetch:
            if date not in existing_dates:
                if not self.dry_run:
                    dv.add_fetched_date(date)



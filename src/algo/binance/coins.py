import logging
import os
import re
import time
import unittest
from abc import ABC, abstractmethod
from typing import Generator, Union, Optional

import pandas as pd
import datetime

from algo.binance.cached_api import symbol_to_ids, get_mcap, RateLimitException
from algo.binance.data_types import DataType
from algo.definitions import ROOT_DIR

basep = ROOT_DIR / 'data' / 'data.binance.vision'

spot_data_folder = basep / 'spot'

exclude_symbols = {'busd',
                   'dai',
                   'tusd',
                   'paxg'  # Gold
                   }

# Some ids correspond to multiple symbols
hardcoded_ids = {'gmx': 'gmx',
                 'bond': 'barnbridge',
                 'firo': 'zcoin',
                 'ont': 'ontology',
                 'sol': 'solana',
                 'dot': 'polkadot',
                 'lrc': 'loopring',
                 'shib': 'shiba-inu',
                 'xno': 'nano',
                 'ltc': 'litecoin',
                 'ftt': 'ftx-token',
                 'dydx': 'dydx',
                 'apt': 'aptos',
                 'eth': 'ethereum',
                 'bnb': 'binancecoin',
                 'ada': 'cardano',
                 'mana': 'decentraland',
                 'doge': 'dogecoin',
                 'xrp': 'ripple',
                 'uni': 'uniswap'
                 }

spot_to_future_names = {'SHIB': '1000SHIB'}


def all_symbols():
    for y in (spot_data_folder / '1d').glob('*USDT'):
        symbol = y.name[:-4].lower()
        if symbol.endswith('down') or symbol.endswith('up') or symbol.endswith('bear') or symbol.endswith('bull'):
            continue
        yield symbol


class MultipleIdException(Exception):
    pass


def top_mcap(date: datetime.date, dry_run: bool = False) -> list[str]:
    logger = logging.getLogger(__name__)

    symbols_map = symbol_to_ids()
    ret = []

    for symbol in all_symbols():
        coin_ids = symbols_map.get(symbol, None)
        if coin_ids is None:
            logger.warning(f'{symbol} not in symbols_map')
            continue
        if symbol in exclude_symbols:
            continue

        if len(coin_ids) > 1:
            if symbol in hardcoded_ids:
                coin_id = hardcoded_ids[symbol]
                assert coin_id in coin_ids
            else:
                logger.error(f'{symbol=}, {coin_ids=} not hardcoded')
                continue
        else:
            coin_id = coin_ids[0]

        while True:
            try:
                if dry_run:
                    info = None
                else:
                    info = get_mcap(coin_id, date)
                if info is not None:
                    ret.append((symbol, info))
                break
            except RateLimitException as e:
                logger.info('Rate Limit Reached, sleeping for 5 seconds')
                time.sleep(5)

    return list(x[0] for x in sorted(ret, key=lambda x: x[1], reverse=True))




class MarketType(ABC):
    @abstractmethod
    def subpath(self) -> str:
        pass


class FutureType(MarketType):
    def subpath(self):
        return 'futures/um'


class SpotType(MarketType):
    def subpath(self):
        return 'spot'


def load_data(pair_name: str, market_type: MarketType, data_type: DataType, start_date: datetime.datetime,
              end_date: datetime.datetime) \
        -> pd.DataFrame:
    folder = basep / market_type.subpath() / 'monthly' / data_type.subpath() / pair_name
    p = re.compile(data_type.filename_pattern(pair_name))

    logger = logging.getLogger(__name__)
    logger.debug(f'Looking for data files in {folder=}')

    dfs = []

    timestamp_col = data_type.timestamp_col()

    for filename in os.listdir(folder):
        parquet_filename = str(filename)

        m = p.match(filename)
        if m:
            year, month = int(m.group(1)), int(m.group(2))

            t0 = datetime.datetime(year=year, month=month, day=1)

            if month == 12:
                next_month = 1
                next_year = year + 1
            else:
                next_month = month + 1
                next_year = year
            t1 = datetime.datetime(year=next_year, month=next_month, day=1)

            if t1 < start_date or t0 > end_date:
                continue

            subdf = pd.read_parquet(folder / parquet_filename)

            obj_cols = [col for col, dt in subdf.dtypes.items() if dt == object]
            assert not obj_cols, filename

            assert max(subdf[timestamp_col] - subdf[timestamp_col].astype(int)) == 0
            subdf[timestamp_col] = subdf[timestamp_col].astype(int)

            close_time = pd.to_datetime(subdf[timestamp_col], unit='ms')
            idx = (close_time >= start_date) & (close_time <= end_date)

            dfs.append(subdf[idx])

    if not dfs:
        raise ValueError(f'No data found in {folder}')

    df = pd.concat(dfs).sort_values(by=timestamp_col)
    df['logret'] = data_type.lookback_logret_op(df)
    df['price'] = data_type.price_op(df)
    df['BuyVolume'] = data_type.buy_volume_op(df)
    return df


class Universe:
    def __init__(self, coins):
        self.coins = coins

    @classmethod
    def make(cls, n_top_coins: int,
             mcap_date: datetime.date) -> 'Universe':
        return cls(top_mcap(mcap_date)[:n_top_coins])


PairDataGenerator = Generator[tuple[str, Optional[pd.DataFrame]], None, None]


def load_universe_data(universe: Union[Universe, list[str]],
                       start_date: datetime.datetime,
                       end_date: datetime.datetime,
                       market_type: MarketType,
                       data_type: DataType,
                       ) -> PairDataGenerator:
    logger = logging.getLogger(__name__)

    if isinstance(universe, Universe):
        pairs = []
        for coin in universe.coins:
            if not isinstance(market_type, SpotType) and coin.upper() in spot_to_future_names:
                pair_fst = spot_to_future_names[coin.upper()]
            else:
                pair_fst = coin.upper()

            pair_name = pair_fst + 'USDT'
            pairs.append(pair_name)
    else:
        pairs = universe

    for pair_name in pairs:
        try:
            subdf = load_data(pair_name, market_type, data_type, start_date, end_date)
        except FileNotFoundError as e:
            logger.error(str(e))
            yield pair_name, None
            continue

        num_nans = subdf.isna().any(axis=1)
        if num_nans.sum() > 0:
            logger.warning(f"Dropping {num_nans.sum() / subdf.shape[0]} percentage of rows with nans for {pair_name}")

        timestamp_col = data_type.timestamp_col()

        yield pair_name, (
            subdf.dropna().
            set_index(timestamp_col).
            sort_index()
        )


class TestSymbols(unittest.TestCase):
    def test_a(self):
        date = datetime.date(year=2022, month=1, day=1)
        top_mcap(date)

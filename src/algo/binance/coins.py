import logging
import os
import re
import time
import unittest
import zipfile
from pathlib import Path
from typing import Generator
import polars as pl

import pandas as pd
import datetime

from algo.binance.cached_api import symbol_to_ids, get_mcap, RateLimitException

basep = Path('/home/lorenzo/data/data.binance.vision')

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


def load_candles(pair_name: str, subpath: str, freq: str, start_date: datetime.datetime, end_date: datetime.datetime):
    folder = basep / subpath / freq / pair_name / freq
    pattern = rf'{pair_name}-{freq}-(\d\d\d\d)-(\d\d).zip'
    p = re.compile(pattern)

    dfs = []

    columns = ['Open time',
               'Open',
               'High',
               'Low',
               'Close',
               'Volume',
               'Close time',
               'Quote asset volume',
               'Number of trades',
               'Taker buy base asset volume',
               'Taker buy quote asset volume',
               'Ignore'
               ]

    for filename in os.listdir(folder):
        filename = str(filename)

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

            csv_filename = filename.replace('.zip', '.csv')
            parquet_filename = filename.replace('.zip', '.parquet')

            if not os.path.exists(folder / csv_filename) and not os.path.exists(folder / parquet_filename):
                with zipfile.ZipFile(folder / str(filename), 'r') as zip_ref:
                    zip_ref.extractall(folder)

            if not os.path.exists(folder / parquet_filename):
                subdf = pd.read_csv(folder / csv_filename)
                subdf.columns = columns
                subdf.to_parquet(folder / parquet_filename)

            subdf = pd.read_parquet(folder / parquet_filename)

            obj_cols = [col for col, dt in subdf.dtypes.items() if dt == object]
            assert not obj_cols, filename

            assert max(subdf['Close time'] - subdf['Close time'].astype(int)) == 0
            subdf['Close time'] = subdf['Close time'].astype(int)

            # test = subdf['Close time'].cast(pl.Datetime).dt.with_time_unit('ns')

            close_time = pd.to_datetime(subdf['Close time'], unit='ms')
            idx = (close_time >= start_date) & (close_time <= end_date)

            dfs.append(subdf[idx])

    if not dfs:
        raise ValueError(f'No data found in {folder}')

    return pd.concat(dfs)


class Universe:
    def __init__(self, coins):
        self.coins = coins

    @classmethod
    def make(cls, n_top_coins: int,
             mcap_date: datetime.date) -> 'Universe':
        return cls(top_mcap(mcap_date)[:n_top_coins])


PairDataGenerator = Generator[tuple[str, pd.DataFrame], None, None]


def load_universe_candles(universe: Universe,
                          start_date: datetime.datetime,
                          end_date: datetime.datetime,
                          freq: str,
                          spot: bool) -> PairDataGenerator:

    logger = logging.getLogger(__name__)

    if spot:
        subpath = 'spot'
    else:
        subpath = 'futures/um'

    for coin in universe.coins:
        if coin.upper() in spot_to_future_names:
            pair_fst = spot_to_future_names[coin.upper()]
        else:
            pair_fst = coin.upper()

        pair_name = pair_fst + 'USDT'

        try:
            subdf = load_candles(pair_name, subpath, freq, start_date, end_date)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        num_nans = subdf.isna().any(axis=1)
        if num_nans.sum() > 0:
            logger.warning(f"Dropping {num_nans.sum() / subdf.shape[0]} percentage of rows with nans for {pair_name}")

        yield pair_name, (
            subdf.dropna().
            sort_values(by='Open time').
            set_index('Close time')
        )


class TestSymbols(unittest.TestCase):
    def test_a(self):
        date = datetime.date(year=2022, month=1, day=1)
        top_mcap(date)

import logging
import time
import unittest
from abc import ABC, abstractmethod
import datetime
from pydantic import BaseModel

from algo.binance.coingecko import symbol_to_ids, get_mcap, RateLimitException, get_mcaps_today, exclude_symbols
from algo.binance.data_types import DataType
from algo.definitions import ROOT_DIR

basep = ROOT_DIR / 'data' / 'data.binance.vision'


class MarketType(BaseModel, ABC):
    @abstractmethod
    def subpath(self) -> str:
        pass


class FutureType(MarketType):
    def subpath(self):
        return 'futures/um'


class SpotType(MarketType):
    def subpath(self):
        return 'spot'


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


def all_symbols(data_type: DataType, market_type: MarketType):
    data_folder = basep / market_type.subpath() / 'monthly' / data_type.subpath()

    for y in data_folder.glob('*USDT'):
        symbol = y.name[:-4].lower()
        if symbol.endswith('down') or symbol.endswith('up') or symbol.endswith('bear') or symbol.endswith('bull'):
            continue
        yield symbol


class MultipleIdException(Exception):
    pass


def top_mcap(date: datetime.date, data_type: DataType,
             market_type: MarketType, dry_run: bool = False) -> list[str]:
    logger = logging.getLogger(__name__)

    symbols_map = symbol_to_ids()
    ret = []

    for symbol in all_symbols(data_type, market_type):
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


class Universe(BaseModel):
    coins: list[str]

    @classmethod
    def make(cls, n_top_coins: int,
             mcap_date: datetime.date,
             data_type: DataType,
             market_type: MarketType) -> 'Universe':
        return cls(coins=top_mcap(mcap_date, data_type, market_type)[:n_top_coins])

    @classmethod
    def make_lookahead(cls, n_top_coins: int) -> 'Universe':
        logger = logging.getLogger(__name__)
        while 1:
            try:
                mcaps = list(get_mcaps_today(datetime.date.today()))
                break
            except RateLimitException as e:
                logger.warning('Rate limit exception, sleeping 5 seconds')
                time.sleep(5)
        return cls(coins=list(x[1] for x in mcaps if x[1].lower() not in exclude_symbols)[:n_top_coins])


class TestSymbols(unittest.TestCase):
    def test_a(self):
        date = datetime.date(year=2022, month=1, day=1)
        top_mcap(date)

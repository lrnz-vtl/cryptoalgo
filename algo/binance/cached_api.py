import datetime
import json
import logging
import os
import unittest
from typing import Optional

import pandas as pd
import requests

RateLimitException = Exception


def persist_to_file(file_name):
    def decorator(original_func):

        if os.path.exists(file_name):
            cache = pd.read_pickle(file_name)
        else:
            cache = {}

        def new_func(*param):
            if param not in cache:
                cache[param] = original_func(*param)
                pd.to_pickle(cache, file_name)
            return cache[param]

        return new_func

    return decorator


@persist_to_file('/home/lorenzo/caches/symbol_to_ids.dat')
def symbol_to_ids() -> dict[str, list[str]]:
    url = 'https://api.coingecko.com/api/v3/coins/list'
    x = requests.get(url)
    coin_list = x.json()
    ret = {}
    for x in coin_list:
        symbol = x['symbol']
        if 'wormhole' in symbol:
            continue
        if 'binance-peg' in symbol:
            continue
        if symbol not in ret:
            ret[symbol] = [x['id']]
        else:
            ret[symbol].append(x['id'])
    return ret


@persist_to_file('/home/lorenzo/caches/get_mcap.dat')
def get_mcap(coin_id: str, date: datetime.date) -> Optional[float]:
    logger = logging.getLogger(__name__)

    date_str = date.strftime("%d-%m-%Y")

    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/history?date={date_str}'
    oi_data = requests.get(url)
    oi_json = oi_data.json()
    if not oi_data.ok:
        if oi_json['status']['error_message'].startswith("You've exceeded the Rate Limit"):
            raise RateLimitException
        raise requests.RequestException(oi_json)
    if 'market_data' not in oi_json:
        logger.warning(f'market_data not in coin {coin_id}: {oi_json}')
        return None
    try:
        return oi_json['market_data']['market_cap']['usd']
    except KeyError as e:
        raise KeyError(oi_data) from e


class TestApi(unittest.TestCase):
    def test_a(self):
        mcap_date = datetime.date(year=2022, month=1, day=1)
        coin_id = 'nusd'
        get_mcap(coin_id, mcap_date)

    def test_b(self):
        mcap_date = datetime.date(year=2022, month=1, day=1)
        coin_id = 'nusd'
        get_mcap(coin_id, mcap_date)

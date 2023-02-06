import logging
from typing import Optional
import datetime
import aiohttp
from dataclasses import dataclass
import requests
import aiohttp.client_exceptions


@dataclass
class QueryParams:
    after_time: Optional[datetime.datetime] = None
    before_time: Optional[datetime.datetime] = None
    min_block: Optional[int] = None
    max_block: Optional[int] = None

    def make_params(self):
        params = {}
        if self.before_time is not None:
            params['before-time'] = self.before_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        if self.after_time is not None:
            params['after-time'] = self.after_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        if self.min_block is not None:
            params['min-round'] = self.min_block
        if self.min_block is not None:
            params['max-round'] = self.max_block
        return params


def get_current_round():
    url = f'https://algoindexer.algoexplorerapi.io/v2/transactions'
    req = requests.get(url=url).json()
    return int(req['current-round'])


class QueryError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


async def query_transactions(session: aiohttp.ClientSession,
                             params: dict,
                             num_queries: Optional[int],
                             query_params: QueryParams):
    logger = logging.getLogger(__name__)

    query = f'https://algoindexer.algoexplorerapi.io/v2/transactions'

    params = {**params, **query_params.make_params()}

    async with session.get(query, params=params) as resp:
        ok = resp.ok
        if not ok:
            msg = f'Session response not OK:, query = {query}, params = {params}'
            raise QueryError(msg)

        try:
            resp = await resp.json()
        except aiohttp.client_exceptions.ContentTypeError as e:
            logger.critical("resp.json failed"
                            f"\n{resp} = resp")
            raise e

    i = 0
    while resp and (num_queries is None or i < num_queries):
        for tx in resp['transactions']:
            yield tx

        if 'next-token' in resp:
            async with session.get(query, params={**params, **{'next': resp['next-token']}}) as resp:
                resp = await resp.json()
        else:
            resp = None
        i += 1

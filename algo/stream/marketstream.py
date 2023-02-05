import os.path

from tinyman.v1.client import TinymanClient
from algo.tools.timestamp import Timestamp
from typing import Optional, Tuple, Iterable, AsyncGenerator, Coroutine, Any
import asyncio
import algosdk
from aiostream import stream
from algo.stream.aggregators import aggregatePrice, AveragePrice
from logging import Logger
from asyncio.exceptions import TimeoutError
from dataclasses import dataclass
from definitions import ROOT_DIR

default_sample_interval = 5
default_log_interval = 5 * 60
MARKETLOG_BASEFOLDER = os.path.join(ROOT_DIR, 'marketData')


class PoolStream:

    def __init__(self, asset1, asset2, client: TinymanClient,
                 logger: Logger = None,
                 sample_interval: int = default_sample_interval,
                 log_interval: int = default_log_interval,
                 log_info: str = None
                 ):

        self.log_info = log_info
        self.aggregate = aggregatePrice(log_interval, logger=logger)
        self.sample_interval = sample_interval
        self.asset1 = asset1
        self.asset2 = asset2
        self.client = client
        self.logger = logger

    async def run(self):
        next(self.aggregate)

        while True:
            pool = None
            try:
                pool = self.client.fetch_pool(self.asset1, self.asset2)
            except algosdk.error.AlgodHTTPError as e:
                self.logger.error(f'fetch_pool({self.asset1, self.asset2}) failed, error={e}, code={e.code}. Skipping.')

            if pool and pool.exists:
                time = Timestamp.get()

                if self.logger is not None:
                    self.logger.debug(f"pair={self.log_info}, time={time.utcnow}")

                maybeRow: Optional[AveragePrice] = self.aggregate.send((time, pool, self.asset1, self.asset2))

                if maybeRow:
                    yield maybeRow

            await asyncio.sleep(self.sample_interval)


@dataclass
class Row:
    asset1: int
    asset2: int
    timestamp: Timestamp
    price: float
    asset1_reserves: int
    asset2_reserves: int


class MultiPoolStream:

    def __init__(self, assetPairs: Iterable[Tuple[int, int]], client: TinymanClient,
                 logger: Logger,
                 sample_interval: int = default_sample_interval,
                 log_interval: int = default_log_interval
                 ):

        self.assetPairs = assetPairs
        self.poolStreams = [
            PoolStream(asset1=pair[0], asset2=pair[1], client=client, sample_interval=sample_interval,
                       log_interval=log_interval, logger=logger, log_info=str(pair)) for pair in assetPairs
        ]

    async def run(self):

        async def withPairInfo(assetPair, poolStream: PoolStream):
            async for x in poolStream.run():
                yield Row(asset1=assetPair[0], asset2=assetPair[1], timestamp=x.timestamp, price=x.price,
                          asset1_reserves=x.asset1_reserves, asset2_reserves=x.asset2_reserves)

        async_generators = [withPairInfo(assetPair, poolStream) for (assetPair, poolStream) in
                            zip(self.assetPairs, self.poolStreams)]

        combine = stream.merge(*async_generators)

        async with combine.stream() as streamer:
            async for row in streamer:
                yield row


def log_stream(async_gen: AsyncGenerator, timeout: Optional[int], logger_fun) -> Coroutine[Any, Any, None]:
    async def run():
        async def foo():
            async for x in async_gen:
                logger_fun(x)

        try:
            await asyncio.wait_for(foo(), timeout=timeout)
        except TimeoutError:
            pass

    return run()

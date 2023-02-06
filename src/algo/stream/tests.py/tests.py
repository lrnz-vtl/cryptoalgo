import unittest
import logging
from algo.stream.marketstream import PoolStream, MultiPoolStream, log_stream
from algo.stream import sqlite
from tinyman.v1.client import TinymanMainnetClient
import asyncio
import uuid
from contextlib import closing
import shutil


class TestStream(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        logging.basicConfig(level=logging.NOTSET)
        self.logger = logging.getLogger("TestStream")

        filename = kwargs.get('filename')
        if filename:
            fh = logging.FileHandler(filename)
            fh.setLevel(logging.INFO)
            self.logger.addHandler(fh)
        super().__init__(*args, **kwargs)

    def test_pool(self, timeout=11, sample_interval=1, log_interval=5):
        asset1 = 0
        asset2 = 226701642

        client = TinymanMainnetClient()

        samplingLogger = logging.getLogger("SamplingLogger")
        samplingLogger.setLevel(logging.FATAL)

        poolStream = PoolStream(asset1=asset1, asset2=asset2, client=client, log_interval=log_interval,
                                sample_interval=sample_interval, logger=samplingLogger)

        def logf(x):
            self.logger.info(x)

        logger_coroutine = log_stream(poolStream.run(), timeout=timeout, logger_fun=logf)
        asyncio.run(logger_coroutine)

    def test_pools(self, timeout=11, sample_interval=1, log_interval=5):
        assetPairs = [
            (0, 226701642),
            (0, 27165954),
            (0, 230946361),
            (0, 287867876),
            (0, 384303832),
        ]

        client = TinymanMainnetClient()

        samplingLogger = logging.getLogger("SamplingLogger")
        samplingLogger.setLevel(logging.FATAL)

        multiPoolStream = MultiPoolStream(assetPairs=assetPairs, client=client, sample_interval=sample_interval,
                                          log_interval=log_interval, logger=samplingLogger)

        run_name = str(uuid.uuid4())

        with sqlite.MarketSqliteLogger(run_name=run_name) as marketLogger:

            marketLogger.create_table(ignore_existing=True)
            logf = lambda x: marketLogger.log(x)

            logger_coroutine = log_stream(multiPoolStream.run(), timeout=timeout, logger_fun=logf)
            asyncio.run(logger_coroutine)

            with closing(marketLogger.con.cursor()) as c:
                c.execute(f"select * from {marketLogger.tablename}")
                for x in c.fetchall():
                    self.logger.info(x)

        shutil.rmtree(marketLogger.base_folder)

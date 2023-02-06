from datetime import datetime
from tinyman.v1.pools import Pool
from algo.tools.timestamp import Timestamp
from datetime import timezone, timedelta
from typing import Optional, Tuple, Generator
import numpy as np
from dataclasses import dataclass


class RunningMean:

    def __init__(self):
        self.x = np.nan
        self.n = 0

    def add(self, x):
        if np.isnan(self.x):
            self.x = 0
        self.x += x
        self.n += 1

    def value(self):
        return self.x / self.n


def time_bucket(timestamp: Timestamp, log_interval):
    time = (timestamp.utcnow - datetime(1970, 1, 1, tzinfo=timezone.utc))
    delta = timedelta(seconds=time.total_seconds() % log_interval)
    return Timestamp(utcnow=timestamp.utcnow - delta, now=timestamp.now - delta)


@dataclass
class AveragePrice:
    timestamp: Timestamp
    price: float
    asset1_reserves: int
    asset2_reserves: int


def aggregatePrice(bucket_delta: int = 60 * 5, logger=None) -> Generator[
    AveragePrice, Tuple[Timestamp, Pool, int, int], None]:
    """
    Very basic time-average price aggregator
    """

    time: Optional[Timestamp] = None
    bucket_delta = bucket_delta
    mean = None
    asset1_mean = None
    asset2_mean = None

    while True:
        t, pool, a1, a2 = (yield)
        t: Timestamp
        pool: Pool

        # Price of buying infinitesimal amount of asset2 in units of asset1, excluding transaction costs
        if pool.asset2_reserves == 0 or pool.asset1_reserves == 0:
            price = np.nan
        else:
            price = pool.asset2_reserves / pool.asset1_reserves
        asset1_reserves = pool.asset1_reserves
        asset2_reserves = pool.asset2_reserves
        if a1 == pool.asset2.id:
            asset1_reserves, asset2_reserves = asset2_reserves, asset1_reserves
            price = 1.0 / price

        if mean is not None:
            t0 = time_bucket(time, bucket_delta)
            t1 = time_bucket(t, bucket_delta)

            if t0.utcnow != t1.utcnow:
                yield AveragePrice(timestamp=t0, price=mean.value(),
                                   asset1_reserves=asset1_mean.value(),
                                   asset2_reserves=asset2_mean.value())
                if logger is not None:
                    logger.debug(f"Number of samples: {mean.n}")
                mean = RunningMean()
                asset1_mean = RunningMean()
                asset2_mean = RunningMean()
        else:
            mean = RunningMean()
            asset1_mean = RunningMean()
            asset2_mean = RunningMean()

        mean.add(price)
        asset1_mean.add(asset1_reserves)
        asset2_mean.add(asset2_reserves)
        time = t

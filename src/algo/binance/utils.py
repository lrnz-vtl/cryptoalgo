import datetime
from dataclasses import dataclass

import pandas as pd


def to_datetime(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, unit='ms')

@dataclass
class TrainTestOptions:
    train_end_time: datetime.datetime
    test_start_time: datetime.datetime
    min_train_period: datetime.timedelta

    def __post_init__(self):
        assert self.train_end_time < self.test_start_time

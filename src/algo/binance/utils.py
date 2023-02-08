import datetime
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def to_datetime(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, unit='ms')


def read_json(p: Path):
    with open(p) as f:
        return json.load(f)


@dataclass
class TrainTestOptions:
    train_end_time: datetime.datetime
    test_start_time: datetime.datetime
    min_train_period: datetime.timedelta

    def __post_init__(self):
        assert self.train_end_time < self.test_start_time

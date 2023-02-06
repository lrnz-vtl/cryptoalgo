from dataclasses import dataclass

import pandas as pd
import datetime
import numpy as np
from scipy.stats.mstats import winsorize
from abc import ABC, abstractmethod
from algo.signals.constants import ASSET_INDEX_NAME, TIME_INDEX_NAME
import warnings
import scipy.linalg as la
from algo.trading.costs import FEE_BPS

warnings.simplefilter(action='ignore', category=FutureWarning)


@dataclass
class ComputedLookaheadResponse:
    ts: pd.Series
    lookahead_time: datetime.timedelta

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.ts, attr)


class LookaheadResponse(ABC):

    @property
    @abstractmethod
    def lookahead_time(self) -> datetime.timedelta:
        pass

    @abstractmethod
    def _call(self, df: pd.DataFrame) -> pd.Series:
        pass

    def __call__(self, df: pd.DataFrame) -> ComputedLookaheadResponse:
        assert df.index.names == [ASSET_INDEX_NAME, TIME_INDEX_NAME]
        ret = self._call(df)
        return ComputedLookaheadResponse(ret, self.lookahead_time)


def shift_forward(ts: pd.Series, minutes_forward: int):
    time_forward = ts.copy().rename('series')
    time_forward.index = time_forward.index.set_levels(
        time_forward.index.levels[1] - datetime.timedelta(minutes=minutes_forward), TIME_INDEX_NAME)

    return ts.to_frame().join(time_forward)['series']


class SimpleResponse(LookaheadResponse):

    def __init__(self, minutes_forward: int, col: str = 'algo_price', start_minutes_forward: int = 0,
                 norm_by_fees=False):
        assert minutes_forward % 5 == 0
        assert start_minutes_forward % 5 == 0
        assert start_minutes_forward < minutes_forward
        self.minutes_forward = minutes_forward
        self.start_minutes_forward = start_minutes_forward
        self.norm_by_fees = norm_by_fees
        self.col = col

    @property
    def lookahead_time(self) -> datetime.timedelta:
        return datetime.timedelta(minutes=self.minutes_forward)

    def _call(self, df: pd.DataFrame) -> pd.Series:
        price = df[self.col].rename('price')

        time_forward = shift_forward(price, self.minutes_forward)
        start_time_forward = shift_forward(price, self.start_minutes_forward)

        # time_forward = price.copy()
        # time_forward.index = time_forward.index.set_levels(
        #     time_forward.index.levels[1] - datetime.timedelta(minutes=self.minutes_forward), TIME_INDEX_NAME)
        #
        # start_time_forward = price.copy()
        # start_time_forward.index = start_time_forward.index.set_levels(
        #     start_time_forward.index.levels[1] - datetime.timedelta(minutes=self.start_minutes_forward),
        #     TIME_INDEX_NAME)

        # prices = price.to_frame().join(time_forward, rsuffix='_forward').join(start_time_forward, rsuffix='_start')
        # resp = (prices['price_forward'] - prices['price_start']) / prices['price_start']

        resp = (time_forward - start_time_forward) / start_time_forward

        if self.norm_by_fees:
            resp = resp / FEE_BPS

        return resp


class VolumeResponse(LookaheadResponse):

    def __init__(self, minutes_forward: int, col: str = 'asset2_amount_gross'):
        assert minutes_forward % 5 == 0
        self.minutes_forward = minutes_forward
        self.col = col

    @property
    def lookahead_time(self) -> datetime.timedelta:
        return datetime.timedelta(minutes=self.minutes_forward)

    def _call(self, df: pd.DataFrame) -> pd.Series:
        volume = df[self.col].rename('volume')
        volume = volume.groupby(ASSET_INDEX_NAME).cumsum()

        time_forward = volume.copy()
        time_forward.index = time_forward.index.set_levels(
            time_forward.index.levels[1] - datetime.timedelta(minutes=self.minutes_forward), TIME_INDEX_NAME)

        volumes = volume.to_frame().join(time_forward, rsuffix='_forward')
        resp = (volumes['volume_forward'] - volumes['volume']) / volumes['volume']

        return resp


def feature_neutralize(X, y, w, strenght=1.0):
    X = np.hstack([X, np.ones(shape=(X.shape[0], 1))])
    Xw = np.einsum('ij,i->ij', X, np.sqrt(w.values))
    yw = np.einsum('i,i->i', y, np.sqrt(w.values))
    betas = la.lstsq(Xw, yw)[0]
    return y - strenght * np.dot(X, betas)


def my_winsorize(y, limits=(0.05, 0.05)):
    resp = y.copy()
    mask = ~np.isnan(resp)
    resp[mask] = winsorize(resp[mask], limits=limits)
    return resp


def drop_extremes(y: pd.Series, limits=0.05):
    lower = y.quantile(limits)
    upper = y.quantile(1.0 - limits)
    return y[(y > lower) & (y < upper)]

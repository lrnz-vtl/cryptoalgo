import numba
import pandas as pd
from ts_tools_algo.features import generate_over_series, ema_provider, ema_emv_vec
from typing import Callable, Union, Optional
from abc import ABC, abstractmethod
from algo.signals.constants import ASSET_INDEX_NAME, TIME_INDEX_NAME
from ts_tools_algo.series import frac_diff
import numpy as np
import datetime


class XAssetFeaturizer(ABC):
    @abstractmethod
    def _call(self, df: pd.DataFrame, response_asset_id: int) -> pd.Series:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def __call__(self, df: pd.DataFrame, response_ids: list[int]) -> pd.Series:
        assert df.index.names == [ASSET_INDEX_NAME, TIME_INDEX_NAME]
        return pd.concat([self._call(df, response_asset_id) for response_asset_id in response_ids], keys=response_ids,
                         names=['asset1']).rename(self.name)


class SingleASAFeaturizer(ABC):

    @abstractmethod
    def _call(self, df: pd.DataFrame) -> pd.Series:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def __call__(self, df: pd.DataFrame, response_ids: list[int]) -> pd.Series:
        assert df.index.names == [ASSET_INDEX_NAME, TIME_INDEX_NAME]
        ret = df[df.index.get_level_values(ASSET_INDEX_NAME).isin(response_ids)]. \
            groupby([ASSET_INDEX_NAME]).apply(lambda x: self._call(x.droplevel(ASSET_INDEX_NAME)).rename(self.name))
        assert all(response_id in ret.index.get_level_values(ASSET_INDEX_NAME).unique() for response_id in response_ids)
        return ret


Featurizer = Union[XAssetFeaturizer, SingleASAFeaturizer]


class IdFeaturizer(SingleASAFeaturizer):

    def __init__(self, col: str):
        self.col = col

    @property
    def name(self):
        return f'{self.col}'

    def _call(self, df: pd.DataFrame):
        return df[self.col]


class RunningSum(SingleASAFeaturizer):

    def __init__(self, minutes: int, col: str, den_col: Optional[str]):
        self.minutes = minutes
        self.col = col
        self.den_col = den_col

    @property
    def name(self):
        if self.den_col:
            return f'{self.col}_{self.den_col}_{self.minutes}'
        else:
            return f'{self.col}_sum_{self.minutes}'

    def _call(self, df: pd.DataFrame):
        dt = datetime.timedelta(seconds=self.minutes * 60)
        ret = df[self.col].rolling(dt).sum()
        if self.den_col:
            ret /= df[self.den_col]
        return ret


class RunningMean(SingleASAFeaturizer):

    def __init__(self, minutes: int, col: str, den_col: Optional[str]):
        self.minutes = minutes
        self.col = col
        self.den_col = den_col

    @property
    def name(self):
        if self.den_col:
            return f'{self.col}_mean_{self.den_col}_{self.minutes}'
        else:
            return f'{self.col}_mean_{self.minutes}'

    def _call(self, df: pd.DataFrame):
        dt = datetime.timedelta(seconds=self.minutes * 60)
        ret = df[self.col].rolling(dt).mean()
        if self.den_col:
            ret /= df[self.den_col]
        return ret


class MAFeaturizerSimple(SingleASAFeaturizer):
    def __init__(self, minutes: int, col: str):
        self.minutes = minutes
        self.col = col

    @property
    def name(self):
        return f'{self.col}_masimple_{self.minutes}'

    def _call(self, df: pd.DataFrame):
        ts = df[self.col]
        ts_seconds = ts.index.values.astype('datetime64[s]').astype('int')
        xs = ts.values
        ema, emv = ema_emv_vec(ts_seconds, xs, self.minutes * 60)
        ma = pd.Series(ema, index=df.index)
        return ma


class MAFeaturizer(SingleASAFeaturizer):
    def __init__(self, minutes: int, col: str):
        self.minutes = minutes
        self.col = col

    @property
    def name(self):
        return f'{self.col}_ma_{self.minutes}'

    def _call(self, df: pd.DataFrame):
        ts = df[self.col]
        ts_seconds = ts.index.values.astype('datetime64[s]').astype('int')
        xs = ts.values
        ema, emv = ema_emv_vec(ts_seconds, xs, self.minutes * 60)
        ma = pd.Series(ema, index=df.index)
        return (ts - ma) / ts


class MSTDFeaturizer(SingleASAFeaturizer):
    def __init__(self, minutes: int, col: str):
        self.minutes = minutes
        self.col = col

    @property
    def name(self):
        return f'{self.col}_mstd_{self.minutes}'

    def _call(self, df: pd.DataFrame):
        ts = df[self.col]
        ts_seconds = ts.index.values.astype('datetime64[s]').astype('int')
        xs = ts.values
        ema, emv = ema_emv_vec(ts_seconds, xs, self.minutes * 60)
        return np.sqrt(emv) / ts


class FracDiffFeaturizer(SingleASAFeaturizer):
    def __init__(self, price_col: str = 'algo_price', d=0.4, thres=0.98):
        self.price_col = price_col
        self.d = d
        self.thres = thres

    @property
    def name(self):
        return f'frac_{self.d}'

    def _call(self, df: pd.DataFrame):
        ts = df[self.price_col]
        return frac_diff(ts, self.d, self.thres) / ts


class FracDiffEMA(SingleASAFeaturizer):
    def __init__(self, minutes: int, d=0.4, price_col: str = 'algo_price', thres=0.98):
        self.price_col = price_col
        self.d = d
        self.thres = thres
        self.minutes = minutes

    @property
    def name(self):
        return f'frac_{self.d}_ma{self.minutes}'

    def _call(self, df: pd.DataFrame):
        ts = df[self.price_col]
        fd = frac_diff(ts, self.d, self.thres).fillna(0)

        startidx = np.where(~fd.isna())[0][0]

        fd_ma = pd.Series(np.nan, index=fd.index)
        fd_ma[startidx:] = pd.Series(generate_over_series(fd[startidx:], ema_provider(self.minutes * 60)),
                                     index=fd.index)
        assert ~fd_ma.isna().any()
        return fd_ma / ts


class SingleXAssetFeaturizer(XAssetFeaturizer):

    @property
    def name(self) -> str:
        return f'{self.ft.name}_{self.feature_id}'

    def _call(self, df: pd.DataFrame, response_asset_id: int) -> pd.Series:
        df = df.loc[self.feature_id]
        return self.ft._call(df)

    def __init__(self, ft: SingleASAFeaturizer, feature_id: int):
        self.ft = ft
        self.feature_id = feature_id


def concat_featurizers(fts: list[Featurizer], response_ids: list[int]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    names = [ft.name for ft in fts]
    assert len(names) == len(set(names))

    def foo(df: pd.DataFrame):
        return pd.concat([ft(df, response_ids) for ft in fts], axis=1).astype(float)

    return foo

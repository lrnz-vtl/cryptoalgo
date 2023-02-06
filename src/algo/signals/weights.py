import pandas as pd
from abc import ABC, abstractmethod
from algo.signals.constants import ASSET_INDEX_NAME


class BaseWeightMaker(ABC):
    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        pass


class DailyWeightMaker(BaseWeightMaker):

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        ws = (df.groupby(['asset1', 'date'])['asset2_reserves'].agg('mean') / (10 ** 6)).rename('weight')

        weights = df[['date']].merge(ws, on=[ASSET_INDEX_NAME, 'date'], how='left')
        weights.index = df.index
        return weights['weight']


class SimpleWeightMaker(BaseWeightMaker):

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return df['asset2_reserves'] / (10 ** 6)

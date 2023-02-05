import logging
import pandas as pd
import datetime
import numpy as np
from algo.dataloading.caching import join_caches_with_priority, make_filter_from_universe
from algo.strategy.analytics import process_market_df
from algo.signals.constants import ASSET_INDEX_NAME, TIME_INDEX_NAME
from algo.signals.weights import BaseWeightMaker
from algo.signals.responses import LookaheadResponse, ComputedLookaheadResponse
from algo.universe.universe import SimpleUniverse
from dataclasses import dataclass
from typing import Optional, Union, Callable
from algo.signals.evaluation import FittableDataStore
from abc import ABC, abstractmethod
from algo.signals.featurizers import Featurizer, concat_featurizers
from ts_tools_algo.series import rolling_min


class DataFilter(ABC):

    @abstractmethod
    def apply(self, df: pd.DataFrame):
        pass

    def __call__(self, df: pd.DataFrame):
        assert df.index.names == [ASSET_INDEX_NAME, TIME_INDEX_NAME]
        return df.groupby(ASSET_INDEX_NAME).apply(lambda x: self.apply(x.droplevel(ASSET_INDEX_NAME)))


class RollingLiquidityFilter(DataFilter):

    def __init__(self, period_days: int = 7, algo_reserves_floor=50000):
        self.period_days = period_days
        self.algo_reserves_floor = algo_reserves_floor

    def apply(self, df: pd.DataFrame):
        period = datetime.timedelta(days=self.period_days)
        rol_min_liq = rolling_min(df['asset2_reserves'], period)
        return (rol_min_liq / 10 ** 6) > self.algo_reserves_floor


def lag_market_data(df: pd.DataFrame, lag_seconds: int):
    df = df.copy()
    df['time'] = df['time'] + lag_seconds
    return df


def process_dfs(dfp: pd.DataFrame, dfv: Optional[pd.DataFrame], ffill_price_minutes,
                volume_aggregators: list[Callable[[pd.DataFrame], pd.DataFrame]]):
    df = process_market_df(dfp, dfv, ffill_price_minutes, make_algo_columns=False, volume_aggregators=volume_aggregators)
    df = df.set_index([ASSET_INDEX_NAME, TIME_INDEX_NAME]).sort_index()
    assert np.all(df['asset2'] == 0)
    return df.drop(columns=['asset2', 'level_1'], errors='ignore')


class AnalysisDataStore:

    def __init__(self,
                 price_caches: list[str],
                 volume_caches: list[str],
                 universe: SimpleUniverse,
                 weight_maker: BaseWeightMaker,
                 ffill_price_minutes: Optional[Union[int, str]],
                 market_lag_seconds: int,
                 volume_aggregators: list[Callable[[pd.DataFrame], pd.DataFrame]],
                 make_df_lagged: bool = True):

        self.features_lag_seconds = market_lag_seconds

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)

        self.asset_ids = [pool.asset1_id for pool in universe.pools]
        assert all(pool.asset2_id == 0 for pool in universe.pools), "Need to provide a Universe with only Algo pools"

        filter_ = make_filter_from_universe(universe)

        dfp = join_caches_with_priority(price_caches, 'prices', filter_)
        if make_df_lagged:
            dfp_lagged = lag_market_data(dfp, self.features_lag_seconds)

        if volume_caches:
            dfv = join_caches_with_priority(volume_caches, 'volumes', filter_)
            if make_df_lagged:
                dfv_lagged = lag_market_data(dfv, self.features_lag_seconds)
        else:
            dfv = None
            dfv_lagged = None

        df = process_dfs(dfp, dfv, ffill_price_minutes, volume_aggregators=volume_aggregators)
        if make_df_lagged:
            df_lagged = process_dfs(dfp_lagged, dfv_lagged, ffill_price_minutes, volume_aggregators=volume_aggregators)

        df = df[df.index.get_level_values(ASSET_INDEX_NAME).isin(self.asset_ids)]
        if make_df_lagged:
            df_lagged = df_lagged[df_lagged.index.get_level_values(ASSET_INDEX_NAME).isin(self.asset_ids)]

        df_ids = df.index.get_level_values(ASSET_INDEX_NAME).unique()
        missing_ids = [asset_id for asset_id in self.asset_ids if asset_id not in df_ids]
        if missing_ids:
            self.logger.error(f"asset ids {missing_ids} are missing from the dataframe generated from the cache")

        self.df: pd.DataFrame = df
        if make_df_lagged:
            self.df_lagged: pd.DataFrame = df_lagged
        self.weights: pd.Series = weight_maker(df)

        assert self.df.index.names == [ASSET_INDEX_NAME, TIME_INDEX_NAME]

    def _make_filters(self, df: Union[pd.DataFrame, pd.Series, ComputedLookaheadResponse], filters: list[DataFilter],
                      filter_nans: bool = False) -> pd.Series:

        filt_idx = pd.Series(True, index=df.index)
        orig_weight = self.weights[self.weights.index.isin(df.index)].sum()

        for filt in filters:
            filt_idx = filt_idx & filt(df)
        self.logger.info('Percentage of data retained after filters: '
                         f'{self.weights[self.weights.index.isin(filt_idx[filt_idx].index)].sum() / orig_weight}')
        if filter_nans:
            if len(df.shape) == 2:
                filt_idx = filt_idx & (~df.isna().any(axis=1))
            else:
                filt_idx = filt_idx & (~df.isna())
            self.logger.info(f'Percentage of data retained after filtering nans: '
                             f'{self.weights[self.weights.index.isin(filt_idx[filt_idx].index)].sum() / orig_weight}')

        return filt_idx

    def _apply_filters(self, df: Union[pd.DataFrame, pd.Series, ComputedLookaheadResponse], filters: list[DataFilter],
                       filter_nans: bool = False) -> Union[pd.DataFrame, pd.Series, ComputedLookaheadResponse]:

        filt_idx = self._make_filters(df, filters, filter_nans)
        if isinstance(df, ComputedLookaheadResponse):
            return ComputedLookaheadResponse(df.ts[filt_idx], df.lookahead_time)
        else:
            return df[filt_idx]

    def make_response(self, response_maker: LookaheadResponse, response_asset_ids: list[int],
                      filter_nans: bool = False) -> ComputedLookaheadResponse:

        assert all(aid in self.asset_ids for aid in response_asset_ids)

        ts = response_maker(self.df.loc[self.df.index.get_level_values(ASSET_INDEX_NAME).isin(response_asset_ids)])

        self.logger.info(f"Percentage of response data after filtering ids: {ts.shape[0] / self.df.shape[0]}")
        ret = self._apply_filters(ts, [], filter_nans)
        return ret

    def make_asset_features(self, featurizers: list[Featurizer],
                            responses: ComputedLookaheadResponse,
                            feature_filters: list[DataFilter],
                            filter_nans: bool = False,
                            post_featurizer: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None) -> pd.DataFrame:

        foo = concat_featurizers(featurizers, list(responses.index.get_level_values(ASSET_INDEX_NAME).unique()))
        df = foo(self.df_lagged)
        if post_featurizer:
            df = post_featurizer(df)
        return self._apply_filters(df, feature_filters, filter_nans)

    def make_trading_filters(self, trading_filters: list[DataFilter]) -> pd.Series:
        return self._make_filters(self.df, trading_filters)

    def make_fittable_data(self,
                           features: pd.DataFrame, response: ComputedLookaheadResponse,
                           trading_filt_idx: pd.Series,
                           oos_time: datetime.datetime,
                           splitting_strategy: str = 'normal'
                           ) -> FittableDataStore:

        common_idx = features.index.intersection(response.index).intersection(trading_filt_idx[trading_filt_idx].index)
        filt_idx = self.df.index.isin(common_idx)

        prefilter_ids = set(self.weights.index.get_level_values(ASSET_INDEX_NAME).unique())
        postfilter_ids = set(self.weights[filt_idx].index.get_level_values(ASSET_INDEX_NAME).unique())

        if prefilter_ids != postfilter_ids:
            self.logger.warning(f'After filtering some ids were dropped! {prefilter_ids.difference(postfilter_ids)}')

        return FittableDataStore(features[features.index.isin(common_idx)],
                                 ComputedLookaheadResponse(response.ts[response.ts.index.isin(common_idx)],
                                                           response.lookahead_time),
                                 self.weights[filt_idx],
                                 oos_time=oos_time,
                                 splitting_strategy=splitting_strategy
                                 )

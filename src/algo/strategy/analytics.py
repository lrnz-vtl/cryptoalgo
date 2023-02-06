import logging
import pandas as pd
import datetime
import numpy as np
from algo.universe.assets import get_decimals
from typing import Optional, Union, Callable


def make_algo_pricevolume(df, make_algo_columns=True):

    volcols = df.columns[df.columns.str.startswith('asset1_amount') | df.columns.str.startswith('asset2_amount')]
    # print(f'volcols = {volcols}')
    df[volcols] = df[volcols].fillna(0)

    # df['mualgo_price'] = df['asset2_reserves'] / df['asset1_reserves']

    def foo(subdf, asset_id):
        assert np.all(df['asset2'] == 0)
        decimals = get_decimals(asset_id)
        subdf['algo_price'] = subdf['asset2_reserves'] / subdf['asset1_reserves'] * 10 ** (decimals - 6)
        subdf['algo_reserves'] = subdf['asset2_reserves'] / (10 ** 6)
        if 'asset2_amount' in subdf.columns:
            subdf['algo_volume'] = subdf['asset2_amount'] / (10 ** 6)
        return subdf

    if make_algo_columns:
        df = df.groupby('asset1').apply(lambda x: foo(x, x.name))

    return df


def timestamp_to_5min(time_col: pd.Series):
    # We do not want lookahead in the data, so each 5 minute slice should contain the data for the past, not the future
    time_5min = ((time_col // 300) + ((time_col % 300) > 0).astype(int)) * 300
    return pd.to_datetime(time_5min, unit='s', utc=True)


def ffill_cols(df: pd.DataFrame, cols: list[str], minutes_limit: Union[int, str], all_times=None):
    if all_times is None:
        all_times = []
        delta = df['time_5min'].max() - df['time_5min'].min()
        for i in range(int(delta.total_seconds() / (5 * 60))):
            all_times.append(df['time_5min'].min() + i * datetime.timedelta(seconds=5 * 60))

        all_times = pd.Series(all_times).rename('time_5min')
        assert len(all_times) == len(set(all_times))

    df = df.merge(all_times, on='time_5min', how='outer')
    df = df.sort_values(by='time_5min')

    df['time_5min_ffilled'] = df['time_5min'].fillna(method='ffill')

    if isinstance(minutes_limit, int):
        assert (minutes_limit % 5 == 0)
        timelimit_idx = ((df['time_5min'] - df['time_5min_ffilled']) <= datetime.timedelta(minutes=minutes_limit))
    elif minutes_limit == 'all':
        timelimit_idx = pd.Series(True, index=df.index)
    else:
        raise ValueError

    for col in cols:
        fill_idx = timelimit_idx & df[col].isna()
        col_ffilled = df[col].fillna(method='ffill')
        df.loc[fill_idx, col] = col_ffilled[fill_idx]

    return df


def ffill_prices(df: pd.DataFrame, minutes_limit: Union[int, str]):
    cols = ['asset1_reserves', 'asset2_reserves', 'issued_liquidity']

    subdf = df[~df[cols].isna().any(axis=1)]

    # assert ~df[cols].isna().any().any()

    if subdf.shape[0] < df.shape[0]:
        logger = logging.getLogger(__name__)
        logger.warning(f'Dropping {df.shape[0] - subdf.shape[0]} rows because htere are nans in cols {cols}')

    ret: pd.DataFrame = subdf.groupby(['asset1', 'asset2']).apply(
        lambda x: ffill_cols(x.drop(columns=['asset1', 'asset2']), cols, minutes_limit)).reset_index()
    ret = ret.dropna(subset=cols)
    return ret


def process_market_df(price_df: pd.DataFrame, volume_df: Optional[pd.DataFrame],
                      ffill_price_minutes: Optional[Union[int, str]],
                      volume_aggregators: list[Callable[[pd.DataFrame], pd.DataFrame]],
                      price_agg_fun='mean',
                      merge_how='left',
                      make_algo_columns: bool = True
                      ):
    logger = logging.getLogger(__name__)

    price_df['time_5min'] = timestamp_to_5min(price_df['time'])

    keys = ['time_5min', 'asset1', 'asset2']

    price_cols = ['asset1_reserves', 'asset2_reserves', 'issued_liquidity']
    price_df = price_df[price_cols + keys].groupby(keys).agg(price_agg_fun).reset_index()

    if ffill_price_minutes:
        in_shape = price_df.shape[0]
        price_df = ffill_prices(price_df, ffill_price_minutes)
        logger.info(f'Forward filled prices, shape ({in_shape}) -> ({price_df.shape[0]})')

    if volume_df is not None:
        volume_agg_dfs = []
        for volume_aggregator in volume_aggregators:
            volume_agg_dfs.append(volume_aggregator(volume_df))

        volume_df['time_5min'] = timestamp_to_5min(volume_df['time'])
        volume_cols = ['asset1_amount', 'asset2_amount']
        new_volume_cols = []

        for col in volume_cols:
            volume_df[f'{col}_gross'] = abs(volume_df[col])

            buy_idx = volume_df[col] > 0
            volume_df.loc[buy_idx, f'{col}_net_buy'] = volume_df.loc[buy_idx, col]
            volume_df.loc[~buy_idx, f'{col}_net_buy'] = 0
            volume_df.loc[~buy_idx, f'{col}_net_sell'] = volume_df.loc[~buy_idx, col]
            volume_df.loc[buy_idx, f'{col}_net_sell'] = 0

            new_volume_cols += [f'{col}_gross', f'{col}_net_buy', f'{col}_net_sell']

        volume_df = volume_df[new_volume_cols + keys].groupby(keys).agg('sum').reset_index()

        df = price_df.merge(volume_df, how=merge_how, on=keys)
        for volume_agg_df in volume_agg_dfs:
            df = df.merge(volume_agg_df, how=merge_how, on=keys)
    else:
        df = price_df

    # assert np.all(df['asset2'] == 0)
    return make_algo_pricevolume(df, make_algo_columns)

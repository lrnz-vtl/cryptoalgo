import logging
import os
import re
from typing import Generator, Union, Optional
import pandas as pd
import datetime
from algo.binance.coins import MarketType, Universe, SpotType, spot_to_future_names
from algo.binance.data_types import DataType, KlineType
from algo.definitions import ROOT_DIR

basep = ROOT_DIR / 'data' / 'data.binance.vision'

PairDataGenerator = Generator[tuple[str, Optional[pd.DataFrame]], None, None]


def load_data(pair_name: str, market_type: MarketType, data_type: DataType, start_date: datetime.datetime,
              end_date: datetime.datetime) \
        -> Optional[pd.DataFrame]:
    folder = basep / market_type.subpath() / 'monthly' / data_type.subpath() / pair_name
    if isinstance(data_type, KlineType):
        folder = folder / data_type.freq
    p = re.compile(data_type.filename_pattern(pair_name))

    logger = logging.getLogger(__name__)
    logger.debug(f'Looking for data files in {folder=}')

    dfs = []

    timestamp_col = data_type.timestamp_col()

    for filename in os.listdir(folder):
        parquet_filename = str(filename)

        m = p.match(filename)
        if m:
            year, month = int(m.group(1)), int(m.group(2))

            t0 = datetime.datetime(year=year, month=month, day=1)

            if month == 12:
                next_month = 1
                next_year = year + 1
            else:
                next_month = month + 1
                next_year = year
            t1 = datetime.datetime(year=next_year, month=next_month, day=1)

            if t1 < start_date or t0 > end_date:
                continue

            subdf = pd.read_parquet(folder / parquet_filename)

            obj_cols = [col for col, dt in subdf.dtypes.items() if dt == object]
            assert not obj_cols, filename

            assert max(subdf[timestamp_col] - subdf[timestamp_col].astype(int)) == 0
            subdf[timestamp_col] = subdf[timestamp_col].astype(int)

            close_time = pd.to_datetime(subdf[timestamp_col], unit='ms')
            idx = (close_time >= start_date) & (close_time <= end_date)

            dfs.append(subdf[idx])

    if not dfs:
        logger.error(f'No data found in {folder} for the chosen period')
        return None

    df = pd.concat(dfs).sort_values(by=timestamp_col)
    df['logret'] = data_type.lookback_logret_op(df)
    df['price'] = data_type.price_op(df)
    df['BuyVolume'] = data_type.buy_volume_op(df)
    return df


def load_universe_data(universe: Union[Universe, list[str]],
                       start_date: datetime.datetime,
                       end_date: datetime.datetime,
                       market_type: MarketType,
                       data_type: DataType,
                       ) -> PairDataGenerator:
    logger = logging.getLogger(__name__)

    if isinstance(universe, Universe):
        pairs = []
        for coin in universe.coins:
            if not isinstance(market_type, SpotType) and coin.upper() in spot_to_future_names:
                pair_fst = spot_to_future_names[coin.upper()]
            else:
                pair_fst = coin.upper()

            pair_name = pair_fst + 'USDT'
            pairs.append(pair_name)
    else:
        pairs = universe

    for pair_name in pairs:
        try:
            subdf = load_data(pair_name, market_type, data_type, start_date, end_date)
        except FileNotFoundError as e:
            logger.error(str(e))
            yield pair_name, None
            continue
        if subdf is None:
            yield pair_name, None
            continue

        num_nans = subdf.isna().any(axis=1)
        if num_nans.sum() > 0:
            logger.warning(f"Dropping {num_nans.sum() / subdf.shape[0]} percentage of rows with nans for {pair_name}")

        timestamp_col = data_type.timestamp_col()

        yield pair_name, (
            subdf.dropna().
            set_index(timestamp_col).
            sort_index()
        )

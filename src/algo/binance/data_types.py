from typing import Union, Literal, Protocol

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, Field

MS_IN_5MIN = 1000 * 60 * 5


class DataType(Protocol):

    def make_subpath_from_pair(self, pair_name: str) -> str:
        ...

    def filename_pattern(self, x: str) -> str:
        ...

    def timestamp_col(self) -> str:
        ...

    def price_op(self, df: pd.DataFrame) -> pd.Series:
        ...

    def lookback_logret_op(self, df: pd.DataFrame) -> pd.Series:
        ...

    def buy_volume_op(self, df: pd.DataFrame) -> pd.Series:
        ...

    def orig_columns(self) -> list[str]:
        ...

    def process_frame(self, df: pl.LazyFrame) -> pl.LazyFrame:
        ...


class KlineType(BaseModel):
    dtype: Literal['kline'] = 'kline'

    freq: str

    def process_frame(self, df: pl.LazyFrame) -> pl.LazyFrame:
        allowed_dtypes = {pl.Int64, pl.Float64, pl.Boolean}
        assert all(t in allowed_dtypes for t in df.dtypes)
        return df

    def buy_volume_op(self, df: pd.DataFrame) -> pd.Series:
        return df['Taker buy base asset volume']

    def orig_columns(self):
        return ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
                'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']

    def filename_pattern(self, pair_name: str) -> str:
        return rf'{pair_name}-{self.freq}-(\d\d\d\d)-(\d\d).parquet'

    def make_subpath_from_pair(self, pair_name: str) -> str:
        return f'klines/{pair_name}/{self.freq}'

    def timestamp_col(self) -> str:
        return 'Close time'

    def price_op(self, df: pd.DataFrame) -> pd.Series:
        return (df['Close'] + df['Open']) / 2.0

    def lookback_logret_op(self, df: pd.DataFrame) -> pd.Series:
        return np.log(df['Close']) - np.log(df['Open'])


class AggTradesType(BaseModel):
    dtype: Literal['agg'] = 'agg'

    # NOTE Some days are missing in the monthly files...
    def process_frame(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return (df
                .with_columns(
            [
                (((pl.col('Timestamp') // MS_IN_5MIN) + 1) * MS_IN_5MIN).alias("Timestamp_5min")
            ])
                .groupby("Timestamp_5min")
                .agg(
            [
                (pl.col('Price') * pl.col('Quantity')).sum().alias('PriceVolume'),
                pl.col('Quantity').sum().alias('Volume'),
                pl.col('Quantity').filter(pl.col('Was the buyer the maker')).sum().fill_null(0).alias('SellVolume')
            ]
        )
                .with_columns(
            [
                (pl.col('PriceVolume') / pl.col('Volume')).alias('vwap'),
                pl.col('Timestamp_5min').cast(pl.Datetime).dt.with_time_unit("ms").alias("datetime_5min"),
            ]
        )
                .select([
            'Timestamp_5min', 'vwap', 'Volume', 'SellVolume'
        ])
                .sort('Timestamp_5min')
                )

    def buy_volume_op(self, df: pd.DataFrame) -> pd.Series:
        return df['Volume'] - df['SellVolume']

    def orig_columns(self):
        return ['Aggregate tradeId', 'Price', 'Quantity', 'First tradeId', 'Last tradeId', 'Timestamp',
                'Was the buyer the maker']

    def lookback_logret_op(self, df: pd.DataFrame) -> pd.Series:
        return np.log(df['vwap']) - np.log(df['vwap'].shift(periods=1)).fillna(0)

    def price_op(self, df: pd.DataFrame) -> pd.Series:
        return df['vwap']

    def make_subpath_from_pair(self, pair_name: str) -> str:
        return f'aggTrades/{pair_name}'

    def filename_pattern(self, pair_name: str) -> str:
        return rf'{pair_name}-aggTrades-(\d\d\d\d)-(\d\d).parquet'

    def timestamp_col(self) -> str:
        return 'Timestamp_5min'


class DataTypeModel(BaseModel):
    t: Union[AggTradesType, KlineType] = Field(..., discriminator='dtype')

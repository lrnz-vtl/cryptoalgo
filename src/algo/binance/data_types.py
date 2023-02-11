from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, Extra

MS_IN_5MIN = 1000 * 60 * 5


class DataType(BaseModel, ABC):

    @abstractmethod
    def subpath(self) -> str:
        pass

    @abstractmethod
    def filename_pattern(self, x: str) -> str:
        pass

    @abstractmethod
    def timestamp_col(self) -> str:
        pass

    @abstractmethod
    def price_op(self, df: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def lookback_logret_op(self, df: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def buy_volume_op(self, df: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def orig_columns(self) -> list[str]:
        pass

    @abstractmethod
    def process_frame(self, df: pl.LazyFrame) -> pl.LazyFrame:
        pass


class KlineType(DataType):
    freq: str

    def process_frame(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df

    def buy_volume_op(self, df: pd.DataFrame) -> pd.Series:
        return df['Taker buy base asset volume']

    def orig_columns(self):
        return ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
                'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']

    def filename_pattern(self, pair_name: str) -> str:
        return rf'{pair_name}-{self.freq}-(\d\d\d\d)-(\d\d).parquet'

    def subpath(self):
        return f'klines'

    def timestamp_col(self) -> str:
        return 'Close time'

    def price_op(self, df: pd.DataFrame) -> pd.Series:
        return (df['Close'] + df['Open']) / 2.0

    def lookback_logret_op(self, df: pd.DataFrame) -> pd.Series:
        return np.log(df['Close']) - np.log(df['Open'])


class AggTradesType(DataType):

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

    def subpath(self):
        return 'aggTrades'

    def filename_pattern(self, pair_name: str) -> str:
        return rf'{pair_name}-aggTrades-(\d\d\d\d)-(\d\d).parquet'

    def timestamp_col(self) -> str:
        return 'Timestamp_5min'

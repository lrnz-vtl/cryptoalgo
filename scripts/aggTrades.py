import argparse
import logging
import polars as pl
from polars import exceptions
from algo.binance.s3_download import BucketDataProcessor

MS_IN_5MIN = 1000 * 60 * 5

def try_process(csv_path, has_header: bool, spot: bool):
    cols = ['Aggregate tradeId', 'Price', 'Quantity', 'First tradeId', 'Last tradeId', 'Timestamp',
            'Was the buyer the maker']
    if spot:
        cols.append('Was the trade the best price match')

    df = (
        pl.scan_csv(csv_path, has_header=has_header, with_column_names=lambda _: cols)
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
    return df.collect()


def process_csv(csv_path, spot: bool):
    try:
        return try_process(csv_path, False, spot)
    except exceptions.ComputeError:
        return try_process(csv_path, True, spot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--spot', action='store_true')
    args = parser.parse_args()

    spot = args.spot

    if spot:
        prefix = 'data/spot/monthly/aggTrades/'
    else:
        prefix = 'data/futures/um/monthly/aggTrades/'

    fmt = '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'

    logging.basicConfig(level=logging.INFO,
                        format=fmt,
                        datefmt='%H:%M:%S'
                        )

    p = BucketDataProcessor(process_csv=lambda x: process_csv(x, spot), prefix=prefix)
    p.run()

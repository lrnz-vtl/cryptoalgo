import argparse
import logging
import polars as pl
from polars import exceptions

from algo.binance.data_types import DataType, AggTradesType, KlineType
from algo.binance.s3_download import BucketDataProcessor


def try_process(csv_path, has_header: bool, spot: bool, data_type: DataType) -> pl.DataFrame:
    cols = data_type.orig_columns()

    if spot and isinstance(data_type, AggTradesType):
        cols.append('Was the trade the best price match')

    return data_type.process_frame(
        pl.scan_csv(csv_path, has_header=has_header, with_column_names=lambda _: cols)
    ).collect()


def process_csv(csv_path, spot: bool, data_type: DataType):
    try:
        return try_process(csv_path, False, spot, data_type)
    except exceptions.ComputeError:
        return try_process(csv_path, True, spot, data_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--spot', action='store_true')
    parser.add_argument('--type', type=str, required=True, choices=['aggTrades', 'Klines'])
    args = parser.parse_args()

    spot = args.spot

    if args.type == 'aggTrades':
        data_type = AggTradesType()
    elif args.type == 'Klines':
        data_type = KlineType('5m')
    else:
        raise ValueError

    fmt = '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'

    logging.basicConfig(level=logging.INFO,
                        format=fmt,
                        datefmt='%H:%M:%S'
                        )

    p = BucketDataProcessor(process_csv=lambda x: process_csv(x, data_type=data_type, spot=spot), data_type=data_type,
                            spot=spot)
    p.run()

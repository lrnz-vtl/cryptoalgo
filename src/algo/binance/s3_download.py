import concurrent.futures
import logging
import os.path
import zipfile
from pathlib import Path
from typing import Callable
import boto3

from algo.binance.data_types import DataType, KlineType, AggTradesType
from algo.definitions import ROOT_DIR
import queue
import polars as pl

bucket_name = 'data.binance.vision'


class BucketDataProcessor:
    def __init__(self, process_csv: Callable[[Path], pl.DataFrame], data_type:DataType, spot:bool):

        self.data_type = data_type

        if isinstance(data_type, KlineType):
            if spot:
                prefix = 'data/spot/monthly/klines/'
            else:
                prefix = 'data/futures/um/monthly/klines/'

        elif isinstance(data_type, AggTradesType):
            if spot:
                prefix = 'data/spot/monthly/aggTrades/'
            else:
                prefix = 'data/futures/um/monthly/aggTrades/'
        else:
            raise ValueError

        assert prefix.startswith('data')
        subpath = prefix[4:]

        self.base_path = Path(str(ROOT_DIR / 'data' / bucket_name) + subpath)

        self.prefix = prefix

        self.process_csv = process_csv

        self.q = queue.Queue()

    def feeq_queue(self):
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)

        logger = logging.getLogger(__name__)

        logger.info(f'Read {self.prefix=}')

        for bucket_object in bucket.objects.filter(Prefix=self.prefix):

            # logger.debug(f'Read unfiltered {bucket_object.key=}')

            if not bucket_object.key.endswith('.zip'):
                continue

            fname = bucket_object.key.split('/')[-1]
            fname_parquet = fname.replace('.zip', '.parquet')

            if isinstance(self.data_type, KlineType):
                pair_name = bucket_object.key.split('/')[-3]
                freq = bucket_object.key.split('/')[-2]
                if freq != self.data_type.freq:
                    continue
                dst_path_parquet = self.base_path / pair_name / freq / fname_parquet
            elif isinstance(self.data_type, AggTradesType):
                pair_name = bucket_object.key.split('/')[-2]
                dst_path_parquet = self.base_path / pair_name / fname_parquet
            else:
                raise ValueError

            if not pair_name.endswith('USDT') and not pair_name.endswith('BUSD'):
                continue

            if os.path.exists(dst_path_parquet):
                continue

            logger.info(f'Processing {bucket_object.key=}')

            arg = (bucket, bucket_object, dst_path_parquet, fname)
            self.q.put(arg)

        return "DONE FEEDING"

    def process_key(self, arg):
        (bucket, bucket_object, dst_path_parquet, fname) = arg

        logger = logging.getLogger(__name__)

        zip_path = dst_path_parquet.parent / fname
        os.makedirs(dst_path_parquet.parent, exist_ok=True)

        bucket.download_file(bucket_object.key, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(Path(zip_path).parent)

        csv_path = dst_path_parquet.parent / fname.replace('.zip', '.csv')
        assert csv_path.exists()

        os.remove(zip_path)

        df = self.process_csv(csv_path)
        df.write_parquet(dst_path_parquet)

        os.remove(csv_path)

        logger.info(f'Saved {dst_path_parquet}')

    def run(self):

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_arg = {
                executor.submit(self.feeq_queue): 'FEEDER DONE'}

            while future_to_arg:
                done, not_done = concurrent.futures.wait(
                    future_to_arg, timeout=5,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )

                while not self.q.empty():
                    arg = self.q.get()
                    future_to_arg[executor.submit(self.process_key, arg)] = arg

                for future in done:
                    arg = future_to_arg[future]
                    try:
                        data = future.result()
                    except Exception as exc:
                        print('%r generated an exception: %s' % (arg, exc))
                    else:
                        if arg == 'FEEDER DONE':
                            print(data)
                        else:
                            print(f'{arg=}')

                    del future_to_arg[future]

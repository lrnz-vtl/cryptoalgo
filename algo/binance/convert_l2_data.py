import sys
from pathlib import Path
import pandas as pd

base_path = Path('/home/lorenzo/data')


if __name__ == '__main__':

    name = 'binance.spot.l2_topk.BTC.USDT.BTCUSDT.2021-12.csv'
    dest_name = '.'.join(name.split('.')[:-1]) + '.parquet'

    fname = base_path / name
    fname_parquet = base_path / dest_name

    assert fname.exists()

    if fname_parquet.exists():
        print(f'{fname_parquet} already exists.')
        sys.exit(0)

    df = pd.read_csv(fname, sep='\t', header=None)
    df.columns = ['timestamp_ms', 'unknown_bool', 'ask', 'bid', 'unknown_int', 'nans']
    del df['nans']


    def parse_row(x0: list):
        x1 = map(lambda y: y.strip(',['), x0)
        x2 = map(lambda y: map(lambda z: float(z), y.split(',')), x1)
        x3 = [item for sublist in x2 for item in sublist]
        return list(x3)


    def process_ob_side(side_name: str):
        xs = df[side_name].str.strip('[]').str.split(']').tolist()
        columns = [item for sublist in
                   [[f'{side_name}_price_{i}', f'{side_name}_qbtc_{i}', f'{side_name}_qusd_{i}'] for i in range(5)] for item
                   in
                   sublist]

        return pd.DataFrame((parse_row(x) for x in xs), columns=columns, index=df.index)


    df = pd.concat([df[['timestamp_ms', 'unknown_bool', 'unknown_int']], process_ob_side('ask'), process_ob_side('bid')], axis=1)
    df.to_parquet(fname_parquet)

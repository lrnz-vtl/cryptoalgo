from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from algo.cpp import shift


data_base = Path('/home/lorenzo/microprice')
fname = 'binance-futures_book_snapshot_5_2020-09-01_BTCUSDT.csv'
fname_parquet = 'binance-futures_book_snapshot_5_2020-09-01_BTCUSDT.parquet'

# start = datetime.datetime(year=2023, month=1, day=2, hour=7, minute=14)
# end = datetime.datetime(year=2023, month=1, day=2, hour=7, minute=16)

# df = pd.read_csv(data_base / fname)
# df.to_parquet(data_base / fname_parquet)
df = pd.read_parquet(data_base / fname_parquet)


def process_df(df):
    # df.columns = ['trade Id', 'price', 'qty', 'quoteQty', 'time', 'isBuyerMaker', 'isBestMatch']
    df['time'] = pd.to_datetime(df['timestamp'], unit='us')
    df = df.set_index('timestamp')
    # idx = ((df.index > start) & (df.index < end))
    # df = df[idx]
    return df

df = process_df(df)

df['mid'] = (df['asks[0].price'] + df['bids[0].price']) / 2.0

df['future_mid'] = shift.shift(df.index.values, df['mid'].values, 1000000 * 6000)
# print(df['future_mid'])

plt.plot(df['time'], df['mid'], label='mid')
plt.plot(df['time'], df['future_mid'], label='future_mid')
plt.legend()
plt.grid()
plt.show()

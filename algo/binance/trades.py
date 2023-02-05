import datetime
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt

data_base = Path('/home/lorenzo/crypto_data')
fname = 'APTUSDT-trades-2023-01-02.csv'
fname2 = 'APTBUSD-trades-2023-01-02.csv'

start = datetime.datetime(year=2023, month=1, day=2, hour=7, minute=14)
end = datetime.datetime(year=2023, month=1, day=2, hour=7, minute=16)

df = pd.read_csv(data_base / fname)
df2 = pd.read_csv(data_base / fname2)
def process_df(df):
    df.columns = ['trade Id', 'price', 'qty', 'quoteQty', 'time', 'isBuyerMaker', 'isBestMatch']
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
    df = df.set_index('timestamp')
    idx = ((df.index > start) & (df.index < end))
    df = df[idx]
    return df

df = process_df(df)
df2 = process_df(df2)

df['price'].plot()
df2['price'].plot()

plt.grid()
plt.show()

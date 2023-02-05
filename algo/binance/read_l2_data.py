from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from binance.convert_l2_data import base_path

names = ['binance.spot.l2_topk.BTC.USDT.BTCUSDT.2021-10.parquet',
         'binance.spot.l2_topk.BTC.USDT.BTCUSDT.2021-11.parquet',
         'binance.spot.l2_topk.BTC.USDT.BTCUSDT.2021-12.parquet']

fnames = [base_path / name for name in names]


def process_file(fname_parquet: Path):
    df = pd.read_parquet(fname_parquet)
    df['time'] = pd.to_datetime(df['timestamp_ms'], unit='ms')

    df['mid'] = (df['ask_price_0'] + df['bid_price_0']) / 2.0
    # seconds_lag = 1
    # future_mid = shift.shift(df['timestamp_ms'].values * 1000, mid.values, seconds_lag * 10**6)
    df['future_mid'] = df['mid'].shift(-1).fillna(df['mid'])

    return df


df = pd.concat((process_file(fname) for fname in fnames))

mid_ret = (df['future_mid'] - df['mid']) / df['mid']

train_idx = np.full(df.shape[0], True)
train_idx[(int(train_idx.shape[0] * 0.66)):] = False
test_idx = ~train_idx

ivw_mid = (df['ask_price_0'] * df['bid_qusd_0'] + df['bid_price_0'] * df['ask_qusd_0']) / (
        df['bid_qusd_0'] + df['ask_qusd_0'])
baseline_pred = ((ivw_mid - df['mid']) / df['mid'])[test_idx]


def make_features(n_prices: int, n_qtys: int, n_shifts: int = 1):
    norm = {}
    for shift in range(n_shifts):
        norm[shift] = 0
        for iq in range(n_qtys):
            norm[shift] += df[f'bid_qusd_{iq}'].shift(shift).fillna(df[f'bid_qusd_{iq}']) + \
                           df[f'ask_qusd_{iq}'].shift(shift).fillna(df[f'ask_qusd_{iq}'])

    features = []
    for ip in range(n_prices):
        for iq in range(n_qtys):
            for shift in range(n_shifts):
                features += [(df[f'ask_price_{ip}'].shift(shift).fillna(df[f'ask_price_{ip}']) / df[f'mid']) *
                             df[f'bid_qusd_{iq}'].shift(shift).fillna(df[f'bid_qusd_{ip}']) / norm[shift],
                             (df[f'bid_price_{ip}'].shift(shift).fillna(df[f'bid_price_{ip}']) / df[f'mid']) *
                             df[f'ask_qusd_{iq}'].shift(shift).fillna(df[f'ask_qusd_{ip}']) / norm[shift]]
    return np.array(features).transpose()


def eval_features(X):
    y = mid_ret
    lm = LinearRegression()
    lm.fit(X=X[train_idx], y=y[train_idx])
    pred = lm.predict(X=X[test_idx])

    sortidx = pred.argsort()
    ypreds = pred[sortidx]
    ytrues = mid_ret[test_idx].values[sortidx]

    nbins = 100
    ypredmeans = [x.mean() for x in np.array_split(ypreds, nbins)]
    ytruemeans = [x.mean() for x in np.array_split(ytrues, nbins)]

    # plt.plot(ypredmeans, ytruemeans)
    # plt.show()

    print(f'corr: {np.corrcoef(pred, mid_ret[test_idx])[0, 1]}')


print(f'Baseline corr: {np.corrcoef(baseline_pred, mid_ret[test_idx])[0, 1]}')

eval_features(make_features(1, 1, 1))
eval_features(make_features(2, 2, 1))

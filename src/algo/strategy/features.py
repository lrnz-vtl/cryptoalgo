from algo.tinychart_data.datastore import DataStore, AssetData
import pandas as pd
from ts_tools_algo.features import exp_average


def compute_moving_average(ts: pd.Series, interval: str = "3h"):
    """Compute the moving average from processed_price_data"""
    return ts.resample(interval).mean().fillna(0).rolling(window=3, min_periods=1).mean()


def make_asset_features(subdf, colname='price'):
    assert len(subdf['asset_id'].unique()) == 1

    subdf['ema_2h'] = exp_average(subdf[colname], 2 * 60 * 60)
    subdf['ema_1h'] = exp_average(subdf[colname], 1 * 60 * 60)
    subdf['ema_30m'] = exp_average(subdf[colname], 30 * 60)

    subdf['MA_2h'] = compute_moving_average(subdf[colname], "2h")
    subdf['MA_15min'] = compute_moving_average(subdf[colname], "15min")

    return subdf


def make_df(ds: DataStore) -> pd.DataFrame:
    def make_single(asset_id: int, data: AssetData):
        df = pd.DataFrame(data=data.slow.price_history)
        df['asset_id'] = asset_id
        return df

    return pd.concat([make_single(asset_id, data) for asset_id, data in ds.items()])

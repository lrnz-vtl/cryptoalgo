import logging
import numpy as np
import sklearn
from sklearn import decomposition
import pandas as pd
from algo.cpp.cseries import shift_forward, compute_ema


class BetaStore:
    def __init__(self, price_ts: pd.Series, min_test_time: int, hours_forward: int):
        """ Index is (pair,time) """

        self.logger = logging.getLogger(__name__)

        def shift_group_forward(x, y, ms):
            return pd.Series(shift_forward(x, y, ms), index=x)

        forward_price_ts = price_ts.groupby('pair').apply(
            lambda x: shift_group_forward(x.index.get_level_values(1), x.values, hours_forward))

        ret_ts: pd.Series = np.log(forward_price_ts) - np.log(price_ts)
        ret_df = ret_ts.unstack(level=0)

        idx = (ret_df.isna().sum(axis=1) <= (ret_df.shape[0] // 2))
        ret_df = ret_df.loc[idx].dropna(axis=1)

        pca = sklearn.decomposition.PCA(n_components=1)
        fitdf = ret_df[ret_df.index < min_test_time]
        pca.fit(fitdf)

        self.mkt_returns = pd.Series(pca.transform(ret_df)[:, 0], index=ret_df.index).rename('mkt_return')

    def compute_beta(self, product_ret_ts: pd.Series) -> float:
        idx = self.mkt_returns.index.intersection(product_ret_ts.index)

        if len(idx) < 0.99 * len(product_ret_ts.index):
            self.logger.warning(f'{len(idx)=}')

        assert len(idx) > 0

        x = self.mkt_returns[idx]
        y = product_ret_ts[idx]

        return (x * y).sum() / (x * x).sum()

    def residualise(self, beta: float, product_ret_ts: pd.Series) -> pd.Series:
        matching_mkt_returns = self.mkt_returns[product_ret_ts.index]
        nan_ratio = matching_mkt_returns.isna().sum() / matching_mkt_returns.shape[0]
        if nan_ratio > 0.01:
            self.logger.warning(f'{nan_ratio=}')
        mkt_component = beta * matching_mkt_returns.fillna(0)
        return product_ret_ts - mkt_component

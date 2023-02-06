import logging
from typing import Optional, Callable

import pandas as pd
import datetime
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, explained_variance_score
from algo.signals.constants import ASSET_INDEX_NAME, TIME_INDEX_NAME
from algo.signals.responses import ComputedLookaheadResponse
from sklearn.decomposition import PCA
from algo.tools.asset_data_store import AssetDataStore, get_asset_datastore
from tinyman.v1.client import TinymanMainnetClient


def any_axis_1(x):
    if (isinstance(x, np.ndarray) and x.ndim == 2) or isinstance(x, pd.DataFrame):
        return x.any(axis=1)
    else:
        return x


def not_nan_mask(*vecs):
    return ~np.any([any_axis_1(np.isnan(x)) for x in vecs], axis=0)


def plot_bins(x, y, w, nbins=20):

    if nbins is None:
        nbins = 20

    sort_idx = x.sort_values().index
    wbins = pd.cut(w[sort_idx].cumsum(), bins=nbins)

    xsums = (x * w).groupby(wbins).sum()
    ysums = (y * w).groupby(wbins).sum()
    wsums = w.groupby(wbins).sum()

    xp = xsums / wsums
    yp = ysums / wsums

    xmin = min(xp)
    xmax = max(xp)
    xs = np.linspace(xmin, xmax, 100)

    plt.plot(xp, yp, label='mean')
    plt.plot(xs, xs, ls='--', c='k')
    plt.grid()


def eval_fitted_model(m, X_test, y_test, w_test, X_full, y_full, w_full, reports=True, reports_args=dict()):
    y_pred = pd.Series(m.predict(X_test), index=X_test.index)
    oos_rsq = r2_score(y_test, y_pred, sample_weight=w_test)
    eps = explained_variance_score(y_test, y_pred, sample_weight=w_test)

    if not reports:
        return oos_rsq

    logger = logging.getLogger(__name__)
    ads = get_asset_datastore()

    logger.info(f'OOS R^2 = {oos_rsq}, eps={eps}')

    corrcoef = corr(y_pred, y_test, w_test)
    logger.info(f'Response correlation = {corrcoef}')

    data = [[corr(X_test[col], y_test, w_test), corr(X_test[col], m.predict(X_test), w_test)] for col in X_test]
    corrs = pd.DataFrame(data, index=X_test.columns, columns=['pred', 'true'])
    # print(corrs)

    sig = m.predict(X_full)

    mean = (sig * w_full).sum() / w_full.sum()
    std = np.sqrt((w_full * (sig - mean) ** 2).sum() / w_full.sum())
    logger.info(f'mean = {mean}, std = {std}')

    plot_bins(y_pred, y_test, w_test, nbins=reports_args.get('nbins'))

    xxw = w_full * sig ** 2
    xyw = w_full * sig * y_full

    asset_ids = w_full.index.get_level_values(ASSET_INDEX_NAME).unique()
    cols = len(asset_ids)
    f, axs = plt.subplots(1, cols, figsize=(10 * cols, 4))

    if cols == 1:
        axs = [axs]

    for asset, ax in zip(asset_ids, axs):
        xxw.loc[asset].groupby(TIME_INDEX_NAME).sum().cumsum().plot(label='signal', ax=ax)
        xyw.loc[asset].groupby(TIME_INDEX_NAME).sum().cumsum().plot(label='response', ax=ax)
        ax.axvline(X_test.index.get_level_values(TIME_INDEX_NAME).min(), ls='--', color='k', label='test cutoff')
        ax.set_title(f'{asset}, {ads.fetch_asset(asset).name}')
        ax.legend()
        ax.grid()
    plt.show()
    plt.clf()

    xxw.groupby(TIME_INDEX_NAME).sum().cumsum().plot(label='signal')
    xyw.groupby(TIME_INDEX_NAME).sum().cumsum().plot(label='response')
    plt.axvline(X_test.index.get_level_values(TIME_INDEX_NAME).min(), ls='--', color='k', label='test cutoff')
    plt.legend()
    plt.grid()
    plt.show()

    return oos_rsq


def cov(x, y, w):
    xmean = (x * w).sum() / w.sum()
    ymean = (y * w).sum() / w.sum()
    return (w * (x - xmean) * (y - ymean)).sum() / w.sum()


def corr(x, y, w):
    xmean = (x * w).sum() / w.sum()
    # print(y,w)
    ymean = (y * w).sum() / w.sum()
    return (w * (x - xmean) * (y - ymean)).sum() / \
           np.sqrt((w * (x - xmean) ** 2).sum() * (w * (y - ymean) ** 2).sum())


def bootstrap_arrays(n: int, index: pd.Index, *vecs: pd.Series):
    assets = index.get_level_values(0).unique()
    np.random.seed(42)
    for i in range(n):
        boot_assets = np.random.choice(assets, len(assets))
        yield tuple(
            np.concatenate([vec[index.get_loc(boot_asset)] for boot_asset in boot_assets]) for vec in vecs)


def bootstrap_betas(features: pd.DataFrame, response: pd.Series, weights: pd.Series):
    for col in features.columns:
        N = 100
        betas = np.zeros(N)

        for i, (x, y, w) in enumerate(bootstrap_arrays(N, features.index, features[col], response, weights)):
            mask = not_nan_mask(x, y, w)
            x, y, w = x[mask], y[mask], w[mask]
            betas[i] = LinearRegression().fit(x[:, None], y, w).coef_

        yield betas


def make_train_val_splits(response: ComputedLookaheadResponse,
                          weights: pd.Series,
                          oos_time: datetime.datetime,
                          splitting_strategy: str,
                          val_frac=0.3,
                          ):
    logger = logging.getLogger(__name__)

    lookahead_time = response.lookahead_time + datetime.timedelta(minutes=5)

    if splitting_strategy == "normal":
        upper_val_ts = oos_time - lookahead_time
        is_weights = weights[weights.index.get_level_values(TIME_INDEX_NAME) <= upper_val_ts]
        is_weights = is_weights.groupby(TIME_INDEX_NAME).agg(sum).sort_index()
    elif splitting_strategy == 'inverted':
        lower_train_ts = oos_time + lookahead_time
        is_weights = weights[weights.index.get_level_values(TIME_INDEX_NAME) >= lower_train_ts]
        is_weights = is_weights.groupby(TIME_INDEX_NAME).agg(sum).sort_index()
    else:
        raise ValueError

    val_idx = (is_weights.cumsum() / is_weights.sum() >= 1-val_frac)
    first_val_ts = val_idx[val_idx].index.get_level_values(TIME_INDEX_NAME).min()
    upper_train_ts = first_val_ts - lookahead_time

    if splitting_strategy == "normal":
        train_idx = weights.index.get_level_values(TIME_INDEX_NAME) <= upper_train_ts
        val_idx = (weights.index.get_level_values(TIME_INDEX_NAME) >= first_val_ts) & \
                  (weights.index.get_level_values(TIME_INDEX_NAME) <= upper_val_ts)
    elif splitting_strategy == "inverted":
        train_idx = (weights.index.get_level_values(TIME_INDEX_NAME) >= lower_train_ts) & \
                    (weights.index.get_level_values(TIME_INDEX_NAME) <= upper_train_ts)
        val_idx = (weights.index.get_level_values(TIME_INDEX_NAME) >= first_val_ts)
    else:
        raise ValueError

    assert len(weights) > train_idx.sum() + val_idx.sum()
    assert ~np.any(train_idx & val_idx)

    logger.info(f'Train, val size = {train_idx.sum()}, {val_idx.sum()}')

    train_times = weights[train_idx].index.get_level_values(TIME_INDEX_NAME)
    test_times = weights[val_idx].index.get_level_values(TIME_INDEX_NAME)

    if splitting_strategy == "normal":
        oos_times = weights[weights.index.get_level_values(TIME_INDEX_NAME) > oos_time].index.get_level_values(TIME_INDEX_NAME)
        assert test_times.max() < oos_times.min()
    elif splitting_strategy == "inverted":
        oos_times = weights[weights.index.get_level_values(TIME_INDEX_NAME) < oos_time].index.get_level_values(
            TIME_INDEX_NAME)
        assert oos_times.max() < test_times.min()
    else:
        raise ValueError

    assert train_times.max() < test_times.min(), f"{train_times.max()}, {test_times.min()}"

    logger.info(f'Train times = {train_times.min(), train_times.max()}, '
                f'val times = {test_times.min(), test_times.max()}')

    return train_idx, val_idx


class FittableDataStore:

    def __init__(self, features: pd.DataFrame, response: ComputedLookaheadResponse, weights: pd.Series,
                 oos_time: datetime.datetime, splitting_strategy: str = 'normal'):
        client = TinymanMainnetClient()
        self.ads = AssetDataStore(client)

        assert np.all(features.index == response.index)
        assert np.all(features.index == weights.index)
        assert np.all(response.index == weights.index)

        self.features = features
        self.response = response
        self.weights = weights

        self.oos_time = oos_time

        if splitting_strategy == 'normal':
            self.oos_idx = self.weights.index.get_level_values(TIME_INDEX_NAME) >= self.oos_time
            self.is_idx = self.weights.index.get_level_values(TIME_INDEX_NAME) <= self.oos_time - response.lookahead_time - datetime.timedelta(minutes=5)
        elif splitting_strategy == 'inverted':
            self.oos_idx = self.weights.index.get_level_values(TIME_INDEX_NAME) <= self.oos_time
            self.is_idx = self.weights.index.get_level_values(TIME_INDEX_NAME) >= self.oos_time + response.lookahead_time + datetime.timedelta(minutes=5)

        self.logger = logging.getLogger(__name__)
        self.full_idx = pd.Series(True, index=self.weights.index)
        self.splitting_strategy = splitting_strategy

    def bootstrap_arrays(self, n: int, *vecs):
        assets = self.weights.index.get_level_values(0).unique()
        np.random.seed(42)
        for i in range(n):
            boot_assets = np.random.choice(assets, len(assets))
            yield tuple(
                np.concatenate([vec[self.weights.index.get_loc(boot_asset)] for boot_asset in boot_assets]) for vec in
                vecs)

    def bootstrap_betas(self):
        return bootstrap_betas(self.features, self.response.ts, self.weights)

    def eval_features(self):
        cols = self.features.columns
        ncols = len(cols)
        f, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 5))

        corr_df = pd.DataFrame(index=cols, columns=cols)
        w = self.weights
        for i, colx in enumerate(cols):
            for coly in cols[i + 1:]:
                x = self.features[colx]
                y = self.features[coly]
                corr_df.loc[colx, coly] = corr(x, y, w)
                corr_df.loc[coly, colx] = corr_df.loc[colx, coly]
            corr_df.loc[colx, colx] = 1
        print(corr_df)

        for betas, ax, name in zip(self.bootstrap_betas(), axs, self.features.columns):
            ax.hist(betas)
            ax.set_title(f'{name}')
            ax.grid()
        plt.show();

    def make_train_val_splits(self, val_frac=0.3):

        return make_train_val_splits(self.response,
                                     self.weights,
                                     val_frac=val_frac,
                                     oos_time=self.oos_time,
                                     splitting_strategy=self.splitting_strategy)

    def predict(self, m, test_idx):
        return pd.Series(m.predict(self.features[test_idx]), index=test_idx.index)

    def test_model(self, m, test_idx):
        return eval_fitted_model(m, self.features[test_idx], self.response[test_idx], self.weights[test_idx],
                                 self.features[test_idx], self.response[test_idx], self.weights[test_idx])

    def fit_model(self, model, train_idx, fitted_response_transform: Optional[Callable[[pd.DataFrame, pd.Series, pd.Series], pd.Series]],
                  reports=True, weight_argname='sample_weight'):

        X_train = self.features[train_idx]
        w_train = self.weights[train_idx]
        y_train = fitted_response_transform(X_train, self.response.ts[train_idx], w_train)

        if reports:
            pca = PCA().fit(X_train)
            self.logger.info(f'Partial variance ratios: {pca.explained_variance_ratio_.cumsum()[:-1]}')

            data = [corr(X_train[col], y_train, w_train) for col in X_train]
            # corrs = pd.Series(data, index=X_train.columns)
            # print(corrs)

        weight_arg = {weight_argname: w_train}
        return model.fit(X_train, y_train, **weight_arg)

    def fit_and_eval_model(self, model, train_idx, test_idx,
                           fitted_response_transform: Optional[Callable[[pd.DataFrame, pd.Series, pd.Series], pd.Series]],
                           eval_response_transform: Optional[Callable[[pd.Series, pd.Series], pd.Series]],
                           test_only=False, reports=True, weight_argname='sample_weight', reports_args = dict()):

        m = self.fit_model(model, train_idx, fitted_response_transform=fitted_response_transform,
                           reports=reports, weight_argname=weight_argname)

        if test_only:
            Xfull, yfull, wfull = self.features[test_idx], self.response.ts[test_idx], self.weights[test_idx]
        else:
            Xfull, yfull, wfull = self.features, self.response, self.weights

        return eval_fitted_model(m, self.features[test_idx],
                                 eval_response_transform(self.response.ts[test_idx], self.weights[test_idx]),
                                 self.weights[test_idx],
                                 Xfull, yfull, wfull, reports=reports, reports_args=reports_args)

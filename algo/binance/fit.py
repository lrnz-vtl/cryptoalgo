import logging
from dataclasses import dataclass
from typing import Callable, Optional, Any
import numpy as np
import pandas as pd
import scipy.stats.mstats
from algo.cpp.cseries import shift_forward, compute_ema, compute_expsum
from sklearn.linear_model._base import LinearModel
from algo.binance.compute_betas import BetaStore
from algo.binance.features import ms_in_hour, FeatureOptions, features_from_data

max_lag_hours = 48

ms_in_month = ms_in_hour * 30 * 24
train_months = 8


class NotEnoughDataException(Exception):
    pass


@dataclass
class TrainTestData:
    ypred: pd.Series
    ytrue: pd.Series


@dataclass
class FitResults:
    train: TrainTestData
    test: TrainTestData
    fitted_model: LinearModel


@dataclass
class ModelOptions:
    get_lm: Callable[[], LinearModel]
    cap_oos_quantile: Optional[float]
    transform_fit_target: Optional[Callable] = None
    transform_model_after_fit: Optional[Callable] = None


@dataclass
class ResidOptions:
    market_pairs: set[str]
    hours_forward: int = 1


@dataclass
class UniverseDataOptions:
    demean: bool
    forward_hour: int
    target_scaler: Optional[Callable[[], Any]]


@dataclass
class FitData:
    train_target: pd.Series
    test_target: pd.Series
    features_train: pd.DataFrame
    features_test: pd.DataFrame


@dataclass
class GlobalPredTargets:
    orig_target_train: pd.Series
    orig_target_test: pd.Series
    global_pred_train: pd.Series
    global_pred_test: pd.Series


@dataclass
class ProductFitData:
    data: FitData
    vol_rescaling: Optional[float]
    global_pred_targets: Optional[GlobalPredTargets]


@dataclass
class UniverseFitData:
    train_targets: dict[str, pd.Series]
    test_targets: dict[str, pd.Series]
    vol_rescalings: Optional[dict[str, float]]


@dataclass
class UniverseFitResults:
    train_targets: dict[str, pd.Series]
    test_targets: dict[str, pd.Series]
    vol_rescalings: Optional[dict[str, float]]
    res_global: Optional[FitResults]


def fit_eval_model(data: FitData,
                   opt: ModelOptions):
    lm = opt.get_lm()

    if opt.transform_fit_target:
        ytrain = pd.Series(opt.transform_fit_target(data.train_target), index=data.train_target.index)
    else:
        ytrain = data.train_target

    lm.fit(data.features_train, ytrain)

    if opt.transform_model_after_fit:
        lm = opt.transform_model_after_fit(lm)

    ypred_train = pd.Series(lm.predict(data.features_train), index=data.train_target.index)
    ypred_test = pd.Series(lm.predict(data.features_test), index=data.test_target.index)

    return FitResults(
        fitted_model=lm,
        train=TrainTestData(ypred=ypred_train.rename('ypred'),
                            ytrue=data.train_target.rename('ytrue')),
        test=TrainTestData(ypred=ypred_test.rename('ypred'),
                           ytrue=data.test_target.rename('ytrue')),
    )


class ProductDataStore:

    def __init__(self, df: pd.DataFrame, ema_options: FeatureOptions):
        self.logger = logging.getLogger(__name__)

        self.price_ts = ((df['Close'] + df['Open']) / 2.0).rename('price')
        self.feature_df = features_from_data(df, ema_options)

        end_train_time = self.feature_df.index.min() + train_months * ms_in_month
        self.start_test_time = end_train_time + max_lag_hours * ms_in_hour

        self.train_idx = self.feature_df.index < end_train_time
        self.test_idx = self.feature_df.index > self.start_test_time

        if self.test_idx.sum() == 0:
            raise NotEnoughDataException()

    def make_target(self, forward_hour: int) -> tuple[pd.Series, pd.Series]:
        fms = ms_in_hour * forward_hour
        forward_price_0 = shift_forward(self.price_ts.index, self.price_ts.values, 1)
        forward_price = shift_forward(self.price_ts.index, self.price_ts.values, fms)
        target = pd.Series(np.log(forward_price) - np.log(forward_price_0), index=self.price_ts.index)

        return target.loc[self.train_idx], target.loc[self.test_idx]


class UniverseDataStore:
    def __init__(self, df: pd.DataFrame, ema_options: FeatureOptions, resid_options: Optional[ResidOptions]):

        self.logger = logging.getLogger(__name__)

        price_ts = ((df['Close'] + df['Open']) / 2.0).rename('price')

        assert len(price_ts.index.names) == 2
        pairs = price_ts.index.get_level_values('pair').unique()

        self.pds = {}
        self.pairs = []

        min_test_time = max(price_ts.index.get_level_values(1))

        mkt_features = []
        for pair in resid_options.market_pairs:
            assert pair in pairs
            mkt_features.append(features_from_data(df.loc[pair], ema_options))

        if mkt_features:
            self.mkt_features = pd.concat(mkt_features, axis=1)
        else:
            self.mkt_features = None

        for pair in pairs:
            if pair in resid_options.market_pairs:
                continue
            try:
                ds = ProductDataStore(df.loc[pair], ema_options)
                min_test_time = min(min_test_time, ds.start_test_time)
            except NotEnoughDataException as e:
                self.logger.warning(f'Not enough data for {pair=}. Skipping.')
                continue

            self.pairs.append(pair)
            self.pds[pair] = ds

        self.bs = None
        if resid_options:
            self.bs = BetaStore(price_ts,
                                min_test_time=min_test_time,
                                hours_forward=resid_options.hours_forward)

    def prepare_data(self, fit_options: UniverseDataOptions) -> UniverseFitData:
        train_targets = {}
        test_targets = {}

        pairs = self.pairs
        for pair, ds in self.pds.items():
            train_targets[pair], test_targets[pair] = ds.make_target(forward_hour=fit_options.forward_hour)

        if self.bs:
            for pair in pairs:
                beta = self.bs.compute_beta(train_targets[pair])
                train_targets[pair] = self.bs.residualise(beta, train_targets[pair])
                test_targets[pair] = self.bs.residualise(beta, test_targets[pair])

        if fit_options.demean:
            train_ret_df = pd.DataFrame(train_targets)
            test_ret_df = pd.DataFrame(test_targets)

            for pair in pairs:
                train_targets[pair] = train_targets[pair] - train_ret_df.mean(axis=1).loc[train_targets[pair].index]
                test_targets[pair] = test_targets[pair] - test_ret_df.mean(axis=1).loc[test_targets[pair].index]

            del train_ret_df
            del test_ret_df

        if fit_options.target_scaler is not None:
            vol_rescalings = {}
            for pair in pairs:
                vol_rescalings[pair] = \
                    fit_options.target_scaler().fit(train_targets[pair].values.reshape(-1, 1)).scale_[0]
                train_targets[pair] /= vol_rescalings[pair]
                test_targets[pair] /= vol_rescalings[pair]
        else:
            vol_rescalings = None

        return UniverseFitData(train_targets=train_targets,
                               test_targets=test_targets,
                               vol_rescalings=vol_rescalings)

    def prepare_data_global(self, ufd: UniverseFitData) -> FitData:
        pairs = self.pairs

        train_target = pd.concat((ufd.train_targets[pair] for pair in pairs), axis=0)
        test_target = pd.concat((ufd.test_targets[pair] for pair in pairs), axis=0)

        features_train = pd.concat((self.pds[pair].feature_df[self.pds[pair].train_idx] for pair in pairs),
                                   axis=0)
        features_test = pd.concat((self.pds[pair].feature_df[self.pds[pair].test_idx] for pair in pairs),
                                  axis=0)

        if self.mkt_features is not None:
            features_train = pd.concat([features_train, self.mkt_features.loc[features_train.index].fillna(0)],
                                       axis=1)
            features_test = pd.concat([features_test, self.mkt_features.loc[features_test.index].fillna(0)], axis=1)

        return FitData(train_target=train_target, test_target=test_target,
                       features_train=features_train, features_test=features_test)

    def gen_product_data(self, ufd: UniverseFitData, global_fit_results: Optional[FitResults]):

        for pair, ds in self.pds.items():
            train_target = ufd.train_targets[pair]
            test_target = ufd.test_targets[pair]

            features_train = ds.feature_df[ds.train_idx]
            features_test = ds.feature_df[ds.test_idx]

            if self.mkt_features is not None:
                features_train = pd.concat([features_train, self.mkt_features.loc[features_train.index].fillna(0)],
                                           axis=1)
                features_test = pd.concat([features_test, self.mkt_features.loc[features_test.index].fillna(0)], axis=1)

            if global_fit_results is not None:
                train_global_pred = global_fit_results.fitted_model.predict(features_train)
                test_global_pred = global_fit_results.fitted_model.predict(features_test)

                train_residual = train_target - train_global_pred
                test_residual = test_target - test_global_pred

                fit_data = FitData(train_residual,
                                   test_residual,
                                   features_train,
                                   features_test,
                                   )

                yield pair, ProductFitData(data=fit_data,
                                           vol_rescaling=ufd.vol_rescalings[pair] if ufd.vol_rescalings else None,
                                           global_pred_targets=GlobalPredTargets(
                                               global_pred_train=train_global_pred,
                                               global_pred_test=test_global_pred,
                                               orig_target_train=train_target,
                                               orig_target_test=test_target)
                                           )

            else:
                fit_data = FitData(train_target,
                                   test_target,
                                   features_train,
                                   features_test,
                                   )
                yield pair, ProductFitData(fit_data,
                                           vol_rescaling=ufd.vol_rescalings[pair] if ufd.vol_rescalings else None,
                                           global_pred_targets=None)


def fit_product(pfd: ProductFitData, model_options: ModelOptions):
    res = fit_eval_model(pfd.data,
                         model_options)

    if pfd.global_pred_targets:
        res.train.ypred = res.train.ypred + pfd.global_pred_targets.global_pred_train
        res.train.ytrue = pfd.global_pred_targets.orig_target_train.rename('ytrue')

        res.test.ypred = res.test.ypred + pfd.global_pred_targets.global_pred_test
        res.test.ytrue = pfd.global_pred_targets.orig_target_test.rename('ytrue')

    # double-check the train prediction is demeaned
    if model_options.cap_oos_quantile is not None:
        wins = scipy.stats.mstats.winsorize(res.train.ypred, model_options.cap_oos_quantile)
        lo = wins.min()
        hi = wins.max()
        res.test.ypred[res.test.ypred < lo] = lo
        res.test.ypred[res.test.ypred > hi] = hi

    if pfd.vol_rescaling is not None:
        res.test.ypred *= pfd.vol_rescaling
        res.test.ytrue *= pfd.vol_rescaling

    assert (res.train.ypred.index == res.train.ytrue.index).all()
    assert (res.test.ypred.index == res.test.ytrue.index).all()

    return res

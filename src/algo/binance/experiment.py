import datetime
import unittest
from pathlib import Path
from typing import Any, Callable, Collection

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from algo.binance.coins import Universe, MarketType
from algo.binance.data_types import DataType
from algo.binance.dataloader import load_universe_data
from algo.binance.features import FeatureOptions, VolumeOptions
from algo.binance.fit import UniverseDataOptions, ResidOptions, UniverseDataStore, ModelOptions, fit_eval_model, \
    fit_product, ProductFitData, ProductFitResult
from algo.binance.model import ProductModel
from algo.binance.utils import TrainTestOptions
from pydantic import BaseModel
from algo.definitions import ROOT_DIR

EXP_BASEP = Path(ROOT_DIR) / 'experiments'


class ExpArgs(BaseModel):
    universe: Universe
    start_date: datetime.datetime
    end_date: datetime.datetime
    feature_options: FeatureOptions
    ro: ResidOptions
    ud_options: UniverseDataOptions
    market_type: MarketType
    data_type: DataType
    tto: TrainTestOptions

    class Config:
        arbitrary_types_allowed = True


def fit_eval_products(product_data: dict[str, ProductFitData], opt: ModelOptions) -> tuple[
    dict[str, ProductFitResult], float]:
    ress = {pair: fit_product(x, opt) for pair, x in product_data.items()}

    totdf = pd.concat((pd.concat([x.res.test.ytrue, x.res.test.ypred], axis=1) for pair, x in ress.items()), axis=0)
    score = r2_score(totdf.ytrue, totdf.ypred)

    return ress, score


class Experiment:
    # @profile
    def __init__(self, args: ExpArgs):
        self.global_fit_results = None
        self.args = args

        generator = load_universe_data(args.universe, args.start_date, args.end_date, args.market_type, args.data_type)

        self.uds = UniverseDataStore(generator, args.feature_options, args.tto, args.ro)

        self.ufd = self.uds.prepare_data(args.ud_options)
        self.global_data = self.uds.prepare_data_global(self.ufd)

    # @profile
    def fit_eval_alpha_global(self, global_opt: ModelOptions):
        global_fit_results = fit_eval_model(self.global_data, global_opt)
        score = r2_score(global_fit_results.test.ytrue, global_fit_results.test.ypred)

        return global_fit_results, score

    # @profile
    def make_product_data(self, global_opt: ModelOptions) -> tuple[float, dict[str, ProductFitData]]:
        self.global_fit_results, score = self.fit_eval_alpha_global(global_opt)

        product_data = {pair: x for pair, x in self.uds.gen_product_data(self.ufd, self.global_fit_results)}
        return score, product_data


class Validator:

    def __init__(self, foo: Callable[[Any], float]):
        self.foo = foo
        self.scores = {}

    def validate(self, params: Collection[Any]):
        for param in params:
            if param not in self.scores:
                self.scores[param] = self.foo(param)

    def current_max(self):
        params = list(self.scores.keys())
        scores = list(self.scores.values())
        idx = np.argmax(scores)
        return params[idx], scores[idx]

    def sorted_items(self):
        params = list(sorted(self.scores.keys()))
        scores = [v for _, v in sorted(self.scores.items())]
        return params, scores


class TestExperiment(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        n_coins = 10
        vol = VolumeOptions(include_imbalance=True, include_logretvol=True)

        feature_options = FeatureOptions(decay_hours=[4, 12, 24, 48, 96], volume_options=vol)
        ro = ResidOptions(market_pairs=['BTCUSDT', 'ETHUSDT'])

        forward_hour = 24

        ud_options = UniverseDataOptions(demean=True,
                                         forward_hour=forward_hour,
                                         vol_scaling=True
                                         )
        start_date = datetime.datetime(year=2022, month=1, day=1)
        end_date = datetime.datetime(year=2022, month=6, day=1)

        tto = TrainTestOptions(
            train_end_time=datetime.datetime(year=2022, month=4, day=1),
            test_start_time=datetime.datetime(year=2022, month=4, day=3),
            min_train_period=datetime.timedelta(days=30 * 2)
        )

        self.exp_args = ExpArgs(
            mcap_date=datetime.date(year=2022, month=1, day=1),
            n_coins=n_coins,
            start_date=start_date,
            end_date=end_date,
            feature_options=feature_options,
            ro=ro,
            ud_options=ud_options,
            spot=False,
            tto=tto
        )
        super().__init__(*args, **kwargs)

    def test_a(self):
        def transform_model_after_fit(lm):
            assert hasattr(lm['ridge'], 'intercept_')
            lm['ridge'].intercept_ = 0
            return lm

        global_opt = ModelOptions(
            get_lm=lambda: make_pipeline(StandardScaler(), Ridge(alpha=0)),
            transform_fit_target=lambda y: winsorize(y, 0.1),
            transform_model_after_fit=transform_model_after_fit,
            cap_oos_quantile=None
        )

        exp = Experiment(self.exp_args)

        score, product_data = exp.make_product_data(global_opt)
        print(f'global {score=}')

        def validate(alpha: float):
            opt = ModelOptions(
                get_lm=lambda: make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
                transform_fit_target=lambda y: winsorize(y, 0.1),
                transform_model_after_fit=transform_model_after_fit,
                cap_oos_quantile=None
            )
            _, score = fit_eval_products(product_data, opt)
            return score

        val = Validator(validate)

        alphas = np.logspace(5, 7, 10)
        val.validate(alphas)

        plt.plot(np.log(list(val.scores.keys())), list(val.scores.values()))
        plt.grid()

        best_alpha, best_score = val.current_max()

        opt = ModelOptions(
            get_lm=lambda: make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)),
            transform_fit_target=lambda y: winsorize(y, 0.1),
            transform_model_after_fit=transform_model_after_fit,
            cap_oos_quantile=None
        )

        ress, score = fit_eval_products(product_data, opt)

        print(f'{score=}')

        for pair, res in ress.items():
            Xtest = product_data[pair].data.features_test
            model = ProductModel.from_fit_results(res, exp.global_fit_results,
                                                  column_names=product_data[pair].data.features_test.columns)
            ytest = model.predict(Xtest)

            assert len(Xtest.columns) == len(set(Xtest.columns))

            np.testing.assert_allclose(ytest, res.res.test.ypred)

    def test_b(self):
        x1 = self.exp_args.json()

        filename = Path(ROOT_DIR) / 'test.txt'

        # with tempfile.TemporaryFile('w') as fp:
        with open(filename, 'w') as fp:
            fp.write(x1)

        with open(filename) as fp:
            x2 = fp.read()
            print(x2)
            assert x1 == x2

            exp_args2 = ExpArgs.parse_raw(x2)
            print(exp_args2)

        exp_args3 = ExpArgs.parse_file(filename)
        assert exp_args3 == self.exp_args

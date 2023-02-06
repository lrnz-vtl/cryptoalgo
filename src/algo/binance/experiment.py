import datetime
import unittest
from dataclasses import dataclass
from typing import Any, Callable, Collection

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from algo.binance.coins import Universe, load_universe_candles
from algo.binance.evaluation import plot_eval
from algo.binance.features import FeatureOptions, VolumeOptions
from algo.binance.fit import UniverseDataOptions, ResidOptions, UniverseDataStore, ModelOptions, fit_eval_model, \
    fit_product, ProductFitData


@dataclass
class ExpArgs:
    mcap_date: datetime.date
    n_coins: int
    start_date: datetime.datetime
    end_date: datetime.datetime
    feature_options: FeatureOptions
    ro: ResidOptions
    ud_options: UniverseDataOptions
    spot: bool


def fit_eval_products(product_data: dict[str, ProductFitData], opt: ModelOptions):
    ress = {pair: fit_product(x, opt) for pair, x in product_data.items()}

    totdf = pd.concat((pd.concat([x.test.ytrue, x.test.ypred], axis=1) for pair, x in ress.items()), axis=0)
    score = r2_score(totdf.ytrue, totdf.ypred)

    return ress, score


class Experiment:
    def __init__(self, args: ExpArgs):
        self.args = args

        universe = Universe.make(args.n_coins, args.mcap_date)

        time_col = 'Close time'

        df = (
            load_universe_candles(universe, args.start_date, args.end_date, '5m', args.spot)
            .set_index(['pair', time_col])
        )
        self.uds = UniverseDataStore(df, args.feature_options, args.ro)

        self.ufd = self.uds.prepare_data(args.ud_options)
        self.global_data = self.uds.prepare_data_global(self.ufd)

    def fit_eval_alpha_global(self, global_opt: ModelOptions):
        global_fit_results = fit_eval_model(self.global_data, global_opt)
        score = r2_score(global_fit_results.test.ytrue, global_fit_results.test.ypred)

        return global_fit_results, score

    def make_product_data(self, global_opt: ModelOptions) -> tuple[float, dict[str, ProductFitData]]:
        global_fit_results, score = self.fit_eval_alpha_global(global_opt)

        product_data = {pair: x for pair, x in self.uds.gen_product_data(self.ufd, global_fit_results)}
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
    def test_a(self):
        vol = VolumeOptions(include_imbalance=True, include_logretvol=True)
        feature_options = FeatureOptions([4, 12, 24, 48, 96], vol)
        ro = ResidOptions(market_pairs=['BTCUSDT', 'ETHUSDT'])

        forward_hour = 24

        ud_options = UniverseDataOptions(demean=True,
                                         forward_hour=forward_hour,
                                         target_scaler=lambda: RobustScaler()
                                         )

        exp_args = ExpArgs(
            mcap_date=datetime.date(year=2022, month=1, day=1),
            n_coins=20,
            start_date=datetime.datetime(year=2022, month=1, day=1),
            end_date=datetime.datetime(year=2023, month=1, day=1),
            feature_options=feature_options,
            ro=ro,
            ud_options=ud_options,
            spot=False
        )

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

        exp = Experiment(exp_args)

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
        plt.show()
        plt.clf()

        best_alpha, best_score = val.current_max()

        opt = ModelOptions(
            get_lm=lambda: make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)),
            transform_fit_target=lambda y: winsorize(y, 0.1),
            transform_model_after_fit=transform_model_after_fit,
            cap_oos_quantile=None
        )

        ress, score = fit_eval_products(product_data, opt)

        print(f'{score=}')

        # plot_eval(ress);

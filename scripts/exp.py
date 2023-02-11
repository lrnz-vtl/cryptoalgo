import argparse
import datetime
import json
import logging
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from algo.binance.coins import SpotType, Universe, MarketType, FutureType
from algo.binance.data_types import KlineType, DataType, AggTradesType
from algo.binance.evaluation import plot_eval
from algo.binance.experiment import ExpArgs, Experiment, fit_eval_products, Validator, EXP_BASEP
from algo.binance.features import VolumeOptions, FeatureOptions
from algo.binance.fit import ResidOptions, UniverseDataOptions, ModelOptions
from algo.binance.model import ProductModel
from algo.binance.utils import TrainTestOptions


def make_exp_args(n_coins: int, market_type: MarketType, data_type: DataType, test: bool,
                  lookahead: bool = False):
    if test:
        vol = VolumeOptions(include_imbalance=False, include_logretvol=False)

        start_date = datetime.datetime(year=2022, month=1, day=1)
        end_date = datetime.datetime(year=2022, month=7, day=1)

        tto = TrainTestOptions(
            train_end_time=datetime.datetime(year=2022, month=5, day=1),
            test_start_time=datetime.datetime(year=2022, month=5, day=3),
            min_train_period=datetime.timedelta(days=30 * 3)
        )

    else:
        vol = VolumeOptions(include_imbalance=True, include_logretvol=True)

        start_date = datetime.datetime(year=2022, month=1, day=1)
        end_date = datetime.datetime(year=2023, month=1, day=1)

        tto = TrainTestOptions(
            train_end_time=datetime.datetime(year=2022, month=8, day=1),
            test_start_time=datetime.datetime(year=2022, month=8, day=3),
            min_train_period=datetime.timedelta(days=30 * 6)
        )

    # feature_options = FeatureOptions(decay_hours=[4, 12, 24, 48, 96], volume_options=vol)
    feature_options = FeatureOptions(decay_hours=[1 / 6, 4, 12, 24, 48, 96], volume_options=vol, include_current=False)
    ro = ResidOptions(market_pairs=['BTCUSDT', 'ETHUSDT'])

    forward_hour = 24

    ud_options = UniverseDataOptions(demean=True,
                                     forward_hour=forward_hour,
                                     vol_scaling=True
                                     )

    if lookahead:
        universe = Universe.make_lookahead(n_coins)
    else:
        mcap_date = datetime.date(year=2022, month=1, day=1)
        universe = Universe.make(n_coins, mcap_date=mcap_date, market_type=market_type, data_type=data_type, )

    exp_args = ExpArgs(
        universe=universe,
        market_type=market_type,
        data_type=data_type,
        start_date=start_date,
        end_date=end_date,
        feature_options=feature_options,
        ro=ro,
        ud_options=ud_options,
        tto=tto
    )

    return exp_args


def run_experiment(exp: Experiment, dst_path: Path):
    logger = logging.getLogger(__name__)

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

    score, product_data = exp.make_product_data(global_opt)
    logger.info(f'global {score=}')

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
    plt.savefig(dst_path / 'scores.png')

    best_alpha, best_score = val.current_max()

    opt = ModelOptions(
        get_lm=lambda: make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)),
        transform_fit_target=lambda y: winsorize(y, 0.1),
        transform_model_after_fit=transform_model_after_fit,
        cap_oos_quantile=None
    )

    ress, score = fit_eval_products(product_data, opt)

    logger.info(f'{score=}')

    models = {}
    for pair, res in ress.items():
        Xtest = product_data[pair].data.features_test
        models[pair] = ProductModel.from_fit_results(res, exp.global_fit_results,
                                                     column_names=Xtest.columns)
    pd.to_pickle(models, dst_path / 'models.pkl')

    if exp.ufd.betas is not None:
        r = json.dumps(exp.ufd.betas, indent=4)
        with open(dst_path / 'betas.json', 'w') as f:
            f.write(r)

    pd.to_pickle(ress, dst_path / 'results.pkl')

    plot_eval({k: v.res for k, v in ress.items()}, dst_path / 'plots')


def run(name: str, n_coins: int, market_type: MarketType, data_type: DataType, test: bool,
        lookahead: bool = False):
    fmt = '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'

    dst_path = EXP_BASEP / name
    os.makedirs(dst_path, exist_ok=True)

    logging.basicConfig(
        filename=dst_path / 'log.log',
        level=logging.DEBUG,
        format=fmt,
        datefmt='%H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    exp_args = make_exp_args(n_coins, market_type, data_type, test, lookahead)

    with open(dst_path / 'exp_args.json', 'w') as f:
        f.write(exp_args.json(indent=4))

    exp = Experiment(exp_args)
    run_experiment(exp, dst_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('name')
    parser.add_argument('--n-coins', type=int, required=True)
    parser.add_argument('--spot', action='store_true')
    parser.add_argument('--agg', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    spot = args.spot
    if spot:
        market_type = SpotType()
    else:
        market_type = FutureType()

    if args.agg:
        data_type = AggTradesType()
    else:
        data_type = KlineType(freq='5m')

    run(args.name, args.n_coins, market_type, data_type=data_type, lookahead=False, test=args.test)


class TestExp(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                            level=logging.INFO)

        self.logger = logging.getLogger(__name__)

    def test_a(self):
        market_type = SpotType()
        data_type = KlineType(freq='5m')
        run(name='test_spot_220211', n_coins=4, market_type=market_type, data_type=data_type, test=True, lookahead=True)

    def test_a1(self):
        market_type = SpotType()
        data_type = AggTradesType()
        run(name='test_spot_agg_220211', n_coins=4, market_type=market_type, data_type=data_type, test=True, lookahead=True)

    def test_a2(self):
        market_type = FutureType()
        data_type = KlineType(freq='5m')
        run(name='test_220211', n_coins=4, market_type=market_type, data_type=data_type, test=True, lookahead=True)

    def test_b(self):
        market_type = SpotType()
        data_type = KlineType(freq='5m')
        run(name='test100_220211', n_coins=100, market_type=market_type, data_type=data_type, test=True, lookahead=True)

    def test_b1(self):
        market_type = SpotType()
        data_type = AggTradesType()
        run(name='test100_220211', n_coins=100, market_type=market_type, data_type=data_type, test=True, lookahead=True)
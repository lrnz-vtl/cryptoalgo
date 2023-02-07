import argparse
import datetime
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from algo.binance.evaluation import plot_eval
from algo.binance.experiment import ExpArgs, Experiment, fit_eval_products, Validator
from algo.binance.features import VolumeOptions, FeatureOptions
from algo.binance.fit import ResidOptions, UniverseDataOptions, ModelOptions
from algo.binance.model import ProductModel
from algo.binance.utils import TrainTestOptions
from algo.definitions import ROOT_DIR

EXP_BASEP = Path(ROOT_DIR) / 'experiments'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('name')
    parser.add_argument('-n', type=int, required=True)
    parser.add_argument('--spot', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    format = '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'

    n_coins = args.n
    spot = args.spot
    name = args.name

    dst_path = EXP_BASEP / name
    os.makedirs(dst_path, exist_ok=True)

    logging.basicConfig(
        filename=dst_path / 'log.log',
        level=logging.INFO,
        format=format,
        datefmt='%H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(format)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)

    if args.test:
        vol = VolumeOptions(include_imbalance=False, include_logretvol=False)

        start_date = datetime.datetime(year=2022, month=1, day=1)
        end_date = datetime.datetime(year=2022, month=6, day=1)

        tto = TrainTestOptions(
            train_end_time=datetime.datetime(year=2022, month=4, day=1),
            test_start_time=datetime.datetime(year=2022, month=4, day=3),
            min_train_period=datetime.timedelta(days=3 * 6)
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

    feature_options = FeatureOptions(decay_hours=[4, 12, 24, 48, 96], volume_options=vol)
    ro = ResidOptions(market_pairs=['BTCUSDT', 'ETHUSDT'])

    forward_hour = 24

    ud_options = UniverseDataOptions(demean=True,
                                     forward_hour=forward_hour,
                                     vol_scaling=True
                                     )

    exp_args = ExpArgs(
        mcap_date=datetime.date(year=2022, month=1, day=1),
        n_coins=n_coins,
        start_date=start_date,
        end_date=end_date,
        feature_options=feature_options,
        ro=ro,
        ud_options=ud_options,
        spot=spot,
        tto=tto
    )
    with open(dst_path / 'exp_args.json', 'w') as f:
        f.write(exp_args.json(indent=4))

    exp = Experiment(exp_args)


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
                                                     column_names=product_data[pair].data.features_test.columns)
    pd.to_pickle(models, dst_path / 'models.pkl')

    if exp.ufd.betas is not None:
        r = json.dumps(exp.ufd.betas, indent=4)
        with open(dst_path / 'betas.json', 'w') as f:
            f.write(r)

    plot_eval({k: v.res for k, v in ress.items()})
    plt.savefig(dst_path / 'evaluations.png')

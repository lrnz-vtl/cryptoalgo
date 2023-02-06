import datetime

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from algo.binance.experiment import ExpArgs, Experiment, fit_eval_products, Validator
from algo.binance.features import VolumeOptions, FeatureOptions
from algo.binance.fit import ResidOptions, UniverseDataOptions, ModelOptions

if __name__ == '__main__':

    plot = False
    spot = False

    vol = VolumeOptions(include_imbalance=False, include_logretvol=False)
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
        spot=spot
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

    if plot:
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

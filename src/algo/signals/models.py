from typing import Callable
import numpy as np
import optuna
import pandas as pd
from matplotlib import pyplot as plt

from algo.signals.constants import ASSET_INDEX_NAME
from algo.signals.evaluation import FittableDataStore
from scipy.stats.mstats import winsorize


class RemoveIntercept:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X, y, **kwargs):
        self.pipeline.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.pipeline.predict(X) - self.pipeline.intercept_

    def __getattr__(self, attr):
        if attr == 'fit':
            return self.fit
        elif attr == 'predict':
            return self.predict
        elif attr == 'fit_predict':
            raise NotImplementedError
        else:
            return getattr(self.pipeline, attr)


class Demean:
    def __init__(self, pipeline, weights):
        self.pipeline = pipeline
        self.weights = weights
        self.wsums = weights.groupby('time_5min').sum()

    def predict(self, X: pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            y = pd.Series(self.pipeline.predict(X), index=X.index)
        elif isinstance(X, np.ndarray):
            y = self.pipeline.predict(X)
        else:
            raise ValueError
        ywsums = (y * self.weights).groupby('time_5min').sum()
        return y - ywsums / self.wsums

    def __getattr__(self, attr):
        if attr == 'predict':
            return self.predict
        elif attr == 'fit_predict':
            raise NotImplementedError
        else:
            return getattr(self.pipeline, attr)


class SimpleTransform:
    def __init__(self, foo):
        self.foo = foo

    def fit(self, X, y, **kwargs):
        return self

    def transform(self, X):
        return self.foo(X)


class Exponentiate:
    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, X):
        return pd.Series(np.exp(self.predictor.predict(X)) - 1, index=X.index)

    def fit(self, *args, **kwargs):
        self.predictor.fit(*args, **kwargs)
        return self


class FeatureSelect:

    def __init__(self, subcols):
        self.subcols = subcols

    def fit(self, X, y, **kwargs):
        return self

    def transform(self, X):
        return X[self.subcols]


class Winsorize:
    def __init__(self, limits):
        self.limits = limits
        self.minmax = []

    def fit(self, X, *args, **kwargs):
        for col in X:
            x_wins = winsorize(X[col], limits=self.limits)
            self.minmax.append((x_wins.min(), x_wins.max()))
        return self

    def transform(self, X, *args, **kwargs):
        assert isinstance(X, pd.DataFrame)
        return pd.DataFrame(
            np.array([np.clip(X[col], mmin, mmax) for (col, (mmin, mmax)) in zip(X.columns, self.minmax)]).transpose(),
            index=X.index,
            columns=X.columns
        )


class KeepColumns:
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, *args, **kwargs):
        self.transformer.fit(X, *args, **kwargs)
        return self

    def transform(self, X):
        return pd.DataFrame(self.transformer.transform(X), columns=X.columns, index=X.index)

    def predict(self, X):
        return pd.Series(self.transformer.predict(X), index=X.index)


class StackedModel:
    def __init__(self, model1, model2, residual_transf):
        self.model1 = model1
        self.model2 = model2
        self.residual_transf = residual_transf

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series):
        y1 = self.model1.fit(X, y, sample_weight=sample_weight).predict(X)
        y1 = pd.Series(y1, index=y.index)
        res = y - y1
        res = self.residual_transf(X, res, sample_weight)
        assert isinstance(res, pd.Series)
        self.model2.fit(X, res, sample_weight=sample_weight)
        return self

    def predict(self, X):
        pred1 = pd.Series(self.model1.predict(X), index=X.index)
        pred2 = self.model2.predict(X)
        return pred1 + pred2


class ById:
    def __init__(self, model_cls, params):
        self.model_cls = model_cls
        self.params = params
        self.fitted_models = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series):
        for aid in X.index.get_level_values(ASSET_INDEX_NAME).unique():
            self.fitted_models[aid] = self.model_cls(**self.params).fit(X.loc[aid], y.loc[aid], sample_weight.loc[aid])
        return self

    def predict(self, X):
        return X.groupby(ASSET_INDEX_NAME).apply(
            lambda x: pd.Series(self.fitted_models[x.name].predict(x), index=x.index.droplevel(0)))


class Tuner:

    def __init__(self, fitds: FittableDataStore,
                 train_idx,
                 test_idx,
                 pipeline_from_trial: Callable,
                 weight_argname,
                 val_error):
        self.study = None
        self.val_error = val_error
        self.fitds = fitds
        self.pipeline_from_trial = pipeline_from_trial
        self.test_idx = test_idx
        self.train_idx = train_idx
        self.weight_argname = weight_argname

    def cv_score(self, pipeline, resp_transf, eval_response_transform):
        X = self.fitds.features[self.train_idx]
        w = self.fitds.weights[self.train_idx]
        y = resp_transf(X, self.fitds.response.ts[self.train_idx], w)

        m = pipeline.fit(X=X, y=y, **{self.weight_argname: w})
        return self.val_error(
            eval_response_transform(self.fitds.response.ts[self.test_idx], self.fitds.weights[self.test_idx]),
            m.predict(self.fitds.features[self.test_idx]),
            sample_weight=self.fitds.weights[self.test_idx]
        )

    def cv_trial_score(self, trial):
        pipeline, resp_transf, eval_response_transform = self.pipeline_from_trial(trial)
        return self.cv_score(pipeline, resp_transf, eval_response_transform)

    def create_study(self) -> None:
        self.study = optuna.create_study()
        optuna.logging.set_verbosity(optuna.logging.ERROR)

    def optimize(self, n_trials: int):
        return self.study.optimize(self.cv_trial_score, n_trials=n_trials)


def plot_study_results(study):
    trdf = study.trials_dataframe()
    pcols = [col for col in trdf if 'params_' in col]
    f, axs = plt.subplots(1, len(pcols), figsize=(5 * len(pcols), 4))

    if len(pcols) == 1:
        axs = [axs]

    for ax, col in zip(axs, pcols):
        ax.scatter(trdf[col], trdf['value'])
        ax.set_title(col)
        ax.grid()
    plt.show()

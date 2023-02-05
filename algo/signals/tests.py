import unittest
from algo.dataloading.caching import make_filter_from_universe, join_caches_with_priority
from algo.signals.datastore import AnalysisDataStore, RollingLiquidityFilter
from algo.signals.evaluation import *
from algo.signals.featurizers import concat_featurizers
from algo.signals.responses import SimpleResponse, my_winsorize
from algo.signals.weights import SimpleWeightMaker
from algo.signals.models import RemoveIntercept
from sklearn.linear_model import LinearRegression
from algo.universe.universe import SimpleUniverse
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from algo.signals.entropy import entropies

smalluniverse_cache_name1 = 'liquid_algo_pools_nousd_prehack_noeth'


class EMALinearStrategy(BaseModel):
    betas: dict[int, float]


class TestAnalysisDs(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        price_cache = '20220209_prehack'
        # volume_cache = '20220209_prehack'

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        ffill_price_minutes = 'all'

        market_lag_seconds = 60

        self.winsor_limit = 0.06

        universe = SimpleUniverse.from_cache(smalluniverse_cache_name)

        self.ds = AnalysisDataStore([price_cache], [], universe, SimpleWeightMaker(),
                                    ffill_price_minutes=ffill_price_minutes,
                                    market_lag_seconds=market_lag_seconds)
        self.minutes = (30, 60, 120)
        self.featurizers = [MAPriceFeaturizer(m) for m in self.minutes]

        respMaker = SimpleResponse(120, 0)
        features = self.ds.make_asset_features(concat_featurizers(self.featurizers))
        response = self.ds.make_response(respMaker)

        self.fitds = self.ds.make_fittable_data(features, response, [RollingLiquidityFilter()], [], True)

        super().__init__(*args, **kwargs)

    def test_features(self):
        cols = len(self.featurizers)
        f, axs = plt.subplots(1, cols, figsize=(4 * cols, 5))

        for betas, ax, minutes in zip(self.fitds.bootstrap_betas(), axs, self.minutes):
            ax.hist(betas)
            ax.set_title(f'minutes = {minutes}')
            ax.grid()
        plt.show();

    def test_fit_and_validate_model(self):
        pipeline = Pipeline(
            (
                ('linreg', RemoveIntercept(TransformResponse(LinearRegression(),
                                                             resp_transform=lambda y: my_winsorize(y, (
                                                                 self.winsor_limit, self.winsor_limit))))),
            )
        )

        train_idx, test_idx = self.fitds.make_train_val_splits()
        self.fitds.fit_and_eval_model(pipeline, train_idx, test_idx, weight_argname=f'linreg__sample_weight')

        self.logger.info(f'betas = {pipeline.named_steps["linreg"].coef_}')


class TestOOS(unittest.TestCase):

    def _make_fitds(self, price_cache: str, universe_name: str):
        universe = SimpleUniverse.from_cache(universe_name)
        ds = AnalysisDataStore([price_cache], [], universe, SimpleWeightMaker(),
                               ffill_price_minutes=self.ffill_price_minutes,
                               market_lag_seconds=self.market_lag_seconds)
        features = ds.make_asset_features(concat_featurizers(self.featurizers))
        response = ds.make_response(self.respMaker)
        return ds.make_fittable_data(features, response, [RollingLiquidityFilter()], [], True)

    def __init__(self, *args, **kwargs):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.ffill_price_minutes = 'all'

        self.market_lag_seconds = 60
        self.winsor_limit = 0.06

        self.minutes = (30, 60, 120)
        self.featurizers = [MAPriceFeaturizer(m) for m in self.minutes]
        self.respMaker = SimpleResponse(120, 0)

        self.pipeline = Pipeline(
            (
                ('linreg', RemoveIntercept(TransformResponse(LinearRegression(),
                                                             resp_transform=lambda y: my_winsorize(y, (
                                                                 self.winsor_limit, self.winsor_limit))))),
            )
        )

        super().__init__(*args, **kwargs)

    def test_oos(self):
        price_cache = '20220209_prehack'
        universe_name = 'liquid_algo_pools_nousd_prehack_noeth'
        ds = self._make_fitds(price_cache, universe_name)

        self.pipeline = ds.fit_model(self.pipeline, ds.full_idx, weight_argname=f'linreg__sample_weight')
        self.logger.info(f'betas = {self.pipeline.named_steps["linreg"].coef_}')

        # price_cache_oos = '20220209'
        universe_name = 'update_20220225'
        price_cache_oos = '20220225'
        ds_oos = self._make_fitds(price_cache_oos, universe_name)
        ds_oos.test_model(self.pipeline, ds_oos.full_idx)


class TestFitTrading(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.cache_names = ['20220209_prehack', '20220209']

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.ffill_price_minutes = 10

        self.universe = SimpleUniverse.from_cache(smalluniverse_cache_name)

        self.minutes = (30, 60, 120)
        self.featurizers = [MAPriceFeaturizer(m) for m in self.minutes]
        self.respMaker = SimpleResponse(120, 5)

        self.model = WinsorizeResponse(LinearRegression())

        super().__init__(*args, **kwargs)

    def test_fit_all(self):
        ds = AnalysisDataStore(self.cache_names, [], self.universe, SimpleWeightMaker(),
                               ffill_price_minutes=self.ffill_price_minutes)
        features = ds.make_asset_features(concat_featurizers(self.featurizers))
        response = ds.make_response(self.respMaker)
        fitds = ds.make_fittable_data(features, response, [RollingLiquidityFilter()], [], True)

        model = WinsorizeResponse(LinearRegression())

        cols = len(self.featurizers)
        f, axs = plt.subplots(1, cols, figsize=(4 * cols, 5))

        for betas, ax, minutes in zip(fitds.bootstrap_betas(), axs, self.minutes):
            ax.hist(betas)
            ax.set_title(f'minutes = {minutes}')
            ax.grid()
        plt.show()

        model = fitds.fit_model(model, fitds.full_idx)
        self.logger.info(model.coef_, model.intercept_)

        fitds.test_model(model, fitds.full_idx)


class TestOne(unittest.TestCase):
    def test_one(self):
        smalluniverse_cache_name = 'update_20220225'
        volume_caches = ['20220225']

        universe = SimpleUniverse.from_cache(smalluniverse_cache_name)

        filter_ = make_filter_from_universe(universe)
        dfv = join_caches_with_priority(volume_caches, 'volumes', filter_)

        test = dfv[dfv['asset1'] == 470842789]
        test['time_5min'] = (test['time'] // (5 * 60))

        times = test['time_5min'].values
        amounts = abs(test['asset2_amount'].values)
        addrs = test['counterparty'].values

        times, values = entropies(times, amounts, addrs)
        print(times[:10], values[:10])

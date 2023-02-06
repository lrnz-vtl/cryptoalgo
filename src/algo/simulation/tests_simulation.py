from __future__ import annotations
import datetime
import unittest
from algo.trading.impact import ASAImpactState, PositionAndImpactState, GlobalPositionAndImpactState, \
    ASAPosition
from algo.trading.signalprovider import DummySignalProvider, EmaSignalProvider, \
    EmaSignalParam, PriceSignalProvider, RandomSignalProvider
from algo.blockchain.stream import stream_from_price_df
from datetime import timezone
from algo.universe.universe import SimpleUniverse
from algo.dataloading.caching import load_algo_pools
from algo.dataloading.caching import make_filter_from_universe
from algo.simulation.simulator import Simulator
from algo.optimizer.optimizerV2 import OptimizerV2
import logging
from algo.simulation.reports import make_simulation_results, make_simulation_reports, \
    SimulationResults, plot_aggregate_values_df
from typing import Callable
from matplotlib import pyplot as plt
from tinyman.v1.pools import Asset




def make_signal_results(make_signal: Callable[[], PriceSignalProvider],
                        price_cache_name: str, universe_cache_name: str,
                        initial_time: datetime.datetime, end_time: datetime.datetime,
                        risk_multiplier:float,
                        initial_algo_position:int):
    risk_coef = risk_multiplier * 10 ** -6
    impact_timescale_seconds = 5 * 60
    simulation_step_seconds = 5 * 60

    initial_mualgo_position = initial_algo_position * 10 ** 6

    seed_time = datetime.timedelta(days=2)

    universe = SimpleUniverse.from_cache(universe_cache_name)

    filter_pair = make_filter_from_universe(universe)
    dfp = load_algo_pools(price_cache_name, 'prices', filter_pair)

    price_stream = stream_from_price_df(dfp, initial_time)
    asset_ids = [pool.asset1_id for pool in universe.pools]
    assert all(pool.asset2_id == 0 for pool in universe.pools)

    pos_impact_states = {
        asset_id: PositionAndImpactState(ASAImpactState(impact_timescale_seconds),
                                         ASAPosition(0))
        for asset_id in asset_ids
    }
    pos_impact_state = GlobalPositionAndImpactState(pos_impact_states, initial_mualgo_position)

    signal_providers = {
        asset_id: make_signal() for asset_id in asset_ids
    }

    def make_optimizer(aid: int):
        asset1 = Asset(aid)
        asset2 = Asset(0)
        return OptimizerV2(asset1=asset1, asset2=asset2, risk_coef=risk_coef)

    simulator = Simulator(universe=universe,
                          pos_impact_state=pos_impact_state,
                          signal_providers=signal_providers,
                          simulation_step_seconds=simulation_step_seconds,
                          seed_time=seed_time,
                          price_stream=price_stream,
                          make_optimizer=make_optimizer
                          )

    results = make_simulation_results(simulator, end_time)
    return results


class TestReports(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        logging.basicConfig(level=logging.ERROR)
        self.logger = logging.getLogger(__name__)

        self.price_cache_name = '20220209_prehack'
        self.universe_cache_name = 'liquid_algo_pools_nousd_prehack'

        short = False
        self.initial_time = datetime.datetime(year=2021, month=10, day=15, tzinfo=timezone.utc)
        if short:
            self.end_time = datetime.datetime(year=2021, month=10, day=18, tzinfo=timezone.utc)
        else:
            self.end_time = datetime.datetime(year=2021, month=12, day=31, tzinfo=timezone.utc)

        self.optimizer_cls = OptimizerV2

        super().__init__(*args, **kwargs)

    def test_fitted_signal(self):
        minutes = (30, 60, 120)
        betas = [-0.2942255, 0.47037815, -0.38620892]



        params = [EmaSignalParam(minute * 60, beta) for minute, beta in zip(minutes, betas)]

        def make_signal():
            return EmaSignalProvider(params, 0.05)

        results = make_signal_results(make_signal, self.price_cache_name, self.universe_cache_name,
                                      self.initial_time, self.end_time)

        # results.save_to_folder('/home/lorenzo/Algotrading/sim_results/is_220223_lag')
        results.save_to_folder('/home/lorenzo/Algotrading/sim_results/is_220223_lag_new')

    def test_liquidation(self):
        initial_position_multiplier = 1 / 100
        risk_coef = 0.000002 * 10 ** -6
        impact_timescale_seconds = 5 * 60
        simulation_step_seconds = 5 * 60
        initial_mualgo_position = 1000000
        seed_time = datetime.timedelta(days=1)
        initial_time = datetime.datetime(year=2021, month=11, day=10, tzinfo=timezone.utc)

        universe = SimpleUniverse.from_cache(self.universe_cache_name)

        filter_pair = make_filter_from_universe(universe)
        dfp = load_algo_pools(self.price_cache_name, 'prices', filter_pair)

        # Just choose some starting positions
        initial_positions = (dfp.groupby('asset1')['asset1_reserves'].mean() * initial_position_multiplier).astype(int)

        price_stream = stream_from_price_df(dfp, initial_time)
        asset_ids = [pool.asset1_id for pool in universe.pools]
        assert all(pool.asset2_id == 0 for pool in universe.pools)

        pos_impact_states = {
            asset_id: PositionAndImpactState(ASAImpactState(impact_timescale_seconds),
                                             ASAPosition(int(initial_positions.loc[asset_id])))
            for asset_id in asset_ids
        }
        pos_impact_state = GlobalPositionAndImpactState(pos_impact_states, initial_mualgo_position)

        signal_providers = {
            asset_id: DummySignalProvider() for asset_id in asset_ids
        }

        simulator = Simulator(universe=universe,
                              pos_impact_state=pos_impact_state,
                              signal_providers=signal_providers,
                              simulation_step_seconds=simulation_step_seconds,
                              risk_coef=risk_coef,
                              seed_time=seed_time,
                              price_stream=price_stream,
                              optimizer_cls=self.optimizer_cls
                              )

        end_time = datetime.datetime(year=2021, month=11, day=20, tzinfo=timezone.utc)

        results = make_simulation_results(simulator, end_time)

        make_simulation_reports(results)

    def test_random_signal(self):
        def make_signal():
            return RandomSignalProvider(0.002)

        results = make_signal_results(make_signal, self.price_cache_name, self.universe_cache_name,
                                      self.initial_time, self.end_time)
        make_simulation_reports(results)


class TestReportsOOS(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        logging.basicConfig(level=logging.ERROR)
        self.logger = logging.getLogger(__name__)

        self.price_cache_name = '20220209'
        self.universe_cache_name = 'liquid_algo_pools_nousd_prehack'

        short = False
        self.initial_time = datetime.datetime(year=2022, month=1, day=25, tzinfo=timezone.utc)
        if short:
            self.end_time = datetime.datetime(year=2022, month=1, day=30, tzinfo=timezone.utc)
        else:
            self.end_time = datetime.datetime(year=2022, month=2, day=22, tzinfo=timezone.utc)

        self.optimizer_cls = OptimizerV2

        super().__init__(*args, **kwargs)

    def test_fitted_signal(self):
        minutes = (30, 60, 120)
        # betas = [-0.2942255, 0.47037815, -0.38620892]
        betas_new = [-0.38339549, 0.54108776, -0.41479177]
        betas = betas_new

        # signal_cap = 0.05
        # risk_multiplier = 0.000002

        signal_cap = 0.1
        risk_multiplier = 0.0000001
        initial_algo_position = 2000

        params = [EmaSignalParam(minute * 60, beta) for minute, beta in zip(minutes, betas)]

        def make_signal():
            return EmaSignalProvider(params, signal_cap)

        results = make_signal_results(make_signal, self.price_cache_name, self.universe_cache_name,
                                      self.initial_time, self.end_time, risk_multiplier, initial_algo_position)

        results.save_to_folder(f'/home/lorenzo/Algotrading/sim_results/oos_220223_lag_new_{initial_algo_position}_{signal_cap}_{risk_multiplier}')

    def test_report(self):
        signal_cap = 0.1
        risk_multiplier = 0.0000001
        initial_algo_position = 2000

        folder = f'/home/lorenzo/Algotrading/sim_results/oos_220223_lag_new_{initial_algo_position}_{signal_cap}_{risk_multiplier}'
        results = SimulationResults.from_folder(folder)
        aggdf = results.make_aggregate_values_df()
        plot_aggregate_values_df(aggdf)
        plt.show()

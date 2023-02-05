from __future__ import annotations
import datetime
import unittest
import logging
from algo.trading.impact import ASAImpactState, PositionAndImpactState, GlobalPositionAndImpactState, \
    ASAPosition
from algo.trading.signalprovider import DummySignalProvider
from algo.blockchain.stream import PoolState, PriceUpdate, stream_from_price_df
from algo.universe.universe import SimpleUniverse
from dataclasses import dataclass
from datetime import timezone
from algo.blockchain.utils import load_algo_pools
from algo.dataloading.caching import make_filter_from_universe
from algo.simulation.simulator import Simulator
from algo.universe.pools import PoolId
from algo.blockchain.utils import datetime_to_int
from algo.trading.trades import TradeInfo
from algo.optimizer.optimizerV2 import OptimizerV2
import json

@dataclass
class SimDebugParameter:
    price: float
    impact_decay_seconds: int
    frac_pool: float
    const_signal_value: float
    alternating: bool = False


class TestSimulator(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.optimizer_cls = OptimizerV2

        super().__init__(*args, **kwargs)

    def _debug_trades(self, params: SimDebugParameter, num_iter=4):
        log_null_trades = False

        asset1_id = 1
        asset2_reserves = 10 ** 12
        risk_coef = 0.000000000001
        impact_timescale_seconds = params.impact_decay_seconds

        simulation_step_seconds = 5 * 60
        initial_mualgo_position = 1000000000000
        universe = SimpleUniverse(pools=[PoolId(asset1_id, 0, "dummy")])
        seed_time = datetime.timedelta(minutes=4)
        initial_time = datetime.datetime(year=2021, month=11, day=10, tzinfo=timezone.utc)
        end_time = datetime.datetime(year=2021, month=11, day=10, minute=30, tzinfo=timezone.utc)
        asset_ids = [pool.asset1_id for pool in universe.pools]
        assert all(pool.asset2_id == 0 for pool in universe.pools)

        logged_trades: list[TradeInfo] = []

        asset1_reserves = int(asset2_reserves / params.price)

        def price_stream():
            for i in range(num_iter):
                lag_seconds = 1 + i * (60 * 5)
                yield PriceUpdate(asset_ids=(asset1_id, 0),
                                  price_update=PoolState(time=datetime_to_int(initial_time) + lag_seconds,
                                                         asset1_reserves=asset1_reserves,
                                                         asset2_reserves=asset2_reserves,
                                                         reverse_order_in_block=0)
                                  )

        pos_impact_states = {
            asset_id: PositionAndImpactState(ASAImpactState(impact_timescale_seconds),
                                             asa_position=ASAPosition(int(params.frac_pool * asset1_reserves)))
            for asset_id in asset_ids
        }
        pos_impact_state = GlobalPositionAndImpactState(pos_impact_states, initial_mualgo_position)
        signal_providers = {
            asset_id: DummySignalProvider(params.const_signal_value, params.alternating) for asset_id in asset_ids
        }
        simulator = Simulator(universe=universe,
                              pos_impact_state=pos_impact_state,
                              signal_providers=signal_providers,
                              simulation_step_seconds=simulation_step_seconds,
                              risk_coef=risk_coef,
                              seed_time=seed_time,
                              price_stream=price_stream(),
                              optimizer_cls=self.optimizer_cls
                              )

        def log_trade(x):
            logged_trades.append(x)

        def log_state(x):
            pass

        simulator.run(end_time, log_trade, log_state)

        return logged_trades

    def test_liquidation(self):

        initial_position_multiplier = 1 / 100

        risk_coef = 0.0000001
        # risk_coef = 0.000000001
        price_cache_name = '20220209_prehack'
        universe_cache_name = 'liquid_algo_pools_nousd_prehack'
        impact_timescale_seconds = 5 * 60
        simulation_step_seconds = 5 * 60
        initial_mualgo_position = 1000000
        seed_time = datetime.timedelta(days=1)
        initial_time = datetime.datetime(year=2021, month=11, day=10, tzinfo=timezone.utc)
        end_time = datetime.datetime(year=2021, month=11, day=20, tzinfo=timezone.utc)

        universe = SimpleUniverse.from_cache(universe_cache_name)

        filter_pair = make_filter_from_universe(universe)
        dfp = load_algo_pools(price_cache_name, 'prices', filter_pair)

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

        def log_trade(x):
            self.logger.info(x)
            with open('test_trades.txt', 'a') as f:
                f.write(str(x.json()))
                f.write('\n')

        def log_state(x):
            pass

        simulator.run(end_time, log_trade, log_state)

    def test_dummy_signal(self):
        log_null_trades = True

        const_signal_value = 0.0001

        risk_coef = 0.0000001
        # risk_coef = 0.000000001
        price_cache_name = '20220209_prehack'
        universe_cache_name = 'liquid_algo_pools_nousd_prehack'
        impact_timescale_seconds = 5 * 60
        simulation_step_seconds = 5 * 60
        initial_mualgo_position = 1000000
        seed_time = datetime.timedelta(days=1)
        initial_time = datetime.datetime(year=2021, month=11, day=10, tzinfo=timezone.utc)
        end_time = datetime.datetime(year=2021, month=11, day=20, tzinfo=timezone.utc)

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
            asset_id: DummySignalProvider(const_signal_value) for asset_id in asset_ids
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

        def log_trade(x):
            self.logger.info(x)

        def log_state(x):
            pass

        simulator.run(end_time, log_trade, log_state)

    def test_price_invariance(self):

        def assert_trades_equal(param0: SimDebugParameter, param1: SimDebugParameter, min_num_trades, num_iter=4):
            trades = [self._debug_trades(param, num_iter) for param in (param0, param1)]
            assert len(trades[0]) == len(trades[1])
            assert len(trades[0]) >= min_num_trades, f"len(trades[0]) = {len(trades[0])}"
            for t0, t1 in zip(trades[0], trades[1]):
                t0.assert_price_covariant(t1), f"\n{t0}, \n{t1}"

        def negate_assert(x):
            try:
                x()
            except AssertionError:
                return
            raise AssertionError

        assert_trades_equal(SimDebugParameter(0.14, 5 * 60, 0.1, 0.0), SimDebugParameter(57, 5 * 60, 0.1, 0.0), 1)

        negate_assert(lambda: assert_trades_equal(SimDebugParameter(0.14, 5 * 60, 0.1, 0.0),
                                                  SimDebugParameter(57, 1 * 60, 0.1, 0.0), 2))

        f_bps = 0.004
        assert_trades_equal(SimDebugParameter(1.4, 5 * 60, 0.0, f_bps), SimDebugParameter(57, 5 * 60, 0.0, f_bps), 1, 3)

        negate_assert(lambda: assert_trades_equal(SimDebugParameter(1.4, 5 * 60, 0.0, 0.01),
                                                  SimDebugParameter(57, 5 * 60, 0.0, 0.011), 1, 3))

        f_bps = 0.004
        assert_trades_equal(SimDebugParameter(1.4, 5 * 60, 0.0, f_bps, True),
                            SimDebugParameter(57, 5 * 60, 0.0, f_bps, True),
                            2,
                            4)

        negate_assert(lambda: assert_trades_equal(SimDebugParameter(1.4, 5 * 60, 0.0, 0.01, True),
                                                  SimDebugParameter(57, 5 * 60, 0.0, 0.011, True),
                                                  2,
                                                  4))

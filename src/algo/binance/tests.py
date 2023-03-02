import datetime
import logging
import unittest

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from algo.binance.backtest import SimulatorCfg, Simulator, SimResults, OptimiserCfg
from algo.binance.features import FeatureOptions, VolumeOptions
from algo.binance.fit import fit_eval_model, UniverseDataOptions, fit_product
from algo.binance.dataloader import load_universe_data
from algo.binance.coins import Universe, all_symbols, top_mcap, symbol_to_ids, DataType, SpotType, FutureType
from algo.binance.fit import UniverseDataStore, ModelOptions, ResidOptions
from algo.binance.sim_reports import plot_results

from algo.binance.data_types import KlineType, AggTradesType
from binance.experiment_runner import run


class TestUniverseDataStore(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        coins = ['btc', 'eth', 'ada', 'xrp', 'dot', 'doge', 'matic', 'algo', 'ltc', 'atom', 'link', 'near', 'bch',
                 'xlm', 'axs', 'vet', 'hbar', 'fil', 'egld', 'theta', 'icp', 'etc', 'xmr', 'xtz', 'aave', 'gala', 'grt',
                 'klay', 'cake', 'ar', 'eos', 'lrc', 'ksm', 'enj', 'qnt', 'amp', 'cvx', 'crv', 'mkr', 'xec', 'kda',
                 'tfuel', 'spell', 'sushi', 'bat', 'neo', 'celo', 'zec', 'osmo', 'chz', 'waves', 'dash', 'fxs', 'nexo',
                 'comp', 'mina', 'yfi', 'iotx', 'xem', 'snx', 'zil', 'rvn', '1inch', 'gno', 'lpt', 'dcr', 'qtum', 'ens',
                 'icx', 'waxp', 'omg', 'ankr', 'scrt', 'sc', 'bnt', 'woo', 'zen', 'iost', 'btg', 'rndr', 'zrx', 'slp',
                 'anc', 'ckb', 'ilv', 'sys', 'uma', 'kava', 'ont', 'hive', 'perp', 'wrx', 'skl', 'flux', 'ren', 'mbox',
                 'ant', 'ray', 'dgb', 'movr', 'nu']
        coins = coins[:4]
        universe = Universe(coins)

        start_date = datetime.datetime(year=2022, month=1, day=1)
        end_date = datetime.datetime(year=2023, month=1, day=1)

        time_col = 'Close time'

        df = load_universe_data(universe, start_date, end_date, '5m')

        df.set_index(['pair', time_col], inplace=True)
        self.df = df

        super().__init__(*args, **kwargs)

    def _aa(self, quantile_cap: float):
        vol = VolumeOptions(include_imbalance=True, include_logretvol=True)
        ema_options = FeatureOptions([4, 12, 24, 48, 96], vol)
        ro = ResidOptions(market_pairs={'BTCUSDT'})

        uds = UniverseDataStore(self.df, ema_options, ro)

        alpha = 1.0

        def get_lm():
            return Ridge(alpha=alpha)

        def transform_fit_target(y):
            return y

        def transform_model_after_fit(lm):
            return lm

        fit_options = UniverseDataOptions(demean=True,
                                          forward_hour=24,
                                          target_scaler=lambda: RobustScaler())
        ufd = uds.prepare_data(fit_options)

        global_opt = ModelOptions(
            get_lm=lambda: Ridge(alpha=0),
            transform_fit_target=transform_fit_target,
            transform_model_after_fit=transform_model_after_fit,
            cap_oos_quantile=None
            # cap_oos_quantile=0.05
        )
        data_global = uds.prepare_data_global(ufd)
        global_fit = fit_eval_model(data_global, global_opt)

        opt = ModelOptions(
            get_lm=get_lm,
            transform_fit_target=transform_fit_target,
            transform_model_after_fit=transform_model_after_fit,
            cap_oos_quantile=quantile_cap
        )
        ress = {}
        for pair, product_data in uds.gen_product_data(ufd, global_fit):
            ress[pair] = fit_product(product_data, opt)

        print(list(ress.values())[0].test.ypred.min(), list(ress.values())[0].test.ypred.max())
        # print(r2_score(list(ress.values())[0].test.ytrue, list(ress.values())[0].test.ypred))

    def test_a(self):
        self._aa(0.00001)

    def test_b(self):
        self._aa(0.4)


class TestSymbols(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
        super().__init__(*args, **kwargs)

    def test_a(self):
        symbols_map = symbol_to_ids()

        for symbol in all_symbols():
            if symbol == 'eth':
                coin_id = symbols_map.get(symbol, None)
                print(f'{symbol=}, {coin_id=}')

    def test_b(self):
        top_mcap(datetime.date(year=2022, month=1, day=1), dry_run=True)


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
        run(name='test_spot_agg_220211', n_coins=4, market_type=market_type, data_type=data_type, test=True,
            lookahead=True)

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


class TestBack(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                            level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

        super().__init__(*args, **kwargs)

    def test_run(self):
        opts: list[OptimiserCfg] = [
            OptimiserCfg(
                comm=0.0001,
                risk_coef=10 ** (-5),
                poslim=10000,
                cash_flat=False,
                mkt_flat=True,
                max_trade_size_usd=1000,
            ),
            # OptimiserCfg(
            #     comm=0.0001,
            #     risk_coef=10 ** (-6),
            #     poslim=10000,
            #     cash_flat=False,
            #     mkt_flat=True,
            #     max_trade_size_usd=1000,
            # )
        ]
        cfg = SimulatorCfg(
            data_type=KlineType(freq='5m'),
            market_type=SpotType(),
            exp_name='test_220211',
            end_time=datetime.datetime(year=2022, month=8, day=15),
            trade_pair_limit=2,
            opts=opts,
            ema_hf_periods=5,
        )
        self.sim = Simulator(cfg)

        res = self.sim.run()
        # plot_results(res, None)

    def test_b(self):
        fname = '/home/lorenzo/algo/sims/spot_slow_flat/results.pkl'
        x: SimResults = pd.read_pickle(fname)
        plot_results(x, None, coefs_list=[0.01])

    def test_opt(self):
        pass


if __name__ == '__main__':
    TestBack().test_run()

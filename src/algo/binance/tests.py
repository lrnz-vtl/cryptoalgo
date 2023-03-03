import datetime
import logging
import os
import unittest
import pytest
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from algo.binance.backtest import SimulatorCfg, Simulator, SimResults, OptimiserCfg, SIMS_BASEP
from algo.binance.features import FeatureOptions, VolumeOptions
from algo.binance.fit import fit_eval_model, UniverseDataOptions, fit_product
from algo.binance.dataloader import load_universe_data
from algo.binance.coins import Universe, all_symbols, top_mcap, symbol_to_ids, DataType, SpotType, FutureType, \
    MarketTypeModel
from algo.binance.fit import UniverseDataStore, ModelOptions, ResidOptions
from algo.binance.sim_reports import plot_results

from algo.binance.data_types import KlineType, AggTradesType, DataTypeModel
from algo.binance.experiment_runner import run


# class TestUniverseDataStore(unittest.TestCase):
#
#     def __init__(self, *args, **kwargs):
#         coins = ['btc', 'eth', 'ada', 'xrp', 'dot', 'doge', 'matic', 'algo', 'ltc', 'atom', 'link', 'near', 'bch',
#                  'xlm', 'axs', 'vet', 'hbar', 'fil', 'egld', 'theta', 'icp', 'etc', 'xmr', 'xtz', 'aave', 'gala', 'grt',
#                  'klay', 'cake', 'ar', 'eos', 'lrc', 'ksm', 'enj', 'qnt', 'amp', 'cvx', 'crv', 'mkr', 'xec', 'kda',
#                  'tfuel', 'spell', 'sushi', 'bat', 'neo', 'celo', 'zec', 'osmo', 'chz', 'waves', 'dash', 'fxs', 'nexo',
#                  'comp', 'mina', 'yfi', 'iotx', 'xem', 'snx', 'zil', 'rvn', '1inch', 'gno', 'lpt', 'dcr', 'qtum', 'ens',
#                  'icx', 'waxp', 'omg', 'ankr', 'scrt', 'sc', 'bnt', 'woo', 'zen', 'iost', 'btg', 'rndr', 'zrx', 'slp',
#                  'anc', 'ckb', 'ilv', 'sys', 'uma', 'kava', 'ont', 'hive', 'perp', 'wrx', 'skl', 'flux', 'ren', 'mbox',
#                  'ant', 'ray', 'dgb', 'movr', 'nu']
#         coins = coins[:4]
#         universe = Universe(coins)
#
#         start_date = datetime.datetime(year=2022, month=1, day=1)
#         end_date = datetime.datetime(year=2023, month=1, day=1)
#
#         time_col = 'Close time'
#
#         df = load_universe_data(universe, start_date, end_date, '5m')
#
#         df.set_index(['pair', time_col], inplace=True)
#         self.df = df
#
#         super().__init__(*args, **kwargs)
#
#     def _aa(self, quantile_cap: float):
#         vol = VolumeOptions(include_imbalance=True, include_logretvol=True)
#         ema_options = FeatureOptions([4, 12, 24, 48, 96], vol)
#         ro = ResidOptions(market_pairs={'BTCUSDT'})
#
#         uds = UniverseDataStore(self.df, ema_options, ro)
#
#         alpha = 1.0
#
#         def get_lm():
#             return Ridge(alpha=alpha)
#
#         def transform_fit_target(y):
#             return y
#
#         def transform_model_after_fit(lm):
#             return lm
#
#         fit_options = UniverseDataOptions(demean=True,
#                                           forward_hour=24,
#                                           target_scaler=lambda: RobustScaler())
#         ufd = uds.prepare_data(fit_options)
#
#         global_opt = ModelOptions(
#             get_lm=lambda: Ridge(alpha=0),
#             transform_fit_target=transform_fit_target,
#             transform_model_after_fit=transform_model_after_fit,
#             cap_oos_quantile=None
#             # cap_oos_quantile=0.05
#         )
#         data_global = uds.prepare_data_global(ufd)
#         global_fit = fit_eval_model(data_global, global_opt)
#
#         opt = ModelOptions(
#             get_lm=get_lm,
#             transform_fit_target=transform_fit_target,
#             transform_model_after_fit=transform_model_after_fit,
#             cap_oos_quantile=quantile_cap
#         )
#         ress = {}
#         for pair, product_data in uds.gen_product_data(ufd, global_fit):
#             ress[pair] = fit_product(product_data, opt)
#
#         print(list(ress.values())[0].test.ypred.min(), list(ress.values())[0].test.ypred.max())
#         # print(r2_score(list(ress.values())[0].test.ytrue, list(ress.values())[0].test.ypred))
#
#     def test_a(self):
#         self._aa(0.00001)
#
#     def test_b(self):
#         self._aa(0.4)
#
#
# class TestSymbols(unittest.TestCase):
#     def __init__(self, *args, **kwargs):
#         logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
#         super().__init__(*args, **kwargs)
#
#     def test_a(self):
#         symbols_map = symbol_to_ids()
#
#         for symbol in all_symbols():
#             if symbol == 'eth':
#                 coin_id = symbols_map.get(symbol, None)
#                 print(f'{symbol=}, {coin_id=}')
#
#     def test_b(self):
#         top_mcap(datetime.date(year=2022, month=1, day=1), dry_run=True)


@pytest.fixture()
def run_exp():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG)
    market_type = FutureType()
    data_type = KlineType(freq='5m')
    run_name = 'test_220211'
    run(name=run_name, n_coins=4, market_type=market_type, data_type=data_type, test=True, lookahead=True)
    return run_name


def test_exp2():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG)
    market_type = FutureType()
    data_type = KlineType(freq='5m')
    run_name = 'test_220211'
    run(name=run_name, n_coins=4, market_type=market_type, data_type=data_type, test=True, lookahead=False)
    return run_name


@pytest.fixture()
def run_sim(run_exp):
    run_name = run_exp
    opts: list[OptimiserCfg] = [
        OptimiserCfg(
            comm=0.0001,
            risk_coef=10 ** (-5),
            poslim=10000,
            cash_flat=False,
            mkt_flat=True,
            max_trade_size_usd=1000,
        ),
    ]
    cfg = SimulatorCfg(
        data_type=DataTypeModel(t=KlineType(freq='5m')),
        market_type=MarketTypeModel(t=FutureType()),
        exp_name=run_name,
        end_time=datetime.datetime(year=2022, month=5, day=13),
        trade_pair_limit=2,
        opts=opts,
        ema_hf_periods=5,
    )
    sim = Simulator(cfg)
    results = sim.run()
    dest_path = SIMS_BASEP / run_name
    os.makedirs(dest_path, exist_ok=True)
    pd.to_pickle(results, dest_path / 'results.pkl')
    return results


def test_reports(run_sim):
    results = run_sim
    plot_results(results, None)


def test_reports_serialised():
    run_name = 'test_220211'
    fname = f'/home/lorenzo/algo/sims/{run_name}/results.pkl'
    results: SimResults = pd.read_pickle(fname)
    plot_results(results, None)

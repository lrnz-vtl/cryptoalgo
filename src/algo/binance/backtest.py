import argparse
import datetime
import logging
import os
# import profile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import BaseModel

from algo.binance.coins import load_universe_candles
from algo.binance.experiment import ExpArgs, EXP_BASEP
from algo.binance.features import features_from_data
from algo.binance.model import ProductModel
from algo.binance.utils import read_json, to_datetime
from algo.cpp.cseries import shift_forward
import cvxpy as cp

from algo.definitions import ROOT_DIR

SIMS_BASEP = ROOT_DIR / 'sims'


@dataclass
class OptimiserCfg:
    comm: float
    risk_coef: float
    poslim: float
    cash_flat: bool
    mkt_flat: bool


class Optimiser:

    def __init__(self, betas: np.array, cfg: OptimiserCfg):
        self.cfg = cfg
        self.update_betas(betas)

    def update_betas(self, betas):
        self.betas = betas
        self.n = len(betas)
        n = self.n
        self.comms = self.cfg.comm * np.ones(n)
        self.risk = self.cfg.risk_coef * np.identity(n)
        self.poslims = self.cfg.poslim * np.ones(n)
        self.ones = np.ones(n)
        self.zeros = np.zeros(n)

    def optimise(self, position: np.array, signal: np.array) -> np.array:
        assert position.shape[0] == self.n
        assert signal.shape[0] == self.n

        # kvbuy = -self.risk * position + (signal - self.comms)
        # kvsell = self.risk * position + (-signal - self.comms)
        # if (kvbuy <= 0).all() and (kvsell <= 0).all():
        #     return self.zeros

        xb = cp.Variable(self.n)
        xs = cp.Variable(self.n)

        cons = [
            xb >= self.zeros,
            xs >= self.zeros,
            (position + xb - xs) <= self.poslims,
            (position + xb - xs) >= -self.poslims
        ]
        if self.cfg.cash_flat:
            cons.append(self.ones @ (position + xb - xs) == 0)
        if self.cfg.mkt_flat:
            cons.append(self.betas @ (position + xb - xs) == 0)

        prob = cp.Problem(
            cp.Minimize((1 / 2) * cp.quad_form(position + xb - xs, self.risk) - signal.T @ (xb - xs) +
                        self.comms @ xb + self.comms @ xs), cons
        )
        prob.solve()
        return xb.value - xs.value


class TradeAcct:
    def __init__(self, betas: np.array,
                 cfg: OptimiserCfg,
                 nsteps: int,
                 n_pairs: int
                 ):
        self.betas = betas
        self.opt = Optimiser(betas, cfg)
        self.qtys = np.zeros((nsteps, n_pairs))
        self.notional = np.zeros((nsteps, n_pairs))
        self.pnl = np.zeros((nsteps, n_pairs))

        self.mkt_exposure = np.zeros(nsteps)

        self.mask = np.full(n_pairs, True)
        self.i = 0

    def remove_pair(self, j):
        self.mask[j] = False
        self.notional[self.i:, j] = self.notional[self.i - 1, j]
        self.pnl[self.i:, j] = self.pnl[self.i - 1, j]
        self.opt.update_betas(self.betas[self.mask])

    def update_step(self):
        self.i += 1

    def trade(self, signals, prices):
        mask = self.mask
        i = self.i

        signals = signals[self.mask]
        prices = prices[self.mask]

        self.pnl[i][mask] = self.qtys[i - 1][mask] * prices - self.notional[i - 1][mask]

        n_tries = 10
        while 1:
            try:
                delta_dollars = self.opt.optimise(self.qtys[i - 1][mask] * prices, signals)
                break
            except Exception as e:
                if n_tries == 0:
                    raise e
                n_tries -= 1

        self.notional[i][mask] = self.notional[i - 1][mask] + delta_dollars
        self.qtys[i][mask] = self.qtys[i - 1][mask] + delta_dollars / prices

        self.mkt_exposure[i] = self.betas[mask] @ (self.qtys[i][mask] * prices)


class SimulatorCfg(BaseModel):
    exp_name: str
    end_time: datetime.datetime
    risk_coefs: list[float]
    cash_flat: bool
    mkt_flat: bool
    trade_pair_limit: Optional[int]


@dataclass
class SimResults:
    accts: dict[float, TradeAcct]
    signals_matrix: np.array
    prices: np.array
    times: np.array
    pairs: list[str]


class Simulator:
    def __init__(self, cfg: SimulatorCfg):
        self.cfg = cfg
        self.exp_path = EXP_BASEP / cfg.exp_name
        assert self.exp_path.exists()
        self.logger = logging.getLogger(__name__)

        betas = read_json(self.exp_path / 'betas.json')

        models: dict[str, ProductModel] = pd.read_pickle(self.exp_path / 'models.pkl')

        exp_args: ExpArgs = ExpArgs.parse_file(self.exp_path / 'exp_args.json')

        start_time = exp_args.tto.test_start_time
        data_start_time = start_time - datetime.timedelta(days=30)
        end_time = self.cfg.end_time

        assert end_time > start_time

        mkt_pairs = exp_args.ro.market_pairs
        mkt_features = []
        for pair, df in load_universe_candles(mkt_pairs, data_start_time, end_time, '5m', exp_args.spot):
            new_mkt_features = features_from_data(df, exp_args.feature_options)
            new_mkt_features.columns = [f'{pair}_{col}' for col in new_mkt_features.columns]
            mkt_features.append(new_mkt_features)
        mkt_features = pd.concat(mkt_features, axis=1)

        trade_pairs = list(models.keys())
        if cfg.trade_pair_limit:
            trade_pairs = trade_pairs[:cfg.trade_pair_limit]
        trade_data = load_universe_candles(trade_pairs, data_start_time, end_time, '5m', exp_args.spot)

        signals = {}
        exec_prices = {}
        for pair, df in trade_data:
            product_features = features_from_data(df, exp_args.feature_options)
            features = pd.concat([product_features, mkt_features.loc[product_features.index].fillna(0)],
                                 axis=1)
            assert all(
                features.columns == models[pair].column_names), f'{features.columns}, {models[pair].column_names}'
            signal = pd.Series(models[pair].predict(features), index=features.index)
            signal = signal[to_datetime(signal.index) > start_time]
            signals[pair] = signal

            price_ts = ((df['Close'] + df['Open']) / 2.0).rename('price')
            exec_price_ts = pd.Series(shift_forward(price_ts.index, price_ts.values, 1), index=df.index)
            exec_price_ts = exec_price_ts[to_datetime(exec_price_ts.index) > start_time]
            exec_prices[pair] = exec_price_ts

            assert (exec_price_ts.index == signal.index).all()

        signals_df = pd.DataFrame(signals)
        exec_prices_df = pd.DataFrame(exec_prices)

        self.removal_times = {}

        for j, col in enumerate(exec_prices_df):
            x = exec_prices_df[col]
            mask = (x.loc[::-1].isna().cumsum() == np.arange(1, x.shape[0] + 1))[::-1]

            if mask.sum() > 0:
                t = mask[mask].index.min()

                if t in self.removal_times:
                    self.removal_times[t].append(j)
                else:
                    self.removal_times[t] = [j]

        self.pairs = signals_df.columns
        self.times = signals_df.index.values

        signals_matrix = signals_df.values
        del signals_df
        exec_prices_matrix = exec_prices_df.values
        del exec_prices_df

        assert signals_matrix.shape == exec_prices_matrix.shape

        self.nsteps = exec_prices_matrix.shape[0]
        self.betas = pd.Series(betas).loc[self.pairs].values

        nan_perc = np.isnan(exec_prices_matrix).sum() / exec_prices_matrix.shape[0] / exec_prices_matrix.shape[1]
        if nan_perc > 0:
            self.logger.warning(f'exec_prices_matrix {nan_perc=}')

        nan_perc = np.isnan(signals_matrix).sum() / signals_matrix.shape[0] / signals_matrix.shape[1]
        if nan_perc > 0:
            self.logger.warning(f'signals_matrix {nan_perc=}')
        signals_matrix[np.isnan(signals_matrix)] = 0

        self.signals_matrix = signals_matrix
        self.exec_prices_matrix = exec_prices_matrix

    def run(self):
        risk_coefs = self.cfg.risk_coefs

        accts: dict[float, TradeAcct] = {coef:
                                             TradeAcct(self.betas,
                                                       OptimiserCfg(comm=0.0001,
                                                                    risk_coef=coef,
                                                                    poslim=1000,
                                                                    cash_flat=self.cfg.cash_flat,
                                                                    mkt_flat=self.cfg.mkt_flat
                                                                    ),
                                                       self.nsteps,
                                                       len(self.pairs)
                                                       )
                                         for coef in risk_coefs}

        for i in range(1, self.nsteps):

            for coef in risk_coefs:
                accts[coef].update_step()

            t = self.times[i]
            if t in self.removal_times:
                for j in self.removal_times[t]:
                    self.logger.warning(f'Removing pair {j=}, {self.pairs[j]=}')
                    for coef in risk_coefs:
                        accts[coef].remove_pair(j)

            for coef in risk_coefs:
                accts[coef].trade(self.signals_matrix[i], self.exec_prices_matrix[i])

        return SimResults(accts=accts,
                          signals_matrix=self.signals_matrix,
                          pairs=self.pairs,
                          times=self.times,
                          prices=self.exec_prices_matrix
                          )


def plot_pair_results(times: np.array,
                      qtys: dict[float, np.array],
                      pnl: dict[float, np.array],
                      signals: np.array,
                      prices: np.array,
                      pair_name: str,
                      dst_path: Optional[Path]):
    cols = 4

    f, axs = plt.subplots(1, cols, figsize=(5 * cols, 5))

    for coef in qtys.keys():
        axs[0].plot(times, qtys[coef], label=coef)
        axs[1].plot(times, pnl[coef], label=coef)
    axs[2].plot(times, signals)
    axs[3].plot(times, prices)

    axs[0].tick_params(labelrotation=35)
    axs[0].grid()
    axs[0].set_title('position')
    axs[0].legend()

    axs[1].tick_params(labelrotation=35)
    axs[1].grid()
    axs[1].set_title('pnl')
    axs[1].legend()

    axs[2].tick_params(labelrotation=35)
    axs[2].grid()
    axs[2].set_title('signal')

    axs[3].tick_params(labelrotation=35)
    axs[3].grid()
    axs[3].set_title('price')

    f.suptitle(pair_name)
    f.tight_layout()

    if dst_path is not None:
        plt.savefig(dst_path/f'{pair_name}.png')
    else:
        plt.show()


def plot_results(x: SimResults, dst_path: Optional[Path]):
    times = to_datetime(x.times)

    plot_pair_results(times=times,
                      qtys={coef: x.qtys.sum(axis=1) for coef, x in x.accts.items()},
                      pnl={coef: x.pnl.sum(axis=1) for coef, x in x.accts.items()},
                      signals=x.signals_matrix.sum(axis=1),
                      prices=x.prices.sum(axis=1),
                      pair_name='total',
                      dst_path=dst_path
                      )

    for j, pair in enumerate(x.pairs):
        plot_pair_results(times=times,
                          qtys={coef: x.qtys[:, j] for coef, x in x.accts.items()},
                          pnl={coef: x.pnl[:, j] for coef, x in x.accts.items()},
                          signals=x.signals_matrix[:, j],
                          prices=x.prices[:, j],
                          pair_name=pair,
                          dst_path=dst_path
                          )


class TestBack(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(__name__)

        super().__init__(*args, **kwargs)

    def test_run(self):
        cfg = SimulatorCfg(exp_name='spot_slow',
                           end_time=datetime.datetime(year=2022, month=9, day=1),
                           trade_pair_limit=2,
                           risk_coefs=[0.001],
                           cash_flat=False,
                           mkt_flat=True
                           )
        self.sim = Simulator(cfg)

        res = self.sim.run()
        plot_results(res, None)

    def test_b(self):
        fname = '/home/lorenzo/algo/sims/spot_slow_test/results.pkl'
        x: SimResults = pd.read_pickle(fname)
        plot_results(x, None)

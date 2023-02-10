import datetime
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.sparse.linalg import ArpackError

from algo.binance.coins import load_universe_candles
from algo.binance.experiment import ExpArgs, EXP_BASEP
from algo.binance.features import features_from_data
from algo.binance.model import ProductModel
from algo.binance.optimiser import OptimiserCfg, Optimiser
from algo.binance.utils import read_json, to_datetime
from algo.cpp.cseries import shift_forward

from algo.definitions import ROOT_DIR

SIMS_BASEP = ROOT_DIR / 'sims'

# Minimum trade size in dollar terms
MIN_TRADE_SIZE = 10


class SimulatorCfg(BaseModel):
    exp_name: str
    end_time: datetime.datetime
    opts: list[OptimiserCfg]
    trade_pair_limit: Optional[int]
    # Half life in units of 5 minutes
    ema_hf_periods: Optional[float] = None


@dataclass
class AcctStats:
    qtys: np.array
    notional: np.array
    pnl: np.array
    mkt_exposure: np.array
    cash_exposure: np.array
    trade_signals: np.array


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
        self.cash_exposure = np.zeros(nsteps)

        self.trade_signals = np.zeros((nsteps, n_pairs))

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

        self.trade_signals[i][self.mask] = signals

        self.pnl[i][mask] = self.qtys[i - 1][mask] * prices - self.notional[i - 1][mask]

        n_tries = 10
        while 1:
            try:
                delta_dollars = self.opt.optimise(self.qtys[i - 1][mask] * prices, signals)
                break
            except ArpackError as e:
                if n_tries == 0:
                    raise e
                n_tries -= 1
            except RuntimeError as e:
                raise e

        delta_dollars[abs(delta_dollars) < MIN_TRADE_SIZE] = 0

        self.notional[i][mask] = self.notional[i - 1][mask] + delta_dollars
        self.qtys[i][mask] = self.qtys[i - 1][mask] + delta_dollars / prices

        self.mkt_exposure[i] = self.betas[mask] @ (self.qtys[i][mask] * prices)

        self.cash_exposure[i] = (self.qtys[i][mask] * prices).sum()

    def serialise(self) -> AcctStats:
        return AcctStats(
            qtys=self.qtys,
            notional=self.notional,
            pnl=self.pnl,
            mkt_exposure=self.mkt_exposure,
            cash_exposure=self.cash_exposure,
            trade_signals=self.trade_signals
        )


@dataclass
class SimResults:
    cfgs_accts: list[tuple[OptimiserCfg, AcctStats]]
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
            exec_price_ts = pd.Series(shift_forward(price_ts.index.values.copy(), price_ts.values.copy(), 1), index=df.index)
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

        self.opt_cfgs = self.cfg.opts

    def run(self):

        accts: list[TradeAcct] = [TradeAcct(self.betas,
                                            cfg=cfg_opt,
                                            nsteps=self.nsteps,
                                            n_pairs=len(self.pairs)
                                            )
                                  for cfg_opt in self.cfg.opts]

        signals = np.zeros(len(self.pairs))

        for i in range(1, self.nsteps):

            for acct in accts:
                acct.update_step()

            t = self.times[i]
            if t in self.removal_times:
                for j in self.removal_times[t]:
                    self.logger.warning(f'Removing pair {j=}, {self.pairs[j]=}')
                    for acct in accts:
                        acct.remove_pair(j)

            raw_signals = self.signals_matrix[i]
            if self.cfg.ema_hf_periods:
                alpha = np.exp(-1.0 / self.cfg.ema_hf_periods)
                signals = alpha * signals + (1 - alpha) * raw_signals
            else:
                signals = raw_signals

            for acct in accts:
                acct.trade(signals, self.exec_prices_matrix[i])

        return SimResults(
            cfgs_accts=list(zip(self.opt_cfgs, (acct.serialise() for acct in accts))),
            signals_matrix=self.signals_matrix,
            pairs=self.pairs,
            times=self.times,
            prices=self.exec_prices_matrix
        )

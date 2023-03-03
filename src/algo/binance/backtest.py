import datetime
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pydantic
from scipy.sparse.linalg import ArpackError

from algo.binance.dataloader import load_universe_data, DataType, MarketType
from algo.binance.experiment import ExpArgs, EXP_BASEP
from algo.binance.features import features_from_data, VolumeZeroError, NanFeatureException
from algo.binance.model import ProductModel
from algo.binance.optimiser import OptimiserCfg, Optimiser
from algo.binance.utils import read_json, to_datetime
from algo.cpp.cseries import shift_forward

from algo.definitions import ROOT_DIR
from algo.binance.coins import MarketTypeModel
from algo.binance.data_types import DataTypeModel
from binance.pfolio_data import PfolioAcctData, PfolioLog

SIMS_BASEP = ROOT_DIR / 'sims'

# Minimum trade size in dollar terms
MIN_TRADE_SIZE = 10


class MyConfig:
    arbitrary_types_allowed = True


@pydantic.dataclasses.dataclass(config=MyConfig)
class SimulatorCfg:
    data_type: DataTypeModel
    market_type: MarketTypeModel
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


@dataclass
class TradeLog:
    pair_idx: int
    dollar_size: float
    signal: float
    price: float
    timestamp: int


class PfolioAcctArray:
    def __init__(self, n_pairs: int, betas: np.array, cfgs: list[OptimiserCfg]):
        self.n_pairs = n_pairs

        self.alive_indices = np.array([j for j in range(self.n_pairs)])
        self.alive_mask = np.full(n_pairs, True)

        self.cfgs = cfgs
        self.betas = betas
        self.optimizers = [Optimiser(betas, cfg) for cfg in cfgs]

        self.n_variations = len(cfgs)
        self.acct_data = PfolioAcctData(n_variations=self.n_variations, n_pairs=n_pairs)

    def remove_pair(self, j):
        self.alive_mask[j] = False
        self.optimizers = [Optimiser(self.betas[self.alive_mask], cfg) for cfg in self.cfgs]
        self.alive_indices = np.array([j for j in range(self.n_pairs) if self.alive_mask[j]])

    def optimise_variation(self, i: int, prices: np.array, signals: np.array):
        n_tries = 10
        while 1:
            try:
                delta_dollars = self.optimizers[i].optimise(self.acct_data.qtys[i, self.alive_mask] * prices, signals)
                break
            except ArpackError as e:
                if n_tries == 0:
                    raise e
                n_tries -= 1
            except RuntimeError as e:
                raise e

        return delta_dollars

    def trade(self, prices: np.array, signals: np.array):
        prices = prices[self.alive_mask]
        signals = signals[self.alive_mask]

        self.acct_data.pnls[:, self.alive_mask] = self.acct_data.qtys[:,
                                                  self.alive_mask] * prices - self.acct_data.notionals[:,
                                                                              self.alive_mask]

        delta_dollars = np.vstack([self.optimise_variation(i, prices, signals) for i in range(self.n_variations)])
        delta_dollars[abs(delta_dollars) < MIN_TRADE_SIZE] = 0

        self.acct_data.notionals[:, self.alive_mask] += delta_dollars
        self.acct_data.qtys[:, self.alive_mask] += delta_dollars / prices

        self.acct_data.mkt_exposures = np.einsum('j,ij,j->i', self.betas[self.alive_mask],
                                                 self.acct_data.qtys[:, self.alive_mask], prices)
        self.acct_data.cash_exposures = np.einsum('ij,j -> i', self.acct_data.qtys[:, self.alive_mask], prices)

        return delta_dollars


VariationCollection = list


@dataclass
class SimResults:
    trade_logs: VariationCollection[list[TradeLog]]
    daily_pfolio_logs: list[PfolioLog]
    pairs: list[str]
    opt_cfgs: VariationCollection[OptimiserCfg]


class Simulator:
    def __init__(self, cfg: SimulatorCfg):
        self.cfg = cfg
        self.exp_path = EXP_BASEP / cfg.exp_name
        assert self.exp_path.exists()
        self.logger = logging.getLogger(__name__)

        betas = read_json(self.exp_path / 'betas.json')

        models: dict[str, ProductModel] = pd.read_pickle(self.exp_path / 'models.pkl')

        exp_args_path = self.exp_path / 'exp_args.json'
        try:
            exp_args: ExpArgs = ExpArgs.parse_file(exp_args_path)
        except pydantic.error_wrappers.ValidationError as e:
            raise ValueError(str(exp_args_path)) from e

        start_time = exp_args.tto.test_start_time
        self.logger.info(f'{start_time=}')
        data_start_time = start_time - datetime.timedelta(days=30)
        end_time = self.cfg.end_time

        assert end_time > start_time

        mkt_pairs = exp_args.ro.market_pairs
        mkt_features = []
        for pair, df in load_universe_data(mkt_pairs, data_start_time, end_time, cfg.market_type.t, cfg.data_type.t):
            if df is None:
                raise RuntimeError(f'Could not load data {mkt_pairs=}, {data_start_time=}, {end_time=}')
            try:
                new_mkt_features = features_from_data(df, exp_args.feature_options)
            except NanFeatureException as e:
                raise NanFeatureException(pair) from e
            except VolumeZeroError as e:
                raise VolumeZeroError(pair) from e
            new_mkt_features.columns = [f'{pair}_{col}' for col in new_mkt_features.columns]
            mkt_features.append(new_mkt_features)
        mkt_features = pd.concat(mkt_features, axis=1)

        trade_pairs = list(models.keys())
        if cfg.trade_pair_limit:
            trade_pairs = trade_pairs[:cfg.trade_pair_limit]
        trade_data = load_universe_data(trade_pairs, data_start_time, end_time, market_type=self.cfg.market_type.t,
                                        data_type=self.cfg.data_type.t)

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

            price_ts = df['price']
            exec_price_ts = pd.Series(shift_forward(price_ts.index.values.copy(), price_ts.values.copy(), 1),
                                      index=df.index)
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

        n_pairs = len(self.pairs)
        n_variations = len(self.cfg.opts)
        pfolio_acct = PfolioAcctArray(n_pairs, self.betas, self.cfg.opts)
        signals = np.zeros(len(self.pairs))

        trade_logs: VariationCollection[list[TradeLog]] = [[] for k in range(n_variations)]
        daily_pfolio_logs = [PfolioLog() for k in range(n_variations)]

        last_date: datetime.date = to_datetime(self.times[0]).date()

        for i in range(1, self.nsteps):
            t = self.times[i]
            date = to_datetime(t).date()
            prices = self.exec_prices_matrix[i]

            if date > last_date:
                for k in range(n_variations):
                    daily_pfolio_logs[k].add_date(date, pfolio_acct.acct_data, k, prices)
            last_date = date

            if t in self.removal_times:
                for j in self.removal_times[t]:
                    self.logger.warning(f'Removing pair {j=}, {self.pairs[j]=}')
                    pfolio_acct.remove_pair(j)

            raw_signals = self.signals_matrix[i]
            if self.cfg.ema_hf_periods:
                alpha = np.exp(-1.0 / self.cfg.ema_hf_periods)
                signals = alpha * signals + (1 - alpha) * raw_signals
            else:
                signals = raw_signals

            dollars_traded = pfolio_acct.trade(signals, prices)

            for v in range(n_variations):
                pair_trade = [(idx[0], d) for idx, d in np.ndenumerate(dollars_traded[v]) if d != 0]

                for j, trade_size in pair_trade:
                    trade_logs[v].append(TradeLog(
                        dollar_size=trade_size,
                        pair_idx=j,
                        signal=signals[j],
                        price=prices[j],
                        timestamp=t,
                    ))

        return SimResults(
            trade_logs=trade_logs,
            daily_pfolio_logs=daily_pfolio_logs,
            pairs=self.pairs,
            opt_cfgs=self.opt_cfgs
        )

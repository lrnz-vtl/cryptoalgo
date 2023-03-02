import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from algo.binance.backtest import SimResults, VariationCollection
from binance.pfolio_data import PfolioLogGlobal


def plot_acct(dates: list[datetime.date],
              qtys: list[np.array],
              pnl: list[np.array],
              notional: list[np.array],
              axs):
    for i in range(len(qtys)):
        axs[0].plot(dates, qtys[i], label=i)
        axs[1].plot(dates, pnl[i], label=i)
        axs[2].plot(dates, notional[i], label=i)

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
    axs[2].set_title('notional')
    axs[2].legend()

    axs[3].tick_params(labelrotation=35)
    axs[3].grid()
    axs[3].set_title('volume')
    axs[3].legend()


def plot_global_results(pfolio_logs_global: VariationCollection[PfolioLogGlobal],
                        dst_path: Optional[Path]):
    cols = 5

    f, axs = plt.subplots(1, cols, figsize=(5 * cols, 5))

    names = ['dollar_qty', 'pnl', 'notional', 'mkt_exposure', 'cash_exposure']

    for k, log in enumerate(pfolio_logs_global):
        axs[0].plot(log.dates, log.dollar_qty, label=k)
        axs[1].plot(log.dates, log.pnl, label=k)
        axs[2].plot(log.dates, log.notional, label=k)
        axs[3].plot(log.dates, log.mkt_exposure, label=k)
        axs[4].plot(log.dates, log.cash_exposure, label=k)

    for name, ax in zip(names, axs):
        ax.tick_params(labelrotation=35)
        ax.grid()
        ax.set_title(name)
        ax.legend()

    f.suptitle('global')
    f.tight_layout()

    if dst_path is not None:
        plt.savefig(dst_path / f'global.png')
    else:
        plt.show()

    matplotlib.pyplot.close()


def plot_pair_results(dates: list[datetime.date],
                      qtys: dict[float, np.array],
                      pnl: dict[float, np.array],
                      notional: dict[float, np.array],
                      volume: dict[float, np.array],
                      signals: np.array,
                      prices: np.array,
                      pair_name: str,
                      dst_path: Optional[Path]):
    cols = 6

    f, axs = plt.subplots(1, cols, figsize=(5 * cols, 5))

    plot_acct(dates,
              qtys,
              pnl,
              notional,
              volume,
              axs)

    axs[4].plot(times, signals)
    axs[4].tick_params(labelrotation=35)
    axs[4].grid()
    axs[4].set_title('signal')

    axs[5].plot(times, prices)
    axs[5].tick_params(labelrotation=35)
    axs[5].grid()
    axs[5].set_title('price')

    f.suptitle(pair_name)
    f.tight_layout()

    if dst_path is not None:
        plt.savefig(dst_path / f'{pair_name}.png')
    else:
        plt.show()

    matplotlib.pyplot.close()


def plot_results(results: SimResults, dst_path: Optional[Path]):

    # volumes = {}
    # for i, (cfg, acct) in enumerate(x.cfgs_accts):
    #     nsteps = acct.notional.shape[0]
    #     shifted = np.zeros((nsteps, len(x.pairs)))
    #     shifted[:(nsteps - 1)] = acct.notional[1:]
    #     volumes[i] = abs(acct.notional - shifted)
    #     volumes[i][-1, :] = 0

    plot_global_results(pfolio_logs_global=[x.compute_global() for x in results.daily_pfolio_logs],
                        dst_path=dst_path
                        )

    # for j, pair in enumerate(x.pairs):
    #     plot_pair_results(dates=dates,
    #                       qtys={i: acct.qtys[:, j] * x.prices[:, j] for i, (coef, acct) in enumerate(x.cfgs_accts)},
    #                       pnl={i: acct.pnl[:, j] for i, (coef, acct) in enumerate(x.cfgs_accts)},
    #                       notional={i: acct.notional[:, j] for i, (coef, acct) in enumerate(x.cfgs_accts)},
    #                       volume={i: volumes[i][:, j] for i, (coef, acct) in enumerate(x.cfgs_accts)},
    #                       signals=x.signals_matrix[:, j],
    #                       prices=x.prices[:, j],
    #                       pair_name=pair,
    #                       dst_path=dst_path
    #                       )


def run_reports(sim_path: Path):
    res: SimResults = pd.read_pickle(sim_path / 'results.pkl')

    plot_path = sim_path / 'plots'
    os.makedirs(plot_path, exist_ok=True)
    plot_results(res, plot_path)

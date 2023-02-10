import os
from pathlib import Path
from typing import Optional
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from algo.binance.utils import to_datetime
import pandas as pd
from algo.binance.backtest import SimResults


def plot_acct(times: np.array,
              qtys: dict[float, np.array],
              pnl: dict[float, np.array],
              notional: dict[float, np.array],
              volume: dict[float, np.array],
              axs):
    for coef in qtys.keys():
        axs[0].plot(times, qtys[coef], label=coef)
        axs[1].plot(times, pnl[coef], label=coef)
        axs[2].plot(times, notional[coef], label=coef)
        axs[3].plot(times, volume[coef].cumsum(), label=coef)

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


def plot_global_results(times: np.array,
                        qtys: dict[float, np.array],
                        pnl: dict[float, np.array],
                        notional: dict[float, np.array],
                        volume: dict[float, np.array],
                        mkt_exposure: dict[float, np.array],
                        cash_exposure: dict[float, np.array],
                        dst_path: Optional[Path]):
    cols = 6

    f, axs = plt.subplots(1, cols, figsize=(5 * cols, 5))

    plot_acct(times,
              qtys,
              pnl,
              notional,
              volume,
              axs)

    for coef in qtys.keys():
        axs[4].plot(times, mkt_exposure[coef], label=coef)
        axs[4].tick_params(labelrotation=35)
        axs[4].grid()
        axs[4].set_title('mkt exposure')
        axs[4].legend()

    for coef in qtys.keys():
        axs[5].plot(times, cash_exposure[coef], label=coef)
        axs[5].tick_params(labelrotation=35)
        axs[5].grid()
        axs[5].set_title('cash exposure')
        axs[5].legend()

    f.suptitle('global')
    f.tight_layout()

    if dst_path is not None:
        plt.savefig(dst_path / f'global.png')
    else:
        plt.show()

    matplotlib.pyplot.close()


def plot_pair_results(times: np.array,
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

    plot_acct(times,
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


def plot_results(x: SimResults, dst_path: Optional[Path], coefs_list: Optional[list[float]] = None):
    times = to_datetime(x.times)

    if coefs_list is not None:
        x.accts = {k: v for k, v in x.accts.items() if k in coefs_list}

    volumes = {}
    for i, (cfg, acct) in enumerate(x.cfgs_accts):
        nsteps = acct.notional.shape[0]
        shifted = np.zeros((nsteps, len(x.pairs)))
        shifted[:(nsteps - 1)] = acct.notional[1:]
        volumes[i] = abs(acct.notional - shifted)
        volumes[i][-1, :] = 0

    plot_global_results(times=times,
                        qtys={i: (acct.qtys * x.prices).sum(axis=1) for i, (coef, acct) in enumerate(x.cfgs_accts)},
                        pnl={i: acct.pnl.sum(axis=1) for i, (coef, acct) in enumerate(x.cfgs_accts)},
                        notional={i: acct.notional.sum(axis=1) for i, (coef, acct) in enumerate(x.cfgs_accts)},
                        volume={i: volumes[i].sum(axis=1) for i, (coef, acct) in enumerate(x.cfgs_accts)},
                        mkt_exposure={i: acct.mkt_exposure for i, (coef, acct) in enumerate(x.cfgs_accts)},
                        cash_exposure={i: acct.cash_exposure for i, (coef, acct) in enumerate(x.cfgs_accts)},
                        dst_path=dst_path
                        )

    for j, pair in enumerate(x.pairs):
        plot_pair_results(times=times,
                          qtys={i: acct.qtys[:, j] * x.prices[:, j] for i, (coef, acct) in enumerate(x.cfgs_accts)},
                          pnl={i: acct.pnl[:, j] for i, (coef, acct) in enumerate(x.cfgs_accts)},
                          notional={i: acct.notional[:, j] for i, (coef, acct) in enumerate(x.cfgs_accts)},
                          volume={i: volumes[i][:, j] for i, (coef, acct) in enumerate(x.cfgs_accts)},
                          signals=x.signals_matrix[:, j],
                          prices=x.prices[:, j],
                          pair_name=pair,
                          dst_path=dst_path)


def run_reports(sim_path: Path):
    res: SimResults = pd.read_pickle(sim_path / 'results.pkl')

    plot_path = sim_path / 'plots'
    os.makedirs(plot_path, exist_ok=True)
    plot_results(res, plot_path)

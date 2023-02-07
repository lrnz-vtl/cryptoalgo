import pandas as pd
from matplotlib import pyplot as plt
from algo.binance.fit import FitResults


def plot_row(xxsum, xysum, x, y, axs):#, title: str):
    axs[0].plot(pd.to_datetime(xxsum.index, unit='ms'), xxsum, label='xxsum');
    axs[0].plot(pd.to_datetime(xysum.index, unit='ms'), xysum, label='xysum');

    axs[1].plot(pd.to_datetime(x.index, unit='ms'), x, label='x');
    axs[1].plot(pd.to_datetime(y.index, unit='ms'), y / y.std() * x.std(), label='y');

    axs[0].legend();
    axs[0].grid();

    axs[1].legend();
    axs[1].grid();

    # f.suptitle(title);
    # plt.show();
    # plt.clf();


def sum_series(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.add(b, fill_value=0)


def plot_eval(ress: dict[str, FitResults]):
    x_tot = pd.Series(dtype=float)
    y_tot = pd.Series(dtype=float)

    xxsum_tot = pd.Series(dtype=float)
    xysum_tot = pd.Series(dtype=float)

    xxsums = {}
    xysums = {}

    for (pair, res) in ress.items():
        xxsums[pair] = (res.test.ypred * res.test.ypred).cumsum()
        xysums[pair] = (res.test.ypred * res.test.ytrue).cumsum()

        if x_tot.shape[0] == 0:
            xxsum_tot = xxsums[pair].copy()
            xysum_tot = xysums[pair].copy()

            x_tot = res.test.ypred.copy()
            y_tot = res.test.ytrue.copy()

        else:
            xxsum_tot = sum_series(xxsum_tot, xxsums[pair])
            xysum_tot = sum_series(xysum_tot, xysums[pair])

            x_tot = sum_series(x_tot, res.test.ypred)
            y_tot = sum_series(y_tot, res.test.ytrue)

    # f, axss = plt.subplots(n_pairs, 2, figsize=(20, 5 * n_pairs));

    n_pairs = len(ress)
    n_rows = n_pairs+1
    fig = plt.figure(figsize=(20, 5 * n_rows))
    subfigs = fig.subfigures(nrows=n_rows, ncols=1)

    subfig = subfigs[0]
    axs = subfig.subplots(nrows=1, ncols=2)
    subfig.suptitle(f'total')
    plot_row(xxsum_tot, xysum_tot, x_tot, y_tot, axs);

    for i, (pair, res) in enumerate(ress.items()):
        subfig = subfigs[i+1]
        axs = subfig.subplots(nrows=1, ncols=2)
        subfig.suptitle(pair)
        plot_row(xxsums[pair], xysums[pair], res.test.ypred, res.test.ytrue, axs);

    # f.tight_layout();

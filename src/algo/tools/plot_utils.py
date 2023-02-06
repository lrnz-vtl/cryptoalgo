import matplotlib.pyplot as plt
from algo.universe.assets import get_name


def plot_asset(ds, asset_id):
    """Show a plot of the historical price of a given asset"""
    plt.figure(figsize=(14, 7))
    name, ticker = get_name(asset_id)
    plt.title(name, fontsize=14)
    plt.ylabel(f'ALGO per {ticker}', fontsize=12)
    plt.plot(ds.data[asset_id]['price_history'].index, ds.data[asset_id]['price_history']['price'].values,
             label='Price', color='C0', alpha=0.65)
    plt.plot(ds.data[asset_id]['MA_15min'].index, ds.data[asset_id]['MA_15min']['price'].values,
             label='Moving Avg (15min)', color='C1', alpha=0.85)
    plt.plot(ds.data[asset_id]['MA_2h'].index, ds.data[asset_id]['MA_2h']['price'].values,
             label='Moving Avg (2h)', color='C3', alpha=0.9)
    plt.grid()
    plt.legend(fontsize=12)
    plt.show()
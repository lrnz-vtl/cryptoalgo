import matplotlib.pyplot as plt
from algo.blockchain.utils import generator_to_df
from algo.blockchain.process_prices import PriceScraper
from tinyman.v1.client import TinymanMainnetClient
from algo.blockchain.algo_requests import QueryParams
import aiohttp


def plot_price(asset1_id: int, asset2_id: int, num_queries: int, timestamp: int = 0,
               inverse: bool = False, savefig: bool = False):

    client = TinymanMainnetClient()
    ps = PriceScraper(client, asset1_id, asset2_id)
    pool = client.fetch_pool(asset1=asset1_id, asset2=asset2_id)

    async with aiohttp.ClientSession() as session:
        df = generator_to_df(ps.scrape(session, timestamp, QueryParams(), num_queries))

    if not inverse:
        price = df.asset2_reserves/df.asset1_reserves
    else:
        price = df.asset1_reserves/df.asset2_reserves

    plt.xlabel('Time')
    if not inverse:
        plt.ylabel(f'{pool.asset2.unit_name} per {pool.asset1.unit_name}')
    else:
        plt.ylabel(f'{pool.asset1.unit_name} per {pool.asset2.unit_name}')
    plt.title(f'{pool.liquidity_asset.name}')
    plt.plot(df['time'], price)
    if savefig:
        plt.savefig(f'{pool.asset1.unit_name}_{pool.asset2.unit_name}.png', bbox_inches="tight")
    else:
        plt.show()

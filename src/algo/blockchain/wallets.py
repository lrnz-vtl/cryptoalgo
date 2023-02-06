from __future__ import annotations
import time, datetime, warnings
import pandas as pd
from typing import Dict, Iterable, Tuple, Optional
from algo.blockchain.algo_requests import QueryParams
from algo.blockchain.stream import DataStream
from algo.strategy.analytics import ffill_cols, timestamp_to_5min
from algo.tools.wallets import get_account_data
from tinyman.v1.client import TinymanMainnetClient
from algo.blockchain.mixedstream import TotalDataLoader
from tinyman.v1.pools import Asset


class WalletValue:

    def __init__(self, cache_name, address):
        self.price_table = None
        self.liquidity_tokens = None
        self.price_df = None
        self.cache_name = cache_name
        self.address = address

        ffilt = lambda x, y: x == 0 or y == 0
        self.tds = TotalDataLoader(cache_name, TinymanMainnetClient(), ffilt)

    async def update(self):
        self.price_df = await self.tds.load()
        self.liquidity_tokens = {}
        client = TinymanMainnetClient()
        for aid in self.price_df['asset1'].unique():
            pool = client.fetch_pool(Asset(aid), Asset(0))
            self.liquidity_tokens[pool.asset1.id] = pool.liquidity_asset.id

        prices_table = self.price_df.copy()
        prices_table['time_5min'] = prices_table['time'] // (5 * 60) * (5 * 60)

        prices_table['liquidity_price'] = 2 * prices_table['asset2_reserves'] / prices_table['issued_liquidity']
        prices_table['price'] = prices_table['asset2_reserves'] / prices_table['asset1_reserves']

        liq_prices_table = prices_table.groupby(['asset1', 'time_5min'])['liquidity_price'].mean()
        liq_prices_table = liq_prices_table.unstack(level=0).fillna(method='ffill')
        liq_prices_table.columns = [self.liquidity_tokens[col] for col in liq_prices_table.columns]

        asset_prices_table = prices_table.groupby(['asset1', 'time_5min'])['price'].mean()
        asset_prices_table = asset_prices_table.unstack(level=0).fillna(method='ffill')
        asset_prices_table[0] = 1

        self.price_table = pd.concat([asset_prices_table, liq_prices_table], axis=1)

    def hist_params(self, query_params):
        datastream = DataStream.from_address(self.address, query_params)

        txns = []

        for _, txn in datastream.next_transaction():
            time = txn['round-time']

            if txn['tx-type'] == 'pay':
                if txn['payment-transaction']['receiver'] == self.address:
                    txns.append((0, - txn['payment-transaction']['amount'], time))
                elif txn['sender'] == self.address:
                    txns.append((0, txn['payment-transaction']['amount'] + txn['fee'], time))
                else:
                    raise ValueError('encountered invalid transaction')
            elif txn['tx-type'] == 'axfer':
                assetid = txn['asset-transfer-transaction']['asset-id']
                if txn['asset-transfer-transaction']['receiver'] == self.address:
                    txns.append((assetid, - txn['asset-transfer-transaction']['amount'], time))
                elif txn['sender'] == self.address:
                    txns.append((0, txn['fee'], time))
                    txns.append((assetid, txn['asset-transfer-transaction']['amount'], time))
                else:
                    raise ValueError('encountered invalid transaction')
            else:
                if txn['sender'] == self.address:
                    txns.append((0, txn['fee'], time))

        return txns

    def historical_wealth(self):
        after_time = datetime.datetime.fromtimestamp(self.price_df['time'].min())
        query_params = QueryParams(after_time=after_time)
        txns = self.hist_params(query_params)

        price_table = self.price_table.copy()

        positions_table = pd.DataFrame(0, index=price_table.index, columns=price_table.columns)
        for asset, amount, time in txns:
            positions_table.loc[positions_table.index >= time, asset] -= amount

        wealth_table = (positions_table * price_table) / 10 ** 6
        wealth = wealth_table.sum(axis=1)
        wealth.index = pd.to_datetime(wealth.index, unit='s', utc=True)

        price_table.index = pd.to_datetime(price_table.index, unit='s', utc=True)

        return price_table, wealth[wealth > 0]

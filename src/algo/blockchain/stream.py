from __future__ import annotations
import logging
from algo.blockchain.process_volumes import PoolTransaction, Swap, is_fee_payment
from algo.blockchain.process_prices import PoolState, get_pool_state_txn
from algo.blockchain.algo_requests import QueryParams
from algo.universe.universe import SimpleUniverse
from typing import Optional, Union, Generator, Any
import pandas as pd
import requests
from dataclasses import dataclass
import datetime
from datetime import timezone
import numpy as np
from algo.blockchain.utils import int_to_tzaware_utc_datetime
import json


def get_pool_transaction_txn(tx: dict, pool_address: str, key: str, asset_id: int):
    receiver, sender = tx[key]['receiver'], tx['sender']

    if pool_address == receiver:
        counterparty = sender
        sign = +1
    elif pool_address == sender:
        counterparty = receiver
        sign = -1
    elif pool_address == tx[key]['close-to']:
        return None
    else:
        raise ValueError(f'pool_address {pool_address} neither in sender nor receiver')

    amount = sign * tx[key]['amount']
    block = tx['confirmed-round']
    return PoolTransaction(amount, asset_id, block, counterparty, tx['tx-type'], tx['round-time'])


class StreamException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class DataStream:
    def __init__(self, universe: SimpleUniverse, query_params: QueryParams, next_token: Optional[str] = None):
        if universe:
            self.universe = universe
            self.pools = {x.address for x in universe.pools}
        self.url = f'https://algoindexer.algoexplorerapi.io/v2/transactions'
        self.params = query_params.make_params()
        if next_token:
            self.params['next'] = next_token

        self.logger = logging.getLogger(__name__)

    @staticmethod
    def from_address(address: str, query_params: QueryParams) -> DataStream:
        datastream = DataStream(None, query_params)
        datastream.params['address'] = address
        datastream.pools = {address}
        return datastream
    
    def next_transaction(self):

        with requests.session() as session:

            while True:
                self.logger.debug('Making new request')

                try:
                    req1 = session.get(url=self.url, params=self.params)
                except requests.exceptions.ChunkedEncodingError as e:
                    raise StreamException("ChunkedEncodingError")

                if not req1.ok:
                    self.logger.critical("session.get response is not OK"
                                         f"\n url = {self.url}")
                    raise StreamException("Response not OK")

                try:
                    req = req1.json()
                except json.decoder.JSONDecodeError as e:
                    self.logger.critical(f"req1.json() failed: "
                                         f"\n req1 = {req1}"
                                         f"\n {e.msg}"
                                         f"\n url = {self.url}"
                                         )
                    raise StreamException("json.decoder.JSONDecodeError")

                first_time = None
                if req['transactions']:
                    first_time = datetime.datetime.fromtimestamp(req['transactions'][0]['round-time'])
                self.logger.debug(f'Queried transaction group, time={first_time}')

                for tx in req['transactions']:
                    pool = None
                    if tx['sender'] in self.pools:
                        pool = tx['sender']
                    elif tx['tx-type'] == 'pay' and tx['payment-transaction']['receiver'] in self.pools:
                        pool = tx['payment-transaction']['receiver']
                    elif tx['tx-type'] == 'axfer' and tx['asset-transfer-transaction']['receiver'] in self.pools:
                        pool = tx['asset-transfer-transaction']['receiver']
                    if pool:
                        yield pool, tx
                if 'next-token' not in req:
                    break
                self.params['next'] = req['next-token']


@dataclass
class PriceOrVolumeUpdate:
    asset_ids: tuple[int, int]
    market_update: Union[PoolState, Swap]


@dataclass
class PriceUpdate:
    asset_ids: tuple[int, int]
    price_update: PoolState


def stream_from_price_df(df: pd.DataFrame, start_time: datetime.datetime) -> Generator[PriceUpdate, Any, Any]:
    assert np.all(
        df.columns == ['time', 'asset1_reserves', 'asset2_reserves', 'block', 'reverse_order_in_block', 'asset1',
                       'asset2']), f'df.columns = {df.columns}'
    df = df.sort_values(by='time')

    assert start_time.tzinfo == timezone.utc

    def row_to_update(row):
        return PriceUpdate(
            asset_ids=(row['asset1'], row['asset2']),
            price_update=PoolState(
                time=row['time'],
                asset1_reserves=row['asset1_reserves'],
                asset2_reserves=row['asset2_reserves'],
                block=row['block'],
                reverse_order_in_block=None
            )
        )

    for _, row in df.iterrows():
        time = int_to_tzaware_utc_datetime(row['time'])

        if time >= start_time:
            yield row_to_update(row)


def only_price(gen: Generator[PriceOrVolumeUpdate, Any, Any]) -> Generator[PriceUpdate, Any, Any]:
    for x in gen:
        if isinstance(x.market_update, PoolState):
            yield PriceUpdate(x.asset_ids, x.market_update)


# FIXME this does not preserve chronological across different pools
def filter_last_prices(gen: Generator[PriceOrVolumeUpdate, Any, Any]) -> Generator[PriceOrVolumeUpdate, Any, Any]:
    last_time: Optional[int] = None
    last_price: Optional[PriceOrVolumeUpdate] = None
    for update in gen:
        if isinstance(update.market_update, Swap):
            yield update
        elif isinstance(update.market_update, PoolState):
            time = update.market_update.time
            if last_price:
                assert time >= last_time
                if time > last_time:
                    yield last_price
            last_price = update
            last_time = time
        else:
            raise ValueError
    if last_price:
        yield last_price


class PoolStateQueue:

    def __init__(self):
        self.app_tx_current_block: list[tuple[tuple[int, int], PoolState]] = []
        self.last_block_time = None

    def flush(self):
        reverse_block_order = len(self.app_tx_current_block) - 1
        for prev_asset_ids, prev_block_ps in self.app_tx_current_block:
            yield PriceOrVolumeUpdate(prev_asset_ids, prev_block_ps.with_reverse_order(reverse_block_order))
            reverse_block_order -= 1
        self.app_tx_current_block = []

    def push_and_yield(self, ps, asset_ids):
        if self.last_block_time:
            assert ps.time >= self.last_block_time
        if self.last_block_time and ps.time > self.last_block_time:
            yield from self.flush()
        self.last_block_time = ps.time
        self.app_tx_current_block.append((asset_ids, ps))


class PriceVolumeStream:
    def __init__(self, data_stream: DataStream):
        self.data_stream = data_stream

        self.address_ids_map = {x.address: (x.asset1_id, x.asset2_id) for x in data_stream.universe.pools}

        self.transaction_in_ = {pool: None for pool in self.address_ids_map.keys()}
        self.transaction_fee_ = {pool: False for pool in self.address_ids_map.keys()}

    def scrape(self) -> Generator[PriceOrVolumeUpdate, Any, Any]:

        def is_transaction_out(tx: PoolTransaction, transaction_in: PoolTransaction,
                               asset1_id: int, asset2_id: int):
            return tx.amount < 0 and tx.counterparty == transaction_in.counterparty \
                   and tx.asset_id != transaction_in.asset_id \
                   and tx.asset_id in [asset1_id, asset2_id] \
                   and not is_fee_payment(tx)

        price_queue = PoolStateQueue()

        for address, tx in self.data_stream.next_transaction():
            pt = None
            asset_ids = self.address_ids_map[address]

            if tx['tx-type'] == 'appl':
                ps = get_pool_state_txn(tx)
                if ps:
                    yield from price_queue.push_and_yield(ps, asset_ids)

            elif tx['tx-type'] == 'pay':
                key = 'payment-transaction'
                pt = get_pool_transaction_txn(tx, address, key, 0)
            elif tx['tx-type'] == 'axfer':
                key = 'asset-transfer-transaction'
                pt = get_pool_transaction_txn(tx, address, key, tx[key]['asset-id'])
            if pt:
                asset1_id = asset_ids[0]
                asset2_id = asset_ids[1]
                # adapted from algo.blockchain.process_volumes.SwapScraper.scrape
                if self.transaction_fee_[address]:
                    if self.transaction_in_[address]:
                        if is_transaction_out(pt, self.transaction_in_[address], asset1_id, asset2_id):
                            if self.transaction_in_[address].asset_id == asset1_id and pt.asset_id == asset2_id:
                                asset1_amount = self.transaction_in_[address].amount
                                asset2_amount = pt.amount
                            elif self.transaction_in_[address].asset_id == asset2_id and pt.asset_id == asset1_id:
                                asset2_amount = self.transaction_in_[address].amount
                                asset1_amount = pt.amount
                            else:
                                raise ValueError
                            assert self.transaction_in_[address].amount > 0 > pt.amount
                            swap = Swap(asset1_amount=asset1_amount,
                                        asset2_amount=asset2_amount,
                                        counterparty=pt.counterparty,
                                        block=pt.block,
                                        time=pt.time)
                            yield PriceOrVolumeUpdate(asset_ids, swap)
                        self.transaction_fee_[address] = False
                        self.transaction_in_[address] = None
                    else:
                        if pt.amount > 0 and not is_fee_payment(pt) and pt.asset_id in [asset1_id, asset2_id]:
                            self.transaction_in_[address] = pt
                        else:
                            self.transaction_in_[address] = None
                elif is_fee_payment(pt):
                    self.transaction_fee_[address] = True

        yield from price_queue.flush()





class PriceVolumeDataStore:

    def __init__(self, price_volume_stream: PriceVolumeStream):

        self.price_volume_stream = price_volume_stream

        self.address_ids_map = self.price_volume_stream.address_ids_map
        self._reset()

    def _gen_df(self, data_map):
        for ids, data in data_map.items():
            df = pd.DataFrame(data)
            df['asset1'] = ids[0]
            df['asset2'] = ids[1]
            yield df

    def _reset(self):
        self.prices_ = {ids: list() for ids in self.address_ids_map.values()}
        self.volumes_ = {ids: list() for ids in self.address_ids_map.values()}

    def volumes(self):
        return pd.concat(self._gen_df(self.volumes_))

    def prices(self):
        return pd.concat(self._gen_df(self.prices_))

    def scrape(self):
        for update in self.price_volume_stream.scrape():
            if isinstance(update.market_update, Swap):
                update_arr = self.volumes_
            elif isinstance(update.market_update, PoolState):
                update_arr = self.prices_
            else:
                raise ValueError
            update_arr[update.asset_ids].append(update.market_update)

    def update(self, prices, volumes):
        self._reset()
        self.scrape()
        return prices.append(self.prices()), volumes.append(self.volumes())

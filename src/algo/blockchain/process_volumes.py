import json
from dataclasses import dataclass
from typing import Optional
from algo.blockchain.algo_requests import query_transactions, QueryParams
from tinyman.v1.client import TinymanClient
from algo.blockchain.base import DataScraper, NotExistentPoolError
from algo.blockchain.cache import DataCacher
from definitions import ROOT_DIR
import datetime
from algo.universe.pools import PoolIdStore
import aiohttp

VOLUME_CACHES_BASEDIR = f'{ROOT_DIR}/caches/volumes'


@dataclass
class PoolTransaction:
    amount: int
    asset_id: int
    block: int
    counterparty: str
    tx_type: str
    time: int


async def query_transactions_for_pool(session: aiohttp.ClientSession,
                                      pool_address: str,
                                      num_queries: int,
                                      query_params: QueryParams
                                      ):
    async for tx in query_transactions(session=session,
                                       params={'address': pool_address},
                                       num_queries=num_queries,
                                       query_params=query_params
                                       ):

        try:
            if tx['tx-type'] == 'axfer':
                # ASA
                key = 'asset-transfer-transaction'
                asset_id = tx[key]['asset-id']

            elif tx['tx-type'] == 'pay':
                # Algo
                key = 'payment-transaction'
                asset_id = 0
            else:
                continue

            receiver, sender = tx[key]['receiver'], tx['sender']

            if pool_address == receiver:
                counterparty = sender
                sign = +1
            elif pool_address == sender:
                counterparty = receiver
                sign = -1
            elif pool_address == tx[key]['close-to']:
                # I haven't understood this case but hopefully it's not too important
                continue
            else:
                raise ValueError(f'pool_address {pool_address} neither in sender nor receiver')

            amount = sign * tx[key]['amount']
            block = tx['confirmed-round']
            yield PoolTransaction(amount, asset_id, block, counterparty, tx['tx-type'], tx['round-time'])

        except Exception as e:
            raise Exception(json.dumps(tx, indent=4)) from e


# Logged swap for a pool, excluding redeeming amounts
@dataclass
class Swap:
    # Signed volume of asset 1 (positive goes into the pool)
    asset1_amount: int
    # Signed volume of asset 2 (positive goes into the pool)
    asset2_amount: int
    counterparty: str
    block: int
    time: int


# TODO Check this is valid, does it also hold for pools without Algo?
def is_fee_payment(tx: PoolTransaction):
    return tx.asset_id == 0 and tx.amount == 2000 and tx.tx_type == 'pay'


class SwapScraper(DataScraper):
    def __init__(self, client: TinymanClient, asset1_id: int, asset2_id: int):

        pool = client.fetch_pool(asset1_id, asset2_id)

        if not pool.exists:
            raise NotExistentPoolError()

        self.liquidity_asset = pool.liquidity_asset.id
        self.asset1_id = asset1_id
        self.asset2_id = asset2_id
        self.address = pool.address

    async def scrape(self, session: aiohttp.ClientSession,
                     timestamp_min: Optional[int],
                     query_params: QueryParams,
                     num_queries: Optional[int] = None):

        def is_transaction_in(tx: PoolTransaction, transaction_out: PoolTransaction):
            return tx.counterparty == transaction_out.counterparty \
                   and tx.asset_id != transaction_out.asset_id \
                   and tx.asset_id in [self.asset1_id, self.asset2_id] \
                   and not is_fee_payment(tx)

        transaction_out: Optional[PoolTransaction] = None
        transaction_in: Optional[PoolTransaction] = None

        async for tx in query_transactions_for_pool(session, self.address, num_queries, query_params=query_params):

            if timestamp_min is not None and tx.time < timestamp_min:
                break

            if transaction_out:
                # We recorded a transaction out and in, looking for a fee payment
                if transaction_in:
                    if is_fee_payment(tx) and tx.counterparty == transaction_in.counterparty:
                        if transaction_in.asset_id == self.asset1_id and transaction_out.asset_id == self.asset2_id:
                            asset1_amount = transaction_in.amount
                            asset2_amount = transaction_out.amount
                        elif transaction_in.asset_id == self.asset2_id and transaction_out.asset_id == self.asset1_id:
                            asset2_amount = transaction_in.amount
                            asset1_amount = transaction_out.amount
                        else:
                            raise ValueError
                        assert transaction_in.amount > 0 > transaction_out.amount

                        yield Swap(asset1_amount=asset1_amount,
                                   asset2_amount=asset2_amount,
                                   counterparty=tx.counterparty,
                                   block=tx.block,
                                   time=tx.time
                                   )
                    transaction_out = None
                    transaction_in = None

                # We recorded a transaction out, looking for a transaction in
                else:
                    # TODO We should account for redeeming excess funds from the pool?
                    if is_transaction_in(tx, transaction_out):
                        transaction_in = tx
                    else:
                        transaction_out = None
            else:
                if tx.amount < 0 and tx.asset_id in [self.asset1_id, self.asset2_id]:
                    transaction_out = tx


class VolumeCacher(DataCacher):

    def __init__(self, client: TinymanClient, pool_id_store: PoolIdStore,
                 date_min: datetime.datetime,
                 date_max: Optional[datetime.datetime],
                 dry_run:bool):
        super().__init__(pool_id_store, VOLUME_CACHES_BASEDIR, client, date_min, date_max, dry_run)

    def make_scraper(self, asset1_id: int, asset2_id: int):
        try:
            return SwapScraper(self.client, asset1_id, asset2_id)
        except NotExistentPoolError as e:
            return None

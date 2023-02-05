from __future__ import annotations
from tinyman.v1.client import TinymanClient
from typing import Optional
import numpy as np
import json
import logging
import requests
from enum import Flag, auto
import dataclasses
from dataclasses import dataclass
from algo.universe.hardcoded import ALGO_DECIMALS
from functools import lru_cache


def get_asset_data(asset_id: int):
    res = requests.get(f'https://algoindexer.algoexplorerapi.io/v2/assets/{asset_id}').json()
    return res['asset']


@lru_cache()
def get_asset_name(asset_id):
    return get_asset_data(asset_id)['params']['name']


@lru_cache()
def get_decimals(asset_id):
    return get_asset_data(asset_id)['params']['decimals']


@dataclass
class AssetInfo:
    id: int
    is_liquidity_token: bool
    name: str
    unit_name: str
    decimals: int
    total_amount: int
    url: str
    is_verified: bool

    @staticmethod
    def from_dict(d):
        d['id'] = int(d['id'])
        return AssetInfo(**d)


class AssetType(Flag):
    LIQUIDITY = auto()
    NOT_LIQUIDITY = auto()
    ALL = LIQUIDITY | NOT_LIQUIDITY


class CandidateASAStore:
    """ This includes both liquidity and not-liquidity token """

    def __init__(self, assets: list[AssetInfo], verified_only: bool):
        self.verified_only = verified_only
        self.assets = assets

    @staticmethod
    def from_scratch(verified_only: bool = True, test: bool = False) -> CandidateASAStore:
        assets = CandidateASAStore._fetch(verified_only, test)
        return CandidateASAStore(assets=assets, verified_only=verified_only)

    @staticmethod
    def from_store(cs: CandidateASAStore, filter_type: AssetType) -> CandidateASAStore:

        def filter_asa(x):
            if x.is_liquidity_token and (filter_type & AssetType.LIQUIDITY):
                return True
            if (not x.is_liquidity_token) and (filter_type & AssetType.NOT_LIQUIDITY):
                return True
            return False

        assets = [x for x in cs.assets if filter_asa(x)]
        return CandidateASAStore(assets, cs.verified_only)

    @staticmethod
    def _fetch(verified_only: bool, test: bool) -> list[AssetInfo]:

        def filter_asa(x):
            if verified_only and not x['is_verified'] and not x['is_liquidity_token']:
                return False
            return True

        url = 'https://mainnet.analytics.tinyman.org/api/v1/assets/?ordering=id'
        results = []
        i = 0
        while url is not None:
            if test and i >= 10:
                break

            res = requests.get(url)
            try:
                res = res.json()
            except json.decoder.JSONDecodeError as e:
                raise json.decoder.JSONDecodeError(f"url={url}", doc=e.doc, pos=e.pos) from e

            url = res['next']
            new = [AssetInfo.from_dict(x) for x in res['results'] if filter_asa(x)]
            results += new

            i += 1

        return results

    def as_dict(self):
        return {
            'verified_only': self.verified_only,
            'assets': [dataclasses.asdict(x) for x in self.assets]
        }


@dataclass
class AssetMarketStats:
    asset_id: int
    asset_name: str
    asset_unit_name: str
    asset_decimals: int
    # Fully diluted market capitalisation (in Algo)
    fdmc: float
    # TODO Add more fields here like in the page https://tinychart.org/asset/470842789


class AssetMarketDataStore:

    def __init__(self, client: TinymanClient, asset_list: list[int]):
        self.client = client
        self.logger = logging.getLogger('AssetMarketDataStore')
        self.data = [x for x in [self.get_info(asset_id) for asset_id in asset_list] if x is not None]
        self.data = sorted(self.data, key=lambda x: x.fdmc)

    def get_info(self, asset_id: int) -> Optional[AssetMarketStats]:

        self.logger.info(f'Retrieving info for {asset_id}')

        asset_info = self.client.fetch_asset(asset_id)

        asset_data = get_asset_data(asset_info.id)
        total_supply = asset_data['params']['total']

        pool_info = self.client.fetch_pool(0, asset_id)

        if pool_info.exists:
            # FIXME These calculations are wrong, I'm not sure why
            if pool_info.asset2_reserves == 0 or pool_info.asset1_reserves == 0:
                price = np.nan
            else:
                price = (pool_info.asset2_reserves / (10 ** asset_info.decimals)) / \
                        (pool_info.asset1_reserves / (10 ** ALGO_DECIMALS))

            fdmc = total_supply * price / (10 ** asset_info.decimals)

            return AssetMarketStats(
                asset_id=asset_info.id,
                asset_name=asset_info.name,
                asset_unit_name=asset_info.unit_name,
                asset_decimals=asset_info.decimals,
                fdmc=fdmc
            )
        else:
            return None

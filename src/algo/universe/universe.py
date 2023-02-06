from __future__ import annotations
import os.path

from tinyman.v1.client import TinymanClient

from algo.universe.pools import PoolInfo, PoolInfoStore, PoolId, load_pool_info, PoolIdStore
import json
import dataclasses

from definitions import ROOT_DIR

UNIVERSE_CACHE_BASEDIR = os.path.join(ROOT_DIR, 'caches', 'universe')
SIMPLEUNIVERSE_CACHE_BASEDIR = os.path.join(ROOT_DIR, 'caches', 'simple_universe')


class SimpleUniverse:
    def __init__(self, pools: list[PoolId]):
        self.pools = pools

    @staticmethod
    def from_cache(universe_cache_name: str) -> SimpleUniverse:
        cache_file = os.path.join(SIMPLEUNIVERSE_CACHE_BASEDIR, universe_cache_name, 'universe.json')
        pool_store = load_pool_info(cache_file)
        return SimpleUniverse(pool_store.pools)

    @staticmethod
    def from_id_list(ids: list[int], client: TinymanClient) -> SimpleUniverse:
        pool_store = PoolIdStore.from_id_list(ids, client)
        return SimpleUniverse(pool_store.pools)


class Universe:

    def __init__(self, pools: list[PoolInfo], poolinfo_cache_name: str):
        self.pools = pools
        self.poolinfo_cache_name = poolinfo_cache_name

    @staticmethod
    def from_cache(universe_cache_name: str) -> Universe:
        fname = os.path.join(UNIVERSE_CACHE_BASEDIR, universe_cache_name, 'universe.json')
        with open(fname) as f:
            data = json.load(f)
        return Universe([PoolInfo(**x) for x in data['pools']], data['poolinfo_cache_name'])

    @staticmethod
    def from_poolinfo_cache(poolinfo_cache_name: str, n_most_liquid: int) -> Universe:
        pis = PoolInfoStore.from_cache(poolinfo_cache_name)

        def key(p: PoolInfo):
            return -(p.current_asset_1_reserves_in_usd + p.current_asset_2_reserves_in_usd) / 2.0

        pools = list(sorted(pis.pools, key=key))[:n_most_liquid]
        return Universe(pools, poolinfo_cache_name)

    def asdicts(self):
        return [dataclasses.asdict(x) for x in self.pools]

    def serialize(self, cache_name: str):
        base_folder = os.path.join(UNIVERSE_CACHE_BASEDIR, cache_name)
        os.makedirs(base_folder, exist_ok=True)
        data = {
            'pools': self.asdicts(),
            'poolinfo_cache_name': self.poolinfo_cache_name
        }

        fname = os.path.join(base_folder, 'universe.json')
        with open(fname, 'w') as f:
            json.dump(data, f, indent=4)

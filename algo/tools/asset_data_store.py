import os
import unittest
from definitions import ROOT_DIR
from tinyman.v1.client import TinymanClient, TinymanMainnetClient
from tinyman.v1.pools import Asset
from dataclasses import asdict
import json
from pathlib import Path

ASSET_CACHE_FILE = os.path.join(ROOT_DIR, 'caches', 'assets.json')


class AssetDataStore:
    def __init__(self, client: TinymanClient):
        self.client = client
        os.makedirs(Path(ASSET_CACHE_FILE).parent, exist_ok=True)

    def fetch_asset(self, asset_id: int) -> Asset:

        if os.path.exists(ASSET_CACHE_FILE):
            with open(ASSET_CACHE_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        if str(asset_id) not in data:
            asset = self.client.fetch_asset(asset_id)
            data[str(asset_id)] = asdict(asset)
            with open(ASSET_CACHE_FILE, 'w') as f:
                json.dump(data, f)
        else:
            asset = Asset(**data[str(asset_id)])
        return asset


asset_data_store = None


def get_asset_datastore() -> AssetDataStore:
    global asset_data_store
    if not asset_data_store:
        asset_data_store = AssetDataStore(client=TinymanMainnetClient())
    return asset_data_store


class TestAssetDataStore(unittest.TestCase):

    def test_data(self):
        aid = 226701642
        from tinyman.v1.client import TinymanMainnetClient
        client = TinymanMainnetClient()
        asset: Asset = AssetDataStore(client).fetch_asset(aid)
        print(asset)

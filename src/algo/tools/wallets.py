from __future__ import annotations
import requests


def get_asset_data(asset_id, testnet=False):
    if testnet:
        asset = requests.get(url=f'https://testnet.algoexplorerapi.io/idx2/v2/assets/{asset_id}').json()
    else:
        asset = requests.get(url=f'https://algoexplorerapi.io/idx2/v2/assets/{asset_id}').json()
    return asset['asset']


def get_decimal(asset_id, testnet=False):
    if testnet:
        asset = requests.get(url=f'https://testnet.algoexplorerapi.io/idx2/v2/assets/{asset_id}').json()
    else:
        asset = requests.get(url=f'https://algoexplorerapi.io/idx2/v2/assets/{asset_id}').json()
    return asset['asset']['params']['decimals']


def get_account_data(address=None, testnet=False):
    """Query wallet data from AlgoExplorer"""

    if testnet:
        url = f'https://testnet.algoexplorerapi.io/v2/accounts/{address}'
    else:
        url = f'https://algoexplorerapi.io/v2/accounts/{address}'

    # as specified here https://algoexplorer.io/api-dev/v2
    data = requests.get(url=url).json()

    # set up dictionary with values for each coin
    coins = {0: data['amount']}  # / 10 ** 6}
    for d in data['assets']:
        coins[d['asset-id']] = d['amount']  # / 10 ** get_decimal(d['asset-id'],testnet)

    # return the assets in the wallet
    # note: we are discarding some info here (rewards, offline, frozen, etc)
    return coins

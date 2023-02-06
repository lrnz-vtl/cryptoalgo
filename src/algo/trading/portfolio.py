from __future__ import annotations
from typing import Dict, Iterable, Tuple
from algo.blockchain.wallets import get_account_data


class Portfolio:
    def __init__(self, address=None, testnet=False):
        """Set up the portfolio from a wallet address"""

        self.testnet = testnet

        if not address:
            # set up a dummy portfolio
            self.coins: Dict[int, int] = {0: 1_000_000}
        else:
            self.address = address
            self.update()

    def update(self):
        self.coins: Dict[int, int] = get_account_data(self.address, self.testnet)

    def __getitem__(self, asset_id):
        return self.coins[asset_id]

    def __len__(self):
        return len(self.coins)

    def __iter__(self) -> Iterable[Tuple[int, int]]:
        return iter(self.coins)

import datetime
from dataclasses import dataclass
import numpy as np
from algo.trading.swapper import AlgoPoolSwap, RedeemedAmounts
from algo.trading.costs import FIXED_FEE_MUALGOS
import pandas as pd
from algo.blockchain.stream import PoolState

# Leading order Taylor expansions of the functions below
impact_deflection_bps_perfraction = 2.0
avg_impact_deflection_bps_perfraction = 1.0


def impact_deflection_bps(asset_pool_percentage: float) -> float:
    """
    asset_pool_percentage: percentage of the token we take out relative to pool assets
    returns -> instantaneous percentage change of price of the token we buy in units of the token we sell
    """
    assert 0 <= asset_pool_percentage <= 1, f"asset_pool_percentage = {asset_pool_percentage}"
    return 1.0 / (1.0 - asset_pool_percentage) ** 2 - 1.0


def avg_impact_deflection_bps(asset_pool_percentage: float) -> float:
    """
    asset_pool_percentage: percentage of the token we take out relative to pool assets
    returns -> The average percentage price deflection paid per a single transaction per token bought
    """
    assert 0 <= asset_pool_percentage <= 1
    return 1.0 / (1.0 - asset_pool_percentage) - 1.0


class ASAImpactState:
    def __init__(self, decay_timescale_seconds: int):
        self.state = 0
        # This does not really matter if s
        self.t = np.datetime64('NaT')
        self.decay_timescale_seconds = decay_timescale_seconds

    def update(self, swap: AlgoPoolSwap, mualgo_reserves: int, asa_reserves: int, t: datetime.datetime):

        if not pd.isnull(self.t):
            delta = t - self.t
            state = self.state * np.exp(- delta.total_seconds() / self.decay_timescale_seconds)
        else:
            state = 0

        if swap.asset_buy == 0:
            assert 0 <= swap.amount_buy <= mualgo_reserves
            algo_price_deflection = impact_deflection_bps(swap.amount_buy / mualgo_reserves)
            state += 1 / (1 + algo_price_deflection) - 1.0
        elif swap.asset_buy > 0:
            assert 0 <= swap.amount_buy <= asa_reserves
            asa_price_deflection = impact_deflection_bps(swap.amount_buy / asa_reserves)
            state += asa_price_deflection
        else:
            raise ValueError

        self.t = t
        self.state = state

    def value(self, t: datetime.datetime):
        if pd.isnull(self.t):
            return 0
        delta = t - self.t
        return self.state * np.exp(- delta.total_seconds() / self.decay_timescale_seconds)


@dataclass
class ASAPosition:
    value: int

    def update(self, traded_swap: AlgoPoolSwap):
        if traded_swap.asset_buy == 0:
            self.value -= traded_swap.amount_sell_with_slippage
            # We can't short
            assert self.value >= 0
        else:
            self.value += traded_swap.amount_buy_with_slippage


@dataclass
class PositionAndImpactState:
    impact: ASAImpactState
    asa_position: ASAPosition

    def update_trade(self, traded_swap: AlgoPoolSwap, mualgo_reserves: int, asa_reserves: int, t: datetime.datetime):
        self.impact.update(traded_swap, mualgo_reserves, asa_reserves, t)
        self.asa_position.update(traded_swap)


@dataclass
class GlobalPositionAndImpactState:
    asa_states: dict[int, PositionAndImpactState]
    mualgo_position: int

    def update_trade(self, asa_id: int, traded_swap: AlgoPoolSwap,
                     mualgo_reserves: int,
                     asa_reserves: int,
                     t: datetime.datetime):

        if traded_swap.asset_buy == 0:
            assert traded_swap.amount_buy_with_slippage >= 0
            self.mualgo_position += traded_swap.amount_buy_with_slippage
        elif traded_swap.asset_buy > 0:
            assert traded_swap.amount_sell_with_slippage >= 0
            self.mualgo_position -= traded_swap.amount_sell_with_slippage

        self.asa_states[asa_id].update_trade(traded_swap, mualgo_reserves, asa_reserves, t)

    def update_redeem(self, asa_id: int, redeemed: RedeemedAmounts):

        self.mualgo_position += redeemed.mualgo_amount
        self.asa_states[asa_id].asa_position.value += redeemed.asa_amount


@dataclass
class ASAPositionImpactLog:
    position: int
    impact_bps: float


@dataclass
class StateLog:
    time: datetime.datetime
    asa_states: dict[int, ASAPositionImpactLog]
    asa_prices: dict[int, float]
    mualgo_position: int

    def __init__(self, time: datetime.datetime, asa_prices: dict[int, PoolState], state: GlobalPositionAndImpactState):
        self.asa_prices = {aid: x.asset2_reserves / x.asset1_reserves for aid, x in asa_prices.items()}
        self.time = time
        self.asa_states = {aid: ASAPositionImpactLog(x.asa_position.value, x.impact.state) for aid, x in
                           state.asa_states.items()}
        self.mualgo_position = state.mualgo_position

from __future__ import annotations
import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from algo.trading.impact import PositionAndImpactState
from algo.trading.costs import FIXED_FEE_MUALGOS
from tinyman.v1.pools import SwapQuote, AssetAmount, Asset
import warnings
from abc import ABC, abstractmethod

# TODO Don't make me global
warnings.simplefilter("error", RuntimeWarning)

# The quadratic impact approximation is not accurate beyond this point
RESERVE_PERCENTAGE_CAP = 0.1

# Only allowed to go 10% above the previous cap after optimisation
RESERVE_PERCENTAGE_CAP_TOLERANCE = 0.1

# Max percentage of algo to sell in a single trade over our total algo holdings
MAX_PERCENTAGE_ALGO_SELL = 0.2

# Below this liquidity in the pool we don't buy (set the signal to zero if positive)
MIN_LIQUIDITY_TO_BUY_ALGOS = 50000


@dataclass
class OptimizedBuy:
    # Optimized amount which maximizes profit
    amount: int
    # The min amount at which trade would be profitable (it's not zero because of the fixed fee)
    min_profitable_amount: int

    def __post_init__(self):
        assert self.amount > 0
        assert self.min_profitable_amount > 0
        assert self.amount >= self.min_profitable_amount


class AssetType(Enum):
    ALGO = 1
    OTHER = 2


@dataclass
class OptimalSwap:
    asset_buy: AssetType
    optimised_buy: OptimizedBuy


def fetch_fixed_input_swap_quote(asset1: Asset, asset2: Asset,
                                 asset1_reserves: int, asset2_reserves: int,
                                 amount_in: AssetAmount,
                                 slippage: float
                                 ) -> SwapQuote:
    asset_in, asset_in_amount = amount_in.asset, amount_in.amount

    if asset_in == asset1:
        asset_out = asset2
        input_supply = asset1_reserves
        output_supply = asset2_reserves
    else:
        asset_out = asset1
        input_supply = asset2_reserves
        output_supply = asset1_reserves

    input_supply = int(input_supply)
    output_supply = int(output_supply)
    # k = input_supply * output_supply
    # ignoring fees, k must remain constant
    # (input_supply + asset_in) * (output_supply - amount_out) = k
    k = input_supply * output_supply
    assert (k >= 0)
    asset_in_amount_minus_fee = (asset_in_amount * 997) / 1000
    swap_fees = asset_in_amount - asset_in_amount_minus_fee
    asset_out_amount = output_supply - (k / (input_supply + asset_in_amount_minus_fee))

    amount_out = AssetAmount(asset_out, int(asset_out_amount))

    quote = SwapQuote(
        swap_type='fixed-input',
        amount_in=amount_in,
        amount_out=amount_out,
        swap_fees=AssetAmount(amount_in.asset, int(swap_fees)),
        slippage=slippage,
    )
    return quote


class BaseOptimizer(ABC):

    @property
    @abstractmethod
    def asset1(self) -> Asset:
        pass

    @property
    @abstractmethod
    def asset2(self) -> Asset:
        pass

    @staticmethod
    @abstractmethod
    def make(asset1: Asset, asset2: Asset, risk_coef: float) -> BaseOptimizer:
        pass

    @abstractmethod
    def optimal_amount_swap(self, signal_bps: float,
                            impact_bps: float,
                            current_asa_position: int,
                            current_asa_reserves: int,
                            current_mualgo_reserves: int,
                            time_dbg: datetime.datetime
                            ) -> Optional[OptimalSwap]:
        pass

    def fixed_sell_swap_quote(self, signal_bps: float,
                              impact_bps: float,
                              current_asa_position: int,
                              current_asa_reserves: int,
                              current_mualgo_reserves: int,
                              current_mualgo_position: int,
                              slippage: float,
                              time_dbg: datetime.datetime) -> Optional[SwapQuote]:

        assert -1 <= impact_bps <= 1

        optimal_swap = self.optimal_amount_swap(signal_bps, impact_bps, current_asa_position,
                                                current_asa_reserves,
                                                current_mualgo_reserves,
                                                time_dbg)

        if optimal_swap is not None:

            if optimal_swap.asset_buy == AssetType.ALGO:
                # What we sell
                asset_in = self.asset1
                input_supply = current_asa_reserves
                output_supply = current_mualgo_reserves
                sell_amount_available = current_asa_position

            else:
                # What we sell
                asset_in = self.asset2
                input_supply = current_mualgo_reserves
                output_supply = current_asa_reserves
                # FIXME What should we subtract here?
                sell_amount_available = min(
                    current_mualgo_position - 2 * FIXED_FEE_MUALGOS,
                    int(current_mualgo_position * MAX_PERCENTAGE_ALGO_SELL)
                )

            # Convert from int64 to int to avoid overflow errors
            input_supply = int(input_supply)
            output_supply = int(output_supply)

            k = input_supply * output_supply
            assert k >= 0

            def asset_in_from_asset_out(asset_out_amount):
                calculated_amount_in_without_fee = (k / (output_supply - asset_out_amount)) - input_supply
                return int(calculated_amount_in_without_fee * 1000 / 997)

            # What we buy
            optimal_asset_out_amount = optimal_swap.optimised_buy.amount
            optimal_asset_in_amount = asset_in_from_asset_out(optimal_asset_out_amount)

            minimal_asset_out_amount = optimal_swap.optimised_buy.min_profitable_amount
            minimal_asset_in_amount = asset_in_from_asset_out(minimal_asset_out_amount)

            if sell_amount_available <= minimal_asset_in_amount:
                return None

            asset_in_amount = int(min(optimal_asset_in_amount, sell_amount_available))

            assert asset_in_amount <= sell_amount_available

            if asset_in_amount > 0:
                quote = fetch_fixed_input_swap_quote(self.asset1, self.asset2,
                                                     current_asa_reserves, current_mualgo_reserves,
                                                     AssetAmount(asset_in, asset_in_amount), slippage)

                # Beyond this the quadratic approximation for the impact cost breaks down
                # (more than 1% error in the calculation)
                cap = RESERVE_PERCENTAGE_CAP * (1 + RESERVE_PERCENTAGE_CAP_TOLERANCE)
                assert quote.amount_out.amount < cap * output_supply, \
                    f"amount_out={quote.amount_out.amount} >= {cap * output_supply} = cap*output_supply"
                return quote

        return None

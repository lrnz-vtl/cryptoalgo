import datetime
import logging
from typing import Optional
from algo.trading.impact import PositionAndImpactState, ASAImpactState
from algo.trading.costs import FIXED_FEE_MUALGOS, FEE_BPS, EXPECTED_SLIPPAGE_BPS, reserves_to_avg_impact_cost_coef
from algo.optimizer.base import BaseOptimizer, OptimizedBuy, RESERVE_PERCENTAGE_CAP, OptimalSwap, AssetType
import numpy as np
import math


def optimal_amount_buy_asset(signal_bps: float,
                             impact_bps: float,
                             current_asset_reserves: int,
                             current_other_asset_reserves: int,
                             quadratic_risk_penalty: float,
                             linear_risk_penalty: float,
                             fixed_fee_other: float) -> Optional[OptimizedBuy]:
    # Do not let impact make us trade more
    if impact_bps < 0:
        impact_bps = 0

    asset_price = current_other_asset_reserves / current_asset_reserves

    f_bps = signal_bps - impact_bps - FEE_BPS - EXPECTED_SLIPPAGE_BPS - linear_risk_penalty

    if f_bps < 0:
        return None

    avg_impact_cost_coef = reserves_to_avg_impact_cost_coef(current_asset_reserves)

    # Expected profit in the reference currency
    # <Profit> = Amount * Price (f_bps - avg_impact_cost_coef * Amount) - quadratic_risk_penalty * Amount^2 - fixed_fee_other

    lin = asset_price * f_bps
    quad = quadratic_risk_penalty + asset_price * avg_impact_cost_coef
    const = fixed_fee_other

    # Debug variables for ASA sell branch
    # avg_impact_cost_coef_dbg = (asset_price * avg_impact_cost_coef) / asset_price
    # quadratic_risk_penalty_dbg = quadratic_risk_penalty / asset_price
    # quad_dbg = quad / asset_price

    # Debug variables for ASA buy branch
    # avg_impact_cost_coef_dbg2 = avg_impact_cost_coef / asset_price
    # quad_dbg2 = quad / (asset_price ** 2)

    # <Profit> = lin * Amount - quad * Amount^2 - const

    amount_profit_argmax = int(lin / (2 * quad))

    # const_dbg = const / asset_price
    # amount_dbg = amount_profit_argmax
    # lin_dbg2 = lin / asset_price
    # amount_dbg2 = amount_profit_argmax * asset_price

    max_profit_other = lin ** 2 / (4 * quad) - const
    if max_profit_other <= 0:
        return None

    min_profitable_amount = (lin - np.sqrt(lin * lin - 4 * const * quad)) / (2 * quad)
    assert min_profitable_amount > 0
    min_profitable_amount = math.ceil(min_profitable_amount)

    capped_amount = int(min(RESERVE_PERCENTAGE_CAP * current_asset_reserves, amount_profit_argmax))
    if capped_amount < min_profitable_amount:
        return None

    return OptimizedBuy(
        amount=capped_amount,
        min_profitable_amount=min_profitable_amount
    )


class OptimizerV1(BaseOptimizer):

    def __init__(self, asset1: int, risk_coef: float):
        self._asset1 = asset1
        self.risk_coef = risk_coef
        assert self.asset1 > 0
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def make(asset1: int, risk_coef: float) -> BaseOptimizer:
        return OptimizerV1(asset1, risk_coef)

    @property
    def asset1(self) -> int:
        return self._asset1

    def optimal_amount_swap(self, signal_bps: float,
                            pos_and_impact_state: PositionAndImpactState,
                            current_asa_reserves: int,
                            current_mualgo_reserves: int,
                            t: datetime.datetime
                            ) -> Optional[OptimalSwap]:

        impact_bps = pos_and_impact_state.impact.value(t)
        current_asa_position = pos_and_impact_state.asa_position

        asa_price_mualgo = current_mualgo_reserves / current_asa_reserves

        # TODO Ideally we should make this also a function of the liquidity, not just the dollar value:
        #  the more illiquid the asset is, the more risky is it to hold it
        quadratic_risk_penalty_asa_buy = self.risk_coef * asa_price_mualgo ** 2
        # TODO Check me
        # linear_risk_penalty_asa_buy = 2.0 * self.risk_coef * current_asa_position.value * asa_price_mualgo ** 2
        linear_risk_penalty_asa_buy = 2.0 * self.risk_coef * current_asa_position.value * asa_price_mualgo

        optimized_asa_buy = optimal_amount_buy_asset(signal_bps=signal_bps,
                                                     impact_bps=impact_bps,
                                                     current_asset_reserves=current_asa_reserves,
                                                     current_other_asset_reserves=current_mualgo_reserves,
                                                     quadratic_risk_penalty=quadratic_risk_penalty_asa_buy,
                                                     linear_risk_penalty=linear_risk_penalty_asa_buy,
                                                     # Upper bound pessimistic estimate for the fixed cost: if we buy now we have to exit later, so pay it twice
                                                     fixed_fee_other=FIXED_FEE_MUALGOS * 2
                                                     )

        # TODO Ideally we should make this also a function of the liquidity, not just the dollar value:
        #  the more illiquid the asset is, the more risky is it to hold it
        quadratic_risk_penalty_algo_buy = self.risk_coef / asa_price_mualgo
        linear_risk_penalty_algo_buy = - 2.0 * self.risk_coef * current_asa_position.value * asa_price_mualgo

        optimized_algo_buy = optimal_amount_buy_asset(signal_bps=1 / (1 + signal_bps) - 1.0,
                                                      impact_bps=1 / (1 + impact_bps) - 1.0,
                                                      current_asset_reserves=current_mualgo_reserves,
                                                      current_other_asset_reserves=current_asa_reserves,
                                                      quadratic_risk_penalty=quadratic_risk_penalty_algo_buy,
                                                      linear_risk_penalty=linear_risk_penalty_algo_buy,
                                                      fixed_fee_other=FIXED_FEE_MUALGOS / asa_price_mualgo
                                                      )

        if not (optimized_asa_buy is None or optimized_algo_buy is None):
            self.logger.error(f"Both buy and sell at time {t}")
        # FIXME
        assert optimized_asa_buy is None or optimized_algo_buy is None, f"{optimized_asa_buy}, {optimized_algo_buy}"

        if optimized_algo_buy is not None:
            return OptimalSwap(AssetType.ALGO, optimized_algo_buy)
        elif optimized_asa_buy is not None:
            return OptimalSwap(AssetType.OTHER, optimized_asa_buy)
        else:
            return None

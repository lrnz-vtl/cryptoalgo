import datetime
import logging
from typing import Optional
from tinyman.assets import Asset
from algo.trading.costs import FIXED_FEE_MUALGOS, FEE_BPS, EXPECTED_SLIPPAGE_BPS, reserves_to_avg_impact_cost_coef
from algo.optimizer.base import BaseOptimizer, OptimizedBuy, RESERVE_PERCENTAGE_CAP, OptimalSwap, AssetType, MIN_LIQUIDITY_TO_BUY_ALGOS
import numpy as np
import math


def optimal_amount_buy_asa(signal_bps: float,
                           impact_bps: float,
                           current_asa_reserves: int,
                           current_mualgo_reserves: int,
                           quadratic_risk_penalty: float,
                           linear_risk_penalty: float,
                           fixed_fee_mualgos: int,
                           is_buy: bool
                           ) -> Optional[OptimizedBuy]:
    assert linear_risk_penalty >= 0
    assert quadratic_risk_penalty > 0

    asa_price = current_mualgo_reserves / current_asa_reserves

    if is_buy:
        f_bps = signal_bps - np.clip(impact_bps, 0, None) - FEE_BPS - EXPECTED_SLIPPAGE_BPS - linear_risk_penalty
        avg_impact_cost_coef = reserves_to_avg_impact_cost_coef(current_asa_reserves)
        amount_cap = RESERVE_PERCENTAGE_CAP * current_asa_reserves
    else:
        f_bps = - signal_bps + np.clip(impact_bps, None, 0) - FEE_BPS - EXPECTED_SLIPPAGE_BPS + linear_risk_penalty
        avg_impact_cost_coef = asa_price * reserves_to_avg_impact_cost_coef(current_mualgo_reserves)
        amount_cap = RESERVE_PERCENTAGE_CAP * current_mualgo_reserves / asa_price

    if f_bps < 0:
        return None

    # Expected profit in mualgo
    # <Profit> = Amount * Price (f_bps - avg_impact_cost_coef * Amount) - quadratic_risk_penalty * Amount^2 - fixed_fee

    lin = asa_price * f_bps
    quad = quadratic_risk_penalty + asa_price * avg_impact_cost_coef
    const = fixed_fee_mualgos

    # <Profit> = lin * Amount - quad * Amount^2 - const

    amount_profit_argmax = int(lin / (2 * quad))

    max_profit_mualgo = lin ** 2 / (4 * quad) - const
    if max_profit_mualgo <= 0:
        return None

    min_profitable_amount = (lin - np.sqrt(lin * lin - 4 * const * quad)) / (2 * quad)
    assert min_profitable_amount > 0
    min_profitable_amount = math.ceil(min_profitable_amount)

    capped_amount = int(min(amount_cap, amount_profit_argmax))
    if capped_amount < min_profitable_amount:
        return None

    return OptimizedBuy(
        amount=capped_amount,
        min_profitable_amount=min_profitable_amount
    )


class OptimizerV2(BaseOptimizer):

    def __init__(self, asset1: Asset, asset2: Asset, risk_coef: float):
        self._asset1 = asset1
        self._asset2 = asset2
        self.risk_coef = risk_coef
        assert self.asset1.id > 0
        assert self.asset2.id == 0
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def make(asset1: Asset, asset2: Asset, risk_coef: float) -> BaseOptimizer:
        return OptimizerV2(asset1, asset2, risk_coef)

    @property
    def asset1(self) -> Asset:
        return self._asset1

    @property
    def asset2(self) -> Asset:
        return self._asset2

    def optimal_amount_swap(self, signal_bps: float,
                            impact_bps: float,
                            current_asa_position: int,
                            current_asa_reserves: int,
                            current_mualgo_reserves: int,
                            time_dbg: datetime.datetime
                            ) -> Optional[OptimalSwap]:

        below_liquidity = False
        if current_mualgo_reserves / 10**6 < MIN_LIQUIDITY_TO_BUY_ALGOS:
            self.logger.warning(f'{time_dbg} Pool for {self.asset1.id} is below threshold liquidity')
            below_liquidity = True
            signal_bps = np.clip(signal_bps, None, 0)

        asa_price_mualgo = current_mualgo_reserves / current_asa_reserves

        # TODO Ideally we should make this also a function of the liquidity, not just the dollar value:
        #  the more illiquid the asset is, the more risky is it to hold it
        quadratic_risk_penalty_asa_buy = self.risk_coef * asa_price_mualgo ** 2
        linear_risk_penalty_asa_buy = 2.0 * self.risk_coef * current_asa_position * asa_price_mualgo

        optimized_asa_buy = optimal_amount_buy_asa(signal_bps=signal_bps,
                                                   impact_bps=impact_bps,
                                                   current_asa_reserves=current_asa_reserves,
                                                   current_mualgo_reserves=current_mualgo_reserves,
                                                   quadratic_risk_penalty=quadratic_risk_penalty_asa_buy,
                                                   linear_risk_penalty=linear_risk_penalty_asa_buy,
                                                   # Upper bound pessimistic estimate for the fixed cost: if we buy now we have to exit later, so pay it twice
                                                   fixed_fee_mualgos=FIXED_FEE_MUALGOS * 2,
                                                   is_buy=True
                                                   )

        optimized_asa_sell = optimal_amount_buy_asa(signal_bps=signal_bps,
                                                    impact_bps=impact_bps,
                                                    current_asa_reserves=current_asa_reserves,
                                                    current_mualgo_reserves=current_mualgo_reserves,
                                                    quadratic_risk_penalty=quadratic_risk_penalty_asa_buy,
                                                    linear_risk_penalty=linear_risk_penalty_asa_buy,
                                                    fixed_fee_mualgos=FIXED_FEE_MUALGOS,
                                                    is_buy=False
                                                    )

        if below_liquidity:
            assert optimized_asa_buy is None

        assert optimized_asa_buy is None or optimized_asa_sell is None, f"{optimized_asa_buy}, {optimized_asa_sell}"

        if optimized_asa_sell is not None:
            optimized_algo_buy = OptimizedBuy(int(optimized_asa_sell.amount * asa_price_mualgo),
                                              int(optimized_asa_sell.min_profitable_amount * asa_price_mualgo))
            return OptimalSwap(AssetType.ALGO, optimized_algo_buy)
        elif optimized_asa_buy is not None:
            return OptimalSwap(AssetType.OTHER, optimized_asa_buy)
        else:
            return None

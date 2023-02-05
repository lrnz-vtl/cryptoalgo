from __future__ import annotations
import datetime
from dataclasses import dataclass
from algo.trading.costs import TradeCostsMualgo, REL_TOL
import math
from pydantic import BaseModel


@dataclass
class PriceInvariantTradeRecord:
    time: datetime.datetime
    asset_buy_id: int
    asset_sell_id: int
    asset_buy_amount: float
    asset_sell_amount: float

    def approx_eq_to(self, right: PriceInvariantTradeRecord):
        return self.time == right.time \
               and self.asset_buy_id == right.asset_buy_id \
               and self.asset_sell_id == right.asset_sell_id \
               and math.isclose(self.asset_sell_amount, right.asset_sell_amount, rel_tol=REL_TOL) \
               and math.isclose(self.asset_buy_amount, right.asset_buy_amount, rel_tol=REL_TOL)


class TradeRecord(BaseModel):
    time: datetime.datetime
    lag_after_update: datetime.timedelta
    asset_buy_id: int
    asset_sell_id: int
    asset_buy_amount: int
    asset_sell_amount: int
    asset_buy_amount_with_slippage: int
    asset_sell_amount_with_slippage: int
    txid: str

    def to_price_invariant(self, asa_price: float) -> PriceInvariantTradeRecord:
        if self.asset_buy_id > 0:
            asset_buy_amount = self.asset_buy_amount * asa_price
        else:
            asset_buy_amount = self.asset_buy_amount
        if self.asset_sell_id > 0:
            asset_sell_amount = self.asset_sell_amount * asa_price
        else:
            asset_sell_amount = self.asset_sell_amount
        return PriceInvariantTradeRecord(self.time, self.asset_buy_id, self.asset_sell_id, asset_buy_amount,
                                         asset_sell_amount)


class TradeInfo(BaseModel):
    trade: TradeRecord
    costs: TradeCostsMualgo
    asa_price: float
    signal_bps: float

    def assert_price_covariant(self, right: TradeInfo):
        assert self.trade.to_price_invariant(asa_price=self.asa_price).approx_eq_to(
            right.trade.to_price_invariant(asa_price=right.asa_price))
        assert self.costs.approx_eq_to(right.costs)
        assert math.isclose(self.signal_bps, right.signal_bps, rel_tol=REL_TOL)

    def price_covariant(self, right: TradeInfo) -> bool:
        try:
            self.assert_price_covariant(right)
        except AssertionError:
            return False
        return True


def read_tradeinfos(tradelog_fname: str) -> list[TradeInfo]:
    with open(tradelog_fname) as f:
        data = f.readlines()
    return [TradeInfo.parse_raw(x.strip()) for x in data]

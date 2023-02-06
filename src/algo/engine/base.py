from __future__ import annotations
import datetime
import logging
from algo.trading.trades import TradeInfo
from algo.trading.impact import GlobalPositionAndImpactState, StateLog
from algo.optimizer.base import BaseOptimizer
from algo.trading.signalprovider import PriceSignalProvider
from algo.blockchain.stream import PoolState
from typing import Callable
from algo.trading.swapper import TimedSwapQuote, MaybeTradedSwap, Swapper
from tinyman.v1.pools import SwapQuote
from abc import ABC, abstractmethod
import urllib.error
import algosdk


# Do not trade if the last successful call of scrape() was older than this
LAG_TRADE_LIMIT_SECONDS = 1 * 60


def lag_ms(dt: datetime.timedelta):
    return int(dt.total_seconds() * 1000)


def validate_swap(opt_swap_quote: SwapQuote, current_mualgo_reserves: int,
                  current_asa_reserves: int, pos_impact_state: GlobalPositionAndImpactState,
                  asset_id: int):
    if opt_swap_quote.amount_out.asset.id == 0:
        out_reserves = current_mualgo_reserves
        sell_position = pos_impact_state.asa_states[asset_id].asa_position.value
    elif opt_swap_quote.amount_out.asset.id > 0:
        assert opt_swap_quote.amount_out.asset.id == asset_id
        out_reserves = current_asa_reserves
        sell_position = pos_impact_state.mualgo_position
    else:
        raise ValueError

    assert opt_swap_quote.amount_out.amount <= out_reserves, \
        f"Buy amount:{opt_swap_quote.amount_out.amount} > pool reserve {out_reserves}: " \
        f"for asset {opt_swap_quote.amount_out.asset.id} in pool {asset_id}"

    assert opt_swap_quote.amount_in.amount <= sell_position


class BaseEngine(ABC):
    logger: logging.Logger
    asset_ids: list[int]
    prices: dict[int, PoolState]
    optimizers: dict[int, BaseOptimizer]
    signal_providers: dict[int, PriceSignalProvider]
    pos_impact_state: GlobalPositionAndImpactState
    swapper: dict[int, Swapper]
    slippage: float
    last_market_state_update: datetime.datetime

    @abstractmethod
    def current_time_prov(self) -> datetime.datetime:
        pass

    def trade_loop(self, log_trade: Callable[[TradeInfo], None], log_state: Callable[[StateLog], None]):
        time_start = self.current_time_prov()
        self.logger.info(f'Entering trade loop.')

        log_state(StateLog(time_start, self.prices, self.pos_impact_state))

        for asset_id in self.asset_ids:

            if asset_id not in self.prices:
                self.logger.warning(f'Price of {asset_id} has never been observed. Skipping trade logic')
                continue

            if asset_id not in self.prices:
                self.logger.warning(f'Price for asset {asset_id} not in data, skipping the trade logic.')
                continue

            opt = self.optimizers[asset_id]

            time_pre_opt = self.current_time_prov()

            signal_bps = self.signal_providers[asset_id].value
            impact_bps = self.pos_impact_state.asa_states[asset_id].impact.value(time_pre_opt)

            current_asa_reserves = self.prices[asset_id].asset1_reserves
            current_mualgo_reserves = self.prices[asset_id].asset2_reserves
            opt_swap_quote = opt.fixed_sell_swap_quote(signal_bps=signal_bps,
                                                       impact_bps=impact_bps,
                                                       current_asa_position=self.pos_impact_state.asa_states[
                                                           asset_id].asa_position.value,
                                                       current_mualgo_position=self.pos_impact_state.mualgo_position,
                                                       current_asa_reserves=current_asa_reserves,
                                                       current_mualgo_reserves=current_mualgo_reserves,
                                                       slippage=self.slippage,
                                                       time_dbg=time_pre_opt)

            time_opt = self.current_time_prov()
            time_since_update = time_opt - self.last_market_state_update
            if time_since_update.total_seconds() > LAG_TRADE_LIMIT_SECONDS:
                self.logger.error('Error the market data is stale, exiting trade loop')
                return

            if opt_swap_quote is not None:

                timed_swap_quote = TimedSwapQuote(last_market_update_time=self.last_market_state_update,
                                                  quote=opt_swap_quote,
                                                  mualgo_reserves_at_opt=current_mualgo_reserves,
                                                  asa_reserves_at_opt=current_asa_reserves)

                validate_swap(opt_swap_quote, current_mualgo_reserves, current_asa_reserves,
                              self.pos_impact_state, asset_id)

                try:
                    maybe_swap: MaybeTradedSwap = self.swapper[asset_id].attempt_transaction(timed_swap_quote)

                except urllib.error.URLError as e:
                    self.logger.critical(f'Transaction for asset {asset_id} failed with urllib.error.URLError: {str(e)}'
                    )
                    continue
                except algosdk.error.AlgodHTTPError as e:
                    self.logger.critical(
                        f'Transaction for asset {asset_id} failed with algosdk.error.AlgodHTTPError, code={e.code}: {str(e)}'
                    )
                    continue

                time_since_start = lag_ms(maybe_swap.time - time_start)
                time_since_opt = lag_ms(maybe_swap.time - timed_swap_quote.last_market_update_time)

                if maybe_swap.swap is not None:

                    traded_swap = maybe_swap.swap

                    trade_costs = traded_swap.make_costs(current_asa_reserves, current_mualgo_reserves, impact_bps)

                    trade_record = traded_swap.make_record(time=maybe_swap.time,
                                                           lag_after_update=maybe_swap.lag_after_update,
                                                           asa_id=asset_id)
                    self.logger.info(
                        f'Traded {time_since_start} ms after loop start, {time_since_opt} ms after last market sync'
                        f'\n{trade_record}')

                    trade_info = TradeInfo(trade=trade_record,
                                           costs=trade_costs,
                                           asa_price=current_mualgo_reserves / current_asa_reserves,
                                           signal_bps=signal_bps)
                    log_trade(trade_info)

                    self.pos_impact_state.update_trade(
                        asset_id,
                        traded_swap,
                        current_mualgo_reserves,
                        current_asa_reserves,
                        maybe_swap.time
                    )

                else:
                    self.logger.info(
                        f'Swapper rejected trade {time_since_start} ms after loop start, {time_since_opt} ms after optimisation'
                        f'for asset {asset_id}'
                    )

        time_end = self.current_time_prov()
        dt = lag_ms(time_end - time_start)
        self.logger.info(f'Exiting trade loop {dt} ms after entering.')


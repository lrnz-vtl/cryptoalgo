from tinyman.v1.client import TinymanClient
import logging
from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from tinyman.v1.pools import SwapQuote
from algo.trading.costs import TradeCostsOther, TradeCostsMualgo
from algo.trading.trades import TradeInfo, TradeRecord
import datetime
from tinyman.v1.optin import prepare_asset_optin_transactions
import algosdk
from enum import Enum


# Max amount of Algo left locked in a pool
MAX_VALUE_LOCKED_ALGOS = 1


def lag_ms(dt: datetime.timedelta):
    return int(dt.total_seconds() * 1000)


@dataclass
class AlgoPoolSwap:
    asset_buy: int
    amount_buy: int
    amount_sell: int
    amount_buy_with_slippage: int
    amount_sell_with_slippage: int
    txid: str
    redeemable_amount: int

    def make_costs(self, current_asa_reserves: int, current_mualgo_reserves: int,
                   impact_before_trade: float) -> TradeCostsMualgo:
        if self.asset_buy == 0:
            price_other = current_asa_reserves / current_mualgo_reserves
        else:
            price_other = current_mualgo_reserves / current_asa_reserves

        if self.asset_buy == 0:
            out_reserves = current_mualgo_reserves
        elif self.asset_buy > 0:
            out_reserves = current_asa_reserves
        else:
            raise ValueError

        return TradeCostsOther(buy_asset=self.asset_buy,
                               buy_amount=self.amount_buy,
                               buy_reserves=out_reserves,
                               buy_asset_price_other=price_other,
                               asa_impact=impact_before_trade,
                               redeemable_amount=self.redeemable_amount
                               ).to_mualgo_basis()

    def make_record(self, time: datetime.datetime, lag_after_update:datetime.timedelta, asa_id: int):
        if asa_id == self.asset_buy:
            asset_sell_id = 0
        else:
            assert self.asset_buy == 0
            asset_sell_id = asa_id

        return TradeRecord(
            time=time,
            asset_buy_id=self.asset_buy,
            asset_sell_id=asset_sell_id,
            asset_buy_amount=self.amount_buy,
            asset_sell_amount=self.amount_sell,
            asset_buy_amount_with_slippage=self.amount_buy_with_slippage,
            asset_sell_amount_with_slippage=self.amount_sell_with_slippage,
            txid=self.txid,
            lag_after_update=lag_after_update
        )


@dataclass
class MaybeTradedSwap:
    swap: Optional[AlgoPoolSwap]
    time: datetime.datetime
    lag_after_update: datetime.timedelta


@dataclass
class TimedSwapQuote:
    last_market_update_time: datetime.datetime
    quote: SwapQuote
    mualgo_reserves_at_opt: int
    asa_reserves_at_opt: int


@dataclass
class RedeemedAmounts:
    asa_amount: int
    mualgo_amount: int


class Swapper(ABC):
    @abstractmethod
    def attempt_transaction(self, quote: TimedSwapQuote) -> MaybeTradedSwap:
        pass

    @abstractmethod
    def fetch_excess_amounts(self, asa_price: float) -> RedeemedAmounts:
        pass


class ExecutionOption(Enum):
    NO_REFRESH = 1
    REFRESH = 2
    REFRESH_AND_RECOMPUTE = 3


class ProductionSwapper(Swapper):
    def __init__(self, aid: int, client: TinymanClient, address: str, key: str, execution_option: ExecutionOption,
                 fetch_redeemable_amounts: bool):
        self.pool = client.fetch_pool(aid, 0)
        self.address = address
        self.aid = aid
        self.key = key
        self.logger = logging.getLogger(__name__)
        self.client = client
        assert self.pool.exists
        # self._client_optin()
        self._asset_optin()
        self.execution_option = execution_option
        self.fetch_redeemable_amounts = fetch_redeemable_amounts

    def _client_optin(self):
        if not self.client.is_opted_in():
            self.logger.info('Account not opted into app, opting in now..')
            transaction_group = self.client.prepare_app_optin_transactions()
            transaction_group.sign_with_private_key(self.address, self.key)
            res = self.client.submit(transaction_group, wait=True)
            self.logger.info(f'Opted into app, {res}')

    def _asset_optin(self):
        acc_info = self.client.algod.account_info(self.address)
        for a in acc_info['assets']:
            if a['asset-id'] == self.aid:
                return

        self.logger.info(f'Account not opted into asset {self.aid}, opting in now..')

        txn_group = prepare_asset_optin_transactions(
            asset_id=self.aid,
            sender=self.address,
            suggested_params=self.client.algod.suggested_params()
        )
        txn_group.sign_with_private_key(self.address, self.key)
        res = self.client.submit(txn_group, wait=True)
        self.logger.info(f'Opted into asset, {res}')

    def attempt_transaction(self, quote: TimedSwapQuote) -> MaybeTradedSwap:

        if self.execution_option == ExecutionOption.REFRESH_AND_RECOMPUTE:
            quote_to_submit = self.pool.fetch_fixed_input_swap_quote(amount_in=quote.quote.amount_in,
                                                                     slippage=quote.quote.slippage)
            if (quote_to_submit.amount_out.amount, quote_to_submit.amount_out_with_slippage.amount) != \
                    (quote.quote.amount_out.amount, quote.quote.amount_out_with_slippage.amount):
                self.logger.warning('\nRecomputed and optimised output quotes differ:'
                                    f'{(quote_to_submit.amount_out.amount, quote_to_submit.amount_out_with_slippage.amount)} '
                                    f'!= {(quote.quote.amount_out.amount, quote.quote.amount_out_with_slippage.amount)}'
                                    f'\nopt reserves: ({quote.asa_reserves_at_opt}, {quote.mualgo_reserves_at_opt})'
                                    f'\nactual reserves: ({self.pool.asset1_reserves}, {self.pool.asset2_reserves})'
                                    f'\nquote={quote.quote}'
                                    f'\nquote_to_submit={quote_to_submit}'
                                    )
        elif self.execution_option == ExecutionOption.REFRESH:
            self.pool.refresh()
            quote_to_submit = quote.quote
        else:
            quote_to_submit = quote.quote

        if self.execution_option in [ExecutionOption.REFRESH, ExecutionOption.REFRESH_AND_RECOMPUTE]:
            if self.pool.asset1_reserves != quote.asa_reserves_at_opt or self.pool.asset2_reserves != quote.mualgo_reserves_at_opt:
                self.logger.warning(f'Refreshed (ASA, Algo) reserves are different from those at optimisation'
                                    f'\n ({self.pool.asset1_reserves, self.pool.asset2_reserves}) != '
                                    f'({quote.asa_reserves_at_opt, quote.mualgo_reserves_at_opt})')

        transaction_group = self.pool.prepare_swap_transactions_from_quote(quote_to_submit)
        transaction_group.sign_with_private_key(self.address, self.key)
        res = self.client.submit(transaction_group, wait=self.fetch_redeemable_amounts)

        time_trade = datetime.datetime.utcnow()

        if isinstance(transaction_group.transactions[3], algosdk.future.transaction.PaymentTxn):
            tx_out = transaction_group.transactions[3].amt
        elif isinstance(transaction_group.transactions[3], algosdk.future.transaction.AssetTransferTxn):
            tx_out = transaction_group.transactions[3].amount
        else:
            raise TypeError(f'transaction_group.transactions[3] should be PaymentTxn or AssetTransferTxn but is '
                            f'{type(transaction_group.transactions[3])}')
        if tx_out != quote_to_submit.amount_out_with_slippage:
            self.logger.warning(
                f'tx_out != quote_to_submit.amount_out_with_slippage, {tx_out} != {quote_to_submit.amount_out_with_slippage}')

        redeemable_amount = None
        if self.fetch_redeemable_amounts:
            assert isinstance(transaction_group.transactions[1], algosdk.future.transaction.ApplicationNoOpTxn)
            fetch_redeem_start = datetime.datetime.utcnow()
            t = self.client.algod.pending_transaction_info(transaction_group.transactions[1].get_txid())
            fetch_redeem_end = datetime.datetime.utcnow()
            self.logger.info(f'Performed fetch_redeemable_amounts in {lag_ms(fetch_redeem_end-fetch_redeem_start)} ms')
            try:
                redeemable_amount = t['local-state-delta'][1]['delta'][0]['value']['uint']
                self.logger.info(f'redeemable_amount = {redeemable_amount}')
            except KeyError as e:
                self.logger.critical(f"txid={res['txid']}"
                                     f"\n{t}"
                                     f"\n{e}")

        lag_after_update = time_trade - quote.last_market_update_time

        self.logger.info(f'Performed transaction in {lag_ms(lag_after_update)} ms after last market sync')

        return MaybeTradedSwap(
            AlgoPoolSwap(
                asset_buy=quote_to_submit.amount_out.asset.id,
                amount_buy=quote_to_submit.amount_out.amount,
                amount_sell=quote_to_submit.amount_in.amount,
                amount_buy_with_slippage=quote_to_submit.amount_out_with_slippage.amount,
                amount_sell_with_slippage=quote_to_submit.amount_in_with_slippage.amount,
                txid=res['txid'],
                redeemable_amount=redeemable_amount
            ),
            time=time_trade,
            lag_after_update=lag_after_update
        )

    def fetch_excess_amounts(self, asa_price: float) -> RedeemedAmounts:

        time = datetime.datetime.utcnow()
        self.logger.debug(f'Entering fetch_excess_amounts for asset {self.aid}')

        excess = self.pool.fetch_excess_amounts(self.address)

        ret = RedeemedAmounts(0, 0)

        for asset, asset_amount in excess.items():

            amount = asset_amount.amount

            if asset.id == 0:
                algo_value = amount / (10 ** 6)
            elif asset.id == self.aid:
                algo_value = amount * asa_price / (10 ** 6)
            else:
                raise ValueError

            if algo_value > MAX_VALUE_LOCKED_ALGOS:
                transaction_group = self.pool.prepare_redeem_transactions(asset_amount)
                transaction_group.sign_with_private_key(self.address, self.key)
                result = self.client.submit(transaction_group, wait=True)
                try:
                    if result['pool-error'] != '':
                        self.logger.error(f"Redemption may have failed with error {result['pool-error']}")
                except KeyError as e:
                    self.logger.critical(f'result={result}')
                    raise e
                self.logger.info(f'Redeemed {asset_amount} from pool {self.aid}, result={result}')

                if asset.id == 0:
                    ret.mualgo_amount = amount
                else:
                    ret.asa_amount = amount

        dt = lag_ms(datetime.datetime.utcnow() - time)
        self.logger.debug(f'Spent {dt} ms inside fetch_excess_amounts')

        return ret


class SimulationSwapper(Swapper):

    def fetch_excess_amounts(self, asa_price: float):
        pass

    def attempt_transaction(self, quote: TimedSwapQuote) -> MaybeTradedSwap:
        return MaybeTradedSwap(
            AlgoPoolSwap(
                asset_buy=quote.quote.amount_out.asset.id,
                amount_buy=quote.quote.amount_out.amount,
                amount_sell=quote.quote.amount_in.amount,
                amount_buy_with_slippage=quote.quote.amount_out_with_slippage.amount,
                amount_sell_with_slippage=quote.quote.amount_in_with_slippage.amount,
                txid="",
                redeemable_amount=-1
            ),
            time=quote.last_market_update_time,
            lag_after_update=datetime.timedelta(seconds=0)
        )

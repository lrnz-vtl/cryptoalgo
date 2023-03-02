import argparse
import logging
import unittest

from algo.binance.coins import SpotType, Universe, MarketType, FutureType
from algo.binance.data_types import KlineType, DataType, AggTradesType
from binance.experiment_runner import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('name')
    parser.add_argument('--n-coins', type=int, required=True)
    parser.add_argument('--spot', action='store_true')
    parser.add_argument('--agg', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    spot = args.spot
    if spot:
        market_type = SpotType()
    else:
        market_type = FutureType()

    if args.agg:
        data_type = AggTradesType()
    else:
        data_type = KlineType(freq='5m')

    run(args.name, args.n_coins, market_type, data_type=data_type, lookahead=False, test=args.test)



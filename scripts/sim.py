import argparse
import datetime
import os
import pandas as pd
from algo.binance.backtest import SIMS_BASEP, SimulatorCfg, Simulator
from algo.binance.sim_reports import run_reports

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('name')
    parser.add_argument('exp_name')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cash-flat', action='store_true')
    parser.add_argument('--ema', type=float)

    args = parser.parse_args()

    dst_path = SIMS_BASEP / args.name
    os.makedirs(dst_path, exist_ok=True)

    if args.test:
        end_time = datetime.datetime(year=2022, month=9, day=1)
        trade_pair_limit = 5
    else:
        end_time = datetime.datetime(year=2023, month=1, day=30)
        trade_pair_limit = None

    cfg = SimulatorCfg(exp_name=args.exp_name,
                       end_time=end_time,
                       trade_pair_limit=trade_pair_limit,
                       risk_coefs=[0.001, 0.01, 0.1],
                       cash_flat=args.cash_flat,
                       mkt_flat=True,
                       ema_hf_periods=args.ema
                       )
    sim = Simulator(cfg)

    with open(dst_path / 'cfg.json', 'w') as f:
        f.write(cfg.json(indent=4))

    res = sim.run()

    pd.to_pickle(res, dst_path / 'results.pkl')

    run_reports(dst_path)

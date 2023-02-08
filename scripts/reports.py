import argparse
from algo.binance.backtest import SIMS_BASEP
from algo.binance.sim_reports import run_reports

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('sim_name')
    args = parser.parse_args()

    sim_name = args.sim_name
    sim_path = SIMS_BASEP / sim_name
    assert sim_path.exists()
    run_reports(sim_path)

import datetime
from dataclasses import dataclass
import numpy as np


@dataclass
class PfolioLogGlobal:
    dollar_qty: np.array
    notional: np.array
    pnl: float
    mkt_exposure: float
    cash_exposure: float


@dataclass
class PairLog:
    dates: list[datetime.date]
    qty: list[float]
    dollar_qty: list[float]
    notional: list[float]
    pnl: list[float]


@dataclass
class PfolioLogGlobal:
    dates: list[datetime.date]
    dollar_qty: list[float]
    notional: list[float]
    pnl: list[float]
    mkt_exposure: list[float]
    cash_exposure: list[float]


class PfolioAcctData:
    def __init__(self, n_variations: int, n_pairs: int):
        shape = (n_variations, n_pairs)

        self.qtys = np.zeros(shape=shape)
        self.notionals = np.zeros(shape=shape)
        self.pnls = np.zeros(shape=shape)

        self.mkt_exposures = np.zeros(n_variations)
        self.cash_exposures = np.zeros(n_variations)


class PfolioLog:
    def __init__(self):
        self.dates: list[datetime.date] = []
        self.qtys: list[np.array] = []
        self.dollar_qtys: list[np.array] = []
        self.notionals: list[np.array] = []
        self.pnls: list[np.array] = []
        self.mkt_exposure: list[float] = []
        self.cash_exposure: list[float] = []

    def add_date(self, date: datetime.date, pfolio_acct: PfolioAcctData, k: int, prices: np.array):
        self.dates.append(date)
        self.qtys.append(pfolio_acct.qtys[k])
        self.dollar_qtys.append(pfolio_acct.qtys[k] * prices)
        self.notionals.append(pfolio_acct.notionals[k])
        self.pnls.append(pfolio_acct.pnls[k])
        self.mkt_exposure.append(pfolio_acct.mkt_exposures[k])
        self.cash_exposure.append(pfolio_acct.cash_exposures[k])

    def select_pair(self, pair_idx: int):
        return PairLog(
            self.dates,
            [x[pair_idx] for x in self.qtys],
            [x[pair_idx] for x in self.dollar_qtys],
            [x[pair_idx] for x in self.notionals],
            [x[pair_idx] for x in self.pnls],
        )

    def compute_global(self) -> PfolioLogGlobal:
        return PfolioLogGlobal(
            self.dates,
            [x.sum() for x in self.dollar_qtys],
            [x.sum() for x in self.notionals],
            [x.sum() for x in self.pnls],
            self.mkt_exposure,
            self.cash_exposure
        )

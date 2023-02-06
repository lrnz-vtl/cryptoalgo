from algo.stream import marketstream
from algo.sqlite.base import BaseSqliteLogger
import arrow
import pandas as pd
from contextlib import closing
from algo.stream.marketstream import MARKETLOG_BASEFOLDER
import os


def convert_arrowdatetime(s):
    return arrow.get(s)


def adapt_arrowdatetime(adt):
    return adt.isoformat()


class MarketSqliteLogger(BaseSqliteLogger):

    def __init__(self, run_name: str):
        self.base_folder = os.path.join(MARKETLOG_BASEFOLDER, run_name)
        self.db_fname = os.path.join(self.base_folder, 'data.db')
        super().__init__('marketData', self.db_fname)

    # CHECK that reserves are int
    def _table_format(self) -> list[tuple[str, str]]:
        return [
            ("asset1", "int"),
            ("asset2", "int"),
            ("asset1_reserves", "int"),
            ("asset2_reserves", "int"),
            ("price", "real"),
            ("now", "timestamp"),
            ("utcnow", "timestamp")
        ]

    def _row_to_tuple(self, row: marketstream.Row) -> tuple:
        return (row.asset1,
                row.asset2,
                row.asset1_reserves,
                row.asset2_reserves,
                row.price,
                row.timestamp.now,
                row.timestamp.utcnow
                )

    def to_dataframe(self) -> pd.DataFrame:
        # This should only be used for small data
        with closing(self.con.cursor()) as c:
            c.execute(f"select * from {self.tablename}")
            data = c.fetchall()

        df = pd.DataFrame(data,
                          columns=['idx', 'asset1', "asset2", "asset1_reserves",
                                   "asset2_reserves", "price", "now", 'utcnow'])
        df["now"] = pd.to_datetime(df["now"])
        df["utcnow"] = pd.to_datetime(df["utcnow"])
        return df

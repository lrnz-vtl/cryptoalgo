import sqlite3
from contextlib import closing
from abc import ABC, abstractmethod
from typing import Any


class BaseSqliteLogger(ABC):

    def __init__(self,
                 tablename: str,
                 dbfile: str
                 ):
        self.tablename = tablename
        self.dbfile = dbfile

    @abstractmethod
    def _table_format(self) -> list[tuple[str, str]]:
        pass

    @abstractmethod
    def _row_to_tuple(self, row: Any) -> tuple:
        pass

    def create_table(self, ignore_existing: bool = False):

        with closing(self.con.cursor()) as c:
            c.execute(f''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{self.tablename}' ''')

            # if the count is 1, then table exists
            if c.fetchone()[0] == 1 and ignore_existing:
                pass
            else:

                c.execute(
                    f"""
                    create table {self.tablename} 
                    (id INTEGER PRIMARY KEY, {', '.join([' '.join(x) for x in self._table_format()])})
                    """)
        self.con.commit()

    def __enter__(self):
        self.con = sqlite3.connect(self.dbfile)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.con.close()

    def log(self, row: Any):
        n = len(self._table_format())
        data = self._row_to_tuple(row)
        with closing(self.con.cursor()) as c:
            c.execute(f"""insert into {self.tablename} values 
                        (NULL, {', '.join(['?'] * n)})""", data)

        self.con.commit()

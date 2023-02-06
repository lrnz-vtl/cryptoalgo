from abc import ABC, abstractmethod
from typing import Optional
import datetime
import aiohttp


class NotExistentPoolError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class DataScraper(ABC):

    @abstractmethod
    async def scrape(self, session: aiohttp.ClientSession,
                     timestamp_min: int,
                     before_time: Optional[datetime.datetime],
                     num_queries: Optional[int] = None):
        pass

from datetime import datetime, timezone
from dataclasses import dataclass
import time as _time

@dataclass
class Timestamp:
    now: datetime
    utcnow: datetime

    @staticmethod
    def get():
        t = _time.time()
        now = datetime.fromtimestamp(t)
        utcnow = datetime.fromtimestamp(t, timezone.utc)
        return Timestamp(now, utcnow)
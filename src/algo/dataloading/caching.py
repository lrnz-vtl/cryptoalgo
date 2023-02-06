import datetime
import glob
import re
import pandas as pd
from pathlib import Path

from algo.universe.universe import SimpleUniverse
from definitions import ROOT_DIR
from typing import Optional, Callable
from algo.blockchain.cache import DateValidator


def load_algo_pools(cache_name: str, data_type: str, filter_pair: Optional[Callable]):
    assert data_type in ['prices', 'volumes']

    pattern = f'{ROOT_DIR}/caches/{data_type}/{cache_name}/*'

    def gen_data():
        for base_dir in glob.glob(pattern):
            name = Path(base_dir).name

            if re.match('[0-9]+_[0-9]+$', name):
                a0, a1 = tuple(int(x) for x in name.split('_'))

                if filter_pair is None or filter_pair(a0, a1):
                    dv = DateValidator(f'{ROOT_DIR}/caches/{data_type}', cache_name, None, (a0, a1))
                    assert not dv.has_gaps()
                    df = pd.read_parquet(base_dir)
                    if not df.empty:
                        df = df.sort_values(by='time')
                        df['asset1'] = a0
                        df['asset2'] = a1
                    yield df

    return pd.concat(gen_data())


def validate_missing_days(df):
    def validate(subdf, name):
        days = pd.to_datetime(subdf['time'], unit='s', utc=True).dt.date.unique()
        day = min(days)
        while day < max(days):
            assert day in days, f"day {day} missing for id {name}"
            day += datetime.timedelta(days=1)

    df.groupby('asset1').apply(lambda x: validate(x, x.name))


def join_caches_with_priority(caches: list[str], data_type: str, filter_pair: Optional[Callable]):
    data = []

    if len(caches) > 1:
        raise NotImplementedError("Need to guarantee the gaps are dealt with properly")

    for cache_priority, cachename in enumerate(caches):
        subdf = load_algo_pools(cachename, data_type, filter_pair=filter_pair)
        subdf['cache_priority'] = cache_priority
        data.append(subdf)
    df = pd.concat(data)
    time_maxes = df.groupby(['asset1', 'cache_priority'])['time'].max()

    filt_idx = pd.Series(True, index=df.index)

    for x, time_max in time_maxes.items():
        aid, priority = x[0], x[1]
        filt_idx &= (df['cache_priority'] <= priority) | (df['asset1'] != aid) | (df['time'] > time_max)

    ret = df[filt_idx].drop(columns='cache_priority').sort_values(by='time')

    return ret


def make_filter_from_universe(universe: SimpleUniverse):
    def filter_pair(a1, a2):
        return tuple(sorted([a1, a2])) in [tuple(sorted([x.asset1_id, x.asset2_id])) for x in universe.pools]

    return filter_pair

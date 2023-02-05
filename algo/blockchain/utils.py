import datetime
import pandas as pd


def datetime_to_int(t: datetime.datetime):
    return int(t.timestamp())


def int_to_datetime(t: int):
    return datetime.datetime.fromtimestamp(t)


def int_to_rfc3339(t: int):
    return datetime.datetime.fromtimestamp(t).isoformat() + 'Z'


def int_to_tzaware_utc_datetime(t: int):
    return datetime.datetime.utcfromtimestamp(t).replace(tzinfo=datetime.timezone.utc)


def generator_to_df(gen, time_columns=('time',)):
    df = pd.DataFrame(gen)
    if df.empty:
        print("DataFrame is empty")
    else:
        for col in time_columns:
            df[col] = pd.to_datetime(df[col], unit='s', utc=True)
    return df


def algo_nousd_filter(a1, a2):
    exclude = [31566704,  # USDC
               312769,  # USDt
               567485181,  # LoudDefi
               ]

    # Select only pairs with Algo
    if a1 in exclude or a2 in exclude \
            or (a1 != 0 and a2 != 0):
        return False
    return True



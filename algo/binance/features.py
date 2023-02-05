from dataclasses import dataclass
from typing import Optional
import numpy as np
from algo.cpp.cseries import shift_forward, compute_ema, compute_expsum
import pandas as pd

ms_in_hour = (10 ** 3) * 60 * 60


@dataclass
class VolumeOptions:
    include_logretvol: bool
    include_imbalance: bool


@dataclass
class FeatureOptions:
    decay_hours: list[int]
    volume_options: Optional[VolumeOptions]
    include_current: bool = True


def features_from_data(df: pd.DataFrame, ema_options: FeatureOptions) -> pd.DataFrame:
    fts = []

    assert len(ema_options.decay_hours) > 0

    price_ts = ((df['Close'] + df['Open']) / 2.0).rename('price')

    def make_ema(dh):
        dms = ms_in_hour * dh
        return compute_ema(price_ts.index, price_ts.values, dms)

    if ema_options.include_current:
        logema0 = np.log(price_ts)
        decays_hours = ema_options.decay_hours
    else:
        logema0 = make_ema(ema_options.decay_hours[0])
        decays_hours = ema_options.decay_hours[1:]

    for dh in decays_hours:
        ema = make_ema(dh)
        ft = np.log(ema) - logema0
        fts.append(pd.Series(ft, index=price_ts.index, name=f'ema_{dh}'))

    if ema_options.volume_options is not None:

        volume_ts = df['Volume']
        volume_norm = volume_ts.median()

        if ema_options.volume_options.include_logretvol:
            logret_ts = (np.log(df['Close']) - np.log(df['Open'])).rename('logret')

            for dh in decays_hours:
                # NOTE Assumes five minutes separated rows
                alpha = 1.0 - np.exp(-1 / (12 * dh))
                logretvol_expsum = compute_expsum((logret_ts * volume_ts).values, alpha)
                # FIXME Do this with no lookahead
                ft = logretvol_expsum / volume_norm
                fts.append(pd.Series(ft, index=price_ts.index, name=f'logretvol_{dh}'))

        if ema_options.volume_options.include_imbalance:
            buy_volume = df['Taker buy base asset volume'] / volume_norm
            sell_volume = df['Volume'] / volume_norm - buy_volume

            for dh in decays_hours:
                # NOTE Assumes five minutes separated rows
                alpha = 1.0 - np.exp(-1 / (12 * dh))
                buy_volume_expsum = compute_expsum(buy_volume.values, alpha)
                sell_volume_expsum = compute_expsum(sell_volume.values, alpha)
                fts.append(pd.Series(buy_volume_expsum, index=price_ts.index, name=f'buyvol_{dh}'))
                fts.append(pd.Series(sell_volume_expsum, index=price_ts.index, name=f'sellvol_{dh}'))

    return pd.concat(fts, axis=1)

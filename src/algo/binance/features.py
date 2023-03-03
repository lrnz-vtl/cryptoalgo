from dataclasses import dataclass
from typing import Optional
import numpy as np
from algo.cpp.cseries import compute_ema, compute_expsum
import pandas as pd
from pydantic import BaseModel

from algo.binance.coins import DataType

ms_in_hour = (10 ** 3) * 60 * 60


class NanFeatureException(ValueError):
    pass


class VolumeZeroError(ValueError):
    pass


@dataclass
class VolumeOptions:
    include_logretvol: bool
    include_imbalance: bool


class FeatureOptions(BaseModel):
    decay_hours: list[float]
    volume_options: Optional[VolumeOptions]
    include_current: bool = True


def features_from_data(df: pd.DataFrame, ema_options: FeatureOptions) -> pd.DataFrame:
    fts = []

    assert len(ema_options.decay_hours) > 0

    price_ts = df['price']
    assert ~price_ts.isna().any()

    def make_ema(dh):
        dms = ms_in_hour * dh
        return compute_ema(price_ts.index.values.copy(), price_ts.values.copy(), dms)

    if ema_options.include_current:
        logema0 = np.log(price_ts)
        decays_hours = ema_options.decay_hours
    else:
        logema0 = np.log(make_ema(ema_options.decay_hours[0]))
        decays_hours = ema_options.decay_hours[1:]

    for dh in decays_hours:
        ema = make_ema(dh)
        ft = np.log(ema) - logema0
        fts.append(pd.Series(ft, index=price_ts.index, name=f'ema_{dh}'))

    if ema_options.volume_options is not None:

        volume_ts = df['Volume']

        assert not volume_ts.isna().any()

        volume_norm = volume_ts.median()
        # FIXME It seems sometimes volumes starts being zero on USDT and transfers to BUSD
        # assert volume_norm > 0
        if not (volume_norm > 0):
            raise VolumeZeroError()

        if ema_options.volume_options.include_logretvol:
            logret_ts = df['logret']
            assert not logret_ts.isna().any()

            assert (logret_ts.index == volume_ts.index).all()

            for dh in decays_hours:
                # NOTE Assumes five minutes separated rows
                alpha = 1.0 - np.exp(-1 / (12 * dh))
                logretvol_expsum = compute_expsum((logret_ts * volume_ts).values.copy(), alpha)
                # FIXME Do this with no lookahead
                ft = logretvol_expsum / volume_norm
                fts.append(pd.Series(ft, index=price_ts.index, name=f'logretvol_{dh}'))

        if ema_options.volume_options.include_imbalance:
            buy_volume = df['BuyVolume'] / volume_norm
            sell_volume = df['Volume'] / volume_norm - buy_volume

            if buy_volume.isna().any():
                raise NanFeatureException('buyvolume')
            assert not sell_volume.isna().any()

            for dh in decays_hours:
                # NOTE Assumes five minutes separated rows
                alpha = 1.0 - np.exp(-1 / (12 * dh))
                buy_volume_expsum = compute_expsum(buy_volume.values.copy(), alpha)
                sell_volume_expsum = compute_expsum(sell_volume.values.copy(), alpha)
                fts.append(pd.Series(buy_volume_expsum, index=price_ts.index, name=f'buyvol_{dh}'))
                fts.append(pd.Series(sell_volume_expsum, index=price_ts.index, name=f'sellvol_{dh}'))

    return pd.concat(fts, axis=1)

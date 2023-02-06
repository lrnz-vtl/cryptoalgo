import datetime
from abc import ABC, abstractmethod
from ts_tools_algo.features import ema_provider
from dataclasses import dataclass
import numpy as np


class PriceSignalProvider(ABC):

    @abstractmethod
    def update(self, t: datetime.datetime, price: float) -> None:
        pass

    @property
    @abstractmethod
    def value(self) -> float:
        pass


class DummySignalProvider(PriceSignalProvider):

    def __init__(self, const: float = 0.0, alternate: bool = False):
        self.const = const
        self.alternate = alternate

    def update(self, t: datetime.datetime, price: float) -> None:
        if self.alternate:
            self.const = -self.const

    @property
    def value(self) -> float:
        return self.const


class PriceEmaFeature:

    def __init__(self, timescale_seconds):
        self.ema_provider = ema_provider(timescale_seconds)
        next(self.ema_provider)
        self.value = None

    def update(self, t: datetime.datetime, price: float):
        ema = self.ema_provider.send((t, price))
        self.value = (price - ema) / price


@dataclass
class EmaSignalParam:
    timescale_seconds: int
    beta: float


class EmaSignalProvider(PriceSignalProvider):

    def __init__(self, params: list[EmaSignalParam], cap_bps: float = 1.0):
        self.features = [PriceEmaFeature(param.timescale_seconds) for param in params]
        self.betas = [param.beta for param in params]
        self.cap_bps = cap_bps

    def update(self, t: datetime.datetime, price: float) -> None:
        for feature in self.features:
            feature.update(t, price)

    @property
    def value(self) -> float:
        value = sum(beta * x.value for beta, x in zip(self.betas, self.features))
        return np.clip(value, -self.cap_bps, self.cap_bps)


class RandomSignalProvider(PriceSignalProvider):

    def __init__(self, std_bps: float):
        self.std_bps = std_bps
        self.state = None

    def update(self, t: datetime.datetime, price: float) -> None:
        self.state = np.random.normal(scale=self.std_bps)

    @property
    def value(self) -> float:
        return self.state


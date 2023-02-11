import logging
from dataclasses import dataclass
import numpy as np
import scipy
from scipy.sparse.linalg import ArpackError
import cvxpy as cp


@dataclass
class OptimiserCfg:
    comm: float
    risk_coef: float
    poslim: float
    cash_flat: bool
    mkt_flat: bool
    max_trade_size_usd: float


class Optimiser:

    def __init__(self, betas: np.array, cfg: OptimiserCfg):
        self.cfg = cfg

        self.betas = betas
        self.n = len(betas)
        n = self.n
        self.comms = self.cfg.comm * np.ones(n)
        self.risk = self.cfg.risk_coef * scipy.sparse.eye(n)
        self.poslims = self.cfg.poslim * np.ones(n)
        self.ones = np.ones(n)
        self.zeros = np.zeros(n)

        self.logger = logging.getLogger(__name__)

    def debug(self, position, signal):
        self.logger.warning(f'{position=}')
        self.logger.warning(f'{signal=}')
        self.logger.warning(f'{self.betas=}')
        self.logger.warning(f'{self.cfg=}')

    def optimise(self, position: np.array, signal: np.array, skip_hard=False) -> np.array:
        assert position.shape[0] == self.n
        assert signal.shape[0] == self.n

        upper_poslims = self.poslims.copy()
        lower_poslims = -self.poslims.copy()
        above_bounds = (position > self.cfg.poslim)
        below_bounds = (position < -self.cfg.poslim)
        upper_poslims[above_bounds] = position[above_bounds]
        lower_poslims[below_bounds] = position[below_bounds]

        xb = cp.Variable(self.n)
        xs = cp.Variable(self.n)

        base_cons = [
            xb >= self.zeros,
            xs >= self.zeros,
            xb <= self.ones * self.cfg.max_trade_size_usd,
            xs <= self.ones * self.cfg.max_trade_size_usd,
            (position + xb - xs) <= upper_poslims,
            (position + xb - xs) >= lower_poslims
        ]

        base_quad_form = cp.quad_form(position + xb - xs, self.risk)

        if not skip_hard:

            cons = base_cons.copy()
            if self.cfg.cash_flat:
                cons.append(self.ones @ (position + xb - xs) == 0)
            if self.cfg.mkt_flat:
                cons.append(self.betas @ (position + xb - xs) == 0)

            prob = cp.Problem(
                cp.Minimize((1 / 2) * base_quad_form - signal.T @ (xb - xs) + self.comms @ xb + self.comms @ xs),
                cons
            )

            # NOTE Can port this to rust
            # prob.solve(solver='OSQP', verbose=True)
            prob.solve(solver='OSQP')

            if xb.value is None or xs.value is None:
                self.logger.warning(f'x is None')
                self.debug(position, signal)
                self.logger.warning(f'Trying with soft penalties on market and cash factors')
            else:
                return xb.value - xs.value

        beta_quad_form = 10 * self.cfg.risk_coef * (self.betas @ self.betas.T)

        prob = cp.Problem(
            cp.Minimize((1 / 2) * (base_quad_form + beta_quad_form) - signal.T @ (xb - xs)
                        + self.comms @ xb + self.comms @ xs),
            base_cons
        )
        prob.solve(solver='OSQP')

        if xb.value is None or xs.value is None:
            self.logger.error(f'Failed after soft constraints.')
            self.debug(position, signal)
            raise RuntimeError(f'Failed after soft constraints.')

        above_bounds = ((position + xb.value - xs.value) > upper_poslims)
        below_bounds = ((position + xb.value - xs.value) < lower_poslims)
        if above_bounds.any() or below_bounds.any():
            self.logger.warning('Outside position bounds')
            self.debug(position, signal)
            self.logger.warning(f'{position[above_bounds]=}, {xb.value[above_bounds]=},  {xs.value[above_bounds]=}')
            self.logger.warning(f'{position[below_bounds]=}, {xb.value[below_bounds]=},  {xs.value[below_bounds]=}')
            # raise RuntimeError(f'Went outside bounds.')

        return xb.value - xs.value

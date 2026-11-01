"""Snare-wire simulator, penalty-collision drive into a resonant HF bank."""

from __future__ import annotations

import numpy as np
from numba import njit

from .modal import ModalBank, ModalParams, _normalize_gains_for_white_noise

_LN1000 = 6.907755278982137


def default_snare_wire_modes(sample_rate: float, seed: int = 2) -> ModalParams:
    """Resonant snare-wire bank."""
    rng = np.random.default_rng(seed)
    # Shimmer core: dense, inharmonic, 3–5 kHz
    core = rng.uniform(2900.0, 5200.0, size=18)
    # High tail: sparse, 5–8 kHz, quieter and shorter
    tail = rng.uniform(5200.0, 8000.0, size=8)
    freqs = np.sort(np.concatenate([core, tail]))
    # Moderate decays: core rings ~40–90 ms, tail shorter ~15–40 ms
    decays = np.where(
        freqs < 5200.0,
        rng.uniform(0.040, 0.090, size=freqs.size),
        rng.uniform(0.015, 0.040, size=freqs.size),
    )
    # Core louder than the high tail so brightness concentrates ~3–5 kHz
    gains = np.where(freqs < 5200.0,
                     rng.uniform(0.7, 1.0, size=freqs.size),
                     rng.uniform(0.25, 0.5, size=freqs.size))
    nyq_limit = 0.48 * sample_rate
    keep = freqs < nyq_limit
    freqs = freqs[keep]; decays = decays[keep]; gains = gains[keep]
    gains = _normalize_gains_for_white_noise(freqs, decays, gains, sample_rate, target_rms=0.5)
    return ModalParams(
        freqs=freqs.astype(np.float64),
        decays=decays.astype(np.float64),
        gains=gains.astype(np.float64),
    )


@njit(cache=True, fastmath=True)
def _collision_exc(
    head: np.ndarray, gap_mult: float, env_coeff: float,
    K: float, alpha: float, beta: float,
    state: np.ndarray,
) -> np.ndarray:
    """One-sided power-law collision drive with a *level-adaptive* contact gap."""
    T = head.shape[0]
    out = np.empty(T, dtype=np.float64)
    prev_eta = state[0]
    env = state[1]
    for t in range(T):
        a = abs(head[t])
        env += env_coeff * (a - env)
        eta = a - gap_mult * env
        if eta > 0.0:
            etadot = (eta - prev_eta)  # discrete penetration velocity (per sample)
            # Power-law contact force
            f = K * eta ** alpha
            # Hunt–Crossley velocity-dependent loss
            hc = 1.0 + beta * etadot
            if hc < 0.0:
                hc = 0.0
            out[t] = f * hc
            prev_eta = eta
        else:
            out[t] = 0.0
            prev_eta = 0.0
    state[0] = prev_eta
    state[1] = env
    return out


class SnareWires:
    """Feed-forward snare-wire voice: penalty-collision drive to resonant bank."""

    def __init__(
        self, sample_rate: float, *,
        gap_mult: float = 1.4, env_ms: float = 80.0,
        stiffness: float = 9.0, alpha: float = 1.7, beta: float = 0.0,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.gap_mult = float(gap_mult)
        self.env_coeff = float(1.0 - np.exp(-1.0 / (env_ms * 1e-3 * self.sample_rate)))
        self.stiffness = float(stiffness)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._state = np.zeros(2, dtype=np.float64)
        self._bank = ModalBank(default_snare_wire_modes(sample_rate), sample_rate)

    def reset(self) -> None:
        self._state[:] = 0.0
        self._bank.reset()

    def process(self, head: np.ndarray) -> np.ndarray:
        drive = _collision_exc(
            np.ascontiguousarray(head, dtype=np.float64),
            self.gap_mult, self.env_coeff,
            self.stiffness, self.alpha, self.beta, self._state,
        )
        return self._bank.process(drive)

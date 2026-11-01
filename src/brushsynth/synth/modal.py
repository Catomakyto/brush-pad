"""Modal resonator bank."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

# ln(1000), for a 60 dB decay in tau seconds
_LN1000 = 6.907755278982137


@dataclass
class ModalParams:
    """Per-mode parameters for a modal resonator bank."""

    freqs: np.ndarray  # (N,) Hz
    decays: np.ndarray  # (N,) seconds (time to decay 60 dB)
    gains: np.ndarray  # (N,) linear mode amplitude

    def __post_init__(self) -> None:
        f = np.asarray(self.freqs, dtype=np.float64)
        t = np.asarray(self.decays, dtype=np.float64)
        g = np.asarray(self.gains, dtype=np.float64)
        if not (f.shape == t.shape == g.shape):
            raise ValueError(
                f"freqs/decays/gains must share shape, got {f.shape} / {t.shape} / {g.shape}"
            )
        if f.ndim != 1:
            raise ValueError("modal parameters must be 1-D")
        if np.any(t <= 0.0):
            raise ValueError("all decay times must be > 0")
        self.freqs = f
        self.decays = t
        self.gains = g

    @property
    def n_modes(self) -> int:
        return int(self.freqs.shape[0])


def apply_spectral_tilt(
    freqs: np.ndarray, gains: np.ndarray, tilt_db_per_oct: float, pivot_hz: float = 500.0,
) -> np.ndarray:
    """Shape modal gains by applying a spectral tilt."""
    if tilt_db_per_oct == 0.0:
        return gains
    oct_offset = np.log2(np.maximum(freqs, 1.0) / pivot_hz)
    weight = 10.0 ** (tilt_db_per_oct * oct_offset / 20.0)
    shaped = gains * weight
    # Preserve overall energy so this knob is orthogonal to loudness
    orig_rms = np.sqrt(np.mean(gains ** 2))
    new_rms = np.sqrt(np.mean(shaped ** 2))
    if new_rms > 0.0:
        shaped = shaped * (orig_rms / new_rms)
    return shaped


def _normalize_gains_for_white_noise(
    freqs: np.ndarray, decays: np.ndarray, gains: np.ndarray, sample_rate: float,
    target_rms: float = 0.5,
) -> np.ndarray:
    """Scale gains so white noise gives the target RMS."""
    R = np.exp(-_LN1000 / (decays * sample_rate))
    denom = np.maximum(1.0 - R * R, 1e-8)
    variance_sum = float(np.sum(gains * gains / denom))
    if variance_sum <= 0.0:
        return gains
    scale = target_rms / np.sqrt(variance_sum)
    return gains * scale


def default_snare_modes(sample_rate: float, seed: int = 0) -> ModalParams:
    """A plausible default modal bank for a 14-inch coated snare drum."""
    rng = np.random.default_rng(seed)

    # Bessel zeros of J_m for (m, n) circular membrane modes, sorted
    bessel = np.array(
        [
            2.405, 3.832, 5.136, 5.520, 6.380, 7.016,
            7.588, 8.417, 8.654, 9.761, 10.173, 10.520,
            11.065, 11.620, 12.339, 13.015, 13.354, 13.589,
            14.372, 14.796, 15.700, 16.038,
        ],
        dtype=np.float64,
    )
    f0 = 180.0
    mem_freqs = bessel * (f0 / bessel[0])
    mem_freqs = mem_freqs * (1.0 + 0.008 * rng.standard_normal(mem_freqs.shape))

    # Shell resonances: broad dense population 300 Hz – 3 kHz
    shell_freqs = np.sort(rng.uniform(300.0, 3000.0, size=40))

    # High-frequency body and head resonances: 1.5 – 6 kHz
    high_freqs = np.sort(rng.uniform(1500.0, 6000.0, size=30))

    freqs = np.concatenate([mem_freqs, shell_freqs, high_freqs])

    # Decay times roughly scale with sqrt(f0/f): low modes ring longer
    decays = 0.8 * np.sqrt(f0 / freqs)
    decays = np.clip(decays, 0.03, 1.2)

    # Amplitude envelope falls off with frequency
    gains = (f0 / freqs) ** 0.4
    gains *= rng.uniform(0.5, 1.5, size=freqs.shape)
    # Membrane modes project louder than shell/body
    gains[: mem_freqs.size] *= 1.4

    # Keep modes strictly below Nyquist with a little margin
    nyq_limit = 0.48 * sample_rate
    keep = freqs < nyq_limit
    freqs = freqs[keep]
    decays = decays[keep]
    gains = gains[keep]

    # Normalize
    gains = _normalize_gains_for_white_noise(freqs, decays, gains, sample_rate)

    return ModalParams(
        freqs=freqs.astype(np.float64),
        decays=decays.astype(np.float64),
        gains=gains.astype(np.float64),
    )


@njit(cache=True, fastmath=True)
def _process_block(
    x: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    gains: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
) -> np.ndarray:
    """Run a block of input samples through the parallel resonator bank."""
    T = x.shape[0]
    N = a1.shape[0]
    out = np.empty(T, dtype=np.float64)
    for t in range(T):
        xt = x[t]
        s = 0.0
        for i in range(N):
            y = a1[i] * y1[i] + a2[i] * y2[i] + gains[i] * xt
            y2[i] = y1[i]
            y1[i] = y
            s += y
        out[t] = s
    return out


class ModalBank:
    """Parallel bank of damped second-order resonators."""

    def __init__(self, params: ModalParams, sample_rate: float) -> None:
        self.sample_rate = float(sample_rate)
        self.params = params
        self._compile_coeffs()
        self.reset()
        # Warm up the JIT so the first real-time callback doesn't block
        self.process(np.zeros(64, dtype=np.float64))
        self.reset()

    def _compile_coeffs(self) -> None:
        fs = self.sample_rate
        f = self.params.freqs
        tau = self.params.decays
        R = np.exp(-_LN1000 / (tau * fs))  # 60 dB decay in tau seconds
        omega = 2.0 * np.pi * f / fs
        self._a1 = (2.0 * R * np.cos(omega)).astype(np.float64)
        self._a2 = (-(R * R)).astype(np.float64)
        self._gains = self.params.gains.astype(np.float64)

    def reset(self) -> None:
        n = self.params.n_modes
        self._y1 = np.zeros(n, dtype=np.float64)
        self._y2 = np.zeros(n, dtype=np.float64)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Filter a block through the bank and return the summed output."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        return _process_block(x, self._a1, self._a2, self._gains, self._y1, self._y2)

    @property
    def n_modes(self) -> int:
        return self.params.n_modes

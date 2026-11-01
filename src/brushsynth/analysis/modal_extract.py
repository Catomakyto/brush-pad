"""Modal parameter extraction from a decayed impulse response."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..synth.modal import ModalParams


@dataclass
class ExtractedMode:
    freq: float  # Hz
    decay: float  # seconds (60 dB)
    amplitude: float  # linear
    phase: float  # radians


def extract_modes(
    ir: np.ndarray,
    sample_rate: float,
    *,
    max_modes: int = 128,
    pencil_param: float | None = None,
    energy_fraction: float = 0.995,
) -> list[ExtractedMode]:
    """Extract modes from a recorded impulse response using Matrix Pencil."""
    x = np.asarray(ir, dtype=np.float64).ravel()
    N = x.shape[0]
    if N < 16:
        raise ValueError("IR is too short for modal extraction")

    # Pencil rank
    default_pencil = min(N // 3, max(3 * max_modes, 64))
    L = int(pencil_param if pencil_param is not None else default_pencil)
    L = int(np.clip(L, max_modes, N - max_modes - 1))

    # Hankel matrices Y1 (rows 0..L-1), Y2 (rows 1..L)
    Y = np.empty((L, N - L), dtype=np.float64)
    for i in range(L):
        Y[i, :] = x[i : i + N - L]
    # SVD-truncate to rank max_modes
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    M = int(min(max_modes, S.size))
    U_m = U[:, :M]

    # Build filtered Y1, Y2 as submatrices shifted by one row in U
    V1 = U_m[:-1, :]
    V2 = U_m[1:, :]
    # Solve V1 @ P = V2 for pencil matrix P
    P, *_ = np.linalg.lstsq(V1, V2, rcond=None)
    z = np.linalg.eigvals(P)

    fs = float(sample_rate)
    dt = 1.0 / fs
    modes: list[ExtractedMode] = []
    for zi in z:
        if np.isclose(zi, 0.0):
            continue
        r = np.abs(zi)
        if r <= 0.0 or r > 1.0 + 1e-6:
            continue  # unstable / numerical noise
        r = min(r, 0.999999)
        # Decay: |z|^n = exp(n log r)
        decay = -6.907755278982137 / (np.log(r) * fs) if np.log(r) < 0 else 10.0
        omega = np.angle(zi) * fs  # rad/s
        freq = omega / (2.0 * np.pi)
        if freq <= 0.0 or freq >= fs * 0.5:
            continue
        modes.append(ExtractedMode(freq=float(freq), decay=float(decay), amplitude=0.0, phase=0.0))

    # Solve for amplitude and phase
    if not modes:
        return modes
    n = np.arange(N)
    basis = []
    for m in modes:
        r = np.exp(-6.907755278982137 / (m.decay * fs))
        omega = 2.0 * np.pi * m.freq / fs
        col_cos = (r ** n) * np.cos(omega * n)
        col_sin = (r ** n) * np.sin(omega * n)
        basis.append(col_cos)
        basis.append(col_sin)
    A = np.vstack(basis).T  # (N, 2K)
    coeffs, *_ = np.linalg.lstsq(A, x, rcond=None)

    enriched: list[tuple[float, ExtractedMode]] = []
    for k, m in enumerate(modes):
        a, b = coeffs[2 * k], coeffs[2 * k + 1]
        amp = float(np.hypot(a, b))
        phase = float(np.arctan2(b, a))
        enriched.append((amp, ExtractedMode(freq=m.freq, decay=m.decay, amplitude=amp, phase=phase)))

    # Sort by descending amplitude
    enriched.sort(key=lambda t: -t[0])
    energies = np.array([a * a for a, _ in enriched])
    total = float(np.sum(energies))
    if total <= 0.0:
        return [m for _, m in enriched]
    cum = np.cumsum(energies) / total
    keep_n = int(np.searchsorted(cum, energy_fraction)) + 1
    return [m for _, m in enriched[:keep_n]]


def modes_to_params(modes: list[ExtractedMode]) -> ModalParams:
    """Convert extracted modes into ModalParams."""
    if not modes:
        raise ValueError("no modes to convert")
    freqs = np.array([m.freq for m in modes], dtype=np.float64)
    decays = np.array([m.decay for m in modes], dtype=np.float64)
    gains = np.array([m.amplitude for m in modes], dtype=np.float64)
    return ModalParams(freqs=freqs, decays=decays, gains=gains)

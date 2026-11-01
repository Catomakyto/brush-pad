"""Perceptual texture statistics for brush-on-snare sound."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Target values and spreads from real brush recordings
TARGETS = {
    "flatness": (0.17, 0.06),  # (target, tolerance)
    "centroid": (2850.0, 500.0),
    "centroid_cv": (0.18, 0.08),
    "crest_db": (15.0, 4.0),
    "rolloff85": (5100.0, 1200.0),
    "env_cv": (0.30, 0.15),
}

# Relative importance of each feature in the composite distance
WEIGHTS = {
    "flatness": 2.0,
    "centroid": 1.5,
    "centroid_cv": 2.0,
    "crest_db": 1.0,
    "rolloff85": 0.5,
    "env_cv": 0.5,
}


@dataclass
class TextureStats:
    flatness: float  # spectral flatness, 0 to 1
    centroid: float  # spectral centroid, Hz (mean over frames)
    centroid_cv: float  # coefficient of variation of per-frame centroid
    crest_db: float  # 20 log10(peak / rms)
    rolloff85: float  # 85%-energy rolloff frequency
    env_cv: float  # envelope variation
    env_mod_peak_hz: float  # dominant envelope-modulation frequency, Hz

    def as_dict(self) -> dict:
        return {
            "flatness": self.flatness,
            "centroid": self.centroid,
            "centroid_cv": self.centroid_cv,
            "crest_db": self.crest_db,
            "rolloff85": self.rolloff85,
            "env_cv": self.env_cv,
            "env_mod_peak_hz": self.env_mod_peak_hz,
        }


def _amp_envelope(y: np.ndarray, sample_rate: float, fc: float = 50.0) -> np.ndarray:
    """Rectify + one-pole low-pass amplitude envelope."""
    r = np.abs(y)
    a = float(np.exp(-2.0 * np.pi * fc / sample_rate))
    env = np.empty_like(r)
    s = 0.0
    for i in range(r.shape[0]):
        s = a * s + (1.0 - a) * r[i]
        env[i] = s
    return env


def analyze_texture(
    audio: np.ndarray, sample_rate: float, *, fft: int = 2048, hop: int = 512,
) -> TextureStats:
    """Compute TextureStats for a mono (or stereo) recording."""
    y = np.asarray(audio, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    peak0 = float(np.max(np.abs(y))) + 1e-12
    y = y / peak0

    win = np.hanning(fft)
    freqs = np.fft.rfftfreq(fft, 1.0 / sample_rate)
    n_frames = 1 + max(0, (y.size - fft) // hop)
    cents = np.empty(n_frames)
    flats = np.empty(n_frames)
    rolls = np.empty(n_frames)
    for i in range(n_frames):
        frame = y[i * hop : i * hop + fft] * win
        mag = np.abs(np.fft.rfft(frame))
        ssum = float(mag.sum()) + 1e-12
        cents[i] = float((freqs * mag).sum() / ssum)
        gm = float(np.exp(np.mean(np.log(mag + 1e-12))))
        am = float(np.mean(mag)) + 1e-12
        flats[i] = gm / am
        cumeng = np.cumsum(mag)
        rolls[i] = float(freqs[int(np.searchsorted(cumeng, 0.85 * cumeng[-1]))])

    if n_frames == 0:
        cents = np.array([0.0]); flats = np.array([1.0]); rolls = np.array([0.0])

    cent_mean = float(cents.mean())
    cent_cv = float(cents.std() / (cent_mean + 1e-12))

    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2))) + 1e-12
    crest = 20.0 * np.log10(peak / rms)

    env = _amp_envelope(y, sample_rate)
    env_cv = float(env.std() / (env.mean() + 1e-12))
    # envelope modulation spectrum, 0.3–40 Hz
    step = max(1, int(sample_rate / 1000.0))
    ed = env[::step]
    ed = ed - ed.mean()
    if ed.size >= 64:
        w = np.hanning(ed.size)
        E = np.abs(np.fft.rfft(ed * w))
        mf = np.fft.rfftfreq(ed.size, step / sample_rate)
        band = (mf >= 0.3) & (mf <= 40.0)
        mod_peak = float(mf[band][int(np.argmax(E[band]))]) if band.any() else 0.0
    else:
        mod_peak = 0.0

    return TextureStats(
        flatness=float(flats.mean()),
        centroid=cent_mean,
        centroid_cv=cent_cv,
        crest_db=float(crest),
        rolloff85=float(rolls.mean()),
        env_cv=env_cv,
        env_mod_peak_hz=mod_peak,
    )


def texture_distance(
    stats: TextureStats,
    targets: dict | None = None,
    weights: dict | None = None,
) -> float:
    """Weighted, tolerance-normalized distance from the brush-texture targets."""
    t = targets or TARGETS
    w = weights or WEIGHTS
    d = stats.as_dict()
    total = 0.0
    for key, (tgt, tol) in t.items():
        if key not in d:
            continue
        total += w.get(key, 1.0) * abs(d[key] - tgt) / max(tol, 1e-9)
    return float(total)

"""Granular brush voice built from real brush recordings."""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import lfilter, resample_poly

_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = str(_ROOT / "recordings" / "brush_ref_karoryfer" / "stir_dl1_rr*.wav")


@dataclass
class GrainConfig:
    sample_rate: float = 48000.0
    grain_ms: float = 90.0  # long grains = smooth
    overlap: int = 4
    speed_knee: float = 6.0  # |v| for half of the loudness range
    speed_gate: float = 0.8  # |v| below this is treated as still
    pressure_curve: float = 0.8  # loudness ∝ pressure**curve
    bright_speed_knee: float = 8.0  # |v| for half-way brightness
    bright_pressure: float = 0.35  # how much pressure (vs speed) drives brightness
    lp_min_hz: float = 6000.0  # soft, slow cut-off
    lp_max_hz: float = 10000.0  # fast/firm cut-off
    rate_min: float = 1.0  # grain playback rate when barely moving
    rate_max: float = 1.06  # rate at a fast sweep
    shelf_hz: float = 3000.0  # "matte" high-shelf
    shelf_db: float = -9.0  # 0 = off
    select_width: float = 0.35  # ± share of the corpus a grain is drawn from
    bright_glide_ms: float = 250.0  # smooths brightness changes so tone never jumps
    attack_ms: float = 6.0
    release_ms: float = 70.0
    gain: float = 0.5  # output level (pre soft-clip)
    seed: int = 0x5717


def _load_corpus(pattern: str, sr: float) -> np.ndarray:
    import soundfile as sf

    files = sorted(f for pat in pattern.split(";") for f in glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"no brush recordings match {pattern}")
    parts = []
    for f in files:
        d, fsr = sf.read(f, dtype="float64")
        if d.ndim > 1:
            d = d.mean(axis=1)
        if int(fsr) != int(sr):
            g = np.gcd(int(sr), int(fsr))
            d = resample_poly(d, int(sr) // g, int(fsr) // g)
        parts.append(d)
    x = np.concatenate(parts)
    # Remove DC/rumble
    a = np.exp(-2.0 * np.pi * 120.0 / sr)
    y = lfilter([(1 + a) / 2, -(1 + a) / 2], [1, -a], x)
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-12)


def _rbj_highshelf(f0: float, sr: float, gain_db: float) -> tuple[np.ndarray, np.ndarray]:
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / sr
    alpha = np.sin(w) / 2 * np.sqrt(2)
    cw, sa = np.cos(w), 2 * np.sqrt(A) * alpha
    b = np.array([A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw),
                  A * ((A + 1) + (A - 1) * cw - sa)])
    a = np.array([(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw),
                  (A + 1) - (A - 1) * cw - sa])
    return b / a[0], a / a[0]


class GrainBrushSynth:
    """Drop-in for BrushSynth in the real-time Engine."""

    def __init__(self, config: GrainConfig | None = None, corpus: str = DEFAULT_CORPUS) -> None:
        cfg = self.config = config or GrainConfig()
        sr = cfg.sample_rate
        self._x = _load_corpus(corpus, sr)
        self._L = int(sr * cfg.grain_ms / 1000.0)
        self._hop = self._L // cfg.overlap
        # Periodic Hann: overlap-adds to a constant (= overlap/2)
        self._win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self._L) / self._L)) * (2.0 / cfg.overlap)
        self._index_grains()
        self._rng = np.random.default_rng(cfg.seed)
        self._acc = np.zeros(self._L * 2 + 8192)
        self._until_next = 0
        self._env = 0.0
        self._lp_state = np.zeros(2)
        self._lp_hz = cfg.lp_min_hz
        self._shelf = _rbj_highshelf(cfg.shelf_hz, sr, cfg.shelf_db) if cfg.shelf_db else None
        self._shelf_zi = np.zeros(2)
        self._bright = 0.0

    # corpus analysis
    def _index_grains(self) -> None:
        x, L = self._x, self._L
        reach = int(L * self.config.rate_max) + 2
        step = L // 8
        starts = np.arange(0, len(x) - reach, step)
        # Transient guard
        sr = self.config.sample_rate
        k = int(0.005 * sr)
        env = np.sqrt(np.convolve(x ** 2, np.ones(k) / k, mode="same"))
        blk = int(0.3 * sr)
        med = np.array([np.median(env[max(0, i - blk // 2): i + blk // 2 + 1: k]) for i in range(0, len(x), k)])
        spike = env[::k][: len(med)] > 3.0 * med
        bad = np.repeat(spike, k)[: len(x)]
        bad_cum = np.concatenate([[0], np.cumsum(bad)])
        keep = (bad_cum[starts + reach] - bad_cum[starts]) == 0
        starts = starts[keep]
        # Brightness descriptor per grain: spectral centroid
        n = 2048
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        w = np.hanning(n)
        cents, rms = [], []
        for s in starts:
            seg = x[s + (L - n) // 2: s + (L - n) // 2 + n]
            mag = np.abs(np.fft.rfft(seg * w))
            cents.append(float((freqs * mag).sum() / (mag.sum() + 1e-12)))
            rms.append(float(np.sqrt(np.mean(x[s:s + L] ** 2))))
        order = np.argsort(cents)
        self._starts = starts[order]
        # Even out grain levels
        r = np.asarray(rms)[order]
        self._norm = np.sqrt(np.median(r) / (r + 1e-9))
        self._centroids = np.asarray(cents)[order]

    # gesture to control
    def _controls(self, vx: float, vy: float, p: float) -> tuple[float, float]:
        cfg = self.config
        s = max(float(np.hypot(vx, vy)) - cfg.speed_gate, 0.0)
        p = min(max(p, 0.0), 1.0)
        loud = (p ** cfg.pressure_curve) * (s / (s + cfg.speed_knee))
        bright = (1 - cfg.bright_pressure) * (s / (s + cfg.bright_speed_knee)) + cfg.bright_pressure * p
        return loud, min(max(bright, 0.0), 1.0)

    def _spawn(self, at: int, bright: float, speed_frac: float) -> None:
        n = len(self._starts)
        centre = bright * (n - 1)
        half = self.config.select_width * n
        i = int(np.clip(centre + self._rng.uniform(-half, half), 0, n - 1))
        cfg = self.config
        rate = cfg.rate_min + (cfg.rate_max - cfg.rate_min) * speed_frac
        pos = self._starts[i] + np.arange(self._L) * rate
        i0 = pos.astype(np.int64)
        fr = pos - i0
        g = self._x[i0] * (1 - fr) + self._x[i0 + 1] * fr
        self._acc[at:at + self._L] += g * self._win * self._norm[i]

    def reset(self) -> None:
        self._acc[:] = 0.0
        self._until_next = 0
        self._env = 0.0
        self._lp_state[:] = 0.0
        self._shelf_zi[:] = 0.0

    def render(
        self,
        vx0: float, vy0: float, p0: float, ang0: float,
        vx1: float, vy1: float, p1: float, ang1: float,
        frames: int,
        x0: float = 0.0, y0: float = 0.0, x1: float = 0.0, y1: float = 0.0,
    ) -> np.ndarray:
        cfg = self.config
        sr = cfg.sample_rate
        loud, bright = self._controls(vx1, vy1, p1)
        if cfg.bright_glide_ms > 0:
            g = 1.0 - np.exp(-frames / (sr * cfg.bright_glide_ms / 1000.0))
            self._bright += g * (bright - self._bright)
            bright = self._bright
        s = max(float(np.hypot(vx1, vy1)) - cfg.speed_gate, 0.0)
        speed_frac = s / (s + cfg.bright_speed_knee)

        # Grains are only scheduled while there is something to hear
        t = self._until_next
        while t < frames:
            if loud > 1e-4 or self._env > 1e-4:
                self._spawn(t, bright, speed_frac)
            t += self._hop
        self._until_next = t - frames
        out = self._acc[:frames].copy()
        self._acc[:-frames] = self._acc[frames:]
        self._acc[-frames:] = 0.0

        # Loudness envelope
        tau = cfg.attack_ms if loud > self._env else cfg.release_ms
        a = np.exp(-1.0 / (sr * tau / 1000.0))
        env = loud + (self._env - loud) * a ** np.arange(1, frames + 1)
        self._env = float(env[-1])
        out *= env

        # Brightness
        target_hz = cfg.lp_min_hz * (cfg.lp_max_hz / cfg.lp_min_hz) ** bright
        self._lp_hz += 0.3 * (target_hz - self._lp_hz)
        c = 1.0 - np.exp(-2.0 * np.pi * self._lp_hz / sr)
        for k in range(2):
            out, _ = lfilter([c], [1.0, c - 1.0], out, zi=[self._lp_state[k] * (1.0 - c)])
            self._lp_state[k] = out[-1]
        if self._shelf is not None:
            out, self._shelf_zi = lfilter(*self._shelf, out, zi=self._shelf_zi)
        return np.tanh(out * cfg.gain).astype(np.float32)

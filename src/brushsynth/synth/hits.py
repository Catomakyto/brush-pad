"""Brush hits: real brushed-snare one-shots (karoryfer Swirly Drums, CC0)."""

from __future__ import annotations

import glob
import re
from collections import deque
from pathlib import Path

import numpy as np
from scipy.signal import lfilter, resample_poly

from .grain_brush import _rbj_highshelf

_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HITS = str(_ROOT / "recordings" / "brush_hits_karoryfer" / "snare_hit_vl*_rr*_top.wav")


class HitPlayer:
    """Polyphonic one-shot player."""

    def __init__(self, sample_rate: float = 48000.0, pattern: str = DEFAULT_HITS, *,
                 gain: float = 1.0, shelf_hz: float = 3000.0, shelf_db: float = -9.0,
                 max_voices: int = 8, seed: int = 0x417) -> None:
        import soundfile as sf

        self.sr = float(sample_rate)
        self.gain = float(gain)
        layers: dict[int, list[np.ndarray]] = {}
        shelf = _rbj_highshelf(shelf_hz, self.sr, shelf_db) if shelf_db else None
        for f in sorted(glob.glob(pattern)):
            m = re.search(r"_vl(\d+)_", Path(f).name)
            if not m:
                continue
            x, fsr = sf.read(f, dtype="float64")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if int(fsr) != int(self.sr):
                g = np.gcd(int(self.sr), int(fsr))
                x = resample_poly(x, int(self.sr) // g, int(fsr) // g)
            if shelf is not None:
                x = lfilter(*shelf, x)
            fade = min(len(x), int(0.05 * self.sr))  # tidy tail, never a click
            x[-fade:] *= np.linspace(1.0, 0.0, fade)
            layers.setdefault(int(m.group(1)), []).append(x)
        if not layers:
            raise FileNotFoundError(f"no hit samples match {pattern}")
        self._layers = [layers[k] for k in sorted(layers)]
        self._rng = np.random.default_rng(seed)
        self._pending: deque[tuple[float, float]] = deque()
        self._voices: list[list] = []  # [sample, position]
        self._max_voices = max_voices

    def trigger(self, velocity: float, gain: float = 1.0) -> None:
        self._pending.append((min(max(float(velocity), 0.0), 1.0), float(gain)))

    def reset(self) -> None:
        self._pending.clear()
        self._voices.clear()

    def render(self, frames: int) -> np.ndarray:
        while self._pending:
            v, g = self._pending.popleft()
            k = int(round(v * (len(self._layers) - 1)))
            rr = self._layers[k]
            self._voices.append([rr[int(self._rng.integers(len(rr)))] * g, 0])
            if len(self._voices) > self._max_voices:
                self._voices.pop(0)
        out = np.zeros(frames)
        alive = []
        for voice in self._voices:
            x, pos = voice
            n = min(frames, len(x) - pos)
            out[:n] += x[pos:pos + n]
            voice[1] = pos + n
            if voice[1] < len(x):
                alive.append(voice)
        self._voices = alive
        return out * self.gain

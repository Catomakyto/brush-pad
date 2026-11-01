"""Two granular brush voices + brush hits."""

from __future__ import annotations

import numpy as np

from ..engine.rtaudio import Gesture, ParamBus
from .grain_brush import GrainBrushSynth, GrainConfig
from .hits import HitPlayer


def _pan(x: float, width: float) -> tuple[float, float]:
    """Constant-power gains for pad x in [-1, 1]."""
    a = (min(max(x * width, -1.0), 1.0) + 1.0) * np.pi / 4.0
    return float(np.cos(a)), float(np.sin(a))


class BrushEnsemble:
    channels = 2

    def __init__(self, sample_rate: float = 48000.0, *, hits: bool = True,
                 hit_gain: float = 1.0, pan_width: float = 0.8,
                 pan_axis: str = "y", pan_flip: bool = False) -> None:
        sr = float(sample_rate)
        self.pan_width = float(pan_width)
        self._axis_y = pan_axis == "y"
        self._flip = -1.0 if pan_flip else 1.0
        self._pan1 = self._pan2 = (np.sqrt(0.5), np.sqrt(0.5))
        self.bus2 = ParamBus()
        self._prev2 = Gesture()
        self.brush1 = GrainBrushSynth(GrainConfig(sample_rate=sr, seed=0x5717))
        self.brush2 = GrainBrushSynth(GrainConfig(sample_rate=sr, seed=0x2B2B))
        self.hits = HitPlayer(sr, gain=hit_gain) if hits else None
        self.hits_r = HitPlayer(sr, gain=hit_gain) if hits else None  # right channel twin

    def _side(self, x: float, y: float) -> float:
        return self._flip * (y if self._axis_y else x)

    def trigger_hit(self, velocity: float, x: float = 0.0, y: float = 0.0) -> None:
        if self.hits is None:
            return
        gl, gr = _pan(self._side(x, y), self.pan_width)
        # Same layer + round-robin on both sides, scaled by pan
        self.hits.trigger(velocity, gl)
        self.hits_r.trigger(velocity, gr)

    def reset(self) -> None:
        self.brush1.reset()
        self.brush2.reset()
        if self.hits is not None:
            self.hits.reset()
            self.hits_r.reset()

    def render(self, vx0, vy0, p0, ang0, vx1, vy1, p1, ang1, frames,
               x0=0.0, y0=0.0, x1=0.0, y1=0.0) -> np.ndarray:
        g0, g1 = self._prev2, self.bus2.snapshot()
        self._prev2 = g1
        b1 = self.brush1.render(vx0, vy0, p0, ang0, vx1, vy1, p1, ang1, frames, x0, y0, x1, y1)
        b2 = self.brush2.render(g0.vx, g0.vy, g0.pressure, g0.angle,
                                g1.vx, g1.vy, g1.pressure, g1.angle, frames, g0.x, g0.y, g1.x, g1.y)
        # Pan follows the brush only while it touches
        if p1 > 0:
            self._pan1 = _pan(self._side(x1, y1), self.pan_width)
        if g1.pressure > 0:
            self._pan2 = _pan(self._side(g1.x, g1.y), self.pan_width)
        out = np.empty((frames, 2))
        out[:, 0] = self._pan1[0] * b1 + self._pan2[0] * b2
        out[:, 1] = self._pan1[1] * b1 + self._pan2[1] * b2
        if self.hits is not None:
            out[:, 0] += self.hits.render(frames)
            out[:, 1] += self.hits_r.render(frames)
        return np.tanh(out).astype(np.float32)

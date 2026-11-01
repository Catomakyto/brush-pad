"""Tests for the bristle excitation model."""

from __future__ import annotations

import numpy as np

from brushsynth.synth.excitation import (
    BristleBrush,
    default_nylon_brush,
)


SR = 48000
FRAMES = 2048


def _render_steady(vx=0.0, vy=0.0, pressure=0.0, ang=0.0, frames=FRAMES):
    brush = BristleBrush(default_nylon_brush(), SR, seed=0)
    # Steady-state
    return brush.render(vx, vy, pressure, ang, vx, vy, pressure, ang, frames)


def test_zero_pressure_is_silent():
    out = _render_steady(vx=1.0, vy=0.0, pressure=0.0)
    assert np.all(np.isfinite(out))
    assert float(np.max(np.abs(out))) == 0.0


def test_zero_velocity_at_any_pressure_decays_to_silence():
    # Even with some residual internal state
    out = _render_steady(vx=0.0, vy=0.0, pressure=0.8, frames=SR // 2)
    tail = out[-4096:]
    rms = float(np.sqrt(np.mean(tail ** 2)))
    assert rms < 1e-6, rms


def test_pressure_up_makes_it_louder():
    low = _render_steady(vx=1.0, vy=0.0, pressure=0.2, frames=SR // 4)
    hi = _render_steady(vx=1.0, vy=0.0, pressure=0.8, frames=SR // 4)
    low_rms = float(np.sqrt(np.mean(low ** 2)))
    hi_rms = float(np.sqrt(np.mean(hi ** 2)))
    assert hi_rms > low_rms * 1.5, (low_rms, hi_rms)


def test_velocity_up_raises_spectral_centroid():
    # Same pressure, 10× velocity difference to audibly brighter swish
    slow = _render_steady(vx=0.3, vy=0.0, pressure=0.6, frames=SR // 2)
    fast = _render_steady(vx=3.0, vy=0.0, pressure=0.6, frames=SR // 2)

    def centroid(x: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(x.size, d=1.0 / SR)
        denom = float(np.sum(spec)) + 1e-12
        return float(np.sum(freqs * spec) / denom)

    c_slow = centroid(slow)
    c_fast = centroid(fast)
    assert c_fast > c_slow * 1.1, (c_slow, c_fast)


def test_reset_restores_initial_behavior():
    brush = BristleBrush(default_nylon_brush(), SR, seed=0)
    a = brush.render(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.6, 0.0, FRAMES)
    # Second call with the same input would see non-zero internal state
    brush.reset()
    b = brush.render(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.6, 0.0, FRAMES)
    assert np.allclose(a, b)


def test_finite_and_bounded():
    out = _render_steady(vx=5.0, vy=5.0, pressure=1.0, frames=SR)
    assert np.all(np.isfinite(out))
    # Sanity: no explosion into absurd magnitudes
    assert float(np.max(np.abs(out))) < 1e4

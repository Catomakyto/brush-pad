"""End-to-end smoke tests for the assembled synth."""

from __future__ import annotations

import numpy as np

from brushsynth.synth.brush_synth import BrushSynth, SynthConfig


def test_synth_produces_audio_under_motion():
    synth = BrushSynth(SynthConfig(sample_rate=48000.0))
    out = synth.render(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.6, 0.0, 48000)
    assert out.dtype == np.float32
    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert rms > 1e-4, rms
    assert np.all(np.isfinite(out))
    # Not clipping hard
    assert float(np.max(np.abs(out))) < 0.98


def test_synth_quiet_without_motion():
    synth = BrushSynth(SynthConfig(sample_rate=48000.0))
    # Run idle for a while to let any residual state decay
    for _ in range(200):
        _ = synth.render(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 256)
    tail = synth.render(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2048)
    rms = float(np.sqrt(np.mean(tail.astype(np.float64) ** 2)))
    assert rms < 1e-5, rms


def test_circle_motion_is_continuous():
    """A sustained circular gesture should produce sustained audio."""
    synth = BrushSynth(SynthConfig(sample_rate=48000.0))
    sr = 48000
    block = 256
    total_frames = sr  # 1 second
    freq = 1.5  # Hz circular rate
    radius = 0.5
    pressure = 0.6

    t0 = 0.0
    prev_vx, prev_vy = 0.0, 0.0
    out = np.zeros(total_frames, dtype=np.float32)
    idx = 0
    while idx < total_frames:
        frames = min(block, total_frames - idx)
        t1 = (idx + frames) / sr
        # Velocity of circle: v = 2π f r * (-sin, cos)
        vx1 = -2.0 * np.pi * freq * radius * np.sin(2.0 * np.pi * freq * t1)
        vy1 = 2.0 * np.pi * freq * radius * np.cos(2.0 * np.pi * freq * t1)
        buf = synth.render(prev_vx, prev_vy, pressure, 0.0, vx1, vy1, pressure, 0.0, frames)
        out[idx : idx + frames] = buf
        prev_vx, prev_vy = vx1, vy1
        t0 = t1
        idx += frames

    # No long silent stretches
    win = 2048
    rms_windows = np.array([
        np.sqrt(np.mean(out[i:i + win] ** 2)) for i in range(0, total_frames - win, win // 2)
    ])
    # Skip the first window (engine warm-up) before checking continuity
    assert np.all(rms_windows[1:] > 1e-4), rms_windows

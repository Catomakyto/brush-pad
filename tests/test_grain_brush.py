"""Granular brush voice: silent when still, louder with pressure, no bangs."""

from __future__ import annotations

import numpy as np

from brushsynth.synth.grain_brush import GrainBrushSynth


def _play(synth, speed, pressure, blocks=200, n=256):
    return np.concatenate([synth.render(speed, 0, pressure, 0, speed, 0, pressure, 0, n)
                           for _ in range(blocks)])


def _rms(y):
    return float(np.sqrt(np.mean(y[len(y) // 4:] ** 2)))


def test_still_brush_is_silent():
    assert _rms(_play(GrainBrushSynth(), 0.0, 0.8)) < 1e-4


def test_louder_with_pressure_and_speed():
    soft = _rms(_play(GrainBrushSynth(), 3.0, 0.15))
    firm = _rms(_play(GrainBrushSynth(), 3.0, 0.85))
    fast = _rms(_play(GrainBrushSynth(), 8.0, 0.15))
    assert firm > 2.5 * soft and fast > soft


def test_steady_stir_has_no_spikes():
    y = _play(GrainBrushSynth(), 4.0, 0.5, blocks=400)[48000:]
    k = 240  # 5 ms
    e = np.sqrt(np.convolve(y ** 2, np.ones(k) / k, "valid"))[::k]
    rises = 20 * np.log10(e[1:] / e[:-1])
    assert rises.max() < 9.0

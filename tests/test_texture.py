"""Property tests for the brush-texture fixes."""

from __future__ import annotations

import numpy as np

from brushsynth.analysis.texture_stats import analyze_texture
from brushsynth.synth.brush_synth import BrushSynth, SynthConfig
from brushsynth.synth.snare_wires import SnareWires, _collision_exc

SR = 48000


def _render_circle(synth: BrushSynth, *, move_position: bool,
                   freq: float = 0.9, radius: float = 0.55,
                   pressure: float = 0.6, seconds: float = 3.0) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    px = radius * np.cos(2 * np.pi * freq * t)
    py = radius * np.sin(2 * np.pi * freq * t)
    vx = -2 * np.pi * freq * radius * np.sin(2 * np.pi * freq * t)
    vy = 2 * np.pi * freq * radius * np.cos(2 * np.pi * freq * t)
    out = np.zeros(n, dtype=np.float32)
    block = 256
    pvx = pvy = 0.0
    ppx, ppy = float(px[0]), float(py[0])
    idx = 0
    while idx < n:
        fr = min(block, n - idx)
        k = idx + fr - 1
        x1 = float(px[k]) if move_position else 0.0
        y1 = float(py[k]) if move_position else 0.0
        x0 = ppx if move_position else 0.0
        y0 = ppy if move_position else 0.0
        out[idx:idx + fr] = synth.render(
            pvx, pvy, pressure, 0.0, float(vx[k]), float(vy[k]), pressure, 0.0,
            fr, x0, y0, x1, y1)
        pvx, pvy = float(vx[k]), float(vy[k])
        ppx, ppy = float(px[k]), float(py[k])
        idx += fr
    return out.astype(np.float64)


def test_snare_wires_fire_on_crossing_head():
    """A head signal that exceeds the contact gap must produce wire output."""
    wires = SnareWires(SR, gap_mult=1.3, stiffness=9.0, alpha=1.7)
    rng = np.random.default_rng(0)
    head = 0.15 * rng.standard_normal(SR // 2)
    out = wires.process(head)
    assert np.all(np.isfinite(out))
    assert float(np.sqrt(np.mean(out ** 2))) > 1e-4


def test_collision_drive_is_sparse_and_one_sided():
    """The power-law collision drive is non-negative and intermittent."""
    rng = np.random.default_rng(1)
    head = 0.1 * rng.standard_normal(SR // 4)
    # gap_mult=1.4, env_coeff for ~80 ms at 48k
    env_coeff = 1.0 - np.exp(-1.0 / (0.08 * SR))
    drive = _collision_exc(head, 1.4, env_coeff, 9.0, 1.7, 0.0, np.zeros(2))
    assert np.all(drive >= 0.0)  # one-sided
    active = float(np.mean(drive > 0.0))
    assert 0.0 < active < 0.5  # intermittent contact


def test_wires_fire_in_full_synth():
    """After auto-calibration the wire gap must sit within head range."""
    synth = BrushSynth(SynthConfig(sample_rate=SR, wires_mix=1.0))
    y_full = _render_circle(synth, move_position=True)
    synth.reset()
    synth.config.wires_mix = 0.0
    y_dry = _render_circle(synth, move_position=True)
    # The wet (with-wires) render must differ audibly from the dry one
    diff = float(np.sqrt(np.mean((y_full - y_dry) ** 2)))
    assert diff > 1e-3, diff


def test_position_motion_animates_centroid():
    """Moving across the coating field changes the spectrum."""
    synth = BrushSynth(SynthConfig(sample_rate=SR))
    moving = analyze_texture(_render_circle(synth, move_position=True), SR)
    synth.reset()
    static = analyze_texture(_render_circle(synth, move_position=False), SR)
    assert moving.centroid_cv > static.centroid_cv * 1.4, (static.centroid_cv, moving.centroid_cv)
    assert moving.centroid_cv > 0.055, moving.centroid_cv


def test_circular_sweep_is_tonal_not_hiss():
    """A sustained circular sweep must be tonal."""
    synth = BrushSynth(SynthConfig(sample_rate=SR))
    stats = analyze_texture(_render_circle(synth, move_position=True), SR)
    assert stats.flatness < 0.35, stats.flatness

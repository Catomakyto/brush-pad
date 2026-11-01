"""Tests for preset voicings, two-hand rendering, and the groove generator."""

from __future__ import annotations

import numpy as np
import pytest

from brushsynth.synth.brush_synth import BrushSynth, SynthConfig
from brushsynth.synth.excitation import BristleBrush, default_nylon_brush
from brushsynth.synth.presets import PRESETS, build_preset

SR = 48000


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42, 123])
def test_bristle_seed_does_not_overflow(seed):
    """Regression: seeds > ~1 used to overflow np.uint64 in _seed_rng."""
    brush = BristleBrush(default_nylon_brush(), SR, seed=seed)
    out = brush.render(0.0, 0.0, 0.6, 0.0, 2.0, 0.0, 0.6, 0.0, 512)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_preset_builds_and_sounds(name):
    synth = build_preset(name, SR)
    out = synth.render(0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.6, 0.0, SR // 2,
                       0.0, 0.0, 0.3, 0.3)
    assert np.all(np.isfinite(out))
    assert float(np.sqrt(np.mean(out.astype(np.float64) ** 2))) > 1e-4
    assert float(np.max(np.abs(out))) < 0.98


def test_render_pair_sums_two_brushes():
    """Two-hand render includes both hands' contributions."""
    synth = BrushSynth(SynthConfig(sample_rate=float(SR)))
    left = (0.0, 0.0, 0.5, 0.0, 3.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.1, 0.0)
    right_silent = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    right_tap = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0)
    synth.reset()
    a = synth.render_pair(left, right_silent, 1024)
    synth.reset()
    b = synth.render_pair(left, right_tap, 1024)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    # Adding a right-hand tap changes the output
    assert not np.allclose(a, b)


def test_render_pair_deterministic_after_reset():
    synth = BrushSynth(SynthConfig(sample_rate=float(SR)))
    L = (0.0, 0.0, 0.5, 0.0, 3.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.1, 0.0)
    R = (0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0)
    a = np.concatenate([synth.render_pair(L, R, 256) for _ in range(8)])
    synth.reset()
    b = np.concatenate([synth.render_pair(L, R, 256) for _ in range(8)])
    assert np.allclose(a, b)


def test_groove_builds_valid_traces():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from render_groove import build_groove

    for style in ("ballad", "medium_swing", "up_swing"):
        left, right = build_groove(style, bpm=120.0, bars=2, sr=SR)
        assert left.shape[1] == 5 and right.shape[1] == 5
        assert left.shape[0] == right.shape[0]
        # right hand carries taps (pressure pulses)
        assert float(right[:, 3].max()) > 0.3

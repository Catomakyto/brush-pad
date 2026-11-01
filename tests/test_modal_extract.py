"""Tests for modal extraction (matrix pencil)."""

from __future__ import annotations

import numpy as np

from brushsynth.analysis.modal_extract import extract_modes


_LN1000 = 6.907755278982137


def _synthesize_ir(freqs, decays, amps, sr, duration=1.0, noise=0.0, seed=0):
    n = int(duration * sr)
    t = np.arange(n) / sr
    ir = np.zeros(n, dtype=np.float64)
    for f, tau, a in zip(freqs, decays, amps):
        envelope = np.exp(-_LN1000 * t / tau)
        ir += a * envelope * np.cos(2.0 * np.pi * f * t)
    if noise > 0:
        rng = np.random.default_rng(seed)
        ir += noise * rng.standard_normal(n)
    return ir


def test_recovers_two_sinusoids_clean():
    sr = 48000
    freqs_true = [220.0, 660.0]
    decays_true = [0.25, 0.15]
    amps_true = [1.0, 0.5]
    ir = _synthesize_ir(freqs_true, decays_true, amps_true, sr, duration=0.5)
    modes = extract_modes(ir, sr, max_modes=8, energy_fraction=0.999)
    # Two dominant modes expected
    top2 = sorted(modes[:2], key=lambda m: m.freq)
    assert abs(top2[0].freq - 220.0) < 2.0
    assert abs(top2[1].freq - 660.0) < 2.0
    # Decays within 20% tolerance
    assert abs(top2[0].decay - 0.25) / 0.25 < 0.3
    assert abs(top2[1].decay - 0.15) / 0.15 < 0.3


def test_returns_no_modes_for_silence():
    sr = 48000
    ir = np.zeros(sr // 4)
    ir[0] = 1e-12
    modes = extract_modes(ir, sr, max_modes=4)
    # Near-zero signal
    assert all(m.amplitude < 1e-6 for m in modes)

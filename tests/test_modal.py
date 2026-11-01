"""Tests for the modal resonator bank."""

from __future__ import annotations

import numpy as np
import pytest

from brushsynth.synth.modal import ModalBank, ModalParams, default_snare_modes


def test_shapes_must_match():
    with pytest.raises(ValueError):
        ModalParams(
            freqs=np.array([100.0, 200.0]),
            decays=np.array([0.5]),
            gains=np.array([1.0, 1.0]),
        )


def test_positive_decays_required():
    with pytest.raises(ValueError):
        ModalParams(
            freqs=np.array([100.0]),
            decays=np.array([0.0]),
            gains=np.array([1.0]),
        )


def test_silence_in_silence_out():
    fs = 48000
    bank = ModalBank(default_snare_modes(fs), fs)
    y = bank.process(np.zeros(1024))
    assert np.all(y == 0.0)


def test_impulse_decays_to_zero():
    """After an impulse, the bank must decay to silence (no unstable pole)."""
    fs = 48000
    bank = ModalBank(default_snare_modes(fs), fs)
    x = np.zeros(fs * 4, dtype=np.float64)
    x[0] = 1.0
    y = bank.process(x)
    # Tail (last 1000 samples) should be much quieter than the head
    head_rms = float(np.sqrt(np.mean(y[:1000] ** 2)))
    tail_rms = float(np.sqrt(np.mean(y[-1000:] ** 2)))
    assert np.all(np.isfinite(y)), "NaN/Inf in modal output"
    assert head_rms > 0.0
    assert tail_rms < head_rms * 1e-3, (head_rms, tail_rms)


def test_impulse_has_expected_peak_frequencies():
    """Impulse-response FFT peaks sit near the modal frequencies."""
    fs = 48000
    params = ModalParams(
        freqs=np.array([220.0, 440.0, 880.0]),
        decays=np.array([0.4, 0.4, 0.4]),
        gains=np.array([1.0, 1.0, 1.0]),
    )
    bank = ModalBank(params, fs)
    x = np.zeros(fs, dtype=np.float64)
    x[0] = 1.0
    y = bank.process(x)
    spectrum = np.abs(np.fft.rfft(y))
    freqs_hz = np.fft.rfftfreq(y.size, d=1.0 / fs)
    # For each target freq, peak should fall within 2 Hz tolerance
    for f_target in (220.0, 440.0, 880.0):
        idx_target = int(np.argmin(np.abs(freqs_hz - f_target)))
        # local window
        lo = max(0, idx_target - 50)
        hi = min(spectrum.size, idx_target + 50)
        idx_peak = lo + int(np.argmax(spectrum[lo:hi]))
        f_peak = freqs_hz[idx_peak]
        assert abs(f_peak - f_target) < 2.0, (f_target, f_peak)


def test_state_resets():
    fs = 48000
    bank = ModalBank(default_snare_modes(fs), fs)
    x = np.zeros(256, dtype=np.float64)
    x[0] = 1.0
    y1 = bank.process(x)
    bank.reset()
    y2 = bank.process(x)
    assert np.allclose(y1, y2)

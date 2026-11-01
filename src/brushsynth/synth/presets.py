"""Named brush voicings."""

from __future__ import annotations

from .brush_synth import BrushSynth, SynthConfig
from .excitation import BristleParams, default_nylon_brush, default_wire_brush


def _thigpen(sr: float) -> tuple[BristleParams, SynthConfig]:
    bp = default_nylon_brush()
    bp.bright_coef = 0.18  # gentle brightness to warm
    bp.coating_bright_depth = 0.42
    bp.bright_vel_knee = 3.6
    cfg = SynthConfig(
        sample_rate=sr,
        tilt_db_per_oct=1.2,
        resonance_mix=0.28,  # mostly smooth friction colour
        dry_cutoff_hz=2900.0,  # warm
        wires_mix=0.16, wire_soft_boost=2.4,
        tap_splay=0.010,  # understated taps
    )
    return bp, cfg


def _papa_jo(sr: float) -> tuple[BristleParams, SynthConfig]:
    bp = default_nylon_brush()
    bp.bright_coef = 0.26  # brighter, driving
    bp.coating_bright_depth = 0.55
    bp.bright_vel_knee = 4.4
    cfg = SynthConfig(
        sample_rate=sr,
        tilt_db_per_oct=2.4,
        resonance_mix=0.30,
        dry_cutoff_hz=3800.0,  # more air
        wires_mix=0.24, wire_soft_boost=2.8,  # crisper wire shimmer
        tap_splay=0.015,  # sharper taps
    )
    return bp, cfg


def _hamilton(sr: float) -> tuple[BristleParams, SynthConfig]:
    bp = default_nylon_brush()
    bp.bright_coef = 0.22
    bp.coating_bright_depth = 0.48
    bp.bright_vel_knee = 4.0
    cfg = SynthConfig(
        sample_rate=sr,
        tilt_db_per_oct=1.6,
        resonance_mix=0.40,  # more drum body / articulate
        dry_cutoff_hz=3200.0,
        wires_mix=0.19, wire_soft_boost=2.4,
        tap_splay=0.014,  # clear taps
    )
    return bp, cfg


def _wire(sr: float) -> tuple[BristleParams, SynthConfig]:
    bp = default_wire_brush()  # fewer, stiffer bristles
    bp.friction_pink = 0.30
    bp.bright_coef = 0.24
    bp.coating_bright_depth = 0.5
    bp.coating_grain_depth = 0.0
    bp.bright_vel_knee = 4.2
    cfg = SynthConfig(
        sample_rate=sr,
        tilt_db_per_oct=2.0,
        resonance_mix=0.32,
        dry_cutoff_hz=3400.0,
        wires_mix=0.22, wire_soft_boost=2.6,
        tap_splay=0.013,
    )
    return bp, cfg


def _live_sweep(sr: float) -> tuple[BristleParams, SynthConfig]:
    """A pure sweep voicing for live playing, with no tap."""
    bp = default_nylon_brush()
    bp.roughness = 0.5
    bp.activity = 0.30
    bp.friction_pink = 0.15
    bp.slip_gain = 0.10
    cfg = SynthConfig(
        sample_rate=sr,
        tap_splay=0.0, tap_body_gain=0.0,  # pure sweep, no tap
        resonance_mix=0.0,  # no modal body ring ("deep")
        wires_mix=0.20,  # snare wire / "metal"
        dry_cutoff_hz=3500.0,  # lighter/airier
        voice_hp_hz=300.0,  # shed the low-mid "deep"
        voice_presence_db=0.0,  # flatten the +5 dB @ 950 Hz hump
        voice_cut_db=0.0,  # keep the 3 kHz sizzle
        voice_air_db=3.0,  # open the top (air = "metal")
    )
    return bp, cfg


PRESETS = {
    "thigpen": _thigpen,
    "papa_jo": _papa_jo,
    "hamilton": _hamilton,
    "wire": _wire,
    "live_sweep": _live_sweep,
}


def preset_params(name: str, sample_rate: float = 48000.0) -> tuple[BristleParams, SynthConfig]:
    """Return the (BristleParams, SynthConfig) for a named voicing."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset '{name}'; choices: {sorted(PRESETS)}")
    return PRESETS[name](float(sample_rate))


def build_preset(name: str, sample_rate: float = 48000.0) -> BrushSynth:
    """Build a BrushSynth configured for a named voicing."""
    bp, cfg = preset_params(name, sample_rate)
    return BrushSynth(cfg, brush_params=bp)

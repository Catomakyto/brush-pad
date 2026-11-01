"""Top-level synth that composes the brush to head to wires signal chain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, sosfilt, sosfilt_zi

from .excitation import BristleBrush, BristleParams, default_nylon_brush
from .modal import (
    ModalBank,
    ModalParams,
    _normalize_gains_for_white_noise,
    apply_spectral_tilt,
    default_snare_modes,
)
from .snare_wires import SnareWires


_SOFT_CLIP_KNEE = 0.85  # transparent below this


def _rbj_highpass(f0: float, sr: float, q: float = 0.707) -> list[float]:
    w0 = 2.0 * np.pi * f0 / sr
    al = np.sin(w0) / (2.0 * q)
    c = np.cos(w0)
    b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
    a = [1 + al, -2 * c, 1 - al]
    return [b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]


def _rbj_peak(f0: float, sr: float, q: float, gain_db: float) -> list[float]:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    al = np.sin(w0) / (2.0 * q)
    c = np.cos(w0)
    b = [1 + al * A, -2 * c, 1 - al * A]
    a = [1 + al / A, -2 * c, 1 - al / A]
    return [b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]


def _rbj_highshelf(f0: float, sr: float, gain_db: float) -> list[float]:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    al = np.sin(w0) / 2.0 * np.sqrt((A + 1 / A) + 2.0)
    c = np.cos(w0)
    tsa = 2.0 * np.sqrt(A) * al
    b = [A * ((A + 1) + (A - 1) * c + tsa), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - tsa)]
    a = [(A + 1) - (A - 1) * c + tsa, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - tsa]
    return [b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]

# If present, this file overrides the placeholder snare modal bank
_DEFAULT_MODES_PATH = Path(__file__).resolve().parents[3] / "models" / "snare_modes.npz"
# If present, this file overrides brush parameters
_DEFAULT_BRUSH_PARAMS_PATH = Path(__file__).resolve().parents[3] / "models" / "brush_params.npz"


def _load_tuned_brush_params(
    path: Path, base: BristleParams,
) -> tuple[BristleParams, dict]:
    """Apply tuned parameters from brush_params.npz to a base BristleParams."""
    if not path.is_file():
        return base, {}
    data = np.load(str(path))
    out = BristleParams(**vars(base))
    def f(k, default):
        return float(data[k]) if k in data.files else float(default)
    # Excitation / bristle knobs
    out.sigma0 = f("sigma0", base.sigma0)
    out.activity = f("activity", base.activity)
    out.slip_gamma = f("slip_gamma", base.slip_gamma)
    out.roughness = f("roughness", base.roughness)
    out.slip_gain = f("slip_gain", base.slip_gain)
    out.friction_pink = f("friction_pink", base.friction_pink)
    out.bright_coef = f("bright_coef", base.bright_coef)
    out.coating_amp_depth = f("coating_amp_depth", base.coating_amp_depth)
    out.coating_bright_depth = f("coating_bright_depth", base.coating_bright_depth)
    out.coating_grain_depth = f("coating_grain_depth", base.coating_grain_depth)
    out.macro_rate_hz = f("macro_rate_hz", base.macro_rate_hz)
    out.macro_depth = f("macro_depth", base.macro_depth)
    out.macro_noise = f("macro_noise", base.macro_noise)
    # Synth-level knobs stored in the same file
    overrides: dict = {}
    for key in ("tilt_db_per_oct", "wires_mix", "wire_gap_rms_mult",
                "wire_stiffness", "wire_alpha", "wire_beta"):
        if key in data.files:
            overrides[key] = float(data[key])
    return out, overrides


def _load_modes_npz(path: Path, sample_rate: float) -> ModalParams | None:
    """Load an extracted snare modal bank if path exists."""
    if not path.is_file():
        return None
    data = np.load(str(path))
    freqs = np.asarray(data["freqs"], dtype=np.float64)
    decays = np.asarray(data["decays"], dtype=np.float64)
    gains = np.asarray(data["gains"], dtype=np.float64)
    # Drop modes that would alias at the target sample rate
    nyq_limit = 0.48 * sample_rate
    keep = (freqs > 0.0) & (freqs < nyq_limit) & (decays > 0.0)
    freqs = freqs[keep]
    decays = decays[keep]
    gains = gains[keep]
    if freqs.size == 0:
        return None
    gains = _normalize_gains_for_white_noise(freqs, decays, gains, sample_rate, target_rms=0.5)
    return ModalParams(freqs=freqs, decays=decays, gains=gains)


@dataclass
class SynthConfig:
    sample_rate: float = 48000.0
    # head_gain is set automatically by BrushSynth._auto_calibrate
    head_gain: float = 1.0
    # Snare-wire base level
    wires_mix: float = 0.18
    # Level-dependent wire compressor
    wire_ref_level: float = 0.30  # head RMS where wire scale = 1.0 (loud end)
    wire_soft_boost: float = 2.4  # max wire multiplier when playing soft
    wire_comp_exp: float = 0.55  # slope of the loudness to wire-scale mapping
    # Brush tap
    tap_splay: float = 0.012  # dp/dt to scrub-velocity gain (0 = off)
    tap_rate_cap: float = 50.0  # max dp/dt used (so sharp edges don't spike)
    tap_env_ms: float = 15.0  # tap scrub decay time
    # The friction scrub alone is a thin blip with no body
    tap_body_gain: float = 0.7  # brief damped body thud on a strike (0 = off)
    tap_wire_gain: float = 2.5  # snare-wire buzz on a strike (the tail + "tss")
    tap_attack_gain: float = 0.30  # crisp contact tick brightness
    # Voicing EQ
    voice_eq: bool = True
    voice_hp_hz: float = 220.0  # high-pass: kill the sub rumble
    voice_presence_db: float = 5.0  # peak @ ~950 Hz: the brush "shhh" presence
    voice_cut_db: float = -6.0  # peak @ ~3 kHz: tame the upper-mid
    voice_air_db: float = -3.0  # high-shelf @ 5 kHz: light air trim
    # How much the drum body *rings* vs merely *colors* the friction
    resonance_mix: float = 0.30
    # Cutoff (Hz) of the smooth dry-friction low-pass
    dry_cutoff_hz: float = 3500.0
    # Pure volume trim layered on top of calibration
    master_gain: float = 1.0
    # Modal-bank spectral tilt in dB/octave
    tilt_db_per_oct: float = 1.5
    # Snare-wire collision parameters
    wire_gap_rms_mult: float = 0.9
    wire_stiffness: float = 9.0
    wire_alpha: float = 1.7
    wire_beta: float = 0.0


class BrushSynth:
    """Brush ensemble excitation to head modal bank to snare wires voice."""

    def __init__(
        self,
        config: SynthConfig | None = None,
        *,
        brush_params: BristleParams | None = None,
        head_params: ModalParams | None = None,
        modes_path: Path | str | None = None,
        brush_params_path: Path | str | None = None,
        auto_calibrate: bool = True,
    ) -> None:
        cfg = config or SynthConfig()
        self.config = cfg

        # Brush/excitation parameters, explicit > tuned file > defaults
        if brush_params is None:
            base = default_nylon_brush()
            path = Path(brush_params_path) if brush_params_path is not None else _DEFAULT_BRUSH_PARAMS_PATH
            tuned, overrides = _load_tuned_brush_params(path, base)
            if path.is_file():
                # The tuned file is authoritative for the globals it stores
                if config is None:
                    for key, val in overrides.items():
                        setattr(cfg, key, val)
                print(f"[synth] loaded tuned brush params from {path}  "
                      f"(sigma0={tuned.sigma0:.0f}, activity={tuned.activity:.2f}, "
                      f"roughness={tuned.roughness:.2f}, pink={tuned.friction_pink:.2f}, "
                      f"tilt={cfg.tilt_db_per_oct:+.1f} dB/oct, wires_mix={cfg.wires_mix:.2f})",
                      flush=True)
            brush_params = tuned
        self.brush = BristleBrush(brush_params, cfg.sample_rate)

        # Head modal bank, explicit > extracted file > placeholder defaults
        if head_params is None:
            path = Path(modes_path) if modes_path is not None else _DEFAULT_MODES_PATH
            extracted = _load_modes_npz(path, cfg.sample_rate)
            if extracted is not None:
                print(f"[synth] loaded extracted snare modes from {path} ({extracted.n_modes} modes)",
                      flush=True)
                head_params = extracted
            else:
                head_params = default_snare_modes(cfg.sample_rate)
        if cfg.tilt_db_per_oct != 0.0:
            tilted = apply_spectral_tilt(head_params.freqs, head_params.gains, cfg.tilt_db_per_oct)
            head_params = ModalParams(freqs=head_params.freqs, decays=head_params.decays, gains=tilted)
        self.head = ModalBank(head_params, cfg.sample_rate)
        self.wires = SnareWires(
            cfg.sample_rate,
            gap_mult=cfg.wire_gap_rms_mult,
            stiffness=cfg.wire_stiffness,
            alpha=cfg.wire_alpha,
            beta=cfg.wire_beta,
        )

        # Smooth "dry" friction path
        self._dry_b, self._dry_a = butter(2, cfg.dry_cutoff_hz / (0.5 * cfg.sample_rate), btype="low")
        self._dry_zi = lfilter_zi(self._dry_b, self._dry_a) * 0.0
        # Level-matching norms (modal vs dry), set by calibration
        self._modal_norm = 1.0
        self._dry_norm = 1.0
        # Voicing EQ
        if cfg.voice_eq:
            sr = cfg.sample_rate
            self._eq_sos = np.array([
                _rbj_highpass(cfg.voice_hp_hz, sr, q=0.7),
                _rbj_peak(950.0, sr, 0.7, cfg.voice_presence_db),
                _rbj_peak(3000.0, sr, 0.6, cfg.voice_cut_db),
                _rbj_highshelf(5000.0, sr, cfg.voice_air_db),
            ], dtype=np.float64)
            self._eq_zi = sosfilt_zi(self._eq_sos) * 0.0
        else:
            self._eq_sos = None
            self._eq_zi = None
        # Slow head-level follower driving the wire compressor
        self._wire_level = 0.0
        # Tap scrub-velocity envelope (attack/decay), shaped from |dp/dt|
        self._splay_env = 0.0
        # Dedicated tap-body resonator
        _bf = np.array([160., 205., 255., 320., 400., 500., 640., 820., 1050.])
        _bd = np.array([0.055, 0.050, 0.045, 0.040, 0.036, 0.032, 0.028, 0.024, 0.020])
        _bg = _normalize_gains_for_white_noise(
            _bf, _bd, np.array([1., 0.95, 0.85, 0.8, 0.72, 0.64, 0.55, 0.46, 0.38]),
            cfg.sample_rate, 0.5)
        self._tap_body = ModalBank(ModalParams(freqs=_bf, decays=_bd, gains=_bg), cfg.sample_rate)
        self._tap_body_rng = np.random.default_rng(0x51)
        self._tap_strike_armed = True
        # Optional second brush (right hand) for two-handed grooves
        self.brush2 = None
        self._splay_env2 = 0.0

        if auto_calibrate:
            self._auto_calibrate()

    _CAL_TARGET_PEAK = 0.5  # target peak for a typical gesture
    _CAL_DURATION_S = 1.5  # length of the calibration gesture

    def _auto_calibrate(self) -> None:
        """Run a short test gesture and scale head_gain."""
        sr = self.config.sample_rate
        n = int(self._CAL_DURATION_S * sr)
        block = 512
        # Medium-velocity circular gesture at pressure 0.6 is representative
        radius = 0.45
        freq = 1.2
        t = np.arange(n) / sr
        px = radius * np.cos(2.0 * np.pi * freq * t)
        py = radius * np.sin(2.0 * np.pi * freq * t)
        vx = -2.0 * np.pi * freq * radius * np.sin(2.0 * np.pi * freq * t)
        vy = 2.0 * np.pi * freq * radius * np.cos(2.0 * np.pi * freq * t)

        prev_vx = prev_vy = 0.0
        prev_x = float(px[0])
        prev_y = float(py[0])
        modal_chunks = []
        dry_chunks = []
        idx = 0
        # Head-only pass
        self._dry_zi = self._dry_zi * 0.0
        try:
            while idx < n:
                frames = min(block, n - idx)
                k = idx + frames - 1
                vx1 = float(vx[k]); vy1 = float(vy[k])
                x1 = float(px[k]); y1 = float(py[k])
                exc = self.brush.render(prev_vx, prev_vy, 0.6, 0.0, vx1, vy1, 0.6, 0.0,
                                        frames, prev_x, prev_y, x1, y1)
                modal_chunks.append(self.head.process(exc))
                dry_chunks.append(self._dry_process(exc))
                prev_vx, prev_vy = vx1, vy1
                prev_x, prev_y = x1, y1
                idx += frames
        finally:
            self.brush.reset()
            self.head.reset()
            self.wires.reset()
            self._dry_zi = self._dry_zi * 0.0

        modal = np.concatenate(modal_chunks)
        dry = np.concatenate(dry_chunks)
        modal_rms = float(np.sqrt(np.mean(modal ** 2)))
        dry_rms = float(np.sqrt(np.mean(dry ** 2)))
        if modal_rms <= 1e-9:
            return
        # Normalize each path to unit RMS so the mix is a clean crossfade
        self._modal_norm = 1.0 / modal_rms
        self._dry_norm = 1.0 / max(dry_rms, 1e-9)
        rm = self.config.resonance_mix
        blended = rm * modal * self._modal_norm + (1.0 - rm) * dry * self._dry_norm
        # Account for the voicing EQ's level change
        if self._eq_sos is not None:
            blended = sosfilt(self._eq_sos, blended)
        blend_peak = float(np.max(np.abs(blended)))
        if blend_peak <= 1e-9:
            return
        self.config.head_gain = self._CAL_TARGET_PEAK / blend_peak
        print(f"[synth] auto-calibrated: head_gain {self.config.head_gain:.4g} "
              f"(resonance_mix {rm:.2f})", flush=True)

    def _dry_process(self, exc: np.ndarray) -> np.ndarray:
        """Smooth low-passed friction path (stateful across blocks)."""
        out, self._dry_zi = lfilter(self._dry_b, self._dry_a, exc, zi=self._dry_zi)
        return out

    def reset(self) -> None:
        self.brush.reset()
        self.head.reset()
        self.wires.reset()
        self._dry_zi = self._dry_zi * 0.0
        self._wire_level = 0.0
        self._splay_env = 0.0
        self._splay_env2 = 0.0
        self._tap_body.reset()
        self._tap_body_rng = np.random.default_rng(0x51)
        self._tap_strike_armed = True
        if self._eq_zi is not None:
            self._eq_zi = self._eq_zi * 0.0
        if self.brush2 is not None:
            self.brush2.reset()

    def render(
        self,
        vx0: float, vy0: float, p0: float, ang0: float,
        vx1: float, vy1: float, p1: float, ang1: float,
        frames: int,
        x0: float = 0.0, y0: float = 0.0, x1: float = 0.0, y1: float = 0.0,
    ) -> np.ndarray:
        """Render frames samples of stereo-compatible mono audio."""
        # A quick pressure change scrubs the bristles
        self._splay_env = self._tap_splay(p0, p1, frames, self._splay_env)
        body_in, tick = self._tap_strike(p0, p1, frames)
        exc = self.brush.render(vx0, vy0, p0, ang0, vx1, vy1, p1, ang1, frames,
                                x0, y0, x1, y1, self._splay_env)
        return self._voice(exc, frames, body_in, tick)

    def _voice(self, exc: np.ndarray, frames: int,
               body_in: np.ndarray | None = None, tick_amp: float = 0.0) -> np.ndarray:
        """Voice a brush excitation through the shared drum body + snare wires."""
        # Blend the ringing modal bank with the smooth dry friction path
        rm = self.config.resonance_mix
        modal = self.head.process(exc) * self._modal_norm
        dry = self._dry_process(exc) * self._dry_norm
        head = (rm * modal + (1.0 - rm) * dry) * self.config.head_gain
        # Level-dependent wire scale
        block_rms = float(np.sqrt(np.mean(head * head))) if frames > 0 else 0.0
        a = 1.0 - np.exp(-frames / (0.25 * self.config.sample_rate))
        self._wire_level += a * (block_rms - self._wire_level)
        ref = self.config.wire_ref_level
        lvl = max(self._wire_level, ref * 0.12)
        wire_scale = float(np.clip((ref / lvl) ** self.config.wire_comp_exp,
                                   0.7, self.config.wire_soft_boost))
        # A strike also rattles the snares
        wire_drive = head
        if body_in is not None and self.config.tap_wire_gain > 0.0:
            wire_drive = head + self.config.tap_wire_gain * body_in
        wires = self.wires.process(wire_drive) * self.config.wires_mix * wire_scale
        raw = (head + wires) * self.config.master_gain
        # Voicing EQ (real-brush band balance) before the limiter
        if self._eq_sos is not None:
            raw, self._eq_zi = sosfilt(self._eq_sos, raw, zi=self._eq_zi)
        # Tap/hit body
        if self.config.tap_body_gain > 0.0:
            bin_ = body_in if body_in is not None else np.zeros(frames)
            raw = raw + self._tap_body.process(bin_) * self.config.tap_body_gain
        # Crisp contact tick (a real hit's attack)
        if tick_amp > 1e-6:
            n = min(int(0.0015 * self.config.sample_rate), frames)
            tick = tick_amp * self._tap_body_rng.standard_normal(n) * \
                np.exp(-np.arange(n) / (0.0008 * self.config.sample_rate))
            raw[:n] = raw[:n] + tick
        # Gentle tanh soft-limiter
        limited = _SOFT_CLIP_KNEE * np.tanh(raw / _SOFT_CLIP_KNEE)
        return limited.astype(np.float32)

    def _tap_strike(self, p0: float, p1: float, frames: int):
        """Detect a strike (fast pressure rise) and return (body_in, tick_amp)."""
        if self.config.tap_body_gain <= 0.0 or frames <= 0:
            return None, 0.0
        rate = abs(p1 - p0) * self.config.sample_rate / frames
        if rate > 8.0 and self._tap_strike_armed:
            hard = min(rate / 40.0, 1.5)
            n = min(int(0.005 * self.config.sample_rate), frames)
            env = np.exp(-np.arange(n) / (0.002 * self.config.sample_rate))
            body_in = np.zeros(frames)
            body_in[:n] = hard * env * self._tap_body_rng.standard_normal(n)
            self._tap_strike_armed = False
            return body_in, self.config.tap_attack_gain * hard
        if rate < 3.0:
            self._tap_strike_armed = True
        return None, 0.0

    def _tap_splay(self, p0: float, p1: float, frames: int, env: float) -> float:
        """Update and return a brush's tap scrub-velocity envelope from |dp/dt|."""
        if self.config.tap_splay <= 0.0 or frames <= 0:
            return 0.0
        rate = min(abs(p1 - p0) * self.config.sample_rate / frames, self.config.tap_rate_cap)
        target = self.config.tap_splay * rate
        env *= float(np.exp(-frames / (self.config.tap_env_ms * 1e-3 * self.config.sample_rate)))
        return max(env, target)

    def render_pair(self, left: tuple, right: tuple, frames: int) -> np.ndarray:
        """Render two brushes into the *same* drum."""
        if self.brush2 is None:
            self.brush2 = BristleBrush(self.brush.params, self.config.sample_rate, seed=7)
        lvx0, lvy0, lp0, la0, lvx1, lvy1, lp1, la1, lx0, ly0, lx1, ly1 = left
        rvx0, rvy0, rp0, ra0, rvx1, rvy1, rp1, ra1, rx0, ry0, rx1, ry1 = right
        self._splay_env = self._tap_splay(lp0, lp1, frames, self._splay_env)
        self._splay_env2 = self._tap_splay(rp0, rp1, frames, self._splay_env2)
        # Taps in a groove come from the right (pattern) hand
        body_in, tick = self._tap_strike(rp0, rp1, frames)
        exc_l = self.brush.render(lvx0, lvy0, lp0, la0, lvx1, lvy1, lp1, la1,
                                  frames, lx0, ly0, lx1, ly1, self._splay_env)
        exc_r = self.brush2.render(rvx0, rvy0, rp0, ra0, rvx1, rvy1, rp1, ra1,
                                   frames, rx0, ry0, rx1, ry1, self._splay_env2)
        return self._voice(exc_l + exc_r, frames, body_in, tick)

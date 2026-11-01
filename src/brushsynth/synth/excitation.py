"""Brush friction excitation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass
class BristleParams:
    """Brush material + geometric parameters (dimensionless / normalized)."""

    n_bristles: int = 128
    fan_radius: float = 1.0  # spatial scale of bristle tips
    sigma0: float = 2000.0  # stick stiffness to slip-rate
    mu: float = 0.4  # kinetic friction coefficient
    pressure_to_load: float = 1.0  # pressure above threshold to bristle force
    output_gain: float = 0.05  # ensemble to modal drive scaling
    activity: float = 0.5  # fraction of bristles actively slipping
    slip_gamma: float = 1.0  # per-slip amplitude exponent (near 1 = smooth)
    # Balance of continuous vs slip
    roughness: float = 1.2  # gain on continuous friction noise
    # Slip component: tiny transient kicks from stick-slip micro-events
    slip_gain: float = 0.06  # gain on slip-state contribution
    # Continuous-friction spectral shaping
    friction_pink: float = 0.25  # 1-pole LP coeff on the noise bed
    # Velocity-scaled HF weight, drives the brightness swish
    bright_coef: float = 0.22
    # Friction-amplitude vs velocity exponent (amp ∝ v^vel_amp_exp)
    vel_amp_exp: float = 1.0
    # Soft knee
    bright_vel_knee: float = 4.0
    # Spatial "coating" field
    coating_amp_depth: float = 0.10  # loudness swell depth (subtle)
    coating_bright_depth: float = 0.45  # brightness swell depth (the slow swish)
    coating_grain_depth: float = 0.0  # fast micro-grain, OFF (it sounds gritty)
    # Slow pressure variation
    macro_rate_hz: float = 0.4  # very slow LFO
    macro_depth: float = 0.10  # small amplitude swing
    macro_noise: float = 0.85  # nearly all noise, almost no sinusoid


def default_nylon_brush() -> BristleParams:
    """Typical nylon jazz brush."""
    return BristleParams(
        n_bristles=128,
        fan_radius=1.0,
        sigma0=2000.0,
        mu=0.4,
        pressure_to_load=1.0,
        output_gain=0.05,
        activity=0.5,
        slip_gamma=1.0,
        roughness=1.2,
        slip_gain=0.06,
        friction_pink=0.25,
        bright_coef=0.22,
        vel_amp_exp=1.0,
        bright_vel_knee=4.0,
        coating_amp_depth=0.10,
        coating_bright_depth=0.45,
        coating_grain_depth=0.0,
        macro_rate_hz=0.4,
        macro_depth=0.05,
        macro_noise=0.85,
    )


def default_wire_brush() -> BristleParams:
    """Typical wire brush: fewer, stiffer bristles."""
    return BristleParams(
        n_bristles=40,
        fan_radius=0.85,
        sigma0=5000.0,
        mu=0.45,
        pressure_to_load=1.0,
        output_gain=0.08,
        activity=0.5,
        slip_gamma=1.0,
        roughness=0.6,
        slip_gain=0.12,
        friction_pink=0.55,
        bright_coef=0.2,
        coating_amp_depth=0.5,
        coating_bright_depth=0.55,
        coating_grain_depth=0.4,
        macro_rate_hz=0.35,
        macro_depth=0.10,
        macro_noise=0.85,
    )


def _make_coating_field(
    n: int, k_lo: float, k_hi: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a band-limited 2-D spatial roughness field as a sum of plane waves."""
    rng = np.random.default_rng(seed)
    kmag = rng.uniform(k_lo, k_hi, size=n)
    kang = rng.uniform(0.0, 2.0 * np.pi, size=n)
    kx = (kmag * np.cos(kang)).astype(np.float64)
    ky = (kmag * np.sin(kang)).astype(np.float64)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float64)
    w = rng.uniform(0.6, 1.0, size=n)
    # RMS of a sum of random-phase sinusoids about sqrt(sum(w^2)/2)
    w = w / np.sqrt(np.sum(w * w) / 2.0 + 1e-12)
    return kx, ky, phase, w.astype(np.float64)


def _fan_layout(n: int, radius: float, seed: int = 1):
    """Return (rest_x, rest_y, engage_threshold) for n bristles."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    rest_x = (r * np.cos(theta)).astype(np.float64)
    rest_y = (r * np.sin(theta)).astype(np.float64)
    r_norm = r / radius
    engage = (r_norm ** 2).astype(np.float64)
    engage = engage * (1.0 + 0.05 * rng.standard_normal(n))
    engage = np.clip(engage, 0.0, 0.99).astype(np.float64)
    return rest_x, rest_y, engage


# Per-sample unconditional state decay
_BASE_DECAY = 0.9995


@njit(cache=True, fastmath=True)
def _eval_field(x: float, y: float,
                kx: np.ndarray, ky: np.ndarray, ph: np.ndarray, w: np.ndarray) -> float:
    """Sum-of-plane-waves spatial field sampled at (x, y)."""
    s = 0.0
    for k in range(kx.shape[0]):
        s += w[k] * np.sin(kx[k] * x + ky[k] * y + ph[k])
    return s


@njit(cache=True, fastmath=True)
def _render_block(
    vx0: float, vy0: float, p0: float, ang0: float,
    x0: float, y0: float,
    vx1: float, vy1: float, p1: float, ang1: float,
    x1: float, y1: float,
    frames: int, sample_rate: float,
    rest_x: np.ndarray,
    rest_y: np.ndarray,
    engage_thresh: np.ndarray,
    state: np.ndarray,
    pink_state: np.ndarray,
    sigma0: float,
    mu: float,
    roughness: float,
    pressure_to_load: float,
    activity: float,
    slip_gamma: float,
    slip_gain: float,
    friction_pink: float,
    bright_coef: float,
    vel_amp_exp: float,
    bright_vel_knee: float,
    splay: float,
    coating_amp_depth: float,
    coating_bright_depth: float,
    coating_grain_depth: float,
    amp_kx: np.ndarray, amp_ky: np.ndarray, amp_ph: np.ndarray, amp_w: np.ndarray,
    bri_kx: np.ndarray, bri_ky: np.ndarray, bri_ph: np.ndarray, bri_w: np.ndarray,
    grn_kx: np.ndarray, grn_ky: np.ndarray, grn_ph: np.ndarray, grn_w: np.ndarray,
    rng_state: np.ndarray,
) -> np.ndarray:
    """Render frames samples of excitation from a linearly-interpolated gesture."""
    M = rest_x.shape[0]
    out = np.zeros(frames, dtype=np.float64)
    inv_fs = 1.0 / sample_rate

    block_duration = frames * inv_fs
    omega_b = (ang1 - ang0) / block_duration if block_duration > 0.0 else 0.0

    # LCG constants (Knuth MMIX), inlined so numba can optimize
    A = np.uint64(6364136223846793005)
    C = np.uint64(1442695040888963407)
    SHIFT = np.uint64(32)
    MASK = np.uint64(0xFFFFFFFF)
    SCALE = 1.0 / 4294967296.0

    rs = rng_state[0]
    # Velocity-colored-noise state
    prev_noise = np.zeros_like(rest_x)

    for t in range(frames):
        alpha = (t + 1.0) / frames
        vx = vx0 + (vx1 - vx0) * alpha
        vy = vy0 + (vy1 - vy0) * alpha
        pressure = p0 + (p1 - p0) * alpha
        ang = ang0 + (ang1 - ang0) * alpha
        px = x0 + (x1 - x0) * alpha
        py = y0 + (y1 - y0) * alpha

        cos_a = np.cos(ang)
        sin_a = np.sin(ang)

        # Sample the spatial coating fields at the current brush position
        v_center = np.sqrt(vx * vx + vy * vy)
        h_amp = _eval_field(px, py, amp_kx, amp_ky, amp_ph, amp_w)
        h_bri = _eval_field(px, py, bri_kx, bri_ky, bri_ph, bri_w)
        h_grn = _eval_field(px, py, grn_kx, grn_ky, grn_ph, grn_w)
        amp_mod = 1.0 + coating_amp_depth * h_amp
        if amp_mod < 0.1:
            amp_mod = 0.1
        # Micro-grain depth grows with sweep speed
        grain_scale = 0.35 + 0.45 * v_center
        if grain_scale > 2.2:
            grain_scale = 2.2
        grain_mod = 1.0 + coating_grain_depth * grain_scale * h_grn
        if grain_mod < 0.1:
            grain_mod = 0.1
        amp_mod *= grain_mod
        # brightness swell in [~0.3, ~1.7]
        bright_mod = 1.0 + coating_bright_depth * h_bri
        if bright_mod < 0.1:
            bright_mod = 0.1

        total = 0.0
        for i in range(M):
            # Global baseline decay (ensures clean silence)
            state[i] *= _BASE_DECAY

            rx = rest_x[i] * cos_a - rest_y[i] * sin_a
            ry = rest_x[i] * sin_a + rest_y[i] * cos_a
            bvx = vx - omega_b * ry
            bvy = vy + omega_b * rx
            v_mag = np.sqrt(bvx * bvx + bvy * bvy)

            thr = engage_thresh[i]
            if pressure <= thr:
                continue
            f_n = (pressure - thr) * pressure_to_load
            if f_n <= 1e-9:
                continue
            f_c = mu * f_n

            # Velocity-dependent slip transient decay
            decay_rate = v_mag * sigma0
            one_minus = decay_rate * inv_fs
            if one_minus > 1.0:
                one_minus = 1.0
            state[i] *= (1.0 - one_minus)

            # Slip-event injection (Bernoulli approximation of Poisson rate)
            slip_rate = v_mag * sigma0 * activity
            p_slip = slip_rate * inv_fs
            if p_slip > 1.0:
                p_slip = 1.0

            rs = rs * A + C
            u = (((rs >> SHIFT) & MASK)) * SCALE
            if u < p_slip:
                rs = rs * A + C
                u2 = (((rs >> SHIFT) & MASK)) * SCALE
                # Raise a uniform to slip_gamma power
                amp_draw = (0.5 + 1.0 * u2) ** slip_gamma
                # Rescale by 1/activity to preserve average power
                state[i] += f_c * amp_draw / max(activity, 1e-3)

            # Slip contribution (small perturbation, scaled down)
            F_i = state[i] * slip_gain

            # Continuous friction noise
            rs = rs * A + C
            u3 = (((rs >> SHIFT) & MASK)) * SCALE - 0.5
            lp = pink_state[i] + (1.0 - friction_pink) * (u3 - pink_state[i])
            pink_state[i] = lp
            du = u3 - prev_noise[i]
            prev_noise[i] = u3
            # Amplitude follows velocity (optionally power-compressed)
            v_amp = v_mag ** vel_amp_exp
            # Brightness follows a soft-saturated velocity
            if bright_vel_knee > 0.0:
                v_bright = bright_vel_knee * np.tanh(v_mag / bright_vel_knee)
            else:
                v_bright = v_mag
            # Warm bed plus an HF channel scaled by velocity
            bright_mix = lp + bright_coef * bright_mod * v_bright * du
            # splay is a tap's bristle-scrub velocity (from dp/dt)
            F_i += roughness * (v_amp + splay) * f_n * bright_mix * 2.0

            total += F_i

        out[t] = total * amp_mod

    rng_state[0] = rs
    return out


class BristleBrush:
    """A virtual brush driving excitation of a membrane resonator bank."""

    def __init__(self, params: BristleParams, sample_rate: float, seed: int = 0) -> None:
        self.params = params
        self.sample_rate = float(sample_rate)
        self._seed = int(seed)
        rest_x, rest_y, engage = _fan_layout(params.n_bristles, params.fan_radius, seed=seed + 1)
        self._rest_x = rest_x
        self._rest_y = rest_y
        self._engage = engage
        self._state = np.zeros(params.n_bristles, dtype=np.float64)
        self._pink_state = np.zeros(params.n_bristles, dtype=np.float64)
        self._rng_state = np.zeros(1, dtype=np.uint64)
        self._seed_rng()

        # Spatial "coating" fields
        self._amp_field = _make_coating_field(8, 1.5, 5.0, seed=seed + 201)
        self._bri_field = _make_coating_field(6, 1.5, 6.0, seed=seed + 202)
        self._grn_field = _make_coating_field(12, 9.0, 26.0, seed=seed + 203)

        # Macro modulator state
        self._macro_rng = np.random.default_rng(seed + 123)
        self._macro_phase = float(self._macro_rng.uniform(0.0, 2.0 * np.pi))
        self._macro_noise_state = 0.0  # leaky-integrated noise (~1 Hz equivalent)
        # Warm JIT
        _render_block(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            8, self.sample_rate,
            self._rest_x, self._rest_y, self._engage, self._state, self._pink_state,
            params.sigma0, params.mu, params.roughness, params.pressure_to_load,
            params.activity, params.slip_gamma, params.slip_gain,
            params.friction_pink, params.bright_coef, params.vel_amp_exp, params.bright_vel_knee,
            0.0,
            params.coating_amp_depth, params.coating_bright_depth, params.coating_grain_depth,
            *self._amp_field, *self._bri_field, *self._grn_field,
            self._rng_state,
        )
        self.reset()

    def _seed_rng(self) -> None:
        # Mask to 64 bits before the uint64 cast
        h = ((self._seed + 1) * 0xC2B2AE3D27D4EB4F) & 0xFFFFFFFFFFFFFFFF
        self._rng_state[0] = np.uint64(h)

    def reset(self) -> None:
        """Return the brush to its just-constructed state."""
        self._state[:] = 0.0
        self._pink_state[:] = 0.0
        self._seed_rng()
        self._macro_rng = np.random.default_rng(self._seed + 123)
        self._macro_phase = float(self._macro_rng.uniform(0.0, 2.0 * np.pi))
        self._macro_noise_state = 0.0

    def _macro_envelope(self, frames: int) -> tuple[float, float]:
        """Return the start and end of the macro gain envelope."""
        p = self.params
        if p.macro_depth <= 0.0:
            return 1.0, 1.0
        dt = frames / self.sample_rate
        # LFO start / end
        s_lfo = np.sin(self._macro_phase)
        self._macro_phase = (self._macro_phase + 2.0 * np.pi * p.macro_rate_hz * dt) % (2.0 * np.pi)
        e_lfo = np.sin(self._macro_phase)
        # Low-pass noise (~1 Hz cutoff)
        s_noise = self._macro_noise_state
        tau = 1.0 / (2.0 * np.pi * 1.0)
        alpha = dt / (tau + dt)
        target = float(self._macro_rng.standard_normal()) * 0.7
        self._macro_noise_state = s_noise + alpha * (target - s_noise)
        e_noise = self._macro_noise_state
        nf = p.macro_noise
        s = 1.0 + p.macro_depth * ((1.0 - nf) * s_lfo + nf * s_noise)
        e = 1.0 + p.macro_depth * ((1.0 - nf) * e_lfo + nf * e_noise)
        lo = max(0.0, 1.0 - p.macro_depth)
        hi = 1.0 + p.macro_depth
        return float(np.clip(s, lo, hi)), float(np.clip(e, lo, hi))

    def render(self, vx0, vy0, p0, ang0, vx1, vy1, p1, ang1, frames: int,
               x0: float = 0.0, y0: float = 0.0, x1: float = 0.0, y1: float = 0.0,
               splay: float = 0.0) -> np.ndarray:
        """Render frames samples of excitation."""
        m0, m1 = self._macro_envelope(frames)
        exc = _render_block(
            float(vx0), float(vy0), float(p0) * m0, float(ang0),
            float(x0), float(y0),
            float(vx1), float(vy1), float(p1) * m1, float(ang1),
            float(x1), float(y1),
            int(frames), self.sample_rate,
            self._rest_x, self._rest_y, self._engage, self._state, self._pink_state,
            self.params.sigma0, self.params.mu,
            self.params.roughness, self.params.pressure_to_load,
            self.params.activity, self.params.slip_gamma, self.params.slip_gain,
            self.params.friction_pink, self.params.bright_coef, self.params.vel_amp_exp,
            self.params.bright_vel_knee, float(splay),
            self.params.coating_amp_depth, self.params.coating_bright_depth,
            self.params.coating_grain_depth,
            *self._amp_field, *self._bri_field, *self._grn_field,
            self._rng_state,
        )
        return exc * self.params.output_gain

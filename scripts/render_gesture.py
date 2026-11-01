"""Offline renderer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from brushsynth.synth.brush_synth import BrushSynth, SynthConfig


SYNTHETIC_GESTURES = {
    "circular_slow": dict(duration=6.0, kind="circle", radius=0.6, freq=0.8, pressure=0.55),
    "circular_fast": dict(duration=6.0, kind="circle", radius=0.5, freq=2.8, pressure=0.7),
    "tap_and_sweep": dict(duration=6.0, kind="tap_sweep"),
    "pressed_release": dict(duration=6.0, kind="pressed_release"),
    "side_accents": dict(duration=6.0, kind="side_accents"),
}


def build_synthetic(name: str, sr: int) -> np.ndarray:
    """Return an (N, 5) array of t, x, y, pressure, angle."""
    spec = SYNTHETIC_GESTURES[name]
    T = spec["duration"]
    n = int(T * sr)
    t = np.linspace(0.0, T, n, endpoint=False)
    x = np.zeros(n)
    y = np.zeros(n)
    p = np.zeros(n)
    a = np.zeros(n)
    kind = spec["kind"]
    if kind == "circle":
        r = spec["radius"]
        f = spec["freq"]
        x = r * np.cos(2.0 * np.pi * f * t)
        y = r * np.sin(2.0 * np.pi * f * t)
        p[:] = spec["pressure"]
        # Soft fade in/out
        env = np.clip(np.minimum(t / 0.3, (T - t) / 0.3), 0.0, 1.0)
        p = p * env
    elif kind == "tap_sweep":
        # Three short taps then a continuous sweep
        p = np.zeros(n)
        for tap_t in (0.4, 0.9, 1.4):
            p += 0.85 * np.exp(-((t - tap_t) / 0.012) ** 2)
        # Sweep from 2.0 s onward: slow spiral outward
        k_sweep = int(2.0 * sr)
        t_s = t[k_sweep:] - 2.0
        r_s = 0.2 + 0.4 * np.clip(t_s / 3.5, 0.0, 1.0)
        x[k_sweep:] = r_s * np.cos(2.0 * np.pi * 0.6 * t_s)
        y[k_sweep:] = r_s * np.sin(2.0 * np.pi * 0.6 * t_s)
        p[k_sweep:] = 0.55 * np.clip(t_s / 0.4, 0.0, 1.0) * np.clip((T - 2.0 - t_s) / 0.4, 0.0, 1.0)
    elif kind == "pressed_release":
        # Hold pressed with subtle motion, then release with a sigh
        x = 0.3 * np.sin(2.0 * np.pi * 0.4 * t)
        y = 0.0 * t
        hold = np.clip(t / 0.5, 0.0, 1.0)
        release_start = 4.5
        release = 1.0 - np.clip((t - release_start) / 0.8, 0.0, 1.0)
        p = 0.9 * hold * release
    elif kind == "side_accents":
        # Side-to-side at moderate rate with three accents
        x = 0.6 * np.sin(2.0 * np.pi * 1.2 * t)
        y = 0.1 * np.sin(2.0 * np.pi * 2.4 * t)
        p = 0.45 * np.ones(n)
        for acc_t in (1.0, 2.5, 4.2):
            k0 = int(acc_t * sr)
            k1 = k0 + int(0.04 * sr)
            p[k0:k1] = 0.95
        env = np.clip(np.minimum(t / 0.3, (T - t) / 0.3), 0.0, 1.0)
        p *= env
    else:
        raise ValueError(f"unknown synthetic gesture '{name}'")
    return np.stack([t, x, y, p, a], axis=1)


def derive_velocity(xy: np.ndarray, sr: int, smoothing: int = 8) -> np.ndarray:
    """Central-difference + box-smooth to produce (vx, vy)."""
    dx = np.gradient(xy[:, 0]) * sr
    dy = np.gradient(xy[:, 1]) * sr
    if smoothing > 1:
        k = np.ones(smoothing) / smoothing
        dx = np.convolve(dx, k, mode="same")
        dy = np.convolve(dy, k, mode="same")
    return np.stack([dx, dy], axis=1)


def render_trace(
    trace: np.ndarray,
    sr: int,
    block: int = 256,
    config: SynthConfig | None = None,
) -> np.ndarray:
    """Render a gesture trace through the synth, block by block."""
    cfg = config or SynthConfig(sample_rate=float(sr))
    synth = BrushSynth(cfg)
    n = trace.shape[0]
    v = derive_velocity(trace[:, 1:3], sr)
    out = np.zeros(n, dtype=np.float32)

    prev_vx = prev_vy = prev_p = prev_ang = 0.0
    prev_x = float(trace[0, 1])
    prev_y = float(trace[0, 2])
    idx = 0
    while idx < n:
        frames = min(block, n - idx)
        k_end = idx + frames - 1
        vx1, vy1 = float(v[k_end, 0]), float(v[k_end, 1])
        x1, y1 = float(trace[k_end, 1]), float(trace[k_end, 2])
        p1 = float(trace[k_end, 3])
        ang1 = float(trace[k_end, 4])
        buf = synth.render(prev_vx, prev_vy, prev_p, prev_ang, vx1, vy1, p1, ang1, frames,
                           prev_x, prev_y, x1, y1)
        out[idx : idx + frames] = buf
        prev_vx, prev_vy, prev_p, prev_ang = vx1, vy1, p1, ang1
        prev_x, prev_y = x1, y1
        idx += frames
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", help="path to .npz with t,x,y,pressure,angle arrays (optional)")
    ap.add_argument("--synthetic", choices=list(SYNTHETIC_GESTURES.keys()),
                    help="generate a canonical evaluation gesture")
    ap.add_argument("-o", "--output", required=True, help="output WAV path")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--block", type=int, default=256)
    args = ap.parse_args()

    if args.synthetic:
        trace = build_synthetic(args.synthetic, args.sr)
    elif args.trace:
        data = np.load(args.trace)
        trace = np.stack([
            data.get("t", np.arange(data["x"].shape[0]) / args.sr),
            data["x"], data["y"],
            data.get("pressure", np.full_like(data["x"], 0.5)),
            data.get("angle", np.zeros_like(data["x"])),
        ], axis=1)
    else:
        ap.error("either --synthetic or a trace file is required")

    audio = render_trace(trace, args.sr, block=args.block)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, args.sr, subtype="PCM_24")
    peak = float(np.max(np.abs(audio)))
    print(f"wrote {args.output}  ({audio.size / args.sr:.2f} s, peak {peak:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

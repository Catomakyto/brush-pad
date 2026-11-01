"""Render two-hand jazz brush grooves."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from brushsynth.synth.brush_synth import BrushSynth, SynthConfig
from brushsynth.synth.presets import build_preset


def _taps(t: np.ndarray, times: list[float], width: float, amp: float,
          accents: dict | None = None) -> np.ndarray:
    """Sum of smooth gaussian pressure pulses at times (seconds)."""
    p = np.zeros_like(t)
    accents = accents or {}
    for i, tt in enumerate(times):
        a = amp * accents.get(i, 1.0)
        p = np.maximum(p, a * np.exp(-((t - tt) / width) ** 2))
    return p


def build_groove(style: str, bpm: float, bars: int, sr: int, swing: float = 0.62):
    """Return (left_trace, right_trace), each (N,5): t,x,y,pressure,angle."""
    beat = 60.0 / bpm
    total = bars * 4 * beat
    n = int(total * sr)
    t = np.linspace(0.0, total, n, endpoint=False)

    lx = np.zeros(n); ly = np.zeros(n); lp = np.zeros(n)
    rx = np.zeros(n); ry = np.zeros(n); rp = np.zeros(n)

    if style == "ballad":
        # LEFT: slow circular stir, ~1 revolution every 2 beats, continuous
        f = 1.0 / (2.0 * beat)
        lx = 0.55 * np.cos(2 * np.pi * f * t)
        ly = 0.55 * np.sin(2 * np.pi * f * t)
        lp[:] = 0.5
        # RIGHT: soft quarter-note taps, slight motion
        times = [b * beat for b in range(bars * 4)]
        rp = _taps(t, times, width=0.020, amp=0.55,
                   accents={i: 1.25 for i in range(0, bars * 4, 2)})  # gentle 1&3
        rx = 0.12 * np.sin(2 * np.pi * (1.0 / beat) * t)

    elif style in ("medium_swing", "up_swing"):
        # LEFT: horizontal back-and-forth "smear"
        f = 1.0 / beat
        lx = 0.5 * np.sin(2 * np.pi * f * t)
        ly = 0.12 * np.sin(2 * np.pi * 2 * f * t)  # slight figure-8
        lp[:] = 0.42 if style == "medium_swing" else 0.36
        # RIGHT: swing ride rhythm
        times = []
        accents = {}
        idx = 0
        for bar in range(bars):
            base = bar * 4 * beat
            for b in (0, 1, 2, 3):
                times.append(base + b * beat)
                if b in (0, 2):
                    accents[idx] = 1.2  # beats 1 & 3 a touch stronger
                idx += 1
                if b in (1, 3):  # swung eighth after 2 and 4
                    times.append(base + (b + swing) * beat)
                    idx += 1
        amp = 0.6 if style == "medium_swing" else 0.5
        w = 0.016 if style == "medium_swing" else 0.012
        rp = _taps(t, times, width=w, amp=amp, accents=accents)
    else:
        raise ValueError(f"unknown style '{style}'")

    z = np.zeros(n)
    left = np.stack([t, lx, ly, lp, z], axis=1)
    right = np.stack([t, rx, ry, rp, z], axis=1)
    return left, right


def _vel(xy: np.ndarray, sr: int, smooth: int = 8) -> np.ndarray:
    dx = np.gradient(xy[:, 0]) * sr
    dy = np.gradient(xy[:, 1]) * sr
    k = np.ones(smooth) / smooth
    return np.stack([np.convolve(dx, k, "same"), np.convolve(dy, k, "same")], axis=1)


def render_groove(left: np.ndarray, right: np.ndarray, sr: int,
                  preset: str | None = None, block: int = 256) -> np.ndarray:
    synth = build_preset(preset, sr) if preset else BrushSynth(SynthConfig(sample_rate=float(sr)))
    n = left.shape[0]
    lv = _vel(left[:, 1:3], sr)
    rv = _vel(right[:, 1:3], sr)
    out = np.zeros(n, dtype=np.float32)
    lpx = float(left[0, 1]); lpy = float(left[0, 2])
    rpx = float(right[0, 1]); rpy = float(right[0, 2])
    lvx = lvy = lp = 0.0
    rvx = rvy = rp = 0.0
    idx = 0
    while idx < n:
        fr = min(block, n - idx); k = idx + fr - 1
        L = (lvx, lvy, lp, 0.0, float(lv[k, 0]), float(lv[k, 1]), float(left[k, 3]), 0.0,
             lpx, lpy, float(left[k, 1]), float(left[k, 2]))
        R = (rvx, rvy, rp, 0.0, float(rv[k, 0]), float(rv[k, 1]), float(right[k, 3]), 0.0,
             rpx, rpy, float(right[k, 1]), float(right[k, 2]))
        out[idx:idx + fr] = synth.render_pair(L, R, fr)
        lvx, lvy, lp = float(lv[k, 0]), float(lv[k, 1]), float(left[k, 3])
        rvx, rvy, rp = float(rv[k, 0]), float(rv[k, 1]), float(right[k, 3])
        lpx, lpy = float(left[k, 1]), float(left[k, 2])
        rpx, rpy = float(right[k, 1]), float(right[k, 2])
        idx += fr
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="medium_swing",
                    choices=["ballad", "medium_swing", "up_swing"])
    ap.add_argument("--preset", default=None, help="voicing (thigpen/papa_jo/hamilton/wire)")
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--bars", type=int, default=4)
    ap.add_argument("--swing", type=float, default=0.63)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    bpm = args.bpm or {"ballad": 62.0, "medium_swing": 132.0, "up_swing": 184.0}[args.style]
    left, right = build_groove(args.style, bpm, args.bars, args.sr, swing=args.swing)
    audio = render_groove(left, right, args.sr, preset=args.preset)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, args.sr, subtype="PCM_24")
    print(f"wrote {args.output}  ({audio.size / args.sr:.1f}s, {args.style} @ {bpm:.0f} BPM, "
          f"preset={args.preset or 'default'}, peak {float(np.max(np.abs(audio))):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Render a scripted brush performance to a WAV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brushsynth.synth.grain_brush import GrainBrushSynth  # noqa: E402

SR, BLOCK = 48000, 256


def performance(sr: int):
    """Yield per-block (speed, pressure) for the scripted sections."""
    def section(dur, speed_fn, press_fn):
        n = int(dur * sr / BLOCK)
        for i in range(n):
            t = i * BLOCK / sr
            yield speed_fn(t, dur), press_fn(t, dur)

    def gap(dur=0.6):
        yield from section(dur, lambda t, d: 0.0, lambda t, d: 0.0)

    # A stir: the hand circles at ~1 Hz
    def stir(v):
        return lambda t, d: v * (0.75 + 0.25 * np.cos(4 * np.pi * t))

    yield from section(4, stir(2.0), lambda t, d: 0.15)
    yield from gap()
    yield from section(4, stir(3.5), lambda t, d: 0.45)
    yield from gap()
    yield from section(4, stir(4.5), lambda t, d: 0.85)
    yield from gap()
    # Sweeps: quick back-and-forth strokes
    yield from section(4, lambda t, d: 7.0 * abs(np.sin(2 * np.pi * t)), lambda t, d: 0.55)
    yield from gap()
    yield from section(6, lambda t, d: 0.5 + 6.0 * t / d, lambda t, d: 0.1 + 0.8 * t / d)
    yield from gap(1.0)


def render(synth) -> np.ndarray:
    out, prev = [], (0.0, 0.0)
    for sp, p in performance(SR):
        out.append(synth.render(prev[0], 0.0, prev[1], 0.0, sp, 0.0, p, 0.0, BLOCK))
        prev = (sp, p)
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="evals/grain/demo.wav")
    ap.add_argument("--play", action="store_true")
    args = ap.parse_args()

    y = render(GrainBrushSynth())
    import soundfile as sf
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, SR)
    print(f"wrote {path}  ({len(y) / SR:.1f} s, peak {np.abs(y).max():.2f})")
    if args.play:
        import sounddevice as sd
        sd.play(y, SR)
        sd.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

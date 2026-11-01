"""Render the evaluation gestures through the current synth."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import soundfile as sf

from brushsynth.analysis.texture_stats import analyze_texture, texture_distance, TARGETS
from brushsynth.synth.brush_synth import BrushSynth, SynthConfig

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_gesture import build_synthetic, render_trace  # noqa: E402

SR = 48000
GESTURES = ["circular_slow", "circular_fast", "tap_and_sweep", "pressed_release", "side_accents"]


def _load(path: str, target_sr: int = SR) -> np.ndarray:
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64)
    if sr != target_sr:
        n = int(len(y) * target_sr / sr)
        y = np.interp(np.linspace(0, len(y), n, endpoint=False), np.arange(len(y)), y)
    return y


def _ref_row() -> None:
    files = sorted(glob.glob("recordings/brush_ref_karoryfer/stir_dl1_rr*.wav"))
    if not files:
        return
    y = np.concatenate([_load(p) for p in files[:3]])
    s = analyze_texture(y, SR)
    print(f"{'REAL stir (ref)':16} flat{s.flatness:.3f} cent{s.centroid:5.0f} "
          f"centCV{s.centroid_cv:.3f} crest{s.crest_db:4.1f} roll{s.rolloff85:5.0f} "
          f"envcv{s.env_cv:.2f} modpk{s.env_mod_peak_hz:4.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default=None, help="eval version dir under evals/our/<name>/")
    args = ap.parse_args()

    tgt = {k: v[0] for k, v in TARGETS.items()}
    print(f"TARGET  flat{tgt['flatness']:.3f} cent{tgt['centroid']:5.0f} "
          f"centCV{tgt['centroid_cv']:.3f} crest{tgt['crest_db']:4.1f} roll{tgt['rolloff85']:5.0f} "
          f"envcv{tgt['env_cv']:.2f}")
    _ref_row()
    print("-" * 96)

    outdir = None
    if args.write:
        outdir = Path("evals/our") / args.write
        outdir.mkdir(parents=True, exist_ok=True)

    dists = []
    for g in GESTURES:
        audio = render_trace(build_synthetic(g, SR), SR)
        y = audio.astype(np.float64)
        s = analyze_texture(y, SR)
        d = texture_distance(s)
        dists.append(d)
        print(f"{g:16} flat{s.flatness:.3f} cent{s.centroid:5.0f} centCV{s.centroid_cv:.3f} "
              f"crest{s.crest_db:4.1f} roll{s.rolloff85:5.0f} envcv{s.env_cv:.2f} "
              f"modpk{s.env_mod_peak_hz:4.1f}  D{d:5.1f}")
        if outdir is not None:
            sf.write(str(outdir / f"{g}.wav"), audio, SR, subtype="PCM_24")

    print("-" * 96)
    print(f"mean texture distance (continuous gestures): "
          f"{np.mean([dists[i] for i, g in enumerate(GESTURES) if g in ('circular_slow','circular_fast','side_accents')]):.2f}")
    if outdir is not None:
        print(f"wrote WAVs → {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

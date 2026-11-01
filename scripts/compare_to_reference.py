"""Compare synth output spectra to reference brush recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from brushsynth.analysis.ref_stats import analyze


def _load(path: str) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float64), int(sr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default="recordings/brush_ref",
                    help="directory of reference WAV/MP3 files")
    ap.add_argument("--synth-dir", default="evals/our/v2_brush_weighted",
                    help="directory of synth-rendered WAV files to compare")
    ap.add_argument("--plot", default=None, help="optional PNG output of LTAS overlay")
    args = ap.parse_args()

    ref_dir = Path(args.ref_dir)
    syn_dir = Path(args.synth_dir)

    refs = []
    for p in sorted(ref_dir.iterdir()):
        if p.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
            continue
        y, sr = _load(str(p))
        if y.size == 0:
            continue
        stats = analyze(y, sr)
        refs.append((p.name, stats))
        print(f"REF  {p.name:35}  tilt={stats.spectral_tilt:+5.1f} dB/oct   crest={stats.crest_db:4.1f} dB   rms={stats.rms:.3f}")

    syns = []
    for p in sorted(syn_dir.iterdir()):
        if p.suffix.lower() != ".wav":
            continue
        y, sr = _load(str(p))
        stats = analyze(y, sr)
        syns.append((p.name, stats))
        print(f"SYN  {p.name:35}  tilt={stats.spectral_tilt:+5.1f} dB/oct   crest={stats.crest_db:4.1f} dB   rms={stats.rms:.3f}")

    print()
    print(f"{'synth':35}  " + "   ".join(r[0][:18] for r in refs))
    for sn, ss in syns:
        dists = [f"{ss.distance(rs):18.3f}" for _, rs in refs]
        print(f"{sn:35}  " + "".join(dists))

    print()
    # Simple aggregate
    for sn, ss in syns:
        avg = float(np.mean([ss.distance(rs) for _, rs in refs]))
        print(f"  {sn:35}  avg dist = {avg:.3f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[warn] matplotlib unavailable: {e}")
            return 0
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, stats in refs:
            ax.semilogx(stats.band_centers, stats.ltas_db, label=f"REF {name}", alpha=0.8)
        for name, stats in syns:
            ax.semilogx(stats.band_centers, stats.ltas_db, "--", label=f"SYN {name}", alpha=0.6)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("LTAS (dB, peak-normalized)")
        ax.set_title("Long-term average spectrum: reference vs synth")
        ax.set_ylim(-60, 3)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="lower center", ncol=2)
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=110)
        print(f"wrote {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

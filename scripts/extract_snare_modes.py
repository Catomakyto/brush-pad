"""Extract a modal bank from one or more snare impulse-response recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from brushsynth.analysis.modal_extract import ExtractedMode, extract_modes


def _trim_impulse(audio: np.ndarray, sr: int, *, pre_ms: float = 2.0, post_ms: float = 300.0) -> np.ndarray:
    """Crop to a window around the strongest onset."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)
    win = max(32, int(0.002 * sr))  # 2 ms envelope window
    kernel = np.ones(win) / win
    energy = np.convolve(audio ** 2, kernel, mode="same")
    peak = int(np.argmax(energy))
    start = max(0, peak - int(pre_ms * 1e-3 * sr))
    end = min(audio.size, peak + int(post_ms * 1e-3 * sr))
    return audio[start:end]


def _merge_modes(
    modes_per_file: list[list[ExtractedMode]],
    freq_tolerance: float = 0.005,
) -> list[ExtractedMode]:
    """Merge mode lists from multiple recordings."""
    flat = [m for lst in modes_per_file for m in lst]
    if not flat:
        return []
    flat.sort(key=lambda m: m.freq)
    clusters: list[list[ExtractedMode]] = [[flat[0]]]
    for m in flat[1:]:
        cluster = clusters[-1]
        ref = cluster[0].freq
        if abs(m.freq - ref) / max(ref, 1.0) <= freq_tolerance:
            cluster.append(m)
        else:
            clusters.append([m])
    merged: list[ExtractedMode] = []
    for cluster in clusters:
        ws = np.array([m.amplitude for m in cluster])
        total = float(np.sum(ws))
        if total <= 0.0:
            merged.append(cluster[0])
            continue
        freq = float(np.sum(np.array([m.freq for m in cluster]) * ws) / total)
        decay = float(np.sum(np.array([m.decay for m in cluster]) * ws) / total)
        amp = float(total / len(cluster))
        merged.append(ExtractedMode(freq=freq, decay=decay, amplitude=amp, phase=0.0))
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+", help="one or more snare IR WAV files")
    ap.add_argument("--max-modes", type=int, default=96,
                    help="cap on total modes retained (kept by descending amplitude)")
    ap.add_argument("--out", default="models/snare_modes.npz", help="output NPZ path")
    ap.add_argument("--energy-fraction", type=float, default=0.995,
                    help="per-file: keep the smallest set of modes carrying this energy fraction")
    ap.add_argument("--pencil-modes", type=int, default=None,
                    help="pencil rank bound (default: 2 * --max-modes)")
    ap.add_argument("--brush-weighting", action="store_true", default=True,
                    help="rebalance gains across log-frequency octaves (default: True). "
                         "Stick-hit IRs over-weight low modes; brushes excite modes more "
                         "uniformly, so we flatten the gain-vs-frequency slope.")
    ap.add_argument("--no-brush-weighting", dest="brush_weighting", action="store_false",
                    help="keep the raw stick-hit-derived gains (for research comparison)")
    args = ap.parse_args()

    wav_paths = [Path(p) for p in args.wavs]
    for p in wav_paths:
        if not p.is_file():
            ap.error(f"not a file: {p}")

    pencil_max = args.pencil_modes or (args.max_modes * 2)
    all_modes: list[list[ExtractedMode]] = []
    ref_sr: int | None = None
    for p in wav_paths:
        audio, sr = sf.read(str(p))
        if ref_sr is None:
            ref_sr = int(sr)
        elif sr != ref_sr:
            # Keep it simple: require a consistent sample rate
            print(f"[warn] {p.name} has sample rate {sr}, expected {ref_sr}; skipping")
            continue
        ir = _trim_impulse(audio, sr)
        print(f"[extract] {p.name}: {ir.size / sr:.2f}s after trim", flush=True)
        modes = extract_modes(ir, sr, max_modes=pencil_max, energy_fraction=args.energy_fraction)
        # Drop unrealistic modes
        modes = [m for m in modes if 20.0 < m.freq < 0.48 * sr and m.decay > 0.01]
        print(f"[extract] {p.name}: {len(modes)} modes retained")
        all_modes.append(modes)

    merged = _merge_modes(all_modes)
    merged.sort(key=lambda m: -m.amplitude)
    merged = merged[: args.max_modes]
    # Sort by frequency for a tidy output file
    merged.sort(key=lambda m: m.freq)
    print(f"[merge] {sum(len(l) for l in all_modes)} input modes -> {len(merged)} output modes")

    freqs = np.array([m.freq for m in merged], dtype=np.float64)
    decays = np.array([m.decay for m in merged], dtype=np.float64)
    gains = np.array([m.amplitude for m in merged], dtype=np.float64)

    if args.brush_weighting:
        # A stick hit strongly excites low fundamentals
        lo_edges = 2 ** np.arange(np.floor(np.log2(max(freqs.min(), 20.0))),
                                  np.ceil(np.log2(freqs.max())) + 1)
        for i in range(len(lo_edges) - 1):
            mask = (freqs >= lo_edges[i]) & (freqs < lo_edges[i + 1])
            if not mask.any():
                continue
            band_rms = float(np.sqrt(np.mean(gains[mask] ** 2)))
            if band_rms > 0.0:
                gains[mask] = gains[mask] / band_rms  # unit-RMS per octave

    if gains.max() > 0:
        gains = gains / gains.max()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, freqs=freqs, decays=decays, gains=gains, sample_rate=float(ref_sr))
    print(f"[write] {out_path}  ({freqs.size} modes)")
    # Quick summary of dominant modes
    for i, m in enumerate(merged[:8]):
        print(f"    mode {i}:  f={m.freq:7.1f} Hz   decay={m.decay:.3f} s   amp={m.amplitude:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

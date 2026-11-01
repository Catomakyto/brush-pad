"""Target-match optimizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import differential_evolution, minimize

from brushsynth.analysis.ref_stats import RefStats, analyze
from brushsynth.synth.brush_synth import BrushSynth, SynthConfig
from brushsynth.synth.excitation import BristleParams, default_nylon_brush


def _render_circle(synth: BrushSynth, duration_s: float = 3.0, seed_offset: int = 0) -> np.ndarray:
    """Render a canonical circular-sweep gesture."""
    sr = int(synth.config.sample_rate)
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    freq = 1.2 + 0.1 * seed_offset  # slight variation
    r = 0.5
    vx = -2.0 * np.pi * freq * r * np.sin(2.0 * np.pi * freq * t)
    vy = 2.0 * np.pi * freq * r * np.cos(2.0 * np.pi * freq * t)
    out = np.zeros(n, dtype=np.float32)
    prev_vx = prev_vy = 0.0
    block = 512
    i = 0
    while i < n:
        k = min(block, n - i)
        ke = i + k - 1
        buf = synth.render(prev_vx, prev_vy, 0.6, 0.0,
                           float(vx[ke]), float(vy[ke]), 0.6, 0.0, k)
        out[i : i + k] = buf
        prev_vx, prev_vy = float(vx[ke]), float(vy[ke])
        i += k
    # Skip the first ~10 ms of warm-up
    return out[int(0.01 * sr):]


def _load_refs(ref_dir: Path) -> list[RefStats]:
    refs: list[RefStats] = []
    for p in sorted(ref_dir.iterdir()):
        if p.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
            continue
        # Skip duplicates (wav version of an mp3)
        if p.suffix.lower() == ".wav" and (p.with_suffix(".mp3")).is_file():
            continue
        y, sr = sf.read(str(p))
        if y.ndim > 1:
            y = y.mean(axis=1)
        refs.append(analyze(y.astype(np.float64), sr))
        print(f"[ref]  {p.name:35}  tilt={refs[-1].spectral_tilt:+4.1f} dB/oct  crest={refs[-1].crest_db:4.1f} dB")
    return refs


# Parameter vector
BOUNDS = np.array([
    [500.0,  20000.0],
    [0.12,   0.8],  # activity, floor well above "saw" regime
    [0.8,    1.8],  # slip_gamma
    [0.0,    0.35],  # roughness
    [-4.0,   14.0],  # tilt_db_per_oct
    [0.0,    1.5],  # wires_mix
])

INIT_X = np.array([3000.0, 0.3, 1.2, 0.12, 6.0, 0.6])


def _build_synth(x: np.ndarray, sample_rate: int) -> BrushSynth:
    sigma0, activity, slip_gamma, roughness, tilt, wires_mix = x
    bp = default_nylon_brush()
    bp.sigma0 = float(sigma0)
    bp.activity = float(activity)
    bp.slip_gamma = float(slip_gamma)
    bp.roughness = float(roughness)
    cfg = SynthConfig(sample_rate=float(sample_rate), wires_mix=float(wires_mix), tilt_db_per_oct=float(tilt))
    return BrushSynth(cfg, brush_params=bp)


def _loss(x: np.ndarray, refs: list[RefStats], sample_rate: int, seeds: int) -> float:
    # Reflect out-of-bounds into the interior
    xc = np.clip(x, BOUNDS[:, 0], BOUNDS[:, 1])
    penalty = 3.0 * float(np.sum(np.maximum(BOUNDS[:, 0] - x, 0) + np.maximum(x - BOUNDS[:, 1], 0)))
    try:
        synth = _build_synth(xc, sample_rate)
    except Exception as e:
        return 1e3 + penalty
    losses = []
    for s in range(seeds):
        synth.reset()
        # Longer render so the macro LFO (~1.8 Hz) completes several cycles
        out = _render_circle(synth, duration_s=5.0, seed_offset=s)
        if not np.all(np.isfinite(out)):
            return 1e3 + penalty
        st = analyze(out.astype(np.float64), sample_rate)
        losses.append(float(np.mean([st.distance(r) for r in refs])))
    return float(np.mean(losses)) + penalty


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default="recordings/brush_ref")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=2,
                    help="render this many circles per evaluation and average their loss")
    ap.add_argument("--out", default="models/brush_params.npz")
    args = ap.parse_args()

    refs = _load_refs(Path(args.ref_dir))
    if not refs:
        ap.error(f"no reference audio found in {args.ref_dir}")

    # Evaluate initial guess
    init_loss = _loss(INIT_X, refs, args.sr, args.seeds)
    print(f"[init] loss = {init_loss:.3f}  x = {INIT_X}")

    best = {"x": INIT_X.copy(), "loss": init_loss, "n": 0}
    def de_cb(xk, convergence):
        best["n"] += 1
        if best["n"] % 5 == 0:
            l = _loss(xk, refs, args.sr, 1)
            print(f"  de iter {best['n']:3d}  loss={l:.3f}  x={np.round(xk, 4)}")
        return False

    # Differential evolution explores aggressively
    de = differential_evolution(
        _loss, BOUNDS, args=(refs, args.sr, args.seeds),
        maxiter=args.iters, popsize=10, tol=0.01, mutation=(0.5, 1.2),
        recombination=0.8, seed=42, polish=False, updating="deferred",
        workers=1, init="sobol", callback=de_cb,
    )
    x_de = np.clip(de.x, BOUNDS[:, 0], BOUNDS[:, 1])
    print(f"[de] best x = {x_de}  loss = {de.fun:.3f}")

    # Nelder-Mead polish on the DE result
    res = minimize(
        _loss, x_de, args=(refs, args.sr, args.seeds + 1),
        method="Nelder-Mead",
        options={"maxiter": 30, "xatol": 0.01, "fatol": 0.03, "adaptive": True},
    )
    x_best = np.clip(res.x, BOUNDS[:, 0], BOUNDS[:, 1])
    final_loss = _loss(x_best, refs, args.sr, max(args.seeds, 4))
    print()
    print(f"[done] init loss {init_loss:.3f}  ->  final loss {final_loss:.3f}")
    print(f"[done] best x = {x_best}")

    # Render one final synth and show stats vs references
    synth = _build_synth(x_best, args.sr)
    out = _render_circle(synth, duration_s=4.0)
    st = analyze(out.astype(np.float64), args.sr)
    print(f"[final-synth]  tilt={st.spectral_tilt:+4.1f} dB/oct  crest={st.crest_db:.1f} dB  rms={st.rms:.3f}")
    for r in refs:
        print(f"  vs ref (tilt={r.spectral_tilt:+4.1f}, crest={r.crest_db:.1f}):  dist={st.distance(r):.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        sigma0=x_best[0], activity=x_best[1], slip_gamma=x_best[2],
        roughness=x_best[3], tilt_db_per_oct=x_best[4], wires_mix=x_best[5],
        final_loss=final_loss, init_loss=init_loss,
    )
    # Also write a JSON twin for human readability
    (out_path.with_suffix(".json")).write_text(json.dumps({
        "sigma0": float(x_best[0]),
        "activity": float(x_best[1]),
        "slip_gamma": float(x_best[2]),
        "roughness": float(x_best[3]),
        "tilt_db_per_oct": float(x_best[4]),
        "wires_mix": float(x_best[5]),
        "final_loss": float(final_loss),
        "init_loss": float(init_loss),
    }, indent=2))
    print(f"wrote {out_path} + {out_path.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

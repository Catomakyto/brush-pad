"""Multi-feature target-match optimizer (texture-aware)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import differential_evolution, minimize

from brushsynth.analysis.ref_stats import RefStats, analyze as analyze_ltas
from brushsynth.analysis.texture_stats import analyze_texture, texture_distance
from brushsynth.synth.brush_synth import BrushSynth, SynthConfig
from brushsynth.synth.excitation import default_nylon_brush

# Continuous-texture gestures the objective is evaluated on
GESTURES = ["circular_slow", "circular_fast", "side_accents"]


# Parameter vector (name, lo, hi)
PARAMS = [
    ("sigma0",               800.0, 6000.0),
    ("activity",             0.12,  0.80),
    ("roughness",            0.20,  2.50),
    ("friction_pink",        0.10,  0.85),
    ("bright_coef",          0.05,  0.50),
    ("coating_amp_depth",    0.00,  0.55),
    ("coating_bright_depth", 0.00,  1.20),
    ("coating_grain_depth",  0.00,  0.55),
    ("tilt_db_per_oct",     -2.00,  5.00),
    ("wires_mix",            0.00,  3.00),
    ("wire_gap_rms_mult",    0.80,  3.00),
]
BOUNDS = np.array([[lo, hi] for _, lo, hi in PARAMS])
NAMES = [n for n, _, _ in PARAMS]
# Start from the hand-tuned point found by the diagnostic sweeps
INIT_X = np.array([2000.0, 0.45, 1.1, 0.30, 0.25, 0.20, 0.70, 0.25, 2.5, 1.0, 1.5])


def _build_synth(x: np.ndarray, sr: int) -> BrushSynth:
    d = dict(zip(NAMES, x))
    bp = default_nylon_brush()
    bp.sigma0 = float(d["sigma0"])
    bp.activity = float(d["activity"])
    bp.roughness = float(d["roughness"])
    bp.friction_pink = float(np.clip(d["friction_pink"], 0.0, 0.95))
    bp.bright_coef = float(d["bright_coef"])
    bp.coating_amp_depth = float(d["coating_amp_depth"])
    bp.coating_bright_depth = float(d["coating_bright_depth"])
    bp.coating_grain_depth = float(d["coating_grain_depth"])
    cfg = SynthConfig(
        sample_rate=float(sr),
        wires_mix=float(d["wires_mix"]),
        tilt_db_per_oct=float(d["tilt_db_per_oct"]),
        wire_gap_rms_mult=float(d["wire_gap_rms_mult"]),
    )
    return BrushSynth(cfg, brush_params=bp)


def _synthetic_trace(name: str, sr: int) -> np.ndarray:
    # Local copy of the canonical gestures
    T = 4.0
    n = int(T * sr)
    t = np.linspace(0.0, T, n, endpoint=False)
    if name == "circular_slow":
        r, f, p = 0.6, 0.8, 0.55
        x = r * np.cos(2 * np.pi * f * t); y = r * np.sin(2 * np.pi * f * t)
        pr = np.full(n, p)
    elif name == "circular_fast":
        r, f, p = 0.5, 2.8, 0.7
        x = r * np.cos(2 * np.pi * f * t); y = r * np.sin(2 * np.pi * f * t)
        pr = np.full(n, p)
    elif name == "side_accents":
        x = 0.6 * np.sin(2 * np.pi * 1.2 * t); y = 0.1 * np.sin(2 * np.pi * 2.4 * t)
        pr = np.full(n, 0.45)
    else:
        raise ValueError(name)
    env = np.clip(np.minimum(t / 0.3, (T - t) / 0.3), 0.0, 1.0)
    pr = pr * env
    return np.stack([t, x, y, pr, np.zeros(n)], axis=1)


def _render(synth: BrushSynth, trace: np.ndarray, sr: int) -> np.ndarray:
    n = trace.shape[0]
    dx = np.gradient(trace[:, 1]) * sr
    dy = np.gradient(trace[:, 2]) * sr
    k = np.ones(8) / 8
    dx = np.convolve(dx, k, mode="same"); dy = np.convolve(dy, k, mode="same")
    out = np.zeros(n, dtype=np.float32)
    pvx = pvy = pp = pa = 0.0
    px = float(trace[0, 1]); py = float(trace[0, 2])
    idx = 0
    block = 256
    while idx < n:
        fr = min(block, n - idx); ke = idx + fr - 1
        out[idx:idx+fr] = synth.render(
            pvx, pvy, pp, pa,
            float(dx[ke]), float(dy[ke]), float(trace[ke, 3]), float(trace[ke, 4]),
            fr, px, py, float(trace[ke, 1]), float(trace[ke, 2]))
        pvx, pvy = float(dx[ke]), float(dy[ke])
        pp, pa = float(trace[ke, 3]), float(trace[ke, 4])
        px, py = float(trace[ke, 1]), float(trace[ke, 2])
        idx += fr
    return out[int(0.2 * sr):]  # drop warm-up


def make_loss(refs: list[RefStats], sr: int, ltas_weight: float = 0.4, verbose: bool = True):
    traces = {g: _synthetic_trace(g, sr) for g in GESTURES}
    state = {"n": 0, "best": np.inf}

    def loss(x: np.ndarray) -> float:
        xc = np.clip(x, BOUNDS[:, 0], BOUNDS[:, 1])
        penalty = 5.0 * float(np.sum(np.maximum(BOUNDS[:, 0] - x, 0) + np.maximum(x - BOUNDS[:, 1], 0)))
        try:
            synth = _build_synth(xc, sr)
        except Exception:
            return 1e3 + penalty
        ds = []
        for g in GESTURES:
            synth.reset()
            y = _render(synth, traces[g], sr).astype(np.float64)
            if not np.all(np.isfinite(y)) or np.max(np.abs(y)) < 1e-6:
                return 1e3 + penalty
            ts = analyze_texture(y, sr)
            td = texture_distance(ts)
            # small LTAS anchor (mean L1 to refs)
            ls = analyze_ltas(y, sr)
            ltas = float(np.mean([
                np.mean(np.abs(ls.ltas_db[:min(ls.ltas_db.size, r.ltas_db.size)]
                               - r.ltas_db[:min(ls.ltas_db.size, r.ltas_db.size)]))
                for r in refs]))
            ds.append(td + ltas_weight * ltas)
        val = float(np.mean(ds)) + penalty
        state["n"] += 1
        if val < state["best"]:
            state["best"] = val
        if verbose and state["n"] % 25 == 0:
            print(f"  eval {state['n']:4d}  cur={val:.3f}  best={state['best']:.3f}", flush=True)
        return val

    return loss


def _load_refs(ref_dir: Path, sr: int) -> list[RefStats]:
    refs = []
    for p in sorted(ref_dir.glob("*")):
        if p.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
            continue
        if p.suffix.lower() == ".wav" and p.with_suffix(".mp3").is_file():
            continue
        y, s = sf.read(str(p))
        if y.ndim > 1:
            y = y.mean(axis=1)
        refs.append(analyze_ltas(y.astype(np.float64), s))
    return refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default="recordings/brush_ref_karoryfer")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--out", default="models/brush_params.npz")
    ap.add_argument("--no-de", action="store_true", help="skip global DE, just Nelder-Mead from defaults")
    args = ap.parse_args()

    refs = _load_refs(Path(args.ref_dir), args.sr)
    if not refs:
        ap.error(f"no reference audio in {args.ref_dir}")
    loss = make_loss(refs, args.sr)

    init_loss = loss(INIT_X)
    print(f"[init] loss={init_loss:.3f}  x={dict(zip(NAMES, np.round(INIT_X,3)))}", flush=True)

    x0 = INIT_X.copy()
    if not args.no_de:
        n_eval = {"n": 0}
        def cb(xk, convergence):
            n_eval["n"] += 1
            if n_eval["n"] % 4 == 0:
                print(f"  de {n_eval['n']:3d}  loss={loss(xk):.3f}", flush=True)
            return False
        de = differential_evolution(
            loss, BOUNDS, maxiter=args.iters, popsize=6, tol=0.005,
            mutation=(0.5, 1.2), recombination=0.8, seed=7, polish=False,
            updating="immediate", workers=1, init="sobol", callback=cb, x0=INIT_X,
        )
        x0 = np.clip(de.x, BOUNDS[:, 0], BOUNDS[:, 1])
        print(f"[de] loss={de.fun:.3f}  x={dict(zip(NAMES, np.round(x0,3)))}", flush=True)

    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"maxiter": 200, "xatol": 0.01, "fatol": 0.02, "adaptive": True})
    x_best = np.clip(res.x, BOUNDS[:, 0], BOUNDS[:, 1])
    final = loss(x_best)
    print(f"\n[done] init {init_loss:.3f} -> final {final:.3f}")
    print(f"[done] x={dict(zip(NAMES, np.round(x_best,4)))}")

    # Report per-gesture stats at the optimum
    synth = _build_synth(x_best, args.sr)
    traces = {g: _synthetic_trace(g, args.sr) for g in GESTURES}
    for g in GESTURES:
        synth.reset()
        y = _render(synth, traces[g], args.sr).astype(np.float64)
        ts = analyze_texture(y, args.sr)
        print(f"  {g:14} flat{ts.flatness:.3f} cent{ts.centroid:5.0f} centCV{ts.centroid_cv:.3f} "
              f"crest{ts.crest_db:4.1f} roll{ts.rolloff85:5.0f} envcv{ts.env_cv:.2f} D{texture_distance(ts):.1f}")

    d = dict(zip(NAMES, [float(v) for v in x_best]))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, final_loss=final, init_loss=init_loss, **d)
    out_path.with_suffix(".json").write_text(json.dumps({**d, "final_loss": float(final)}, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

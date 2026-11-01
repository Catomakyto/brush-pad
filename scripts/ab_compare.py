"""Blind listening test harness."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_DBFS = -20.0
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg"}


def _load_mono(path: Path, sr: int = 48000) -> np.ndarray:
    y, s = sf.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64)
    if s != sr:
        n = int(len(y) * sr / s)
        y = np.interp(np.linspace(0, len(y), n, endpoint=False), np.arange(len(y)), y)
    return y


def _match_loudness(y: np.ndarray, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    rms = float(np.sqrt(np.mean(y ** 2))) + 1e-12
    gain = (10.0 ** (target_dbfs / 20.0)) / rms
    out = y * gain
    peak = float(np.max(np.abs(out)))
    if peak > 0.99:  # guard against clipping after matching
        out *= 0.99 / peak
    return out


def _clips(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in AUDIO_EXT
                  and not (p.suffix.lower() == ".wav" and p.with_suffix(".mp3").is_file()))


def _write(path: Path, y: np.ndarray, sr: int = 48000) -> None:
    sf.write(str(path), y.astype(np.float32), sr, subtype="PCM_24")


def run_turing(synth_dirs: list[Path], out: Path, seed: int, clip_s: float) -> None:
    rng = np.random.default_rng(seed)
    pool = []  # (label, path)
    for d in synth_dirs:
        # Label each folder synth or real
        is_synth = "our" in d.parts or "synth" in d.name.lower()
        for p in _clips(d):
            pool.append(("synth" if is_synth else "real", p))
    rng.shuffle(pool)
    out.mkdir(parents=True, exist_ok=True)
    key = []
    rows = []
    for i, (label, p) in enumerate(pool, 1):
        y = _match_loudness(_load_mono(p))
        if clip_s > 0:
            y = y[: int(clip_s * 48000)]
        name = f"trial_{i:02d}.wav"
        _write(out / name, y)
        key.append({"trial": i, "file": name, "truth": label, "source": str(p)})
        rows.append({"trial": i, "your_guess (real/synth)": "", "confidence (1-5)": "", "notes": ""})
    (out / "key.json").write_text(json.dumps(key, indent=2))
    with (out / "scoresheet.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"turing: {len(pool)} trials → {out}")
    print(f"  listen to trial_*.wav, mark real/synth in scoresheet.csv, THEN open key.json")


def run_paired(a_dir: Path, b_dir: Path, out: Path, seed: int, clip_s: float) -> None:
    rng = np.random.default_rng(seed)
    a_map = {p.stem: p for p in _clips(a_dir)}
    b_map = {p.stem: p for p in _clips(b_dir)}
    common = sorted(set(a_map) & set(b_map))
    if not common:
        raise SystemExit(f"no matching basenames between {a_dir} and {b_dir}")
    out.mkdir(parents=True, exist_ok=True)
    key = []
    rows = []
    for i, stem in enumerate(common, 1):
        ya = _match_loudness(_load_mono(a_map[stem]))
        yb = _match_loudness(_load_mono(b_map[stem]))
        if clip_s > 0:
            ya = ya[: int(clip_s * 48000)]; yb = yb[: int(clip_s * 48000)]
        swap = bool(rng.integers(0, 2))  # randomize which is A
        _write(out / f"trial_{i:02d}_A.wav", yb if swap else ya)
        _write(out / f"trial_{i:02d}_B.wav", ya if swap else yb)
        key.append({"trial": i, "technique": stem,
                    "A": "b" if swap else "a", "B": "a" if swap else "b"})
        rows.append({"trial": i, "technique": stem,
                     "prefer (A/B)": "", "confidence (1-5)": "", "notes": ""})
    (out / "key.json").write_text(json.dumps(key, indent=2))
    with (out / "scoresheet.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"paired: {len(common)} trials → {out}  (A/a={a_dir.name}, B/b={b_dir.name})")


def run_play(folder: Path) -> None:
    """Interactively play a generated test folder and score you at the end."""
    import sounddevice as sd  # lazy, only needed here

    key = json.loads((folder / "key.json").read_text())
    paired = "A" in key[0]
    print(f"\n  Blind {'A/B preference' if paired else 'real-vs-synth'} test — "
          f"{len(key)} trials in {folder}")
    print("  [enter]=replay   q=quit   x=skip   "
          + ("a/b = prefer A or B\n" if paired else "r=real   s=synth\n"))

    def play_trial(k) -> None:
        if paired:
            y, sr = sf.read(str(folder / f"trial_{k['trial']:02d}_A.wav"))
            print("    A …"); sd.play(y, sr); sd.wait(); sd.sleep(300)
            y, sr = sf.read(str(folder / f"trial_{k['trial']:02d}_B.wav"))
            print("    B …"); sd.play(y, sr); sd.wait()
        else:
            y, sr = sf.read(str(folder / k["file"]))
            sd.play(y, sr); sd.wait()

    valid = {"a", "b"} if paired else {"r", "s"}
    answers = []
    for k in key:
        label = f" ({k['technique']})" if paired else ""
        print(f"  Trial {k['trial']:2d}{label}:")
        play_trial(k)
        while True:
            ans = input("    > ").strip().lower()
            if ans in ("q", "quit"):
                print("  quit early."); break
            if ans in ("", "p"):
                play_trial(k); continue
            if ans == "x":
                answers.append((k, None)); break
            if ans in valid:
                answers.append((k, ans)); break
            print(f"    (type {'a or b' if paired else 'r or s'})")
        else:
            continue
        if ans in ("q", "quit"):
            break

    scored = [(k, a) for k, a in answers if a is not None]
    if not scored:
        print("\n  no answers recorded."); return
    if paired:
        ours = sum(1 for k, a in scored if k[a.upper()] == "a")
        print(f"\n  You preferred OURS (the --a folder) on {ours}/{len(scored)} trials.")
    else:
        gk = {"r": "real", "s": "synth"}
        correct = sum(1 for k, a in scored if gk[a] == k["truth"])
        n = len(scored)
        print(f"\n  Score: {correct}/{n} correct ({100 * correct / n:.0f}%).")
        print("  ~50% = you can't reliably tell synth from real — that's the win.")
        print("  Much higher = you can still spot the synth; tell me which trials and why.")
        fooled = [k["trial"] for k, a in scored if gk[a] != k["truth"] and k["truth"] == "synth"]
        if fooled:
            print(f"  Synth trials you thought were REAL (fooled you): {fooled}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    pl = sub.add_parser("play", help="interactively run + score a generated test folder")
    pl.add_argument("folder")
    t = sub.add_parser("turing")
    t.add_argument("dirs", nargs="+", help="folders of clips (synth dirs under evals/our, others = real)")
    t.add_argument("-o", "--output", required=True)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--clip-s", type=float, default=6.0, help="trim each clip to N s (0=full)")
    p = sub.add_parser("paired")
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--clip-s", type=float, default=0.0)
    args = ap.parse_args()

    if args.mode == "play":
        run_play(Path(args.folder))
    elif args.mode == "turing":
        run_turing([Path(d) for d in args.dirs], Path(args.output), args.seed, args.clip_s)
    else:
        run_paired(Path(args.a), Path(args.b), Path(args.output), args.seed, args.clip_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

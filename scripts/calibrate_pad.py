"""Calibrate the pad from real brushing: rest to feather to medium to firm."""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from brushsynth.input.rawframe import (  # noqa: E402
    AlphaBeta2D,
    RawMapConfig,
    SerialRawSource,
    SimulatedRawSource,
    extract_blobs,
    fill_dead_lines,
    parse_raw_frame,
)

TARGET_SPEED = 4.0  # model |v| for a normal medium sweep
FEATHER_PRESSURE = 0.15  # feather target, 0 to 1
DEAD_ROWS = (3, 15)  # pad R4, R16 (0-based)
DEAD_COLS = (0,)  # pad C8 (0-based)


def find_port() -> str | None:
    ports = sorted(glob.glob("/dev/cu.usbmodem*"))
    return ports[0] if ports else None


class FrameFeed:
    """Background reader keeping (t, grid) pairs."""

    def __init__(self, source) -> None:
        self.frames: list[tuple[float, np.ndarray]] = []
        self.stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(source,), daemon=True)
        self._thread.start()

    def _run(self, source) -> None:
        for _, buf in source.stream(self.stop):
            f = parse_raw_frame(buf)
            if f is not None:
                self.frames.append((time.perf_counter(), f.grid.astype(np.float64)))

    def capture(self, seconds: float) -> list[tuple[float, np.ndarray]]:
        t0 = time.perf_counter()
        time.sleep(seconds)
        return [(t, g) for t, g in self.frames if t >= t0]


def phase(feed: FrameFeed, title: str, instruction: str, seconds: float, auto: bool):
    print(f"\n=== {title} ===\n{instruction}")
    if not auto:
        input("Press Enter, then start…")
    for k in (3, 2, 1):
        print(f"  {k}…", flush=True)
        time.sleep(0.6)
    print(f"  GO — {seconds:.0f} s", flush=True)
    got = feed.capture(seconds)
    print(f"  captured {len(got)} frames")
    if len(got) < 20:
        raise SystemExit("Too few frames — is the Teensy streaming? (check with view_pad.py)")
    return got


def blob_series(frames, baseline, threshold):
    """(t, force, cx_norm, cy_norm) for frames with a blob."""
    out = []
    for t, g in frames:
        act = fill_dead_lines(np.maximum(g - baseline, 0.0), DEAD_ROWS, DEAD_COLS)
        blobs = extract_blobs(act, threshold=threshold, max_blobs=1)
        if blobs:
            b = blobs[0]
            h, w = g.shape
            out.append((t, b.force, b.cx / max(w - 1, 1) * 2 - 1, b.cy / max(h - 1, 1) * 2 - 1))
    return out


def sim_feed() -> FrameFeed:
    # Rest 4 s, feather 6 s, medium 6 s, firm 6 s
    rate = 300.0
    segs = [(4.6, 0.0, 0.0), (7.8, 0.08, 0.6), (7.8, 0.35, 1.0), (7.8, 0.9, 0.8)]
    xs, ys, ps = [], [], []
    for dur, p, speed in segs:
        t = np.arange(int(dur * rate)) / rate
        xs.append(0.5 * np.cos(2 * np.pi * speed * t / 2))
        ys.append(0.5 * np.sin(2 * np.pi * speed * t / 2))
        ps.append(np.full_like(t, p))
    src = SimulatedRawSource(np.concatenate(xs), np.concatenate(ys), np.concatenate(ps),
                             rate_hz=rate, noise=1.5, realtime=True, loop=False)
    return FrameFeed(src)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--sim", action="store_true", help="dry run on a simulated pad")
    ap.add_argument("--out", default="calibration/pad.json")
    args = ap.parse_args()

    if args.sim:
        feed = sim_feed()
    else:
        port = args.port or find_port()
        if port is None:
            print("No /dev/cu.usbmodem* found — is the Teensy plugged in?")
            return 1
        feed = FrameFeed(SerialRawSource(port))
    time.sleep(0.5)

    rest = phase(feed, "1/4 REST", "Hands and brush OFF the pad.", 3.0, args.sim)
    grids = np.stack([g for _, g in rest])
    baseline = grids.mean(axis=0)
    noise = grids.std(axis=0)
    threshold = float(max(6.0, 6.0 * np.median(noise)))
    print(f"  rest: mean {baseline.mean():.0f} counts, typical noise {np.median(noise):.1f}"
          f" → blob_threshold {threshold:.1f}")
    noisy = np.argwhere(noise > 4 * max(np.median(noise), 0.5))
    if len(noisy):
        cells = ", ".join(f"R{r + 1}C{c + 1}" for r, c in noisy[:8])
        print(f"  warning: unusually noisy cells (loose wire? touching?): {cells}")
    hot = np.argwhere(baseline > np.median(baseline) + 400)
    if len(hot):
        cells = ", ".join(f"R{r + 1}C{c + 1}" for r, c in hot[:8])
        print(f"  warning: cells resting well above the rest of the pad (touching at rest?): {cells}")

    feather = blob_series(phase(feed, "2/4 FEATHER", "Your softest ballad stir, keep it moving.",
                                6.0, args.sim), baseline, threshold)
    medium = blob_series(phase(feed, "3/4 MEDIUM", "A normal medium-tempo sweep.",
                               6.0, args.sim), baseline, threshold)
    firm = blob_series(phase(feed, "4/4 FIRM", "Firm, pressed sweeps — your loudest.",
                             6.0, args.sim), baseline, threshold)
    feed.stop.set()

    for name, s in (("feather", feather), ("medium", medium), ("firm", firm)):
        if len(s) < 10:
            raise SystemExit(f"{name}: the pad barely saw the brush ({len(s)} frames). "
                             "Check wiring in view_pad.py before calibrating.")

    f_force = np.array([s[1] for s in feather])
    min_contact = float(max(2 * threshold, 0.5 * np.percentile(f_force, 10)))
    force_full = float(np.percentile([s[1] for s in firm], 90))

    # Medium-phase sweep speed
    tr = AlphaBeta2D(RawMapConfig().tau)
    speeds = []
    for t, _, x, y in medium:
        vx, vy = tr.update(x, y, t)
        speeds.append(np.hypot(vx, vy))
    med_speed = float(np.median(speeds[5:])) if len(speeds) > 10 else 1.0
    vgain = float(np.clip(TARGET_SPEED / max(med_speed, 1e-3), 0.2, 20.0))

    # Feather median should land near FEATHER_PRESSURE
    ratio = float(np.clip(np.median(f_force) / force_full, 1e-4, 0.99))
    gamma = float(np.clip(np.log(FEATHER_PRESSURE) / np.log(ratio), 0.25, 1.0))

    cfg = RawMapConfig(blob_threshold=threshold, baseline_margin=threshold,
                       min_contact_force=min_contact, force_full=force_full,
                       pressure_gamma=gamma, velocity_gain=vgain, dead_rows=DEAD_ROWS, dead_cols=DEAD_COLS)
    print("\n=== RESULT ===")
    print(f"  feather force p10/p50 : {np.percentile(f_force, 10):.0f} / {np.median(f_force):.0f}")
    print(f"  firm force p90        : {force_full:.0f}")
    print(f"  feather → pressure ≈ {ratio ** gamma:.2f} (curve gamma {gamma:.2f})")
    print(f"  medium sweep speed {med_speed:.2f} → velocity_gain {vgain:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    print(f"  saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

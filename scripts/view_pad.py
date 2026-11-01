"""Live heat-map of the pad, the bring-up and debugging window."""

from __future__ import annotations

import argparse
import glob
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from brushsynth.input.rawframe import (  # noqa: E402
    BaselineTracker,
    SerialRawSource,
    SimulatedRawSource,
    extract_blobs,
    fill_dead_lines,
    parse_raw_frame,
)

CELL = 40
MARGIN = 30


def find_port() -> str | None:
    ports = sorted(glob.glob("/dev/cu.usbmodem*"))
    return ports[0] if ports else None


def sim_source() -> SimulatedRawSource:
    t = np.linspace(0.0, 8.0, 8 * 400)
    x = 0.6 * np.cos(2 * np.pi * t / 2.0)
    y = 0.4 * np.sin(2 * np.pi * t / 2.0)
    p = 0.5 + 0.4 * np.sin(2 * np.pi * t / 3.0)
    return SimulatedRawSource(x, y, p, noise=1.5, realtime=True, loop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--full", type=float, default=600.0,
                    help="activation (counts) shown as full white")
    ap.add_argument("--threshold", type=float, default=6.0, help="blob threshold (counts)")
    ap.add_argument("--dead-rows", type=int, nargs="*", default=[4, 16],
                    help="1-based rows to fill from neighbours (broken strips); default: 4 16")
    ap.add_argument("--dead-cols", type=int, nargs="*", default=[1],
                    help="1-based columns to fill from neighbours; default: 1")
    args = ap.parse_args()

    if args.sim:
        src = sim_source()
    else:
        port = args.port or find_port()
        if port is None:
            print("No /dev/cu.usbmodem* found — is the Teensy plugged in? (or use --sim)")
            return 1
        print(f"reading {port}")
        src = SerialRawSource(port)

    latest: dict = {"frame": None, "n": 0, "bad": 0}
    stop = threading.Event()

    def reader() -> None:
        for _, buf in src.stream(stop):
            f = parse_raw_frame(buf)
            if f is None:
                latest["bad"] += 1
                continue
            latest["frame"] = f
            latest["n"] += 1

    threading.Thread(target=reader, daemon=True).start()

    import pygame
    pygame.display.init()
    pygame.font.init()
    font = pygame.font.SysFont("Menlo", 11)
    screen = pygame.display.set_mode((16 * CELL + 2 * MARGIN, 16 * CELL + 2 * MARGIN))
    pygame.display.set_caption("brush pad")
    clock = pygame.time.Clock()

    baseline = BaselineTracker()
    show_numbers = False

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                stop.set()
                return 0
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_n:
                show_numbers = not show_numbers
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                baseline = BaselineTracker()

        screen.fill((18, 18, 22))
        f = latest["frame"]
        if f is not None:
            act = baseline.update(f.grid)
            fill_dead_lines(act, [r - 1 for r in args.dead_rows], [c - 1 for c in args.dead_cols])
            h, w = f.grid.shape
            for r in range(h):
                for c in range(w):
                    v = min(act[r, c] / args.full, 1.0)
                    shade = int(40 + 215 * v)
                    rect = (MARGIN + c * CELL, MARGIN + r * CELL, CELL - 2, CELL - 2)
                    pygame.draw.rect(screen, (shade, shade, int(shade * 0.9)), rect)
                    if show_numbers:
                        txt = font.render(str(int(f.grid[r, c])), True, (230, 90, 60))
                        screen.blit(txt, (rect[0] + 3, rect[1] + 12))
            for i, b in enumerate(extract_blobs(act, threshold=args.threshold)):
                px = MARGIN + b.cx * CELL + CELL / 2 - 1
                py = MARGIN + b.cy * CELL + CELL / 2 - 1
                color = (255, 70, 70) if i == 0 else (70, 160, 255)
                pygame.draw.circle(screen, color, (int(px), int(py)), 8, 3)
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    raise SystemExit(main())

"""Two-brush tracking and hit detection on synthetic 16×8 frames."""

from __future__ import annotations

import numpy as np

from brushsynth.input.multibrush import MultiBrushProcessor
from brushsynth.input.rawframe import RawMapConfig

REST = 680.0


def _grid(blobs, h=16, w=8):
    """blobs: list of (col, row, peak counts)."""
    jj, ii = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    g = np.full((h, w), REST)
    for cx, cy, amp in blobs:
        g += amp * np.exp(-((jj - cx) ** 2 + (ii - cy) ** 2) / (2 * 0.9 ** 2))
    return g


def _proc():
    return MultiBrushProcessor(RawMapConfig(force_full=6000.0, min_contact_force=100.0,
                                            blob_threshold=8.0, baseline_margin=8.0))


def test_two_brushes_keep_their_identity_while_moving():
    pr, dt = _proc(), 1 / 300
    pr.update(_grid([]), 0.0)
    xs = []
    for k in range(300):  # left brush slides down, right brush slides up
        t = (k + 1) * dt
        g, _ = pr.update(_grid([(1.5, 2 + 10 * k / 300, 400), (6.0, 13 - 10 * k / 300, 400)]), t)
        xs.append((g[0].x, g[1].x, g[0].pressure > 0, g[1].pressure > 0))
    xs = np.array(xs[10:])
    assert xs[:, 2].all() and xs[:, 3].all()  # both brushes sound throughout
    # One voice stays on the left column
    assert (np.sign(xs[:, 0]) == np.sign(xs[0, 0])).all()
    assert (np.sign(xs[:, 1]) == np.sign(xs[0, 1])).all() and np.sign(xs[0, 0]) != np.sign(xs[0, 1])


def test_fast_tap_fires_one_hit_slow_set_down_none():
    pr, dt = _proc(), 1 / 300
    hits, t = [], 0.0
    for _ in range(60):  # brush in the air for 0.2 s
        pr.update(_grid([]), t)
        t += dt
    for k in range(60):  # tap: 0 to full in ~10 ms, hold, lift
        t += dt
        amp = min(k / 3, 1.0) * 900 if k < 30 else 0.0
        hits += pr.update(_grid([(4, 8, amp)]), t)[1]
    assert len(hits) == 1 and hits[0][0] > 0.2
    pr2, slow = _proc(), []
    for k in range(60):
        pr2.update(_grid([]), k * dt)
    for k in range(150):  # set the brush down over 0.5 s
        slow += pr2.update(_grid([(4, 8, 900 * k / 150)]), (k + 61) * dt)[1]
    assert slow == []


def test_split_bristles_are_one_brush():
    pr = _proc()
    pr.update(_grid([]), 0.0)
    for k in range(20):
        g, _ = pr.update(_grid([(3.0, 8, 300), (4.6, 8, 300)]), (k + 1) / 300)
    assert g[0].pressure > 0 and g[1].pressure == 0

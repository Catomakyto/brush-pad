"""Two brushes + hits from raw pad frames."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from ..engine.rtaudio import Gesture, ParamBus
from .rawframe import (
    AlphaBeta2D,
    BaselineTracker,
    RawLog,
    RawMapConfig,
    fill_dead_lines,
    parse_raw_frame,
)


@dataclass
class MultiConfig:
    blur: bool = True
    pressure_smooth_ms: float = 12.0
    stir_floor: float = 0.08  # leftover after a lift reads 0.06 to 0.10
    hit_stir_mute_ms: float = 200.0  # a hit's own splat isn't a stir
    hit_mute_radius: float = 0.7  # normalized radius
    min_sep_taxels: float = 4.0  # peaks closer than this are one brush
    second_min_ratio: float = 0.35  # 2nd brush needs this share of the 1st's force
    # Hit settings
    hit_level: float = 0.20  # pressure a landing must reach
    hit_window_ms: float = 45.0  # time allowed to reach it
    hit_floor: float = 0.05  # below this the brush counts as "in the air"
    hit_min_air_ms: float = 100.0  # minimum time in the air
    hit_full_rate: float = 90.0  # landing rate for velocity 1
    hit_dedupe_ms: float = 60.0  # merge peaks from one tap


def _peak_contacts(act: np.ndarray, threshold: float, min_sep: float,
                   max_n: int = 2) -> list[tuple[float, float, float]]:
    """Contacts as hills, not connected patches."""
    h, w = act.shape
    p = np.pad(act, 1, mode="constant", constant_values=-np.inf)
    neigh = np.max(np.stack([p[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
                             for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                             if dy or dx]), axis=0)
    ys, xs = np.nonzero((act >= neigh) & (act >= 2 * threshold))
    order = np.argsort(-act[ys, xs])
    peaks: list[tuple[int, int]] = []
    for k in order:
        y, x = int(ys[k]), int(xs[k])
        if all(np.hypot(x - px, y - py) >= min_sep for py, px in peaks):
            peaks.append((y, x))
        if len(peaks) == max_n:
            break
    if not peaks:
        return []
    cy, cx = np.nonzero(act >= threshold)
    v = act[cy, cx]
    d = np.stack([np.hypot(cx - px, cy - py) for py, px in peaks])
    owner = np.argmin(d, axis=0)
    out = []
    for i in range(len(peaks)):
        m = owner == i
        f = float(v[m].sum())
        if f > 0:
            out.append((float((cx[m] * v[m]).sum() / f), float((cy[m] * v[m]).sum() / f), f))
    out.sort(key=lambda c: c[2], reverse=True)
    return out


def _blur(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    a = 0.25 * p[:-2, 1:-1] + 0.5 * p[1:-1, 1:-1] + 0.25 * p[2:, 1:-1]
    p = np.pad(a, ((0, 0), (1, 1)), mode="edge")
    return 0.25 * p[:, :-2] + 0.5 * p[:, 1:-1] + 0.25 * p[:, 2:]


class _Voice:
    def __init__(self, cfg: RawMapConfig) -> None:
        self.tracker = AlphaBeta2D(cfg.tau)
        self.air_since: float | None = None  # when it last went below the floor
        self.t_leave: float | None = None  # last frame it was below the floor
        self.p_smooth = 0.0
        self.mute_until = -1e9
        self.release()

    def release(self) -> None:
        self.active = False
        self.tracker.reset()
        self.px = self.py = 0.0
        self.vx = self.vy = 0.0


class MultiBrushProcessor:
    """Raw frame to (two Gesture, list of (hit velocity, pad x, pad y))."""

    def __init__(self, config: RawMapConfig | None = None, multi: MultiConfig | None = None) -> None:
        self.cfg = config or RawMapConfig()
        self.m = multi or MultiConfig()
        c = self.cfg
        self.baseline = BaselineTracker(c.baseline_margin, c.baseline_coeff, c.baseline_creep)
        self.voices = [_Voice(c), _Voice(c)]
        self._t_last: float | None = None
        self._last_hit = -1e9

    def _pressure(self, force: float) -> float:
        c = self.cfg
        p = (force - c.force_floor) / max(c.force_full - c.force_floor, 1e-6)
        return float(min(max(p, 0.0), 1.0)) ** c.pressure_gamma

    def update(self, grid: np.ndarray, t: float) -> tuple[list[Gesture], list[tuple[float, float, float]]]:
        c, m = self.cfg, self.m
        dt = 1.0 / 300.0 if self._t_last is None else min(max(t - self._t_last, 1e-4), 0.1)
        self._t_last = t
        act = self.baseline.update(grid)
        fill_dead_lines(act, c.dead_rows, c.dead_cols)
        if m.blur:
            act = _blur(act)
        h, w = act.shape
        merged = []
        for cx, cy, f in _peak_contacts(act, c.blob_threshold, m.min_sep_taxels):
            if f < c.min_contact_force:
                continue
            nx = (cx / (w - 1)) * 2.0 - 1.0 if w > 1 else 0.0
            ny = (cy / (h - 1)) * 2.0 - 1.0 if h > 1 else 0.0
            merged.append([nx, -ny if c.invert_y else ny, f])
        if len(merged) > 1 and merged[1][2] < m.second_min_ratio * merged[0][2]:
            merged = merged[:1]

        # Match contacts to voices: keep identities by nearest distance
        assign: dict[int, list] = {}
        free = list(merged)
        active = [i for i, v in enumerate(self.voices) if v.active]
        if len(free) == 2 and len(active) == 2:
            v0, v1 = self.voices
            d_keep = np.hypot(free[0][0] - v0.px, free[0][1] - v0.py) + np.hypot(free[1][0] - v1.px, free[1][1] - v1.py)
            d_swap = np.hypot(free[1][0] - v0.px, free[1][1] - v0.py) + np.hypot(free[0][0] - v1.px, free[0][1] - v1.py)
            assign = {0: free[0], 1: free[1]} if d_keep <= d_swap else {0: free[1], 1: free[0]}
            free = []
        else:
            for i in active:
                if not free:
                    break
                v = self.voices[i]
                j = int(np.argmin([np.hypot(g[0] - v.px, g[1] - v.py) for g in free]))
                assign[i] = free.pop(j)
            for i, v in enumerate(self.voices):
                if free and i not in assign and not v.active:
                    assign[i] = free.pop(0)

        gestures, hits = [], []
        for i, v in enumerate(self.voices):
            g = assign.get(i)
            p_raw = self._pressure(g[2]) if g is not None else 0.0
            vh = self._hit(v, p_raw, t)
            if vh:  # mute every brush near the hit
                for u in self.voices:
                    if u is v or np.hypot(u.px - g[0], u.py - g[1]) < m.hit_mute_radius:
                        u.mute_until = t + m.hit_stir_mute_ms / 1000.0
            hits += [(vel, g[0], g[1]) for vel in vh]
            if g is None:
                v.release()
                v.p_smooth = 0.0
                v.vx *= 0.5
                v.vy *= 0.5
                gestures.append(Gesture(x=v.px, y=v.py, vx=v.vx, vy=v.vy, pressure=0.0))
                continue
            nx, ny = g[0], g[1]
            if v.active and np.hypot(nx - v.px, ny - v.py) > c.max_jump:
                v.tracker.reset()
                v.tracker.update(nx, ny, t)
                vx, vy = v.vx, v.vy
            else:
                vx, vy = v.tracker.update(nx, ny, t)
            mrs = c.max_raw_speed
            v.vx, v.vy = float(np.clip(vx, -mrs, mrs)), float(np.clip(vy, -mrs, mrs))
            a = float(np.exp(-dt / max(m.pressure_smooth_ms / 1000.0, 1e-6)))
            v.p_smooth = p_raw if not v.active else a * v.p_smooth + (1 - a) * p_raw
            v.px, v.py, v.active = nx, ny, True
            p_out = max(v.p_smooth - m.stir_floor, 0.0) / (1.0 - m.stir_floor)
            if t < v.mute_until:
                p_out = 0.0
            gestures.append(Gesture(x=nx, y=ny, vx=v.vx * c.velocity_gain, vy=v.vy * c.velocity_gain,
                                    pressure=p_out))
        return gestures, hits

    def _hit(self, v: _Voice, p: float, t: float) -> list[float]:
        m = self.m
        if p < m.hit_floor:
            if v.air_since is None:
                v.air_since = t
            v.t_leave = t
            return []
        fired: list[float] = []
        if (v.t_leave is not None and p >= m.hit_level
                and t - v.t_leave <= m.hit_window_ms / 1000.0):
            air = v.t_leave - (v.air_since if v.air_since is not None else v.t_leave)
            if air >= m.hit_min_air_ms / 1000.0 and t - self._last_hit > m.hit_dedupe_ms / 1000.0:
                rate = (p - m.hit_floor) / (t - v.t_leave + 1.0 / 300.0)
                fired = [float(min(max(rate / m.hit_full_rate, 0.2), 1.0))]
                self._last_hit = t
            v.t_leave = None
        if p >= m.hit_level:
            v.air_since = None
        return fired


class MultiBrushDriver:
    """Raw source to MultiBrushProcessor to two buses + a hit callback."""

    def __init__(self, buses: list[ParamBus], source, config: RawMapConfig | None = None,
                 multi: MultiConfig | None = None, *, on_hit=None, log: RawLog | None = None) -> None:
        self.buses = buses
        self.source = source
        self.processor = MultiBrushProcessor(config, multi)
        self.on_hit = on_hit
        self.log = log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.n_frames = 0
        self.n_bad = 0
        self.n_hits = 0

    def _run(self) -> None:
        for t, raw in self.source.stream(self._stop):
            if self._stop.is_set():
                break
            frame = parse_raw_frame(raw)
            if frame is None:
                self.n_bad += 1
                continue
            if self.log is not None:
                self.log.add(t, frame)
            gestures, hits = self.processor.update(frame.grid, t)
            for bus, g in zip(self.buses, gestures):
                bus.write(g)
            for vel, x, y in hits:
                self.n_hits += 1
                if self.on_hit is not None:
                    self.on_hit(vel, x, y)
            self.n_frames += 1

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

"""Raw-frame input path for the pad prototype: dumb firmware, smart host."""

from __future__ import annotations

import struct
import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..engine.rtaudio import Gesture, ParamBus

_SYNC0 = 0xB3
_SYNC1 = 0x55
_HDR_FMT = "<BBHBBH"  # sync0, sync1, seq, w, h, piezo
_HDR_SIZE = struct.calcsize(_HDR_FMT)  # 8


def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF)."""
    crc = 0xFFFF
    for b in data:
        crc ^= (b & 0xFF) << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


def frame_size(w: int, h: int) -> int:
    return _HDR_SIZE + 2 * w * h + 2


@dataclass
class RawFrame:
    """One raw taxel frame from the pad."""

    seq: int
    grid: np.ndarray  # (h, w) uint16, force-positive
    piezo: int = 0

    @property
    def w(self) -> int:
        return int(self.grid.shape[1])

    @property
    def h(self) -> int:
        return int(self.grid.shape[0])

    def pack(self) -> bytes:
        h, w = self.grid.shape
        hdr = struct.pack(_HDR_FMT, _SYNC0, _SYNC1, self.seq & 0xFFFF,
                          w, h, self.piezo & 0xFFFF)
        payload = np.ascontiguousarray(self.grid, dtype="<u2").tobytes()
        body = hdr[2:] + payload  # CRC covers everything after the sync bytes
        return hdr + payload + struct.pack("<H", crc16_ccitt(body))


def parse_raw_frame(buf: bytes) -> RawFrame | None:
    """Parse one complete frame."""
    if len(buf) < _HDR_SIZE or buf[0] != _SYNC0 or buf[1] != _SYNC1:
        return None
    _, _, seq, w, h, piezo = struct.unpack(_HDR_FMT, buf[:_HDR_SIZE])
    if not (1 <= w <= 64 and 1 <= h <= 64) or len(buf) != frame_size(w, h):
        return None
    if crc16_ccitt(buf[2:-2]) != struct.unpack("<H", buf[-2:])[0]:
        return None
    grid = np.frombuffer(buf, dtype="<u2", count=w * h, offset=_HDR_SIZE)
    return RawFrame(seq=seq, grid=grid.reshape(h, w).copy(), piezo=piezo)


class BaselineTracker:
    """Per-taxel no-touch reference: seed on first frame, adapt ONLY while idle."""

    def __init__(self, touch_margin: float = 6.0, adapt_coeff: float = 0.02,
                 creep_coeff: float = 0.0) -> None:
        self.touch_margin = float(touch_margin)
        self.adapt_coeff = float(adapt_coeff)
        # Very slow adaptation even while "touched"
        self.creep_coeff = float(creep_coeff)
        self._baseline: np.ndarray | None = None

    def update(self, grid: np.ndarray) -> np.ndarray:
        """Return the activation image (force above baseline, ≥ 0)."""
        g = grid.astype(np.float64)
        if self._baseline is None or self._baseline.shape != g.shape:
            self._baseline = g.copy()
            return np.zeros_like(g)
        act = g - self._baseline
        idle = act < self.touch_margin
        self._baseline[idle] += self.adapt_coeff * act[idle]
        if self.creep_coeff > 0.0:
            self._baseline[~idle] += self.creep_coeff * act[~idle]
        np.maximum(act, 0.0, out=act)
        return act


def fill_dead_lines(act: np.ndarray, rows=(), cols=()) -> np.ndarray:
    """Fill dead rows and columns from live neighbors."""
    for axis, dead in ((0, rows), (1, cols)):
        n = act.shape[axis]
        dead_set = {i for i in dead if 0 <= i < n}
        for i in dead_set:
            nbrs = [j for j in (i - 1, i + 1) if 0 <= j < n and j not in dead_set]
            fill = np.mean([np.take(act, j, axis=axis) for j in nbrs], axis=0) if nbrs else 0.0
            if axis == 0:
                act[i, :] = fill
            else:
                act[:, i] = fill
    return act


@dataclass
class BlobFeatures:
    """Weighted-moment features of one contact blob (taxel units)."""

    force: float  # Σ activation
    cx: float  # centroid, taxel coords (col axis)
    cy: float  # centroid, taxel coords (row axis)
    spread: float  # rms radius around the centroid
    orient: float  # principal-axis angle, radians in (-π/2, π/2]
    area: int  # active taxel count


def extract_blobs(
    activation: np.ndarray, *, threshold: float = 6.0, max_blobs: int = 2,
) -> list[BlobFeatures]:
    """Find connected contact blobs (4-neighbor) and their weighted moments."""
    h, w = activation.shape
    active = activation >= threshold
    seen = np.zeros_like(active, dtype=bool)
    blobs: list[BlobFeatures] = []
    for sy in range(h):
        for sx in range(w):
            if not active[sy, sx] or seen[sy, sx]:
                continue
            # BFS this component
            stack = [(sy, sx)]
            seen[sy, sx] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and active[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = np.array([c[0] for c in cells], dtype=np.float64)
            xs = np.array([c[1] for c in cells], dtype=np.float64)
            ps = activation[ys.astype(int), xs.astype(int)]
            force = float(ps.sum())
            if force <= 0.0:
                continue
            cx = float((xs * ps).sum() / force)
            cy = float((ys * ps).sum() / force)
            dx = xs - cx
            dy = ys - cy
            mu20 = float((dx * dx * ps).sum() / force)
            mu02 = float((dy * dy * ps).sum() / force)
            mu11 = float((dx * dy * ps).sum() / force)
            spread = float(np.sqrt(max(mu20 + mu02, 0.0)))
            orient = 0.5 * float(np.arctan2(2.0 * mu11, mu20 - mu02))
            blobs.append(BlobFeatures(force=force, cx=cx, cy=cy, spread=spread,
                                      orient=orient, area=len(cells)))
    blobs.sort(key=lambda b: b.force, reverse=True)
    return blobs[:max_blobs]


class AlphaBeta2D:
    """Critically-damped α-β (g-h) tracker: smooth position to smooth velocity."""

    def __init__(self, tau: float = 0.008) -> None:
        self.tau = float(tau)
        self.reset()

    def reset(self) -> None:
        self._init = False
        self._x = self._y = 0.0
        self._vx = self._vy = 0.0
        self._t: float | None = None

    def update(self, x: float, y: float, t: float) -> tuple[float, float]:
        if not self._init:
            self._init = True
            self._x, self._y, self._t = x, y, t
            self._vx = self._vy = 0.0
            return 0.0, 0.0
        dt = max(t - (self._t or t), 1e-4)
        th = float(np.exp(-dt / self.tau))
        g = 1.0 - th * th
        hh = (1.0 - th) ** 2
        for axis in ("x", "y"):
            pos = x if axis == "x" else y
            est = self._x if axis == "x" else self._y
            vel = self._vx if axis == "x" else self._vy
            pred = est + vel * dt
            r = pos - pred
            est = pred + g * r
            vel = vel + (hh / dt) * r
            if axis == "x":
                self._x, self._vx = est, vel
            else:
                self._y, self._vy = est, vel
        self._t = t
        return self._vx, self._vy


@dataclass
class RawMapConfig:
    """Calibration for raw frames to Gesture."""

    blob_threshold: float = 6.0  # counts above baseline to count as active
    min_contact_force: float = 25.0  # Σ counts for a blob to count as contact
    baseline_margin: float = 6.0  # activation below this lets the baseline adapt
    baseline_coeff: float = 0.02  # per-frame baseline adaptation
    baseline_creep: float = 0.002  # per-frame adaptation while touched
    force_floor: float = 0.0  # blob force to pressure 0
    force_full: float = 6000.0  # blob force to pressure 1
    pressure_gamma: float = 1.0  # pressure = linear**gamma
    velocity_gain: float = 1.0  # normalized-units/s to model |v|
    tau: float = 0.018  # α-β tracker time constant (s)
    max_raw_speed: float = 14.0  # speed clamp
    max_jump: float = 0.3  # centroid step per frame
    invert_y: bool = False
    dead_rows: tuple[int, ...] = ()  # 0-based strips with a broken connection
    dead_cols: tuple[int, ...] = ()


class RawFrameProcessor:
    """Stateful raw frame to Gesture: baseline to blob to features to map."""

    def __init__(self, config: RawMapConfig | None = None) -> None:
        self.cfg = config or RawMapConfig()
        self.baseline = BaselineTracker(self.cfg.baseline_margin, self.cfg.baseline_coeff,
                                        self.cfg.baseline_creep)
        self.tracker = AlphaBeta2D(self.cfg.tau)
        self.last_blobs: list[BlobFeatures] = []
        self._px = self._py = 0.0
        self._vx = self._vy = 0.0
        self._in_contact = False

    def update(self, frame: RawFrame, t: float) -> Gesture:
        cfg = self.cfg
        act = self.baseline.update(frame.grid)
        fill_dead_lines(act, cfg.dead_rows, cfg.dead_cols)
        blobs = extract_blobs(act, threshold=cfg.blob_threshold, max_blobs=2)
        self.last_blobs = blobs

        if not blobs or blobs[0].force < cfg.min_contact_force:
            self._in_contact = False
            self.tracker.reset()
            self._vx *= 0.5
            self._vy *= 0.5
            return Gesture(x=self._px, y=self._py, vx=self._vx, vy=self._vy,
                           pressure=0.0, angle=0.0, tilt=0.0)

        b = blobs[0]
        w, h = frame.w, frame.h
        nx = (b.cx / (w - 1)) * 2.0 - 1.0 if w > 1 else 0.0
        ny = (b.cy / (h - 1)) * 2.0 - 1.0 if h > 1 else 0.0
        if cfg.invert_y:
            ny = -ny

        # A big jump is a glitch, not hand motion
        if self._in_contact and np.hypot(nx - self._px, ny - self._py) > cfg.max_jump:
            self.tracker.reset()
            self.tracker.update(nx, ny, t)
            vx, vy = self._vx, self._vy
        else:
            vx, vy = self.tracker.update(nx, ny, t)
        mrs = cfg.max_raw_speed
        vx = float(min(max(vx, -mrs), mrs))
        vy = float(min(max(vy, -mrs), mrs))

        pressure = (b.force - cfg.force_floor) / max(cfg.force_full - cfg.force_floor, 1e-6)
        pressure = float(min(max(pressure, 0.0), 1.0)) ** cfg.pressure_gamma

        self._px, self._py = nx, ny
        self._vx, self._vy = vx, vy
        self._in_contact = True
        vg = cfg.velocity_gain
        return Gesture(x=nx, y=ny, vx=vx * vg, vy=vy * vg,
                       pressure=pressure, angle=float(b.orient), tilt=0.0)


class SimulatedRawSource:
    """Synthesize raw frames from a gesture trace."""

    def __init__(self, x, y, pressure, *, w: int = 16, h: int = 16,
                 rate_hz: float = 400.0, amp_full: float = 900.0,
                 sigma_taxels: float = 1.2, noise: float = 0.0,
                 offsets: np.ndarray | None = None, seed: int = 3,
                 realtime: bool = False, loop: bool = False) -> None:
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.p = np.asarray(pressure, dtype=np.float64)
        if not (len(self.x) == len(self.y) == len(self.p)):
            raise ValueError("x, y, pressure must be equal length")
        self.w, self.h = int(w), int(h)
        self.rate_hz = float(rate_hz)
        self.dt = 1.0 / self.rate_hz
        self.amp_full = float(amp_full)
        self.sigma = float(sigma_taxels)
        self.noise = float(noise)
        self.realtime = realtime
        self.loop = loop
        self._rng = np.random.default_rng(seed)
        if offsets is None:
            offsets = self._rng.uniform(0.0, 4.0, size=(self.h, self.w))
        self.offsets = np.asarray(offsets, dtype=np.float64)

    def render_grid(self, x: float, y: float, pressure: float) -> np.ndarray:
        """One force-positive uint16 grid for a gesture sample."""
        cx = (float(np.clip(x, -1, 1)) + 1.0) * 0.5 * (self.w - 1)
        cy = (float(np.clip(y, -1, 1)) + 1.0) * 0.5 * (self.h - 1)
        jj, ii = np.meshgrid(np.arange(self.w), np.arange(self.h))
        blob = np.exp(-(((jj - cx) ** 2 + (ii - cy) ** 2) / (2.0 * self.sigma ** 2)))
        g = self.offsets + self.amp_full * float(np.clip(pressure, 0, 1)) * blob
        if self.noise > 0.0:
            g = g + self._rng.normal(0.0, self.noise, size=g.shape)
        return np.clip(g, 0.0, 4095.0).astype(np.uint16)

    def stream(self, stop: threading.Event | None = None):
        import time
        seq = 0
        i = 0
        t = 0.0
        n = len(self.x)
        wall0 = time.perf_counter() if self.realtime else 0.0
        while stop is None or not stop.is_set():
            if i >= n:
                if not self.loop:
                    return
                i = 0
            grid = self.render_grid(self.x[i], self.y[i], self.p[i])
            yield t, RawFrame(seq=seq, grid=grid).pack()
            seq = (seq + 1) & 0xFFFF
            i += 1
            t += self.dt
            if self.realtime:
                dt_sleep = (wall0 + t) - time.perf_counter()
                if dt_sleep > 0:
                    time.sleep(dt_sleep)


class SerialRawSource:
    """Read raw frames from the Teensy over USB serial (lazy pyserial)."""

    def __init__(self, port: str, baud: int = 2_000_000) -> None:
        self.port = port
        self.baud = baud

    def stream(self, stop: threading.Event | None = None):
        import time

        import serial

        ser = serial.Serial(self.port, self.baud, timeout=0.05)
        buf = bytearray()
        try:
            while stop is None or not stop.is_set():
                chunk = ser.read(4096)
                if chunk:
                    buf.extend(chunk)
                while True:
                    if len(buf) < _HDR_SIZE:
                        break
                    if buf[0] != _SYNC0 or buf[1] != _SYNC1:
                        del buf[0]
                        continue
                    w, h = buf[4], buf[5]  # [sync0 sync1][seq lo hi][w][h]
                    if not (1 <= w <= 64 and 1 <= h <= 64):
                        del buf[0]
                        continue
                    need = frame_size(w, h)
                    if len(buf) < need:
                        break
                    frame = bytes(buf[:need])
                    if parse_raw_frame(frame) is None:
                        del buf[0]
                        continue
                    del buf[:need]
                    yield time.perf_counter(), frame
        finally:
            ser.close()


@dataclass
class RawLog:
    """In-memory raw-frame log."""

    times: list = field(default_factory=list)
    frames: list = field(default_factory=list)  # list[RawFrame]

    def add(self, t: float, frame: RawFrame) -> None:
        self.times.append(float(t))
        self.frames.append(frame)

    def save(self, path) -> None:
        grids = np.stack([f.grid for f in self.frames]) if self.frames else \
            np.zeros((0, 0, 0), dtype=np.uint16)
        np.savez_compressed(
            path, t=np.asarray(self.times), grids=grids,
            seq=np.asarray([f.seq for f in self.frames], dtype=np.uint32),
            piezo=np.asarray([f.piezo for f in self.frames], dtype=np.uint16),
        )


class LoggedRawSource:
    """Replay a saved RawLog npz through the same pipeline."""

    def __init__(self, path, *, realtime: bool = False) -> None:
        d = np.load(path)
        self.t = d["t"]
        self.grids = d["grids"]
        self.seq = d["seq"]
        self.piezo = d["piezo"]
        self.realtime = realtime

    def stream(self, stop: threading.Event | None = None):
        import time
        wall0 = time.perf_counter()
        t0 = float(self.t[0]) if len(self.t) else 0.0
        for i in range(len(self.t)):
            if stop is not None and stop.is_set():
                return
            t = float(self.t[i])
            frame = RawFrame(seq=int(self.seq[i]), grid=self.grids[i],
                             piezo=int(self.piezo[i]))
            if self.realtime:
                dt_sleep = (wall0 + (t - t0)) - time.perf_counter()
                if dt_sleep > 0:
                    time.sleep(dt_sleep)
            yield t, frame.pack()


class RawInputDriver:
    """Raw source to RawFrameProcessor to ParamBus."""

    def __init__(self, bus: ParamBus, source, config: RawMapConfig | None = None,
                 *, log: RawLog | None = None) -> None:
        self.bus = bus
        self.source = source
        self.processor = RawFrameProcessor(config)
        self.log = log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.gesture = Gesture()
        self.n_frames = 0
        self.n_bad = 0

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
            g = self.processor.update(frame, t)
            self.gesture = g
            self.bus.write(g)
            self.n_frames += 1

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def run(self) -> None:
        self._stop.clear()
        self._run()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

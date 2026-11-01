"""Hardware brush-controller input: pressure-matrix pad to gesture bus."""

from __future__ import annotations

import struct
import threading
from dataclasses import dataclass

import numpy as np

from ..engine.rtaudio import Gesture, ParamBus

_SYNC0 = 0xB2
_SYNC1 = 0x55
# sync0,sync1
_PACKET_FMT = "<BBHHHHHhBB"
PACKET_SIZE = struct.calcsize(_PACKET_FMT)  # 16
_BODY_FMT = "<HHHHHhB"  # bytes [2..14]


def crc8(data: bytes) -> int:
    """CRC-8 (poly 0x07, init 0x00), MSB-first."""
    crc = 0
    for b in data:
        crc ^= b & 0xFF
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


@dataclass
class HardwarePacket:
    """One frame from the pad (raw sensor units, before mapping)."""

    seq: int = 0
    cx: int = 0  # centroid x, 0..65535 over pad width
    cy: int = 0  # centroid y, 0..65535 over pad height
    force: int = 0  # total normal force, raw
    spread: int = 0  # contact-area / bristle-spread proxy, raw
    orient: int = 0  # principal-axis angle
    contact: int = 0  # 1 = touching, 0 = lifted

    def pack(self) -> bytes:
        body = struct.pack(
            _BODY_FMT, self.seq & 0xFFFF, self.cx & 0xFFFF, self.cy & 0xFFFF,
            self.force & 0xFFFF, self.spread & 0xFFFF, int(self.orient), self.contact & 0xFF,
        )
        return bytes((_SYNC0, _SYNC1)) + body + bytes((crc8(body),))

    @classmethod
    def parse(cls, buf: bytes) -> "HardwarePacket | None":
        """Parse a 16-byte frame."""
        if len(buf) != PACKET_SIZE or buf[0] != _SYNC0 or buf[1] != _SYNC1:
            return None
        body = buf[2:15]
        if crc8(body) != buf[15]:
            return None
        seq, cx, cy, force, spread, orient, contact = struct.unpack(_BODY_FMT, body)
        return cls(seq=seq, cx=cx, cy=cy, force=force, spread=spread,
                   orient=orient, contact=contact)


def encode_packet(
    x: float, y: float, pressure: float, angle: float = 0.0, *, seq: int = 0,
    force_floor: float = 0.0, force_full: float = 65535.0,
    spread: float = 0.0, contact: bool | None = None,
) -> HardwarePacket:
    """Build a packet from normalized gesture values (inverse of the mapper)."""
    cx = int(round((float(np.clip(x, -1, 1)) + 1.0) * 0.5 * 65535.0))
    cy = int(round((float(np.clip(y, -1, 1)) + 1.0) * 0.5 * 65535.0))
    force = int(round(force_floor + float(np.clip(pressure, 0, 1)) * (force_full - force_floor)))
    orient = int(round(np.degrees(angle) * 100.0))
    orient = max(-18000, min(18000, orient))
    touch = (pressure > 1e-6) if contact is None else bool(contact)
    return HardwarePacket(seq=seq, cx=max(0, min(65535, cx)), cy=max(0, min(65535, cy)),
                          force=max(0, min(65535, force)), spread=int(spread),
                          orient=orient, contact=1 if touch else 0)


@dataclass
class MappingConfig:
    """Calibration for raw packet to Gesture."""

    force_floor: float = 0.0  # raw force at rest contact to pressure 0
    force_full: float = 65535.0  # raw force at full musical body to pressure 1
    velocity_gain: float = 1.0  # centroid-motion to model |v|
    velocity_smoothing: float = 0.3  # one-pole on derived velocity per frame
    max_raw_speed: float = 14.0  # clamp on each raw velocity component
    invert_y: bool = False


class GestureMapper:
    """Stateful raw-packet to Gesture mapper."""

    def __init__(self, config: MappingConfig | None = None) -> None:
        self.cfg = config or MappingConfig()
        self.reset()

    def reset(self) -> None:
        self._px = 0.0
        self._py = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._pt: float | None = None
        self._have_prev = False

    def update(self, pkt: HardwarePacket, t: float) -> Gesture:
        cfg = self.cfg
        nx = pkt.cx / 65535.0 * 2.0 - 1.0
        ny = pkt.cy / 65535.0 * 2.0 - 1.0
        if cfg.invert_y:
            ny = -ny
        angle = float(np.radians(pkt.orient / 100.0))

        if not pkt.contact:
            # Lifted: no force
            self._vx *= 0.5
            self._vy *= 0.5
            self._have_prev = False
            self._pt = t
            return Gesture(x=self._px, y=self._py, vx=self._vx, vy=self._vy,
                           pressure=0.0, angle=angle, tilt=0.0)

        pressure = (pkt.force - cfg.force_floor) / max(cfg.force_full - cfg.force_floor, 1e-6)
        pressure = float(min(max(pressure, 0.0), 1.0))

        if self._have_prev and self._pt is not None:
            dt = max(t - self._pt, 1e-4)
            mrs = cfg.max_raw_speed
            rvx = float(min(max((nx - self._px) / dt, -mrs), mrs))
            rvy = float(min(max((ny - self._py) / dt, -mrs), mrs))
            sm = cfg.velocity_smoothing
            self._vx = sm * rvx + (1.0 - sm) * self._vx
            self._vy = sm * rvy + (1.0 - sm) * self._vy
        else:
            self._vx = self._vy = 0.0

        self._px, self._py = nx, ny
        self._pt = t
        self._have_prev = True
        vg = cfg.velocity_gain
        return Gesture(x=nx, y=ny, vx=self._vx * vg, vy=self._vy * vg,
                       pressure=pressure, angle=angle, tilt=0.0)


class SimulatedPacketSource:
    """Yield packets from a gesture trace so the pipeline runs with no hardware."""

    def __init__(self, x, y, pressure, angle=None, *, rate_hz: float = 500.0,
                 realtime: bool = False, loop: bool = False,
                 force_floor: float = 0.0, force_full: float = 65535.0) -> None:
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.p = np.asarray(pressure, dtype=np.float64)
        self.ang = (np.zeros_like(self.x) if angle is None
                    else np.asarray(angle, dtype=np.float64))
        n = len(self.x)
        if not (len(self.y) == len(self.p) == len(self.ang) == n):
            raise ValueError("x, y, pressure, angle must be equal length")
        self.rate_hz = float(rate_hz)
        self.dt = 1.0 / self.rate_hz
        self.realtime = realtime
        self.loop = loop
        self.force_floor = force_floor
        self.force_full = force_full

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
            pkt = encode_packet(
                self.x[i], self.y[i], self.p[i], self.ang[i], seq=seq,
                force_floor=self.force_floor, force_full=self.force_full,
            )
            yield t, pkt.pack()
            seq = (seq + 1) & 0xFFFF
            i += 1
            t += self.dt
            if self.realtime:
                target = wall0 + t
                dt_sleep = target - time.perf_counter()
                if dt_sleep > 0:
                    time.sleep(dt_sleep)


class SerialPacketSource:
    """Read framed packets from a real controller over USB serial."""

    def __init__(self, port: str, baud: int = 1_000_000) -> None:
        self.port = port
        self.baud = baud

    def stream(self, stop: threading.Event | None = None):
        import time

        import serial  # pyserial

        ser = serial.Serial(self.port, self.baud, timeout=0.05)
        buf = bytearray()
        try:
            while stop is None or not stop.is_set():
                chunk = ser.read(256)
                if chunk:
                    buf.extend(chunk)
                # Frame: find sync, emit complete packets, drop garbage
                while len(buf) >= PACKET_SIZE:
                    if buf[0] != _SYNC0 or buf[1] != _SYNC1:
                        del buf[0]
                        continue
                    frame = bytes(buf[:PACKET_SIZE])
                    if HardwarePacket.parse(frame) is None:
                        del buf[0]  # bad CRC/framing, resync by one byte
                        continue
                    del buf[:PACKET_SIZE]
                    yield time.perf_counter(), frame
        finally:
            ser.close()


class HardwareInputDriver:
    """Packet source to GestureMapper to ParamBus."""

    def __init__(self, bus: ParamBus, source, mapping: MappingConfig | None = None) -> None:
        self.bus = bus
        self.source = source
        self.mapper = GestureMapper(mapping)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.gesture = Gesture()
        self.n_packets = 0
        self.n_bad = 0

    def _run(self) -> None:
        for t, raw in self.source.stream(self._stop):
            if self._stop.is_set():
                break
            pkt = HardwarePacket.parse(raw)
            if pkt is None:
                self.n_bad += 1
                continue
            g = self.mapper.update(pkt, t)
            self.gesture = g
            self.bus.write(g)
            self.n_packets += 1

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

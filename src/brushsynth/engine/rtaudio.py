"""Real-time audio engine and gesture parameter bus."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd


@dataclass
class Gesture:
    """One sample of brush-hand state."""

    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    pressure: float = 0.0
    angle: float = 0.0
    tilt: float = 0.0


_FIELDS = ("x", "y", "vx", "vy", "pressure", "angle", "tilt")


class ParamBus:
    """Shared gesture state between input thread and audio thread."""

    def __init__(self) -> None:
        self._arr = np.zeros(len(_FIELDS), dtype=np.float64)

    def write(self, g: Gesture) -> None:
        a = self._arr
        a[0] = g.x
        a[1] = g.y
        a[2] = g.vx
        a[3] = g.vy
        a[4] = g.pressure
        a[5] = g.angle
        a[6] = g.tilt

    def snapshot(self) -> Gesture:
        a = self._arr
        return Gesture(
            x=float(a[0]), y=float(a[1]),
            vx=float(a[2]), vy=float(a[3]),
            pressure=float(a[4]), angle=float(a[5]),
            tilt=float(a[6]),
        )


class Engine:
    """Real-time audio engine wrapping the synth and sounddevice."""

    def __init__(
        self,
        synth,
        *,
        sample_rate: int = 48000,
        block_size: int = 256,
        device: int | str | None = None,
    ) -> None:
        self.synth = synth
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.device = device
        self.bus = ParamBus()
        self._stream: sd.OutputStream | None = None
        self._last_gesture = Gesture()
        # Track callback stats for debugging
        self._underrun_count = 0
        self._lock = threading.Lock()

    def _callback(self, outdata, frames, time_info, status) -> None:
        if status:
            if status.output_underflow:
                self._underrun_count += 1
            print(f"[audio] {status}", flush=True)
        g = self.bus.snapshot()
        prev = self._last_gesture
        buf = self.synth.render(
            prev.vx, prev.vy, prev.pressure, prev.angle,
            g.vx, g.vy, g.pressure, g.angle,
            frames,
            prev.x, prev.y, g.x, g.y,
        )
        if buf.ndim == 2:
            outdata[:] = buf
        else:
            outdata[:, 0] = buf
        self._last_gesture = g

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=int(getattr(self.synth, "channels", 1)),  # stereo synths set channels = 2
            dtype="float32",
            callback=self._callback,
            device=self.device,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    @property
    def underrun_count(self) -> int:
        return self._underrun_count

    def __enter__(self) -> "Engine":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

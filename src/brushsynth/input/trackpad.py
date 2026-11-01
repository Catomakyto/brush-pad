"""Trackpad gesture driver: a Force Touch trackpad as a single-point brush."""

from __future__ import annotations

from ..engine.rtaudio import ParamBus
from .mock import MockConfig, MockInputDriver


class TrackpadPressure:
    """Live trackpad force, updated by an NSEvent local monitor."""

    def __init__(self) -> None:
        self._force = 0.0
        self._monitor = None
        self.n_pressure = 0
        self.n_drag = 0
        self.peak = 0.0

    def value(self) -> float:
        return self._force

    def start(self) -> None:
        import AppKit

        self._AppKit = AppKit
        mask = (
            AppKit.NSEventMaskPressure
            | AppKit.NSEventMaskLeftMouseDown
            | AppKit.NSEventMaskLeftMouseDragged
            | AppKit.NSEventMaskLeftMouseUp
        )

        def handler(event):
            etype = event.type()
            if etype == AppKit.NSEventTypeLeftMouseUp:
                # Finger lifted, force returns to zero
                self._force = 0.0
                return event
            try:
                p = float(event.pressure())
            except Exception:
                p = 0.0
            if etype == AppKit.NSEventTypePressure:
                self.n_pressure += 1
            else:
                self.n_drag += 1
            # Force-click pushes pressure past 1.0
            p = 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)
            self._force = p
            if p > self.peak:
                self.peak = p
            return event

        self._monitor = AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            mask, handler
        )

    def stop(self) -> None:
        if self._monitor is not None:
            self._AppKit.NSEvent.removeMonitor_(self._monitor)
            self._monitor = None


class TrackpadInputDriver:
    """Force Touch trackpad to gesture bus."""

    def __init__(self, bus: ParamBus, config: MockConfig | None = None) -> None:
        self.bus = bus
        self.pressure = TrackpadPressure()
        self._driver = MockInputDriver(
            bus,
            config,
            pressure_source=self.pressure.value,
            caption="brushsynth — trackpad input",
        )

    def stop(self) -> None:
        self._driver.stop()

    def run(self) -> None:
        self.pressure.start()
        try:
            self._driver.run()
        finally:
            self.pressure.stop()
            if self.pressure.n_pressure == 0 and self.pressure.n_drag == 0:
                print(
                    "[trackpad] WARNING: no pressure/drag events seen — did you "
                    "press-and-drag on the trackpad with the window focused?",
                    flush=True,
                )
            else:
                print(
                    f"[trackpad] peak force {self.pressure.peak:.3f}  "
                    f"(pressure-events={self.pressure.n_pressure}, "
                    f"drag-events={self.pressure.n_drag})",
                    flush=True,
                )

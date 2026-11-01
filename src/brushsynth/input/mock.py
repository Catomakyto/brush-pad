"""Mock gesture driver."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..engine.rtaudio import Gesture, ParamBus


@dataclass
class MockConfig:
    window_size: int = 480
    control_rate_hz: float = 200.0
    velocity_smoothing: float = 0.35
    # Multiplier on smoothed velocity
    velocity_gain: float = 1.0
    # Clamp each raw velocity component before smoothing
    max_raw_speed: float = 14.0
    pressure_attack: float = 6.0  # 1/s
    pressure_release: float = 4.0  # 1/s
    angle_slew_rate: float = 3.0  # rad/s
    # One-pole smoothing on an *external* pressure source
    pressure_smoothing: float = 0.6
    show_hud: bool = True


class MockInputDriver:
    """Runs a pygame window."""

    def __init__(
        self,
        bus: ParamBus,
        config: MockConfig | None = None,
        *,
        pressure_source: Callable[[], float] | None = None,
        caption: str = "brushsynth — mock input",
    ) -> None:
        self.bus = bus
        self.cfg = config or MockConfig()
        self._stop = threading.Event()
        self.gesture = Gesture()
        # Optional external pressure
        self.pressure_source = pressure_source
        self.caption = caption

    def stop(self) -> None:
        """Ask the pygame loop to exit at the next iteration."""
        self._stop.set()

    def run(self) -> None:
        """Block this (main) thread running the pygame event loop."""
        self._stop.clear()
        self._run_loop()

    def _run_loop(self) -> None:
        # Import pygame lazily so unit tests don't pull it in
        import os
        import pygame

        os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "80,80")

        # Init only display and font
        print("[mock-input] pygame.display.init() ...", flush=True)
        pygame.display.init()
        pygame.font.init()
        print(f"[mock-input] init done; drivers: video={pygame.display.get_driver() if pygame.display.get_init() else None}",
              flush=True)
        size = self.cfg.window_size
        print(f"[mock-input] calling set_mode({size},{size})", flush=True)
        screen = pygame.display.set_mode((size, size))
        print(f"[mock-input] set_mode returned driver={pygame.display.get_driver()}", flush=True)
        pygame.display.set_caption(self.caption)

        screen.fill((10, 10, 12))
        pygame.display.flip()
        for _ in range(3):
            pygame.event.pump()

        w, h = pygame.display.get_surface().get_size()
        print(f"[mock-input] window {w}x{h} ready", flush=True)

        clock = pygame.time.Clock()
        font = pygame.font.SysFont("menlo", 14)

        # State
        x, y = 0.0, 0.0
        vx, vy = 0.0, 0.0
        pressure = 0.0
        pressure_target = 0.0
        tap_pulse = 0.0  # quick decaying pressure pulse for taps
        angle = 0.0
        angle_target = 0.0
        prev_t = time.perf_counter()
        prev_x, prev_y = 0.0, 0.0

        running = True
        while running and not self._stop.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    tap_pulse = 0.9  # spacebar = a brush tap
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pressure_target = 1.0
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    pressure_target = 0.0
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    tap_pulse = 0.9  # right-click = a brush tap too
                elif event.type == pygame.MOUSEWHEEL:
                    pressure_target = float(np.clip(pressure_target + 0.1 * event.y, 0.0, 1.0))

            # Sample current input
            mx, my = pygame.mouse.get_pos()
            nx = (mx / size) * 2.0 - 1.0
            ny = 1.0 - (my / size) * 2.0
            now = time.perf_counter()
            dt = max(now - prev_t, 1e-4)
            mrs = self.cfg.max_raw_speed
            raw_vx = float(np.clip((nx - prev_x) / dt, -mrs, mrs))
            raw_vy = float(np.clip((ny - prev_y) / dt, -mrs, mrs))
            sm = self.cfg.velocity_smoothing
            vx = sm * raw_vx + (1.0 - sm) * vx
            vy = sm * raw_vy + (1.0 - sm) * vy

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                angle_target -= self.cfg.angle_slew_rate * dt
            if keys[pygame.K_RIGHT]:
                angle_target += self.cfg.angle_slew_rate * dt

            if self.pressure_source is not None:
                # Real force sensor: read latest value, one-pole smooth it
                raw_p = float(np.clip(self.pressure_source(), 0.0, 1.0))
                pressure += self.cfg.pressure_smoothing * (raw_p - pressure)
                pressure_out = float(np.clip(pressure, 0.0, 1.0))
            else:
                # Smooth pressure (sustained contact for sweeps/scratches)
                rate = self.cfg.pressure_attack if pressure_target > pressure else self.cfg.pressure_release
                pressure += (pressure_target - pressure) * (1.0 - np.exp(-rate * dt))
                # Tap pulse
                tap_pulse *= float(np.exp(-dt / 0.025))  # ~25 ms decay
                pressure_out = float(np.clip(pressure + tap_pulse, 0.0, 1.0))
            # Smooth angle (critically damped pull toward target)
            angle += (angle_target - angle) * (1.0 - np.exp(-8.0 * dt))

            prev_t = now
            prev_x, prev_y = nx, ny

            # Push to bus
            vg = self.cfg.velocity_gain
            g = Gesture(
                x=float(nx), y=float(ny),
                vx=float(vx * vg), vy=float(vy * vg),
                pressure=pressure_out,
                angle=float(angle),
                tilt=0.0,
            )
            self.gesture = g
            self.bus.write(g)

            # Draw
            screen.fill((10, 10, 12))
            # Brush cursor
            cx = int((nx + 1.0) * 0.5 * size)
            cy = int((1.0 - ny) * 0.5 * size)
            radius = int(6 + 40 * pressure)
            col_intensity = int(60 + 180 * pressure)
            pygame.draw.circle(screen, (col_intensity, col_intensity, col_intensity), (cx, cy), radius, width=2)
            # Heading tick
            hx = int(cx + radius * np.cos(angle))
            hy = int(cy - radius * np.sin(angle))
            pygame.draw.line(screen, (220, 120, 50), (cx, cy), (hx, hy), 2)

            if self.cfg.show_hud:
                hud = (
                    f"pos ({nx:+.2f}, {ny:+.2f})   |v| {np.hypot(vx, vy) * vg:.2f}  "
                    f"(gain x{vg:.1f})\n"
                    f"pressure {pressure:.2f}   angle {angle:+.2f} rad"
                )
                for i, line in enumerate(hud.split("\n")):
                    surf = font.render(line, True, (210, 210, 210))
                    screen.blit(surf, (12, 12 + i * 18))
                help_lines = [
                    "move: sweep   hold LMB: press   scroll: bias pressure",
                    "SPACE / right-click: tap   ←/→: rotate   Esc: quit",
                ]
                for i, line in enumerate(help_lines):
                    surf = font.render(line, True, (120, 120, 120))
                    screen.blit(surf, (12, size - 18 * (len(help_lines) - i) - 10))

            pygame.display.flip()
            clock.tick(self.cfg.control_rate_hz)

        pygame.quit()
        self._stop.set()

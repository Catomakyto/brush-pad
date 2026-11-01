"""Probe: does this Mac's trackpad report live pressure to us."""

from __future__ import annotations

import time

import AppKit
import pygame


def main() -> int:
    # Shared mutable state written by the NSEvent monitor
    state = {"pressure": 0.0, "stage": 0, "n_pressure": 0, "n_drag": 0, "raw": 0.0}

    mask = (
        AppKit.NSEventMaskPressure
        | AppKit.NSEventMaskLeftMouseDragged
        | AppKit.NSEventMaskLeftMouseDown
        | AppKit.NSEventMaskLeftMouseUp
    )

    def handler(event):
        etype = event.type()
        try:
            p = float(event.pressure())
        except Exception:
            p = 0.0
        state["raw"] = p
        if etype == AppKit.NSEventTypePressure:
            state["n_pressure"] += 1
            state["pressure"] = p
            try:
                state["stage"] = int(event.stage())
            except Exception:
                state["stage"] = 0
        else:
            state["n_drag"] += 1
            # Dragged events also carry pressure on tablets/Force Touch
            state["pressure"] = p
        return event  # pass the event through untouched

    monitor = AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(mask, handler)

    pygame.init()
    size = 420
    screen = pygame.display.set_mode((size, size))
    pygame.display.set_caption("trackpad pressure probe")
    font = pygame.font.SysFont("menlo", 15)
    clock = pygame.time.Clock()

    print("Press/drag on the trackpad while this window is focused. Esc to quit.", flush=True)
    peak = 0.0
    t0 = time.perf_counter()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        p = state["pressure"]
        peak = max(peak, p)
        screen.fill((12, 12, 16))
        bar_h = int(min(p, 2.0) / 2.0 * (size - 40))
        pygame.draw.rect(screen, (60, 200, 120), (size - 60, size - 20 - bar_h, 30, bar_h))
        lines = [
            f"pressure {p:5.3f}   peak {peak:5.3f}   stage {state['stage']}",
            f"pressure-events {state['n_pressure']}   drag-events {state['n_drag']}",
            f"elapsed {time.perf_counter() - t0:5.1f}s",
            "",
            "If pressure moves when you press: Force Touch works.",
            "If it stays 0: pressure isn't reaching us.",
        ]
        for i, ln in enumerate(lines):
            screen.blit(font.render(ln, True, (210, 210, 210)), (14, 14 + i * 20))
        pygame.display.flip()
        clock.tick(120)

    AppKit.NSEvent.removeMonitor_(monitor)
    pygame.quit()
    print(f"peak pressure seen: {peak:.3f}  "
          f"(pressure-events={state['n_pressure']}, drag-events={state['n_drag']})",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

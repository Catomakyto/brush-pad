"""Interactive mouse-driven demo."""

from __future__ import annotations

import argparse
import sys

from brushsynth.engine.rtaudio import Engine
from brushsynth.input.mock import MockConfig, MockInputDriver
from brushsynth.synth import BrushSynth, SynthConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", type=int, default=48000, help="sample rate")
    ap.add_argument("--block", type=int, default=256, help="audio block size")
    ap.add_argument("--device", default=None, help="sounddevice output device index/name")
    ap.add_argument("--no-audio", action="store_true",
                    help="skip audio start (useful for debugging the pygame window alone)")
    args = ap.parse_args()

    synth = BrushSynth(config=SynthConfig(sample_rate=float(args.sr)))
    engine = Engine(synth, sample_rate=args.sr, block_size=args.block, device=args.device)
    driver = MockInputDriver(engine.bus, MockConfig(window_size=480))

    print(
        f"Starting audio at {args.sr} Hz, block {args.block}. "
        "Move the mouse to sweep. Hold LMB to press down. Esc to quit.",
        flush=True,
    )
    try:
        if not args.no_audio:
            # Audio runs on its own thread via sounddevice's callback
            engine.start()
        else:
            print("(--no-audio: skipping audio engine)", flush=True)
        # pygame MUST run on the main thread on macOS (SDL/Cocoa)
        driver.run()
    except KeyboardInterrupt:
        pass
    finally:
        driver.stop()
        if not args.no_audio:
            engine.stop()
            if engine.underrun_count:
                print(f"[audio] {engine.underrun_count} underruns", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Live trackpad-driven sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Fall back to the src tree if the install is missing
try:
    import brushsynth  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brushsynth.engine.rtaudio import Engine
from brushsynth.input.mock import MockConfig
from brushsynth.input.trackpad import TrackpadInputDriver
from brushsynth.synth import BrushSynth
from brushsynth.synth.presets import preset_params


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", type=int, default=48000, help="sample rate")
    ap.add_argument("--block", type=int, default=256, help="audio block size")
    ap.add_argument("--device", default=None, help="sounddevice output device index/name")
    ap.add_argument("--window", type=int, default=480, help="pad window size in px")
    ap.add_argument("--vgain", type=float, default=8.0,
                    help="velocity gain: scales trackpad motion into the synth's |v|≈3–5 range")
    ap.add_argument("--no-audio", action="store_true",
                    help="skip audio start (debug the window/pressure alone)")
    args = ap.parse_args()

    # The live_sweep voice
    brush, cfg = preset_params("live_sweep", float(args.sr))
    synth = BrushSynth(config=cfg, brush_params=brush)
    engine = Engine(synth, sample_rate=args.sr, block_size=args.block, device=args.device)
    # Gentler smoothing than the mouse mock
    driver = TrackpadInputDriver(engine.bus, MockConfig(
        window_size=args.window,
        velocity_smoothing=0.14,
        pressure_smoothing=0.28,
        angle_slew_rate=0.0,
        velocity_gain=args.vgain,
    ))

    print(
        f"Starting audio at {args.sr} Hz, block {args.block}. "
        "Press-and-drag on the trackpad to sweep; press harder for more body. "
        "Esc to quit.",
        flush=True,
    )
    try:
        if not args.no_audio:
            engine.start()
        else:
            print("(--no-audio: skipping audio engine)", flush=True)
        # pygame + the NSEvent monitor MUST run on the main thread on macOS
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

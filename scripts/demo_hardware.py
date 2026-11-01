"""Live sweep driven by the hardware brush controller, or a simulator."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Fall back to the src tree if the install is missing
try:
    import brushsynth  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brushsynth.engine.rtaudio import Engine
from brushsynth.input.hardware import (
    HardwareInputDriver,
    MappingConfig,
    SerialPacketSource,
    SimulatedPacketSource,
)
from brushsynth.input.rawframe import (
    RawInputDriver,
    RawLog,
    RawMapConfig,
    SerialRawSource,
    SimulatedRawSource,
)
from brushsynth.synth import BrushSynth
from brushsynth.synth.ensemble import BrushEnsemble
from brushsynth.synth.grain_brush import GrainBrushSynth, GrainConfig
from brushsynth.input.multibrush import MultiBrushDriver
from brushsynth.synth.presets import preset_params


def _circular_gesture(ctrl_hz: float, dur: float, radius: float, freq: float, pressure: float):
    """A steady circular stir."""
    n = int(ctrl_hz * dur)
    th = np.linspace(0.0, 2.0 * np.pi * freq * dur, n)
    x = radius * np.cos(th)
    y = radius * np.sin(th)
    p = np.full(n, pressure)
    ang = np.zeros(n)
    return x, y, p, ang


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", type=int, default=48000, help="sample rate")
    ap.add_argument("--block", type=int, default=256, help="audio block size")
    ap.add_argument("--device", default=None, help="sounddevice output device")
    ap.add_argument("--serial", default=None, help="serial port, processed-packet mode")
    ap.add_argument("--serial-raw", default=None, help="serial port, RAW-frame mode (dev default on real HW)")
    ap.add_argument("--raw", action="store_true", help="simulate RAW taxel frames (full pad pipeline)")
    ap.add_argument("--baud", type=int, default=1_000_000, help="serial baud")
    ap.add_argument("--rate", type=float, default=500.0, help="simulated packet/frame rate (Hz)")
    ap.add_argument("--vgain", type=float, default=1.0, help="velocity gain (calibrate to |v|≈3–5)")
    ap.add_argument("--cal", default=None, help="calibration JSON from calibrate_pad.py (raw modes)")
    ap.add_argument("--log", default=None, help="save raw frames to this .npz for offline replay")
    ap.add_argument("--seconds", type=float, default=0.0, help="run duration; 0 = until Ctrl-C")
    ap.add_argument("--no-audio", action="store_true", help="skip audio (pipeline smoke test)")
    ap.add_argument("--brushes", type=int, choices=(1, 2), default=2,
                    help="pad (--serial-raw): 2 = two-brush tracking + hits (default), 1 = single brush")
    ap.add_argument("--no-hits", action="store_true", help="two-brush mode without tap/hit samples")
    ap.add_argument("--pan-flip", action="store_true", help="swap left/right of the two-brush stereo")
    ap.add_argument("--synth", choices=("grain", "model"), default="grain",
                    help="grain = real brush recordings re-played by the gesture (default); "
                         "model = the physical-model BrushSynth (live_sweep preset)")
    args = ap.parse_args()

    multi = bool(args.serial_raw) and args.brushes == 2 and args.synth == "grain"
    if multi:
        synth = BrushEnsemble(float(args.sr), hits=not args.no_hits, pan_flip=args.pan_flip)
    elif args.synth == "grain":
        synth = GrainBrushSynth(GrainConfig(sample_rate=float(args.sr)))
    else:
        brush, cfg = preset_params("live_sweep", float(args.sr))
        synth = BrushSynth(config=cfg, brush_params=brush)
    engine = Engine(synth, sample_rate=args.sr, block_size=args.block, device=args.device)

    log = RawLog() if args.log else None
    if args.serial_raw:
        port = args.serial_raw
        if port == "auto":
            import glob
            ports = sorted(glob.glob("/dev/cu.usbmodem*"))
            if not ports:
                print("No /dev/cu.usbmodem* found — is the Teensy plugged in?", file=sys.stderr)
                return 1
            port = ports[0]
        if args.cal:
            import dataclasses
            import json
            fields = {f.name for f in dataclasses.fields(RawMapConfig)}
            raw_cfg = RawMapConfig(**{k: v for k, v in json.load(open(args.cal)).items()
                                      if k in fields})
        else:
            raw_cfg = RawMapConfig(velocity_gain=args.vgain)
        source = SerialRawSource(port, max(args.baud, 2_000_000))
        if multi:
            driver = MultiBrushDriver([engine.bus, synth.bus2], source, raw_cfg,
                                      on_hit=synth.trigger_hit, log=log)
        else:
            driver = RawInputDriver(engine.bus, source, raw_cfg, log=log)
        args.serial_raw = port
        mode = "two brushes + hits" if multi else "one brush"
        print(f"RAW frames from {args.serial_raw} ({mode}).", flush=True)
    elif args.raw:
        # Full pad pipeline
        rate = min(args.rate, 400.0)
        x, y, p, _ = _circular_gesture(rate, dur=8.0, radius=0.6, freq=0.8, pressure=0.55)
        idle = int(0.25 * rate)
        import numpy as _np
        x = _np.concatenate([_np.zeros(idle), x])
        y = _np.concatenate([_np.zeros(idle), y])
        p = _np.concatenate([_np.zeros(idle), p])
        source = SimulatedRawSource(x, y, p, rate_hz=rate, noise=2.0,
                                    realtime=True, loop=True)
        driver = RawInputDriver(engine.bus, source,
                                RawMapConfig(force_full=8000.0, velocity_gain=args.vgain),
                                log=log)
        print("Simulated RAW-frame sweep (full pad pipeline, no hardware).", flush=True)
    elif args.serial:
        source = SerialPacketSource(args.serial, args.baud)
        driver = HardwareInputDriver(engine.bus, source, MappingConfig(velocity_gain=args.vgain))
        print(f"Reading controller on {args.serial} @ {args.baud} baud.", flush=True)
    else:
        x, y, p, ang = _circular_gesture(args.rate, dur=8.0, radius=0.6, freq=0.8, pressure=0.55)
        source = SimulatedPacketSource(x, y, p, ang, rate_hz=args.rate, realtime=True, loop=True)
        driver = HardwareInputDriver(engine.bus, source, MappingConfig(velocity_gain=args.vgain))
        print("Simulated circular sweep (no hardware). Use --raw for the pad pipeline.", flush=True)

    print(f"Audio {args.sr} Hz block {args.block}. Ctrl-C to stop.", flush=True)
    try:
        if not args.no_audio:
            engine.start()
        driver.start()
        if args.seconds > 0:
            time.sleep(args.seconds)
        else:
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        driver.stop()
        if not args.no_audio:
            engine.stop()
        n_ok = getattr(driver, "n_packets", None) or getattr(driver, "n_frames", 0)
        hits = f" hits={driver.n_hits}" if hasattr(driver, "n_hits") else ""
        print(f"[hardware] frames={n_ok} bad={driver.n_bad}{hits} "
              f"underruns={engine.underrun_count}", file=sys.stderr)
        if log is not None and log.frames:
            log.save(args.log)
            print(f"[hardware] raw log saved: {args.log} ({len(log.frames)} frames)",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

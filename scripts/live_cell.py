"""Live bar display for bring-up firmware (bringup_1x1 / bringup_16x1)."""

from __future__ import annotations

import glob
import re
import sys
import time

import serial


def main() -> int:
    ports = sorted(glob.glob("/dev/cu.usbmodem*"))
    if not ports:
        print("No /dev/cu.usbmodem* found — is the Teensy plugged in?")
        return 1
    s = serial.Serial(ports[0], 115200, timeout=0.2)
    time.sleep(0.3)
    s.reset_input_buffer()
    s.write(b"t")
    buf = ""
    try:
        while True:
            buf += s.read(4096).decode(errors="replace")
            frames = buf.split("--- frame")
            if len(frames) < 3:
                continue
            last = frames[-2]  # newest complete frame
            buf = "--- frame" + frames[-1]
            rows = re.findall(r"^R\d+((?:[ \t]+\d+)+)", last, re.M)
            out = ["  ".join(f"{int(v):5d}" for v in r.split()) for r in rows]
            sys.stdout.write("\x1b[H\x1b[J" + "\n".join(out) + "\n")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        s.write(b"b")
        s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

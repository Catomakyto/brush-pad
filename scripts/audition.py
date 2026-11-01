"""Play every .wav in a folder in turn, announcing each filename."""

from __future__ import annotations

import glob
import sys

import soundfile as sf


def main() -> int:
    import sounddevice as sd  # lazy

    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(folder.rstrip("/") + "/*.wav"))
    if not files:
        print(f"no .wav files in {folder}")
        return 1
    print(f"{len(files)} clips in {folder}   ([enter]=next, r=replay, q=quit)\n")
    for f in files:
        print(f"  ▶ {f}")
        while True:
            y, s = sf.read(f)
            sd.play(y, s)
            sd.wait()
            ans = input("    [enter=next / r=replay / q=quit] ").strip().lower()
            if ans == "r":
                continue
            if ans == "q":
                return 0
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

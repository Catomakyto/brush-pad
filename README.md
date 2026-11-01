SORRY I DIDN'T DO A VERSION-CONTROL! BUT DON'T WORRY, MY AP CS TEACHER WOULD FLAME ME FIRST!

# brush pad

A pressure-sensing pad and real-time synthesizer for playing jazz brushes.

## Status (2026-10-31)

Still working on the sound! 


## Run it

Hardware:

```bash
uv sync
pio run -d firmware/brushpad -e half_16x8 -t upload     # flash the 16x8 scanner
.venv/bin/python scripts/view_pad.py                    # heat map 
.venv/bin/python scripts/calibrate_pad.py               # rest / feather / medium / firm 
.venv/bin/python scripts/demo_hardware.py --serial-raw auto --cal calibration/pad.json
```

No hardware:

```bash
.venv/bin/python scripts/demo_hardware.py --raw         # simulated pad
.venv/bin/python scripts/render_grain_demo.py           #a scripted performance to evals/grain/demo.wav
.venv/bin/python -m pytest                              # tests
```

Firmware environments: `half_16x8` (current), `teensy40` (16x16), `bringup_1x1`, `bringup_1x4`, `bringup_1x8`, `bringup_16x1`, and `diag` (wiring check).

## Layout

- `firmware/brushpad/` - Teensy scanner and wiring diagnostic
- `src/brushsynth/input/` - frame parsing, baseline, dead-strip fill, brush tracking (`rawframe.py`, `multibrush.py`)
- `src/brushsynth/synth/` - `grain_brush.py` (brush voice), `hits.py` (brush hits), `ensemble.py` (two brushes + hits). `brush_synth.py`, `excitation.py`, `modal.py` are the earlier physical-model synth, kept for reference (replaced because it sounded metallic)
- `scripts/` - heat map, calibration, live demo, A/B rendering
- `evals/grain/ab/` - A/B clips used to choose the final voicing

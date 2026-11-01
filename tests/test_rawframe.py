"""Tests for the raw-frame input path (input/rawframe.py)."""

from __future__ import annotations

import numpy as np
import pytest

from brushsynth.engine.rtaudio import ParamBus
from brushsynth.input.rawframe import (
    AlphaBeta2D,
    BaselineTracker,
    LoggedRawSource,
    RawFrame,
    RawInputDriver,
    RawLog,
    RawMapConfig,
    RawFrameProcessor,
    SerialRawSource,
    SimulatedRawSource,
    crc16_ccitt,
    extract_blobs,
    frame_size,
    parse_raw_frame,
)


# protocol


def test_serial_source_locks_on_through_junk_and_high_seq():
    """Real byte stream through a pty."""
    import os
    import threading
    import time
    import tty

    pytest.importorskip("serial")
    master, slave = os.openpty()
    tty.setraw(slave)
    grids = [np.full((16, 16), 680 + k, np.uint16) for k in range(5)]
    stop = threading.Event()

    def writer():
        time.sleep(0.3)  # pyserial flushes input on open
        os.write(master, b"\x00junk\xb3")
        for k, g in enumerate(grids):
            os.write(master, RawFrame(seq=300 + k, grid=g).pack())
            if k == 2:
                os.write(master, b"\xb3\x55garbage")

    threading.Thread(target=writer, daemon=True).start()
    timer = threading.Timer(5.0, stop.set)
    timer.start()
    got = []
    for _, buf in SerialRawSource(os.ttyname(slave), baud=115200).stream(stop):
        got.append(parse_raw_frame(buf))
        if len(got) == len(grids):
            stop.set()
    timer.cancel()
    assert [f.seq for f in got] == [300, 301, 302, 303, 304]
    assert all(np.array_equal(f.grid, g) for f, g in zip(got, grids))

def test_crc16_ccitt_known_vector():
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_frame_sizes():
    assert frame_size(16, 16) == 522
    assert frame_size(8, 8) == 138


def test_raw_frame_roundtrip_16x16():
    rng = np.random.default_rng(0)
    grid = rng.integers(0, 4096, size=(16, 16)).astype(np.uint16)
    f = RawFrame(seq=4321, grid=grid, piezo=777)
    buf = f.pack()
    assert len(buf) == frame_size(16, 16)
    parsed = parse_raw_frame(buf)
    assert parsed is not None
    assert parsed.seq == 4321
    assert parsed.piezo == 777
    assert np.array_equal(parsed.grid, grid)


def test_raw_frame_roundtrip_8x8():
    grid = np.arange(64, dtype=np.uint16).reshape(8, 8)
    parsed = parse_raw_frame(RawFrame(seq=1, grid=grid).pack())
    assert parsed is not None and parsed.w == 8 and parsed.h == 8
    assert np.array_equal(parsed.grid, grid)


def test_raw_frame_rejects_bad_sync_crc_and_length():
    grid = np.zeros((8, 8), dtype=np.uint16)
    good = bytearray(RawFrame(seq=5, grid=grid).pack())
    bad_sync = bytearray(good)
    bad_sync[0] ^= 0xFF
    assert parse_raw_frame(bytes(bad_sync)) is None
    bad_crc = bytearray(good)
    bad_crc[10] ^= 0x01
    assert parse_raw_frame(bytes(bad_crc)) is None
    assert parse_raw_frame(bytes(good[:-3])) is None


# baseline

def test_baseline_seeds_and_tracks_idle_drift():
    bt = BaselineTracker(touch_margin=6.0, adapt_coeff=0.2)
    g0 = np.full((8, 8), 100.0)
    act = bt.update(g0)
    assert np.all(act == 0.0)  # seeded
    # Slow drift within the margin gets absorbed
    g1 = np.full((8, 8), 104.0)
    for _ in range(40):
        act = bt.update(g1)
    assert float(act.max()) < 1.0


def test_baseline_frozen_under_press():
    bt = BaselineTracker(touch_margin=6.0, adapt_coeff=0.2)
    bt.update(np.full((8, 8), 100.0))
    pressed = np.full((8, 8), 100.0)
    pressed[4, 4] = 300.0
    for _ in range(50):
        act = bt.update(pressed)
    # The pressed taxel's baseline must NOT have eaten the signal
    assert act[4, 4] > 150.0


# blob features

def _gaussian_blob(w, h, cx, cy, amp, sx, sy=None, theta=0.0):
    sy = sx if sy is None else sy
    jj, ii = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    dx, dy = jj - cx, ii - cy
    c, s = np.cos(theta), np.sin(theta)
    u = c * dx + s * dy
    v = -s * dx + c * dy
    return amp * np.exp(-(u * u / (2 * sx * sx) + v * v / (2 * sy * sy)))


def test_blob_centroid_accuracy_subtaxel():
    act = _gaussian_blob(16, 16, cx=7.3, cy=4.6, amp=800.0, sx=1.2)
    blobs = extract_blobs(act, threshold=6.0)
    assert len(blobs) == 1
    b = blobs[0]
    assert b.cx == pytest.approx(7.3, abs=0.05)
    assert b.cy == pytest.approx(4.6, abs=0.05)
    assert 0.8 < b.spread < 2.2
    assert b.area > 10


def test_blob_orientation_of_elongated_contact():
    theta = np.radians(30.0)
    act = _gaussian_blob(16, 16, cx=7.5, cy=7.5, amp=400.0, sx=2.2, sy=0.8, theta=theta)
    b = extract_blobs(act, threshold=6.0)[0]
    assert b.orient == pytest.approx(theta, abs=np.radians(6.0))


def test_two_blob_segmentation_orders_by_force():
    act = (_gaussian_blob(16, 16, cx=3.0, cy=3.0, amp=300.0, sx=1.0)
           + _gaussian_blob(16, 16, cx=12.0, cy=12.0, amp=700.0, sx=1.0))
    blobs = extract_blobs(act, threshold=6.0, max_blobs=2)
    assert len(blobs) == 2
    assert blobs[0].force > blobs[1].force
    assert blobs[0].cx == pytest.approx(12.0, abs=0.1)
    assert blobs[1].cx == pytest.approx(3.0, abs=0.1)


def test_no_blob_below_threshold():
    act = _gaussian_blob(16, 16, cx=8.0, cy=8.0, amp=4.0, sx=1.2)  # feather < threshold
    assert extract_blobs(act, threshold=6.0) == []


# processor: mapping, contact, velocity

def _run_processor(proc, source, n):
    """Feed n frames from a SimulatedRawSource through the processor."""
    out = []
    for i, (t, raw) in enumerate(source.stream()):
        if i >= n:
            break
        frame = parse_raw_frame(raw)
        assert frame is not None
        out.append(proc.update(frame, t))
    return out


def test_processor_position_and_pressure_mapping():
    n_idle, n_press = 10, 30
    x = np.concatenate([np.zeros(n_idle), np.full(n_press, 0.5)])
    y = np.concatenate([np.zeros(n_idle), np.full(n_press, -0.25)])
    p = np.concatenate([np.zeros(n_idle), np.full(n_press, 0.6)])
    src = SimulatedRawSource(x, y, p, noise=0.0)
    proc = RawFrameProcessor(RawMapConfig(force_full=8000.0))
    gs = _run_processor(proc, src, n_idle + n_press)
    g = gs[-1]
    assert g.x == pytest.approx(0.5, abs=0.03)
    assert g.y == pytest.approx(-0.25, abs=0.03)
    assert 0.3 < g.pressure < 0.9
    # Idle frames really were no-contact
    assert gs[n_idle - 1].pressure == 0.0


def test_processor_contact_gate_and_no_recontact_spike():
    n = 10
    x = np.concatenate([np.zeros(n), np.full(20, -0.5), np.zeros(5), np.full(3, 0.5)])
    y = np.zeros_like(x)
    p = np.concatenate([np.zeros(n), np.full(20, 0.6), np.zeros(5), np.full(3, 0.6)])
    src = SimulatedRawSource(x, y, p, noise=0.0)
    proc = RawFrameProcessor(RawMapConfig(force_full=8000.0))
    gs = _run_processor(proc, src, len(x))
    lifted = gs[n + 20 + 2]
    assert lifted.pressure == 0.0
    recontact = gs[n + 20 + 5]  # first frame back in contact
    assert recontact.pressure > 0.0
    assert abs(recontact.vx) < 0.5  # tracker reset


def test_velocity_noise_at_rest_below_feel_bound():
    # Brush at rest: |v| noise < 0.15 model units
    n_idle, n_hold = 10, 800  # 2 s at 400 Hz
    x = np.concatenate([np.zeros(n_idle), np.full(n_hold, 0.2)])
    y = np.concatenate([np.zeros(n_idle), np.full(n_hold, 0.1)])
    p = np.concatenate([np.zeros(n_idle), np.full(n_hold, 0.5)])
    src = SimulatedRawSource(x, y, p, noise=2.0, seed=11)
    proc = RawFrameProcessor(RawMapConfig(force_full=8000.0))
    gs = _run_processor(proc, src, n_idle + n_hold)
    v = np.array([[g.vx, g.vy] for g in gs[n_idle + 200:]])
    assert float(np.abs(v).std(axis=0).max()) < 0.15
    assert float(np.abs(v).max()) < 0.5


def test_velocity_tracks_circle_with_low_lag():
    # Velocity group delay ≤ ~20 ms
    rate, dur, r, f = 400.0, 2.0, 0.6, 0.8
    n_idle = 10
    tt = np.arange(int(rate * dur)) / rate
    x = np.concatenate([np.zeros(n_idle), r * np.cos(2 * np.pi * f * tt)])
    y = np.concatenate([np.zeros(n_idle), r * np.sin(2 * np.pi * f * tt)])
    p = np.concatenate([np.zeros(n_idle), np.full(tt.size, 0.5)])
    src = SimulatedRawSource(x, y, p, noise=0.0)
    proc = RawFrameProcessor(RawMapConfig(force_full=8000.0))
    gs = _run_processor(proc, src, len(x))

    skip = n_idle + int(0.5 * rate)  # discard the settling transient
    vx_est = np.array([g.vx for g in gs[skip:]])
    t_est = (np.arange(len(gs)) / rate)[skip:]
    vx_true = -r * 2 * np.pi * f * np.sin(2 * np.pi * f * (t_est - n_idle / rate))

    # Amplitude sanity
    assert float(np.abs(vx_est).max()) == pytest.approx(r * 2 * np.pi * f, rel=0.15)

    # Lag via cross-correlation peak
    a = vx_est - vx_est.mean()
    b = vx_true - vx_true.mean()
    lags = np.arange(-40, 41)
    corr = [float(np.dot(a[max(0, -k):len(a) - max(0, k)],
                         b[max(0, k):len(b) - max(0, -k)])) for k in lags]
    lag_ms = -lags[int(np.argmax(corr))] / rate * 1000.0
    assert abs(lag_ms) <= 20.0


# end to end + logging

def test_simulated_raw_source_drives_bus_end_to_end():
    rate, r, f = 400.0, 0.6, 0.8
    n_idle, n_move = 20, 200
    tt = np.arange(n_move) / rate
    x = np.concatenate([np.zeros(n_idle), r * np.cos(2 * np.pi * f * tt)])
    y = np.concatenate([np.zeros(n_idle), r * np.sin(2 * np.pi * f * tt)])
    p = np.concatenate([np.zeros(n_idle), np.full(n_move, 0.5)])
    src = SimulatedRawSource(x, y, p, noise=1.0)
    bus = ParamBus()
    drv = RawInputDriver(bus, src, RawMapConfig(force_full=8000.0))
    drv.run()
    assert drv.n_frames == n_idle + n_move
    assert drv.n_bad == 0
    g = bus.snapshot()
    assert np.hypot(g.x, g.y) == pytest.approx(r, abs=0.05)
    assert g.pressure > 0.2


def test_driver_counts_bad_frames():
    class GarbageSource:
        def stream(self, stop=None):
            yield 0.0, b"\x00" * frame_size(8, 8)
            yield 0.0025, RawFrame(seq=1, grid=np.zeros((8, 8), dtype=np.uint16)).pack()

    drv = RawInputDriver(ParamBus(), GarbageSource())
    drv.run()
    assert drv.n_bad == 1
    assert drv.n_frames == 1


def test_raw_log_save_and_replay(tmp_path):
    x = np.concatenate([np.zeros(5), np.full(10, 0.3)])
    y = np.zeros_like(x)
    p = np.concatenate([np.zeros(5), np.full(10, 0.5)])
    src = SimulatedRawSource(x, y, p, noise=0.0)
    log = RawLog()
    drv = RawInputDriver(ParamBus(), src, log=log)
    drv.run()
    path = tmp_path / "frames.npz"
    log.save(path)

    replay = LoggedRawSource(path)
    frames = [parse_raw_frame(raw) for _, raw in replay.stream()]
    assert len(frames) == len(x)
    assert all(fr is not None for fr in frames)
    assert np.array_equal(frames[-1].grid, log.frames[-1].grid)


def test_pressure_gamma_lifts_soft_end_and_keeps_full_scale():
    cfg_lin = RawMapConfig(force_full=4000.0)
    cfg_cur = RawMapConfig(force_full=4000.0, pressure_gamma=0.5)
    src = SimulatedRawSource([0.0] * 3, [0.0] * 3, [0.0, 0.05, 1.0], amp_full=900.0)
    frames = [RawFrame(seq=i, grid=src.render_grid(0.0, 0.0, p))
              for i, p in enumerate([0.0, 0.05, 1.0])]
    lin, cur = RawFrameProcessor(cfg_lin), RawFrameProcessor(cfg_cur)
    out_lin = [lin.update(f, i * 0.0025) for i, f in enumerate(frames)]
    out_cur = [cur.update(f, i * 0.0025) for i, f in enumerate(frames)]
    assert out_cur[1].pressure == pytest.approx(out_lin[1].pressure ** 0.5, rel=1e-6)
    assert out_cur[1].pressure > out_lin[1].pressure
    assert out_cur[2].pressure == pytest.approx(1.0)


def test_fill_dead_lines_bridges_a_broken_row():
    from brushsynth.input.rawframe import fill_dead_lines

    act = np.zeros((16, 8))
    act[6, 3], act[8, 3] = 100.0, 60.0  # brush straddling dead row 7
    fill_dead_lines(act, rows=(7,))
    assert act[7, 3] == 80.0
    (b,) = extract_blobs(act, threshold=6.0, max_blobs=1)
    assert b.area == 3 and 6.5 < b.cy < 7.5  # one blob spanning the gap, not two

    edge = np.ones((4, 4))
    fill_dead_lines(edge, rows=(0,), cols=(3,))
    assert edge[0, 0] == 1.0 and edge[2, 3] == 1.0  # edges copy their single neighbor

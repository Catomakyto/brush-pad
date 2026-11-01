"""Tests for the hardware brush-controller input path (input/hardware.py)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from brushsynth.engine.rtaudio import ParamBus
from brushsynth.input.hardware import (
    PACKET_SIZE,
    GestureMapper,
    HardwareInputDriver,
    HardwarePacket,
    MappingConfig,
    SimulatedPacketSource,
    crc8,
    encode_packet,
)


def test_packet_size_is_16():
    assert PACKET_SIZE == 16


def test_packet_roundtrip():
    p = HardwarePacket(seq=1234, cx=40000, cy=25000, force=50000,
                       spread=321, orient=-4500, contact=1)
    parsed = HardwarePacket.parse(p.pack())
    assert parsed == p


def test_packet_rejects_bad_sync():
    raw = bytearray(HardwarePacket(cx=1, contact=1).pack())
    raw[0] ^= 0xFF
    assert HardwarePacket.parse(bytes(raw)) is None


def test_packet_rejects_corruption():
    raw = bytearray(HardwarePacket(seq=7, cx=100, force=200, contact=1).pack())
    raw[5] ^= 0x01  # flip a body byte to CRC mismatch
    assert HardwarePacket.parse(bytes(raw)) is None


def test_packet_rejects_wrong_length():
    assert HardwarePacket.parse(b"\xb2\x55short") is None


def test_crc8_known_and_stable():
    # Deterministic and order-sensitive
    assert crc8(b"") == 0
    assert crc8(b"\x01\x02\x03") == crc8(b"\x01\x02\x03")
    assert crc8(b"\x01\x02") != crc8(b"\x02\x01")


def test_encode_roundtrips_position_and_pressure():
    p = encode_packet(0.5, -0.25, 0.6, angle=np.radians(30.0))
    m = GestureMapper()
    g = m.update(p, t=0.0)
    assert g.x == pytest.approx(0.5, abs=1e-3)
    assert g.y == pytest.approx(-0.25, abs=1e-3)
    assert g.pressure == pytest.approx(0.6, abs=1e-3)
    assert g.angle == pytest.approx(np.radians(30.0), abs=1e-3)


def test_pressure_clips_to_unit_range():
    m = GestureMapper(MappingConfig(force_floor=1000.0, force_full=40000.0))
    below = m.update(HardwarePacket(force=500, contact=1), t=0.0)
    assert below.pressure == 0.0
    m.reset()
    above = m.update(HardwarePacket(force=60000, contact=1), t=0.0)
    assert above.pressure == 1.0


def test_velocity_derived_from_motion():
    # Move +0.1 in nx over 10 ms to raw 10 u/s
    m = GestureMapper(MappingConfig(velocity_gain=1.0, velocity_smoothing=0.3))
    m.update(encode_packet(0.0, 0.0, 0.5), t=0.00)
    g = m.update(encode_packet(0.1, 0.0, 0.5), t=0.01)
    assert g.vx == pytest.approx(3.0, abs=0.05)
    assert g.vy == pytest.approx(0.0, abs=1e-6)


def test_velocity_gain_scales():
    m = GestureMapper(MappingConfig(velocity_gain=2.0, velocity_smoothing=0.3))
    m.update(encode_packet(0.0, 0.0, 0.5), t=0.00)
    g = m.update(encode_packet(0.1, 0.0, 0.5), t=0.01)
    assert g.vx == pytest.approx(6.0, abs=0.1)


def test_raw_speed_clamp_kills_spikes():
    # A huge jump in a tiny dt must be clamped
    m = GestureMapper(MappingConfig(max_raw_speed=14.0, velocity_smoothing=1.0))
    m.update(encode_packet(-1.0, 0.0, 0.5), t=0.0)
    g = m.update(encode_packet(1.0, 0.0, 0.5), t=1e-4)  # 2.0 / 1e-4 = 20000 u/s raw
    assert abs(g.vx) <= 14.0 + 1e-6


def test_contact_gate_zeroes_pressure_and_holds_position():
    m = GestureMapper()
    m.update(encode_packet(0.3, 0.4, 0.7), t=0.0)
    lifted = m.update(encode_packet(0.3, 0.4, 0.7, contact=False), t=0.01)
    assert lifted.pressure == 0.0
    assert lifted.x == pytest.approx(0.3, abs=1e-3)
    assert lifted.y == pytest.approx(0.4, abs=1e-3)


def test_recontact_does_not_spike_velocity():
    # After a lift
    m = GestureMapper()
    m.update(encode_packet(-0.5, 0.0, 0.5), t=0.00)
    m.update(encode_packet(-0.5, 0.0, 0.5, contact=False), t=0.01)  # lift
    g = m.update(encode_packet(0.5, 0.0, 0.5), t=0.02)  # re-touch far away
    assert g.vx == 0.0  # no baseline to velocity starts at 0


def test_simulated_source_drives_bus_end_to_end():
    # A circular sweep through the whole pipeline
    n = 200
    th = np.linspace(0, 2 * np.pi, n)
    x = 0.6 * np.cos(th)
    y = 0.6 * np.sin(th)
    p = np.full(n, 0.5)
    src = SimulatedPacketSource(x, y, p, rate_hz=500.0, realtime=False, loop=False)
    bus = ParamBus()
    drv = HardwareInputDriver(bus, src)
    drv.run()  # finite trace, returns at the end
    assert drv.n_packets == n
    assert drv.n_bad == 0
    g = bus.snapshot()
    # Last point sits on the circle of radius 0.6
    assert np.hypot(g.x, g.y) == pytest.approx(0.6, abs=0.02)
    assert g.pressure == pytest.approx(0.5, abs=1e-2)


def test_driver_counts_bad_frames():
    class GarbageSource:
        def stream(self, stop=None):
            yield 0.0, b"\x00" * PACKET_SIZE  # bad sync
            yield 0.001, encode_packet(0.0, 0.0, 0.5).pack()  # good

    bus = ParamBus()
    drv = HardwareInputDriver(bus, GarbageSource())
    drv.run()
    assert drv.n_bad == 1
    assert drv.n_packets == 1

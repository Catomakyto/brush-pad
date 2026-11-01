"""Reference-audio statistics for target matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RefStats:
    sample_rate: float
    band_centers: np.ndarray  # (K,) Hz, 1/12-octave band centers
    ltas_db: np.ndarray  # (K,) dB, long-term avg spectrum, normalized
    spectral_tilt: float  # dB/oct
    crest_db: float  # 20*log10(peak/rms)
    rms: float  # linear
    duration_s: float

    def distance(self, other: "RefStats", use_crest: bool = False) -> float:
        """Return a single scalar distance between two stat sets."""
        k = min(self.ltas_db.size, other.ltas_db.size)
        ltas_l1 = float(np.mean(np.abs(self.ltas_db[:k] - other.ltas_db[:k])))
        tilt_err = abs(self.spectral_tilt - other.spectral_tilt)
        d = ltas_l1 + 0.3 * tilt_err
        if use_crest:
            crest_err = abs(self.crest_db - other.crest_db)
            d += 0.2 * crest_err
        return d


def _fractional_octave_centers(
    lo_hz: float = 40.0, hi_hz: float = 18000.0, fraction: int = 12,
) -> np.ndarray:
    ratio = 2.0 ** (1.0 / fraction)
    n = int(np.ceil(np.log2(hi_hz / lo_hz) * fraction)) + 1
    return lo_hz * ratio ** np.arange(n)


def compute_ltas(
    audio: np.ndarray,
    sample_rate: float,
    *,
    fft_size: int = 8192,
    hop: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (band_centers_hz, ltas_db) for a mono recording."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)
    if audio.size < fft_size:
        fft_size = max(1024, 2 ** int(np.floor(np.log2(max(audio.size, 1024)))))
        hop = fft_size // 2

    window = np.hanning(fft_size)
    n_frames = max(1, 1 + (audio.size - fft_size) // hop)
    mag_sum = np.zeros(fft_size // 2 + 1)
    for i in range(n_frames):
        frame = audio[i * hop : i * hop + fft_size] * window
        mag = np.abs(np.fft.rfft(frame))
        mag_sum += mag
    mag_mean = mag_sum / max(n_frames, 1)

    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    centers = _fractional_octave_centers()
    # Assign each FFT bin to its nearest 1/12-octave band (on log axis)
    with np.errstate(divide="ignore"):
        log_centers = np.log2(centers)
        log_freqs = np.log2(np.maximum(freqs, 1e-6))
    band_idx = np.clip(
        np.round((log_freqs - log_centers[0]) * 12).astype(int),
        0, centers.size - 1,
    )
    band_power = np.zeros(centers.size)
    band_count = np.zeros(centers.size)
    # Use only bins from 40 Hz to 18 kHz
    valid = (freqs >= 20.0) & (freqs <= 22000.0)
    np.add.at(band_power, band_idx[valid], mag_mean[valid] ** 2)
    np.add.at(band_count, band_idx[valid], 1.0)
    band_power = band_power / np.maximum(band_count, 1.0)
    band_mag = np.sqrt(band_power)

    # Drop bands that saw zero bins (sub-40-Hz region at low fs)
    nonzero = band_count > 0
    centers_out = centers[nonzero]
    band_mag = band_mag[nonzero]

    ltas_db = 20.0 * np.log10(np.maximum(band_mag, 1e-12))
    ltas_db -= np.max(ltas_db)
    return centers_out, ltas_db


def compute_spectral_tilt(centers_hz: np.ndarray, ltas_db: np.ndarray,
                          band: tuple[float, float] = (100.0, 8000.0)) -> float:
    """Least-squares slope of LTAS over band in dB per octave."""
    lo, hi = band
    mask = (centers_hz >= lo) & (centers_hz <= hi)
    x = np.log2(centers_hz[mask])
    y = ltas_db[mask]
    if x.size < 3:
        return 0.0
    slope, *_ = np.polyfit(x, y, 1)
    return float(slope)


def compute_crest_db(audio: np.ndarray) -> float:
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim > 1:
        a = a.mean(axis=1)
    peak = float(np.max(np.abs(a)))
    rms = float(np.sqrt(np.mean(a ** 2)))
    if rms < 1e-12:
        return 0.0
    return 20.0 * np.log10(peak / rms)


def analyze(audio: np.ndarray, sample_rate: float) -> RefStats:
    """Compute a RefStats from mono (or stereo) audio."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)
    centers, ltas = compute_ltas(audio, sample_rate)
    tilt = compute_spectral_tilt(centers, ltas)
    crest = compute_crest_db(audio)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    dur = audio.size / sample_rate
    return RefStats(
        sample_rate=float(sample_rate),
        band_centers=centers,
        ltas_db=ltas,
        spectral_tilt=tilt,
        crest_db=crest,
        rms=rms,
        duration_s=dur,
    )

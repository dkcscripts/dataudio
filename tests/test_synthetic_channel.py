"""Synthetic channel degradation tests.

These simulate the kinds of distortion a real phone speaker -> air ->
phone microphone -> recording app pipeline might introduce, without
needing actual hardware:

  - additive noise (background noise, mic self-noise)
  - clock drift / resampling (recorder and player run on different clocks)
  - amplitude scaling (quiet recording, distance from speaker)
  - hard clipping (recording gain too high)
  - band-limiting (mic/speaker frequency response roll-off)
  - combinations of the above

Each test documents the level at which decoding still succeeds and,
where relevant, the level at which it correctly fails (via CRC) rather
than silently producing corrupted output.
"""

import os
import sys
import random

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import audio_utils as au
from encoder import build_audio
from decoder import decode_payload


def make_payload(num_bytes=900, seed=42):
    random.seed(seed)
    return bytes(random.randrange(256) for _ in range(num_bytes))


def encode_reference(num_bytes=900, seed=42, filename="test.bin"):
    payload = make_payload(num_bytes, seed)
    audio = build_audio(filename, payload)
    return filename, payload, audio


def resample_drift(audio: np.ndarray, ppm: float) -> np.ndarray:
    """Simulate a recorder clock running `ppm` parts-per-million off from
    the nominal sample rate, by resampling the signal in time."""
    factor = 1.0 + ppm * 1e-6
    n_new = int(round(len(audio) / factor))
    x_old = np.linspace(0, 1, len(audio), endpoint=False)
    x_new = np.linspace(0, 1, n_new, endpoint=False)
    return np.interp(x_new, x_old, audio)


# ---------------------------------------------------------------------------
# Additive noise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("noise_std", [0.01, 0.05, 0.1, 0.15, 0.2])
def test_additive_noise_tolerable_levels(noise_std):
    filename, payload, audio = encode_reference()
    rng = np.random.default_rng(1)
    noisy = audio + rng.normal(0, noise_std, len(audio))

    fname, decoded = decode_payload(noisy)
    assert fname == filename
    assert decoded == payload


def test_additive_noise_extreme_fails_safely():
    filename, payload, audio = encode_reference()
    rng = np.random.default_rng(1)
    noisy = audio + rng.normal(0, 0.6, len(audio))

    with pytest.raises(ValueError):
        decode_payload(noisy)


# ---------------------------------------------------------------------------
# Clock drift (resampling)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ppm", [20, 50, 100])
def test_clock_drift_tolerable_levels(ppm):
    # Real-world crystal oscillators typically drift by tens of ppm; this
    # covers that range plus a healthy margin.
    filename, payload, audio = encode_reference(num_bytes=3000)  # long enough to accumulate drift
    drifted = resample_drift(audio, ppm)

    fname, decoded = decode_payload(drifted)
    assert fname == filename
    assert decoded == payload


@pytest.mark.parametrize("ppm", [200, 1000])
def test_clock_drift_extreme_fails_or_degrades(ppm):
    # Beyond ~100-200ppm, drift within a 20s resync interval starts to
    # approach a symbol width, which can alias the per-segment
    # symbol-count estimate in decoder.py and corrupt enough bytes in a
    # single RS block to exceed its correction capacity. This is a known
    # boundary of the current design (RESYNC_INTERVAL trades pilot
    # overhead for drift tolerance) - real crystal drift is expected to
    # stay well under this (tens of ppm).
    filename, payload, audio = encode_reference(num_bytes=3000)
    drifted = resample_drift(audio, ppm)

    try:
        fname, decoded = decode_payload(drifted)
        assert fname == filename and decoded == payload
    except ValueError:
        pass  # fails safely via RS/CRC - also an acceptable outcome


# ---------------------------------------------------------------------------
# Amplitude scaling (quiet recording / distance from speaker)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1.0, 0.5, 0.1, 0.02])
def test_amplitude_scaling(scale):
    filename, payload, audio = encode_reference()
    scaled = audio * scale

    fname, decoded = decode_payload(scaled)
    assert fname == filename
    assert decoded == payload


# ---------------------------------------------------------------------------
# Hard clipping (recording gain too high)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("clip_level", [0.9, 0.7, 0.5])
def test_hard_clipping_tolerable_levels(clip_level):
    filename, payload, audio = encode_reference()
    clipped = np.clip(audio, -clip_level, clip_level)

    fname, decoded = decode_payload(clipped)
    assert fname == filename
    assert decoded == payload


# ---------------------------------------------------------------------------
# Band-limiting (mic/speaker frequency roll-off)
# ---------------------------------------------------------------------------

def lowpass(audio: np.ndarray, cutoff_hz: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    b, a = butter(4, cutoff_hz / (au.SAMPLE_RATE / 2), btype="low")
    return filtfilt(b, a, audio)


@pytest.mark.parametrize("cutoff_hz", [12000, 10000])
def test_bandlimiting_above_pilot_survives(cutoff_hz):
    filename, payload, audio = encode_reference()
    filtered = lowpass(audio, cutoff_hz)

    fname, decoded = decode_payload(filtered)
    assert fname == filename
    assert decoded == payload


def test_bandlimiting_below_pilot_fails_safely():
    # A cutoff below the 9000Hz pilot tone removes it almost entirely,
    # so the preamble can no longer be detected - fails safely rather
    # than silently producing corrupted output.
    filename, payload, audio = encode_reference()
    filtered = lowpass(audio, 2000)

    with pytest.raises(ValueError):
        decode_payload(filtered)


# ---------------------------------------------------------------------------
# Combined realistic degradation
# ---------------------------------------------------------------------------

def test_combined_realistic_degradation():
    filename, payload, audio = encode_reference(num_bytes=3000)
    rng = np.random.default_rng(2)

    degraded = resample_drift(audio, 150)
    degraded = degraded * 0.4
    degraded = degraded + rng.normal(0, 0.03, len(degraded))
    degraded = lowpass(degraded, 11000)

    fname, decoded = decode_payload(degraded)
    assert fname == filename
    assert decoded == payload

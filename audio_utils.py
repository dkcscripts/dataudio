"""
Shared constants and low-level DSP / framing helpers for dataudio.

Modulation scheme
------------------
8 parallel channels, each 4-FSK (2 bits/channel/symbol) => 16 bits per
symbol period == exactly 2 bytes per symbol. 20ms symbols (50 baud) =>
raw throughput of 800 bps.

32 total tones spread across 900-8000 Hz (8 channels x 4 tones, raised
from an original 300Hz floor after real recordings showed a low-frequency
noise hump swamping the lowest channel), Gray-coded so a single
misdetected adjacent tone flips only one bit.

A 9000 Hz pilot tone (outside the data band) is used for:
  - the initial preamble (~1.5s)
  - periodic resync bursts (~0.2s) every ~20s of data audio, so the
    decoder can re-lock timing against clock drift on long recordings.

Forward error correction: real-device testing showed ~15-20% per-byte
error rates on some channels (likely speaker intermodulation distortion),
too high for CRC-only detection to ever succeed. Everything is now
Reed-Solomon coded, RS(255,128), correcting up to 63/255 (~24.7%)
erroneous bytes per block. See RS_K/RS_NSYM/RS_N above.

Frame layout (all lengths in bytes unless noted):

    [PREAMBLE: 9000Hz pilot, PREAMBLE_DURATION seconds]
    [OUTER HEADER: one fixed-size RS(255,128) block containing
        filename (64, null-padded) | payload_len (4) | crc32 (4),
        plus a trailing 0x00 alignment byte -> OUTER_HEADER_TOTAL_BYTES
    ]
    [MAIN BODY: (payload + payload_crc32) split into RS_K-byte chunks
        (last one zero-padded), each RS-encoded to an RS_N-byte block,
        modulated as parallel 4-FSK symbols, with a 9000Hz resync burst
        spliced in every RESYNC_INTERVAL seconds of data audio
    ]
    [TERMINATOR: TERMINATOR_REPEATS symbols, every channel transmits
                 TERMINATOR_SYMBOL_VALUE]

Both the outer header and the main body are padded to even byte length
(one trailing 0x00) so the bitstream always divides evenly into 2-byte
symbol periods. The decoder always knows exactly how many bytes to read
for the outer header (fixed size) and, once it's RS-decoded, exactly how
many RS blocks make up the main body (from payload_len) - no incremental
byte-by-byte parsing is needed.
"""

from __future__ import annotations

import zlib
import numpy as np
import reedsolo

# ---------------------------------------------------------------------------
# Core signal parameters
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100
SYMBOL_DURATION = 0.02  # seconds (20 ms)
SYMBOL_SAMPLES = int(round(SAMPLE_RATE * SYMBOL_DURATION))  # 882 samples

NUM_CHANNELS = 8
FREQS_PER_CHANNEL = 4
TOTAL_FREQS = NUM_CHANNELS * FREQS_PER_CHANNEL  # 32
BITS_PER_SYMBOL_PERIOD = NUM_CHANNELS * 2  # 16 bits = 2 bytes

# FREQ_MIN was originally 300 Hz, but real phone-mic recordings showed a
# strong low-frequency noise hump (~100-600 Hz, handling/room noise) that
# reliably swamped the lowest channel's tones. Raised to clear it with margin.
FREQ_MIN = 900.0
FREQ_MAX = 8000.0
_ALL_FREQS = np.linspace(FREQ_MIN, FREQ_MAX, TOTAL_FREQS)
CHANNEL_FREQS = [
    _ALL_FREQS[c * FREQS_PER_CHANNEL:(c + 1) * FREQS_PER_CHANNEL]
    for c in range(NUM_CHANNELS)
]

# 2-bit value -> local frequency index within a channel's 4 tones.
# This particular ordering is self-inverse (its own decode table).
GRAY_ORDER = [0, 1, 3, 2]

# ---------------------------------------------------------------------------
# Pilot tone (preamble / resync) parameters
# ---------------------------------------------------------------------------

PILOT_FREQ = 9000.0
PREAMBLE_DURATION = 1.5   # seconds
RESYNC_DURATION = 0.2     # seconds
RESYNC_INTERVAL = 20.0    # seconds of data audio between resync bursts
RESYNC_INTERVAL_SYMBOLS = int(round(RESYNC_INTERVAL / SYMBOL_DURATION))  # 1000

# ---------------------------------------------------------------------------
# Amplitude / envelope shaping
# ---------------------------------------------------------------------------

AMPLITUDE = 0.8
RAMP_DURATION = 0.0025  # seconds (2.5 ms)
RAMP_SAMPLES = int(round(RAMP_DURATION * SAMPLE_RATE))

# ---------------------------------------------------------------------------
# Terminator
# ---------------------------------------------------------------------------

TERMINATOR_SYMBOL_VALUE = 3  # every channel transmits this 2-bit value
TERMINATOR_REPEATS = 16

# ---------------------------------------------------------------------------
# Forward error correction (Reed-Solomon)
# ---------------------------------------------------------------------------
#
# Real-device testing showed per-byte error rates around 15-20% on a couple
# of the 8 parallel channels (likely intermodulation distortion from a
# cheap phone speaker driving 8 simultaneous tones). CRC-only detection
# cannot succeed under that error rate for anything but tiny payloads, so
# every meaningful chunk of data is now protected with RS(255,128):
# 128 data bytes + 127 parity bytes per 255-byte block, correcting up to
# 63 erroneous bytes per block (~24.7%).

RS_K = 128       # data bytes per RS block
RS_NSYM = 127    # parity bytes per RS block
RS_N = RS_K + RS_NSYM  # 255
_RSC = reedsolo.RSCodec(RS_NSYM, nsize=RS_N)

# Fixed-size outer header (filename + payload length), RS-protected as a
# single block, so the decoder can read it deterministically right after
# the preamble without needing to know its length in advance.
OUTER_FILENAME_FIELD = 64  # bytes, null-padded/truncated utf-8 filename
OUTER_HEADER_PLAIN = OUTER_FILENAME_FIELD + 4 + 4  # filename + payload_len + crc32
assert OUTER_HEADER_PLAIN <= RS_K
OUTER_HEADER_RS_BYTES = RS_N          # one RS block
OUTER_HEADER_TOTAL_BYTES = RS_N + 1   # + 1 pad byte so it's even (symbol-aligned)

# ---------------------------------------------------------------------------
# File size policy
# ---------------------------------------------------------------------------

MAX_FILE_SIZE = 2 * 1024 * 1024       # 2 MB hard cap
WARN_FILE_SIZE = 500 * 1024           # warn above this


def estimate_duration_seconds(payload_len: int, filename_len: int = 16) -> float:
    """Rough estimate of total audio duration for a given payload size,
    including Reed-Solomon FEC overhead."""
    plain_main = payload_len + 4  # + payload crc
    num_blocks = max(1, -(-plain_main // RS_K))  # ceil div
    main_rs_len = num_blocks * RS_N
    if main_rs_len % 2:
        main_rs_len += 1
    total_data_len = OUTER_HEADER_TOTAL_BYTES + main_rs_len
    num_symbols = total_data_len // 2
    data_duration = num_symbols * SYMBOL_DURATION
    num_resyncs = int(data_duration // RESYNC_INTERVAL)
    return PREAMBLE_DURATION + data_duration + num_resyncs * RESYNC_DURATION + \
        TERMINATOR_REPEATS * SYMBOL_DURATION


# ---------------------------------------------------------------------------
# Waveform generation
# ---------------------------------------------------------------------------

def apply_ramp(waveform: np.ndarray) -> np.ndarray:
    """Apply a raised-cosine fade-in/out to avoid clicks between symbols."""
    n = len(waveform)
    r = min(RAMP_SAMPLES, n // 2)
    if r <= 0:
        return waveform
    out = waveform.copy()
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(r) / r))
    out[:r] *= ramp
    out[-r:] *= ramp[::-1]
    return out


def generate_tone(freq: float, duration: float, amplitude: float = AMPLITUDE) -> np.ndarray:
    n = int(round(duration * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    return apply_ramp(wave)


def generate_symbol(values: list[int]) -> np.ndarray:
    """values: NUM_CHANNELS ints in [0,3] (2 bits each).

    Returns a normalized (peak == AMPLITUDE) waveform of SYMBOL_SAMPLES.
    """
    assert len(values) == NUM_CHANNELS
    t = np.arange(SYMBOL_SAMPLES) / SAMPLE_RATE
    combined = np.zeros(SYMBOL_SAMPLES)
    for ch, val in enumerate(values):
        idx = GRAY_ORDER[val]
        freq = CHANNEL_FREQS[ch][idx]
        combined += np.sin(2 * np.pi * freq * t)
    combined = apply_ramp(combined)
    peak = np.max(np.abs(combined))
    if peak > 0:
        combined = combined * (AMPLITUDE / peak)
    return combined


# ---------------------------------------------------------------------------
# Byte <-> symbol-value conversion
# ---------------------------------------------------------------------------

def bytes_to_symbol_values(data: bytes) -> list[list[int]]:
    """Split data (even length) into per-symbol lists of NUM_CHANNELS
    2-bit values, MSB-first."""
    assert len(data) % 2 == 0, "data must be padded to even length"
    symbols = []
    for i in range(0, len(data), 2):
        word = (data[i] << 8) | data[i + 1]
        values = []
        for ch in range(NUM_CHANNELS):
            shift = (NUM_CHANNELS - 1 - ch) * 2
            values.append((word >> shift) & 0b11)
        symbols.append(values)
    return symbols


def symbol_values_to_bytes(symbols: list[list[int]]) -> bytes:
    out = bytearray()
    for values in symbols:
        word = 0
        for ch, v in enumerate(values):
            shift = (NUM_CHANNELS - 1 - ch) * 2
            word |= (v & 0b11) << shift
        out.append((word >> 8) & 0xFF)
        out.append(word & 0xFF)
    return bytes(out)


# ---------------------------------------------------------------------------
# CRC / Reed-Solomon framing helpers
# ---------------------------------------------------------------------------

def crc32(data: bytes) -> bytes:
    return zlib.crc32(data).to_bytes(4, "big")


def rs_encode_block(data: bytes) -> bytes:
    """data must be exactly RS_K bytes (pad with zeros first if shorter).
    Returns an RS_N-byte codeword."""
    assert len(data) == RS_K
    return bytes(_RSC.encode(data))


def rs_decode_block(codeword: bytes, stats: list | None = None) -> bytes:
    """codeword must be exactly RS_N bytes. Returns the original RS_K-byte
    data, correcting up to RS_NSYM // 2 erroneous bytes. Raises ValueError
    if the block has too many errors to correct.

    If `stats` is given, appends a dict per call describing the outcome:
    {"ok": True, "errors_corrected": N} or {"ok": False, "error": str}.
    """
    assert len(codeword) == RS_N
    try:
        decoded, _, errata = _RSC.decode(codeword)
    except reedsolo.ReedSolomonError as e:
        if stats is not None:
            stats.append({"ok": False, "error": str(e)})
        raise ValueError(f"Reed-Solomon block uncorrectable: {e}") from e
    if stats is not None:
        stats.append({"ok": True, "errors_corrected": len(errata)})
    return bytes(decoded)[:RS_K]


def build_outer_header(filename: str, payload_len: int) -> bytes:
    """Fixed-size, RS-protected block containing the filename and payload
    length, always OUTER_HEADER_TOTAL_BYTES long (symbol-aligned)."""
    fname_bytes = filename.encode("utf-8")
    if len(fname_bytes) > OUTER_FILENAME_FIELD:
        raise ValueError(f"Filename too long (max {OUTER_FILENAME_FIELD} utf-8 bytes)")
    fname_field = fname_bytes.ljust(OUTER_FILENAME_FIELD, b"\x00")
    body = fname_field + payload_len.to_bytes(4, "big")
    body += crc32(body)
    assert len(body) == OUTER_HEADER_PLAIN
    padded = body.ljust(RS_K, b"\x00")
    codeword = rs_encode_block(padded)
    return codeword + b"\x00"  # pad to even length for symbol alignment


def parse_outer_header(block: bytes, stats: list | None = None):
    """block must be OUTER_HEADER_TOTAL_BYTES long. Returns (filename, payload_len)."""
    codeword = block[:RS_N]
    plain = rs_decode_block(codeword, stats=stats)
    fname_field = plain[:OUTER_FILENAME_FIELD]
    fname = fname_field.rstrip(b"\x00").decode("utf-8")
    p = OUTER_FILENAME_FIELD
    payload_len = int.from_bytes(plain[p:p + 4], "big")
    p += 4
    body = plain[:p]
    header_crc = plain[p:p + 4]
    if crc32(body) != header_crc:
        raise ValueError("Outer header CRC mismatch after RS correction (unexpected).")
    return fname, payload_len


def encode_main_body(payload: bytes) -> bytes:
    """RS-encodes (payload + payload_crc32) in RS_K-byte blocks, padding
    the last block with zeros. Result length is a multiple of RS_N, plus
    one trailing pad byte if that's odd (for symbol alignment)."""
    plain = payload + crc32(payload)
    out = bytearray()
    for i in range(0, len(plain), RS_K):
        chunk = plain[i:i + RS_K].ljust(RS_K, b"\x00")
        out += rs_encode_block(chunk)
    if len(out) % 2:
        out += b"\x00"
    return bytes(out)


def num_main_body_blocks(payload_len: int) -> int:
    plain_len = payload_len + 4
    return max(1, -(-plain_len // RS_K))  # ceil div


def decode_main_body(data: bytes, payload_len: int, stats: list | None = None) -> bytes:
    """data must contain exactly num_main_body_blocks(payload_len) RS blocks
    (RS_N bytes each), ignoring any trailing alignment pad byte. Returns
    the validated payload bytes."""
    num_blocks = num_main_body_blocks(payload_len)
    plain = bytearray()
    for i in range(num_blocks):
        codeword = data[i * RS_N:(i + 1) * RS_N]
        plain += rs_decode_block(codeword, stats=stats)
    payload = bytes(plain[:payload_len])
    payload_crc = bytes(plain[payload_len:payload_len + 4])
    if crc32(payload) != payload_crc:
        raise ValueError("Payload CRC mismatch after RS correction (unexpected).")
    return payload


# ---------------------------------------------------------------------------
# Decoding: per-symbol FFT-based demodulation
# ---------------------------------------------------------------------------

_INV_GRAY = GRAY_ORDER  # self-inverse for this specific ordering


def decode_symbol(window: np.ndarray) -> list[int]:
    """Given a SYMBOL_SAMPLES-length audio window, return the NUM_CHANNELS
    2-bit values by picking the strongest tone in each channel's block."""
    n = len(window)
    hann = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(window * hann))
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)

    values = []
    for ch in range(NUM_CHANNELS):
        mags = []
        for f in CHANNEL_FREQS[ch]:
            b = int(round(f * n / SAMPLE_RATE))
            b = max(0, min(b, len(spectrum) - 1))
            # small neighborhood max to tolerate slight frequency drift
            lo, hi = max(0, b - 1), min(len(spectrum), b + 2)
            mags.append(spectrum[lo:hi].max())
        idx = int(np.argmax(mags))
        values.append(_INV_GRAY[idx])
    return values


def goertzel_magnitude(samples: np.ndarray, freq: float) -> float:
    """Single-bin magnitude via Goertzel algorithm (used for pilot detection)."""
    n = len(samples)
    if n == 0:
        return 0.0
    k = int(round(freq * n / SAMPLE_RATE))
    w = 2 * np.pi * k / n
    coeff = 2 * np.cos(w)
    s_prev, s_prev2 = 0.0, 0.0
    for x in samples:
        s = x + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    power = s_prev2 ** 2 + s_prev ** 2 - coeff * s_prev * s_prev2
    return float(np.sqrt(max(power, 0.0)))

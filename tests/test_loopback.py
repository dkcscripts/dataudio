import os
import sys
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import audio_utils as au
from encoder import build_audio
from decoder import decode_payload


def test_roundtrip_small_binary():
    random.seed(42)
    payload = bytes(random.randrange(256) for _ in range(500))
    filename = "test_file.bin"

    audio = build_audio(filename, payload)
    fname, decoded = decode_payload(audio)

    assert fname == filename
    assert decoded == payload


def test_roundtrip_empty_file():
    payload = b""
    filename = "empty.txt"

    audio = build_audio(filename, payload)
    fname, decoded = decode_payload(audio)

    assert fname == filename
    assert decoded == payload


def test_roundtrip_crosses_resync_boundary():
    # Force at least one resync burst by generating enough symbols.
    random.seed(1)
    num_bytes = (au.RESYNC_INTERVAL_SYMBOLS + 50) * 2
    payload = bytes(random.randrange(256) for _ in range(num_bytes))
    filename = "big.bin"

    audio = build_audio(filename, payload)
    fname, decoded = decode_payload(audio)

    assert fname == filename
    assert decoded == payload


def test_bit_symbol_conversion_roundtrip():
    random.seed(7)
    data = bytes(random.randrange(256) for _ in range(64))
    symbols = au.bytes_to_symbol_values(data)
    restored = au.symbol_values_to_bytes(symbols)
    assert restored == data


def test_minor_corruption_is_corrected_by_fec():
    # A single corrupted symbol only touches a handful of bytes within one
    # 255-byte Reed-Solomon block, well within its ~24.7% correction
    # capacity - decoding should succeed transparently.
    payload = b"hello world"
    filename = "hi.txt"
    audio = build_audio(filename, payload)

    rng = np.random.default_rng(0)
    mid = len(audio) // 2
    audio = audio.copy()
    audio[mid:mid + au.SYMBOL_SAMPLES] = rng.uniform(-1, 1, au.SYMBOL_SAMPLES)

    fname, decoded = decode_payload(audio)
    assert fname == filename
    assert decoded == payload


def test_heavy_corruption_fails_safely():
    # Corrupting a large contiguous stretch of the main body exceeds the
    # RS block's correction capacity and should raise, not return wrong data.
    payload = bytes(random.randrange(256) for _ in range(2000))
    filename = "corrupt.bin"
    audio = build_audio(filename, payload)

    rng = np.random.default_rng(0)
    audio = audio.copy()
    start = len(audio) // 3
    span = au.SYMBOL_SAMPLES * 200  # ~200 symbols = way more than one RS block can fix
    audio[start:start + span] = rng.uniform(-1, 1, span)

    try:
        decode_payload(audio)
        raised = False
    except ValueError:
        raised = True
    assert raised

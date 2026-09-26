#!/usr/bin/env python3
"""Decode a WAV file produced by encoder.py back into the original file."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf

import audio_utils as au


def load_mono(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != au.SAMPLE_RATE:
        # Simple linear resample to the expected sample rate.
        duration = len(audio) / sr
        new_len = int(round(duration * au.SAMPLE_RATE))
        x_old = np.linspace(0, duration, len(audio), endpoint=False)
        x_new = np.linspace(0, duration, new_len, endpoint=False)
        audio = np.interp(x_new, x_old, audio)
    return audio


def find_pilot_bursts(audio: np.ndarray, threshold_ratio: float = 0.4):
    """Return list of (start_sample, end_sample) for contiguous regions
    where the 9000 Hz pilot tone dominates, at SYMBOL_SAMPLES granularity."""
    block = au.SYMBOL_SAMPLES
    num_blocks = len(audio) // block
    if num_blocks == 0:
        return []
    mags = np.empty(num_blocks)
    for i in range(num_blocks):
        seg = audio[i * block:(i + 1) * block]
        mags[i] = au.goertzel_magnitude(seg, au.PILOT_FREQ)

    peak = mags.max()
    if peak <= 0:
        return []
    threshold = threshold_ratio * peak
    is_pilot = mags > threshold

    bursts = []
    i = 0
    n = len(is_pilot)
    while i < n:
        if is_pilot[i]:
            j = i
            while j < n and is_pilot[j]:
                j += 1
            bursts.append((i * block, j * block))
            i = j
        else:
            i += 1
    return bursts


def _iter_symbol_bytes(segments, audio):
    """Yields 2-byte chunks (one per decoded symbol) across all data
    segments in order, using per-segment drift-corrected symbol timing."""
    for seg_start, seg_end in segments:
        seg = audio[seg_start:seg_end]
        # Rescale the local symbol length to the segment's actual measured
        # duration. Each inter-resync segment is known (by construction) to
        # contain an integer number of symbols; dividing the *measured*
        # segment length by that count self-corrects for clock drift
        # accumulated since the last pilot burst, instead of assuming the
        # nominal SYMBOL_SAMPLES holds exactly.
        expected_symbols = int(round(len(seg) / au.SYMBOL_SAMPLES))
        if expected_symbols <= 0:
            continue
        local_symbol_len = len(seg) / expected_symbols
        for i in range(expected_symbols):
            start = int(round(i * local_symbol_len))
            end = int(round((i + 1) * local_symbol_len))
            window = seg[start:end]
            if len(window) < au.SYMBOL_SAMPLES // 2:
                continue
            values = au.decode_symbol(window)
            yield au.symbol_values_to_bytes([values])


def _read_n_bytes(gen, n):
    buf = bytearray()
    for chunk in gen:
        buf.extend(chunk)
        if len(buf) >= n:
            return bytes(buf[:n])
    raise ValueError("Ran out of audio before enough data was received.")


def decode_payload(audio: np.ndarray, diagnostics: dict | None = None):
    if diagnostics is None:
        diagnostics = {}

    bursts = find_pilot_bursts(audio)
    diagnostics["bursts"] = [
        {"start_sample": s, "end_sample": e, "duration_s": (e - s) / au.SAMPLE_RATE}
        for s, e in bursts
    ]
    if not bursts:
        raise ValueError("Preamble not found: no 9000Hz pilot tone detected.")

    preamble_start, preamble_end = bursts[0]
    preamble_len_s = (preamble_end - preamble_start) / au.SAMPLE_RATE
    if preamble_len_s < au.PREAMBLE_DURATION * 0.5:
        raise ValueError(
            f"First pilot burst too short to be the preamble "
            f"({preamble_len_s:.2f}s detected, expected ~{au.PREAMBLE_DURATION}s)."
        )

    # Data segments: gaps between pilot bursts, plus the tail after the last one.
    segments = []
    prev_end = preamble_end
    for b_start, b_end in bursts[1:]:
        segments.append((prev_end, b_start))
        prev_end = b_end
    segments.append((prev_end, len(audio)))
    diagnostics["segments"] = [
        {
            "start_sample": s, "end_sample": e,
            "duration_s": (e - s) / au.SAMPLE_RATE,
            "symbols": int(round((e - s) / au.SYMBOL_SAMPLES)),
        }
        for s, e in segments
    ]

    sym_gen = _iter_symbol_bytes(segments, audio)

    outer_stats = []
    diagnostics["outer_header_rs"] = outer_stats
    outer_block = _read_n_bytes(sym_gen, au.OUTER_HEADER_TOTAL_BYTES)
    fname, payload_len = au.parse_outer_header(outer_block, stats=outer_stats)
    diagnostics["filename"] = fname
    diagnostics["payload_len"] = payload_len

    num_blocks = au.num_main_body_blocks(payload_len)
    main_body_len = num_blocks * au.RS_N
    if main_body_len % 2:
        main_body_len += 1
    main_body = _read_n_bytes(sym_gen, main_body_len)

    main_stats = []
    diagnostics["main_body_rs"] = main_stats
    payload = au.decode_main_body(main_body, payload_len, stats=main_stats)

    # Best-effort terminator sanity check (not required for correctness,
    # since block/length accounting above is authoritative).
    expected_term = [au.TERMINATOR_SYMBOL_VALUE] * au.NUM_CHANNELS
    try:
        term_bytes = _read_n_bytes(sym_gen, 2)
        term_values = au.bytes_to_symbol_values(term_bytes)[0]
        diagnostics["terminator_ok"] = (term_values == expected_term)
        if term_values != expected_term:
            print("Warning: terminator pattern mismatch (sanity check failed); "
                  "payload CRC passed so proceeding anyway.", file=sys.stderr)
    except ValueError:
        diagnostics["terminator_ok"] = None  # not enough trailing audio to check

    return fname, payload


def print_diagnostics(diagnostics: dict):
    print("\n--- debug diagnostics ---", file=sys.stderr)

    bursts = diagnostics.get("bursts", [])
    print(f"Pilot bursts found: {len(bursts)}", file=sys.stderr)
    for i, b in enumerate(bursts):
        kind = "preamble" if i == 0 else "resync"
        print(f"  [{i}] {kind}: samples {b['start_sample']}-{b['end_sample']} "
              f"({b['duration_s']:.3f}s)", file=sys.stderr)

    segments = diagnostics.get("segments", [])
    print(f"Data segments: {len(segments)}", file=sys.stderr)
    for i, s in enumerate(segments):
        print(f"  [{i}] {s['duration_s']:.3f}s (~{s['symbols']} symbols)", file=sys.stderr)

    def summarize_rs(stats, label):
        if not stats:
            print(f"{label}: no blocks processed", file=sys.stderr)
            return
        ok = [s for s in stats if s["ok"]]
        failed = [s for s in stats if not s["ok"]]
        print(f"{label}: {len(stats)} block(s), {len(failed)} failed", file=sys.stderr)
        if ok:
            counts = [s["errors_corrected"] for s in ok]
            print(f"  errors corrected per block: min={min(counts)} "
                  f"max={max(counts)} avg={sum(counts) / len(counts):.1f} "
                  f"(capacity is {au.RS_NSYM // 2} per block)", file=sys.stderr)
        for i, s in enumerate(stats):
            if not s["ok"]:
                print(f"  block {i}: FAILED - {s['error']}", file=sys.stderr)

    summarize_rs(diagnostics.get("outer_header_rs", []), "Outer header RS")
    summarize_rs(diagnostics.get("main_body_rs", []), "Main body RS")

    if "filename" in diagnostics:
        print(f"Filename: {diagnostics['filename']!r}", file=sys.stderr)
    if "payload_len" in diagnostics:
        print(f"Payload length: {diagnostics['payload_len']} bytes", file=sys.stderr)
    if "terminator_ok" in diagnostics:
        print(f"Terminator sanity check: {diagnostics['terminator_ok']}", file=sys.stderr)
    print("--- end diagnostics ---\n", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Decode a dataudio WAV file back into a file.")
    parser.add_argument("input", help="Path to the input WAV file")
    parser.add_argument("-o", "--output-dir", default=".", help="Directory to write the decoded file into")
    parser.add_argument("--debug", action="store_true", help="Print extra diagnostics on failure")
    args = parser.parse_args()

    audio = load_mono(args.input)

    diagnostics = {}
    try:
        fname, payload = decode_payload(audio, diagnostics=diagnostics)
    except ValueError as e:
        if args.debug:
            print_diagnostics(diagnostics)
        print(f"Decode failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.debug:
        print_diagnostics(diagnostics)

    out_path = os.path.join(args.output_dir, fname)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(payload)

    print(f"Decoded '{fname}' ({len(payload)} bytes) -> '{out_path}'")


if __name__ == "__main__":
    main()

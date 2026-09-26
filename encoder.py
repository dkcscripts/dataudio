#!/usr/bin/env python3
"""Encode a file into a WAV file using the dataudio acoustic protocol."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf

import audio_utils as au


def build_audio(filename: str, payload: bytes) -> np.ndarray:
    outer_header = au.build_outer_header(filename, len(payload))
    main_body = au.encode_main_body(payload)
    data = outer_header + main_body
    symbols = au.bytes_to_symbol_values(data)

    chunks = [au.generate_tone(au.PILOT_FREQ, au.PREAMBLE_DURATION)]

    since_last_resync = 0
    for values in symbols:
        chunks.append(au.generate_symbol(values))
        since_last_resync += 1
        if since_last_resync >= au.RESYNC_INTERVAL_SYMBOLS:
            chunks.append(au.generate_tone(au.PILOT_FREQ, au.RESYNC_DURATION))
            since_last_resync = 0

    terminator_values = [au.TERMINATOR_SYMBOL_VALUE] * au.NUM_CHANNELS
    for _ in range(au.TERMINATOR_REPEATS):
        chunks.append(au.generate_symbol(terminator_values))

    return np.concatenate(chunks)


def main():
    parser = argparse.ArgumentParser(description="Encode a file into a dataudio WAV file.")
    parser.add_argument("input", help="Path to the input file to encode")
    parser.add_argument("output", help="Path to the output WAV file")
    parser.add_argument("--amplitude", type=float, default=None,
                         help=f"Override peak amplitude (default {au.AMPLITUDE})")
    parser.add_argument("-y", "--yes", action="store_true",
                         help="Skip confirmation prompts (warnings are still printed)")
    args = parser.parse_args()

    if args.amplitude is not None:
        au.AMPLITUDE = args.amplitude

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input, "rb") as f:
        payload = f.read()

    size = len(payload)
    if size > au.MAX_FILE_SIZE:
        print(f"Error: file size {size} bytes exceeds hard limit of "
              f"{au.MAX_FILE_SIZE} bytes (2 MB).", file=sys.stderr)
        sys.exit(1)

    filename = os.path.basename(args.input)
    est = au.estimate_duration_seconds(size, len(filename.encode("utf-8")))

    if size > au.WARN_FILE_SIZE:
        print(f"Warning: file is {size / 1024:.1f} KB (> 500 KB). "
              f"Estimated transmission time: {est / 60:.1f} minutes "
              f"({est / 3600:.2f} hours). Proceeding...", file=sys.stderr)
    else:
        print(f"Estimated transmission time: {est / 60:.1f} minutes.", file=sys.stderr)

    print(f"Encoding '{args.input}' ({size} bytes) -> '{args.output}' ...")
    audio = build_audio(filename, payload)
    sf.write(args.output, audio, au.SAMPLE_RATE, subtype="PCM_16")
    print(f"Wrote {args.output} ({len(audio) / au.SAMPLE_RATE:.1f}s of audio).")


if __name__ == "__main__":
    main()

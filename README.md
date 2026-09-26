# dataudio

Encode a file into a WAV file that can be played through a speaker,
recorded by another device's microphone (e.g. a phone), transferred,
and decoded back into the original file - no network required.

## How it works

- **Modulation**: 8 parallel audio channels, each 4-FSK modulated
  (2 bits/channel/symbol) => 16 bits (2 bytes) per 20ms symbol =>
  ~800 bps raw throughput.
- **Frequencies**: 32 tones spread across 900–8000 Hz, Gray-coded per
  channel (a single misdetected adjacent tone flips only one bit).
  The floor was raised from an initial 300 Hz after real recordings
  showed a low-frequency noise hump (~100–600 Hz, mic/handling noise)
  reliably swamping the lowest channel.
- **Sync**: a 9000 Hz pilot tone (outside the data band) marks the
  initial preamble (~1.5s) and periodic resync bursts (~0.2s every
  ~20s of data), so the decoder can re-lock timing against clock
  drift on long recordings.
- **Forward error correction**: real-device testing showed ~15-20%
  per-byte error rates on some channels (likely speaker
  intermodulation distortion from driving 8 simultaneous tones).
  CRC-only detection cannot succeed at that error rate, so the entire
  frame is Reed-Solomon coded, RS(255,128) - 128 data bytes + 127
  parity bytes per block, correcting up to 63/255 (~24.7%) erroneous
  bytes per block. CRC-32 is layered on top as a final integrity
  check after RS correction.
- **Filename** is preserved (fixed 64-byte field in the header).

See the `audio_utils.py` module docstring for the exact frame layout
(preamble / outer header / main body / terminator).

## Usage

```bash
pip install -r requirements.txt

# Encode
python encoder.py path/to/file.pdf output.wav

# Play output.wav through a speaker, record it with another device
# (e.g. a phone voice memo / camera app), transfer the recording back,
# convert it to WAV if needed (e.g. via ffmpeg), then:

# Decode (writes the original filename into the given directory)
python decoder.py recording.wav -o ./decoded/

# Add --debug for a diagnostic report (pilot bursts, segment timing,
# per-block RS error-correction counts, which block failed if any):
python decoder.py recording.wav -o ./decoded/ --debug
```

On decode failure, the tool fails loudly (non-zero exit, no partial
output written) rather than silently producing corrupted data.

## Limits

- Hard cap: 2 MB per file (`audio_utils.MAX_FILE_SIZE`).
- Files over 500 KB print an estimated-duration warning but proceed
  automatically (`audio_utils.WARN_FILE_SIZE`).
- Rough throughput after RS overhead: ~400 bps effective. A 2248-byte
  test file took 50.7s of audio (up from 24.8s pre-FEC).

## Practical note: encoded WAV file size

Because reliability was prioritized over speed, and FEC roughly
doubles the audio duration on top of the already-conservative
modulation, this trades throughput hard for robustness. At
44.1kHz/16-bit mono:

- ~2.2 KB payload → ~50s of audio
- 2 MB payload → very roughly ~11-12 hours of audio, ~3.5 GB WAV file

Plan storage and transfer time accordingly - this is meant for
small-to-medium files (configs, keys, archives) where reliability over
an acoustic channel matters more than speed.

## Testing

```bash
pip install pytest
pytest tests/
```

- `tests/test_loopback.py` - encodes a random payload straight to an
  in-memory waveform and decodes it again (no recording involved), to
  validate the codec logic itself: round trips, RS-corrected minor
  corruption, safely-failing heavy corruption, and crossing a resync
  boundary.
- `tests/test_synthetic_channel.py` - degrades the encoded waveform in
  ways a real speaker/air/microphone/recording-app pipeline would
  (noise, clock drift via resampling, amplitude scaling, clipping,
  band-limiting) and checks decoding still succeeds within realistic
  bounds, and fails safely beyond them.

**Empirically validated tolerances:**
- Additive noise: reliable up to ~0.2 std (signal peak is 0.8)
- Clock drift: reliable up to ~100-200ppm (typical crystal drift is
  tens of ppm); beyond that, drift within a single 20s resync interval
  can approach a full symbol width, aliasing the per-segment
  symbol-count estimate in `decoder.py` and corrupting enough bytes in
  one RS block to exceed its correction capacity. If real-world
  testing ever shows this is a problem, shortening `RESYNC_INTERVAL`
  in `audio_utils.py` trades a bit more pilot-tone overhead for more
  drift headroom.
- Amplitude: works from full scale down to ~2% (very quiet recordings)
- Clipping: tolerates hard clipping down to ±0.5
- Band-limiting: survives a low-pass at 10kHz (well below the 9kHz
  pilot); fails safely if filtered below ~2kHz
- **Real hardware**: confirmed working on an actual speaker → air →
  phone microphone recording → file transfer → decode round trip,
  after two real-world-driven fixes (frequency floor, then Reed-Solomon
  FEC - see "Design history" below).

## Design history / known issues found via real-device testing

1. **Low-frequency noise**: first real recording showed the lowest
   data channel (originally 300-1045 Hz) swamped by ambient/handling
   noise below ~600 Hz. Fixed by raising `FREQ_MIN` to 900 Hz.
2. **Per-channel error rate too high for CRC-only**: second real
   recording showed ~16-19% per-byte error rates concentrated on 2 of
   8 channels, consistent with speaker intermodulation distortion.
   CRC-only detection (the original v1 design) can never succeed at
   that error rate for any payload of practical size - fixed by adding
   Reed-Solomon FEC, verified against the exact measured error profile
   via simulation (20/20 trials) before the real-world confirmation.

If you hit further real-world failures, run the decoder with
`--debug` and share the output plus the recording - the diagnostic
report (pilot burst positions/durations, segment timing, per-block RS
error counts, which block failed) is usually enough to pinpoint the
next issue quickly, the same way it was used for the two fixes above.

## Project layout

```
audio_utils.py    # constants, modulation, RS/CRC framing, FFT decode
encoder.py        # CLI: file -> WAV
decoder.py        # CLI: WAV -> file (+ --debug diagnostics)
requirements.txt
tests/
  test_loopback.py
  test_synthetic_channel.py
```

# Envelope-Based Morse Segmentation (1/3/7 Rule)

This document explains the simple envelope thresholding baseline that segments dots, dashes, and gaps directly from the waveform and decodes them with the 1/3/7 timing rule.

## What the script does
- Load WAVs from a JSONL manifest (`audio_filepath`, optional `text`, optional `wpm`).
- Compute a smoothed amplitude envelope (abs → moving average).
- Pick a threshold at `median + ratio * (p95 - median)` so it sits inside the orange envelope band.
- Binarize to on/off, extract run lengths.
- Estimate the dot time unit from median run lengths (or `wpm` if provided).
- Classify runs:
  - Dots vs dashes: split at `1.6 * median(on_lengths)`.
  - Gaps: gap1/gap3/gap7 based on median(off_lengths) with 1×/1.8×/3.5× cutoffs.
- Decode runs to Morse → text using `audio2morse.data.morse_map.MORSE_CODE`.
- Save a diagnostic plot per file (waveform, envelope, threshold, colored spans) plus CSVs of segments and decodes.

Colors in the plot:
- Dot (on) spans: green.
- Dash (on) spans: red.
- Gaps (off) spans: purple.
- Orange line: envelope; red dashed line: threshold.

## Usage
Requires the venv and `PYTHONPATH=src`:
```bash
PYTHONPATH=src .venv/bin/python analyses/segment_envelope.py \
  --manifest ab123_example/manifests/test.jsonl \
  --out-dir ab123_example/segments \
  --smooth-ms 10 \
  --threshold-ratio 0.3
```

Key knobs:
- `--smooth-ms`: envelope smoothing window; larger smooths out micro-oscillations.
- `--threshold-ratio`: raise/lower the cut inside the envelope band.

Outputs (under `--out-dir`):
- `*_segments.png`: waveform + envelope with colored spans.
- `segments.csv`: start/end/kind/on/units for every span.
- `decodes.csv`: decoded text per file.

## Current reference run
Using the command above on the synthetic set `ab123_example/` (10 variants of “ALPHA BRAVO 123” at varied WPM/frequency/amplitude), all samples decode correctly. Plots/CSVs are in `ab123_example/segments/`.

## When it fails and what to tweak
- If spans are chopped too finely: increase `--smooth-ms` or lower `--threshold-ratio`.
- If long gaps are not promoted to gap3/gap7: the off-run cutoffs are in `classify_runs` in `analyses/segment_envelope.py` (multipliers 1×/1.8×/3.5× of the median off length).
- If dots/dashes are confused: adjust the `1.6` multiplier on the median on length (same file).

## Self-supervised pretrain hook
To inject the 1/3/7 prior into a DL model without full labels, use the helper:
```bash
PYTHONPATH=src .venv/bin/python analyses/train_self_supervised_envelope.py \
  --train-manifest ab123_example/manifests/test.jsonl \
  --val-manifest   ab123_example/manifests/test.jsonl \
  --smooth-ms 10 --threshold-ratio 0.3 --mask-weight 0.5 \
  --out-dir outputs/self_supervised_envelope_demo
```
It trains `MultiTaskCTCCountsModel` with an auxiliary mask head (on/off envelope) plus CTC on pseudo text (the manifest’s `text` can be produced by the envelope segmenter). The mask labels are recomputed on the fly from the envelope using the same smoothing/threshold parameters.

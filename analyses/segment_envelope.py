"""
Rule-based segmentation of Morse audio using the amplitude envelope.

Given a manifest, this script:
- Computes a smoothed amplitude envelope from each WAV.
- Thresholds to on/off.
- Estimates the dot time unit (from WPM if present, otherwise median short runs).
- Quantizes run lengths to units and classifies runs as dot, dash, gap (1/3/7).
- Decodes to dots/dashes and text (if MORSE_CODE mapping covers chars).
- Saves a plot with waveform, envelope, threshold, and shaded on/off runs.
- Writes a CSV of segments (start/end in seconds, type).

Example:
    PYTHONPATH=src .venv/bin/python analyses/segment_envelope.py \
      --manifest ab123_example/manifests/test.jsonl \
      --out-dir ab123_example/segments
"""
import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from audio2morse.data.morse_map import MORSE_CODE


def load_manifest(path: Path) -> List[dict]:
    """Load JSONL manifest into a list of dicts."""
    with path.open() as fp:
        return [json.loads(line) for line in fp if line.strip()]


def two_cluster_split(lengths: List[int]) -> float | None:
    """Simple 1D 2-means; returns the midpoint between the two centroids."""
    if len(lengths) < 2:
        return None
    c1, c2 = float(min(lengths)), float(max(lengths))
    for _ in range(12):
        cluster1 = [x for x in lengths if abs(x - c1) <= abs(x - c2)]
        cluster2 = [x for x in lengths if abs(x - c1) > abs(x - c2)]
        if not cluster1 or not cluster2:
            break
        new_c1, new_c2 = float(np.mean(cluster1)), float(np.mean(cluster2))
        if abs(new_c1 - c1) + abs(new_c2 - c2) < 1e-3:
            break
        c1, c2 = new_c1, new_c2
    if c1 == c2:
        return None
    low, high = (c1, c2) if c1 < c2 else (c2, c1)
    return 0.5 * (low + high)


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float32) / float(win)
    pad = win // 2
    x_pad = np.pad(x, (pad, win - 1 - pad), mode="reflect")
    return np.convolve(x_pad, kernel, mode="valid")


def run_lengths(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs = []
    current = int(mask[0])
    length = 1
    for v in mask[1:]:
        if int(v) == current:
            length += 1
        else:
            runs.append((current, length))
            current = int(v)
            length = 1
    runs.append((current, length))
    return runs


def estimate_unit(on_runs: List[int], off_runs: List[int], sr: int, wpm: float | None) -> float:
    if wpm and wpm > 0:
        return 1.2 / float(wpm)
    pool = []
    if on_runs:
        pool.extend(sorted(on_runs)[: max(1, len(on_runs) // 2)])
    if off_runs:
        pool.extend(sorted(off_runs)[: max(1, len(off_runs) // 2)])
    if not pool:
        return 0.05
    median_samples = float(np.median(pool))
    return median_samples / sr


def classify_runs(
    runs: List[Tuple[int, int]],
    sr: int,
    unit: float,
    on_split: float | None,
    off_split: float | None,
) -> List[dict]:
    unit_samples = unit * sr if unit else 0.0
    segments = []
    t_cursor = 0
    for on, length in runs:
        start = t_cursor
        end = t_cursor + length
        t_cursor = end
        if on == 1:
            if on_split:
                kind = "dot" if length <= on_split else "dash"
            else:
                units = length / unit_samples if unit_samples > 0 else 0
                kind = "dot" if units < 2 else "dash"
            units = length / unit_samples if unit_samples > 0 else 0
        else:
            units = length / unit_samples if unit_samples > 0 else 0
            if off_split:
                if length > 3.5 * off_split:
                    kind = "gap7"
                elif length > 1.8 * off_split:
                    kind = "gap3"
                else:
                    kind = "gap1"
            else:
                if units >= 6:
                    kind = "gap7"
                elif units >= 3:
                    kind = "gap3"
                else:
                    kind = "gap1"
        segments.append({"on": bool(on), "start": start, "end": end, "kind": kind, "units": units})
    return segments


def decode_segments(segments: List[dict]) -> str:
    chars = []
    current = []
    for seg in segments:
        if seg["on"]:
            current.append("." if seg["kind"] == "dot" else "-")
        else:
            if seg["kind"] in ("gap3", "gap7"):
                if current:
                    morse = "".join(current)
                    # invert MORSE_CODE mapping
                    inv = {v: k for k, v in MORSE_CODE.items()}
                    chars.append(inv.get(morse, "?"))
                    current = []
                if seg["kind"] == "gap7":
                    chars.append(" ")
    if current:
        inv = {v: k for k, v in MORSE_CODE.items()}
        chars.append(inv.get("".join(current), "?"))
    return "".join(chars).strip()


def process_sample(sample: dict, out_dir: Path, smooth_ms: float, thr_ratio: float):
    wav, sr = sf.read(sample["audio_filepath"], always_2d=True)
    mono = wav.mean(axis=1)
    env = np.abs(mono)
    win = int(sr * smooth_ms / 1000.0)
    env_s = moving_average(env, win)
    med = np.median(env_s)
    p95 = np.percentile(env_s, 95)
    thr = med + thr_ratio * (p95 - med)
    mask = (env_s > thr).astype(np.int32)
    runs = run_lengths(mask)
    on_runs = [l for on, l in runs if on == 1]
    off_runs = [l for on, l in runs if on == 0]
    unit = estimate_unit(on_runs, off_runs, sr, sample.get("wpm"))
    # Dot/dash split: scaled median of on-run lengths to bias toward dot.
    on_split = float(np.median(on_runs) * 1.6) if on_runs else None
    # Gap split: median of off-run lengths.
    off_split = float(np.median(off_runs)) if off_runs else None

    segments = classify_runs(runs, sr, unit, on_split, off_split)
    decoded = decode_segments(segments)

    # Plot
    t = np.arange(len(mono)) / sr
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(t, mono, linewidth=0.5, color="tab:blue")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title(f"{Path(sample['audio_filepath']).name} | decoded: {decoded}")
    ax[1].plot(t[: len(env_s)], env_s, linewidth=0.8, color="tab:orange", label="envelope")
    ax[1].axhline(thr, color="red", linestyle="--", linewidth=0.8, label="threshold")
    y_min, y_max = env_s.min(), env_s.max()
    colors = {"dot": "green", "dash": "red", "gap1": "purple", "gap3": "purple", "gap7": "purple"}
    for seg in segments:
        color = colors.get(seg["kind"], "gray")
        alpha = 0.25 if seg["on"] else 0.15
        ax[1].axvspan(seg["start"] / sr, seg["end"] / sr, color=color, alpha=alpha)
    ax[1].set_ylabel("Envelope")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{Path(sample['audio_filepath']).stem}_segments.png")
    plt.close(fig)

    # Segments CSV
    seg_rows = []
    for seg in segments:
        seg_rows.append(
            {
                "audio": sample["audio_filepath"],
                "start_s": seg["start"] / sr,
                "end_s": seg["end"] / sr,
                "kind": seg["kind"],
                "on": seg["on"],
                "units": seg["units"],
            }
        )
    return decoded, seg_rows


def main():
    parser = argparse.ArgumentParser(description="Segment Morse audio via envelope thresholding and 1/3/7 rules.")
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for plots/segments.")
    parser.add_argument("--smooth-ms", type=float, default=5.0, help="Envelope smoothing window (ms).")
    parser.add_argument("--threshold-ratio", type=float, default=0.4, help="Median + ratio*(max-median).")
    args = parser.parse_args()

    segments_out = []
    decodes = []
    for sample in load_manifest(args.manifest):
        decoded, seg_rows = process_sample(sample, args.out_dir, args.smooth_ms, args.threshold_ratio)
        decodes.append({"audio": sample["audio_filepath"], "decoded": decoded, "text": sample.get("text", "")})
        segments_out.extend(seg_rows)

    import csv

    seg_path = args.out_dir / "segments.csv"
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    with seg_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["audio", "start_s", "end_s", "kind", "on", "units"])
        writer.writeheader()
        writer.writerows(segments_out)

    dec_path = args.out_dir / "decodes.csv"
    with dec_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["audio", "decoded", "text"])
        writer.writeheader()
        writer.writerows(decodes)
    print(f"Wrote plots+segments to {args.out_dir}, decodes to {dec_path}, segments to {seg_path}")


if __name__ == "__main__":
    main()

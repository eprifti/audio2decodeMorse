"""
Timing-based Morse count baseline (no learned model).

This script estimates the Morse "dot" time unit from audio, then counts
characters by walking on/off durations using the 1/3/7-unit rules.

Example:
    PYTHONPATH=src python3 analyses/count_from_timing.py \
      --train data/datasets/simple_baseline/manifests/train.jsonl \
      --val   data/datasets/simple_baseline/manifests/val.jsonl \
      --test  data/datasets/simple_baseline/manifests/test.jsonl \
      --out outputs/timing_counts.csv
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import soundfile as sf


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    # Use reflect padding to avoid edge dips
    padded = np.pad(x, (win // 2, win - 1 - win // 2), mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    # Cheap Hilbert via scipy is unavailable; approximate with magnitude of analytic signal
    # using FFT-based approach. For speed and to avoid deps, fall back to absolute value
    # of signal smoothed by moving average (handled upstream).
    return np.abs(x)


def binarize_envelope(
    env: np.ndarray,
    ratio: float = 0.6,
    percentile: float | None = None,
    method: str = "static",
    mad_k: float = 3.0,
) -> Tuple[np.ndarray, float]:
    if method == "adaptive":
        med = float(np.median(env))
        mad = float(np.median(np.abs(env - med))) + 1e-8
        thr = med + mad_k * mad
    elif percentile is not None:
        thr = float(np.percentile(env, percentile))
    else:
        med = float(np.median(env))
        mx = float(env.max())
        thr = med + ratio * (mx - med)
    if thr <= 0:
        thr = np.mean(env) if np.mean(env) > 0 else 1e-6
    return env > thr, thr


def run_lengths(mask: np.ndarray) -> List[Tuple[bool, int]]:
    runs: List[Tuple[bool, int]] = []
    current = bool(mask[0])
    length = 1
    for v in mask[1:]:
        if bool(v) == current:
            length += 1
        else:
            runs.append((current, length))
            current = bool(v)
            length = 1
    runs.append((current, length))
    return runs


def merge_short_gaps(runs: List[Tuple[bool, int]], sr: int, unit: float, merge_gap_units: float) -> List[Tuple[bool, int]]:
    if unit <= 0 or merge_gap_units <= 0:
        return runs
    merged: List[Tuple[bool, int]] = []
    i = 0
    max_gap = merge_gap_units * unit
    while i < len(runs):
        on, length = runs[i]
        if not on:
            gap = length / sr
            if gap < max_gap and merged and i + 1 < len(runs) and runs[i + 1][0]:
                prev_on, prev_len = merged.pop()
                next_on, next_len = runs[i + 1]
                merged.append((True, prev_len + length + next_len))
                i += 2
                continue
        merged.append((on, length))
        i += 1
    return merged


def estimate_unit(on_durs: List[float], off_durs: List[float], wpm: float | None) -> float:
    # If WPM is known, use the textbook mapping: dot = 1.2 / WPM seconds.
    if wpm and wpm > 0:
        return 1.2 / float(wpm)
    pool = []
    if on_durs:
        pool.extend(sorted(on_durs)[: max(1, len(on_durs) // 2)])
    if off_durs:
        pool.extend(sorted(off_durs)[: max(1, len(off_durs) // 2)])
    if not pool:
        return 0.05  # safe default for ~24 WPM
    return float(np.median(pool))


def quantize_units(duration: float, unit: float, allowed: List[int]) -> int:
    if unit <= 0:
        return allowed[0]
    ratio = duration / unit
    return min(allowed, key=lambda x: abs(ratio - x))


def predict_count(
    path: Path,
    wpm: float | None,
    smooth_ms: float = 8.0,
    min_run_ms: float = 6.0,
    thr_ratio: float = 0.6,
    percentile_thr: float | None = None,
    threshold_method: str = "static",
    mad_k: float = 3.0,
    merge_gap_units: float = 0.5,
) -> int:
    wav, sr = sf.read(path, always_2d=True)
    mono = wav.mean(axis=1)
    env = hilbert_envelope(mono)
    win = int(sr * smooth_ms / 1000.0)
    smooth = moving_average(env, win)
    mask, _ = binarize_envelope(
        smooth,
        ratio=thr_ratio,
        percentile=percentile_thr,
        method=threshold_method,
        mad_k=mad_k,
    )

    # Drop extremely short blips that are almost surely noise.
    min_run = max(1, int((min_run_ms / 1000.0) * sr))
    raw_runs = run_lengths(mask)
    runs = [(on, length) for on, length in raw_runs if length >= min_run]
    if not runs:
        return 0

    on_durs = [l / sr for on, l in runs if on]
    off_durs = [l / sr for on, l in runs if not on]
    unit = estimate_unit(on_durs, off_durs, wpm)
    runs = merge_short_gaps(runs, sr, unit, merge_gap_units)

    count = 0
    in_char = False
    for on, length in runs:
        dur = length / sr
        if on:
            if not in_char:
                count += 1
                in_char = True
        else:
            gap_units = quantize_units(dur, unit, [1, 3, 7])
            if gap_units < 1.5:
                # very short gap, stay in character
                continue
            if gap_units >= 5.5:
                in_char = False
            elif gap_units >= 2.5:
                in_char = False
    return count


def iter_manifest(path: Path, limit: int | None = None):
    with path.open("r") as fp:
        for i, line in enumerate(fp):
            if limit is not None and i >= limit:
                break
            yield json.loads(line)


def process_manifest(
    manifest: Path,
    partition: str,
    limit: int | None = None,
    smooth_ms: float = 8.0,
    min_run_ms: float = 6.0,
    thr_ratio: float = 0.6,
    percentile_thr: float | None = None,
    threshold_method: str = "static",
    mad_k: float = 3.0,
    merge_gap_units: float = 0.5,
) -> List[dict]:
    rows = []
    for sample in iter_manifest(manifest, limit=limit):
        audio_path = Path(sample["audio_filepath"])
        true_count = len(sample.get("text", ""))
        wpm = sample.get("wpm")
        pred = predict_count(
            audio_path,
            wpm,
            smooth_ms=smooth_ms,
            min_run_ms=min_run_ms,
            thr_ratio=thr_ratio,
            percentile_thr=percentile_thr,
            threshold_method=threshold_method,
            mad_k=mad_k,
            merge_gap_units=merge_gap_units,
        )
        err = pred - true_count
        rows.append(
            {
                "partition": partition,
                "audio_filepath": str(audio_path),
                "true_count": true_count,
                "predicted_count": pred,
                "error": err,
                "abs_error": abs(err),
                "wpm": wpm,
            }
        )
    return rows


def summarize(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        return {"n": 0, "mae": None, "rmse": None}
    abs_err = np.array([r["abs_error"] for r in rows], dtype=np.float32)
    err = np.array([r["error"] for r in rows], dtype=np.float32)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    return {"n": len(rows), "mae": mae, "rmse": rmse}


def main():
    parser = argparse.ArgumentParser(description="Count characters from timing heuristics (no model).")
    parser.add_argument("--train", type=Path, help="Train manifest JSONL")
    parser.add_argument("--val", type=Path, help="Val manifest JSONL")
    parser.add_argument("--test", type=Path, help="Test manifest JSONL")
    parser.add_argument("--out", type=Path, default=Path("outputs/timing_counts.csv"), help="Where to write per-utt CSV")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap per split for quick checks")
    parser.add_argument("--smooth-ms", type=float, default=8.0, help="Smoothing window for envelope (ms)")
    parser.add_argument("--min-run-ms", type=float, default=6.0, help="Minimum on/off duration to keep (ms)")
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=0.4,
        help="Threshold = median + ratio*(max-median); raise to suppress noise blips.",
    )
    parser.add_argument(
        "--threshold-method",
        choices=["static", "adaptive"],
        default="static",
        help="Use median+ratio or adaptive median+MAD thresholding.",
    )
    parser.add_argument(
        "--mad-k",
        type=float,
        default=3.0,
        help="Multiplier for MAD when threshold-method=adaptive.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=None,
        help="Optional envelope percentile to use as threshold (overrides ratio if set).",
    )
    parser.add_argument(
        "--merge-gap-units",
        type=float,
        default=0.5,
        help="Merge on-runs separated by gaps shorter than this many units (helps prevent over-splitting).",
    )
    args = parser.parse_args()

    all_rows: List[dict] = []
    for partition, path in (("train", args.train), ("val", args.val), ("test", args.test)):
        if path is None:
            continue
        print(f"Processing {partition} from {path} (limit={args.limit})")
        rows = process_manifest(
            path,
            partition,
            limit=args.limit,
            smooth_ms=args.smooth_ms,
            min_run_ms=args.min_run_ms,
            thr_ratio=args.threshold_ratio,
            percentile_thr=args.threshold_percentile,
            threshold_method=args.threshold_method,
            mad_k=args.mad_k,
            merge_gap_units=args.merge_gap_units,
        )
        all_rows.extend(rows)
        stats = summarize(rows)
        fmt = lambda x: "n/a" if x is None else f"{x:.4f}"
        print(f"{partition}: n={stats['n']} MAE={fmt(stats['mae'])} RMSE={fmt(stats['rmse'])}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["partition", "audio_filepath", "true_count", "predicted_count", "error", "abs_error", "wpm"],
        )
        writer.writeheader()
        writer.writerows(all_rows)
    total_stats = summarize(all_rows)
    fmt = lambda x: "n/a" if x is None else f"{x:.4f}"
    print(f"All: n={total_stats['n']} MAE={fmt(total_stats['mae'])} RMSE={fmt(total_stats['rmse'])}")
    print(f"Wrote per-utterance counts to {args.out}")


if __name__ == "__main__":
    main()

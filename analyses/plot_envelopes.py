"""
Plot waveform + amplitude envelope for a manifest of Morse samples.

Example:
    PYTHONPATH=src .venv/bin/python analyses/plot_envelopes.py \
      --manifest ab123_example/manifests/test.jsonl \
      --out-dir ab123_example/envelopes
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float32) / float(win)
    pad = win // 2
    x_pad = np.pad(x, (pad, win - 1 - pad), mode="reflect")
    return np.convolve(x_pad, kernel, mode="valid")


def load_manifest(path: Path):
    with path.open() as fp:
        for line in fp:
            yield json.loads(line)


def plot_envelope(sample: dict, out_dir: Path, smooth_ms: float = 5.0, thr_ratio: float = 0.4):
    wav, sr = sf.read(sample["audio_filepath"], always_2d=True)
    mono = wav.mean(axis=1)
    env = np.abs(mono)
    win = int(sr * smooth_ms / 1000.0)
    env_s = moving_average(env, win)
    thr = np.median(env_s) + thr_ratio * (env_s.max() - np.median(env_s))
    t = np.arange(len(mono)) / sr

    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(t, mono, linewidth=0.5, color="tab:blue")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title(f"WAV: {Path(sample['audio_filepath']).name}")
    ax[1].plot(t, env_s, linewidth=0.8, color="tab:orange", label="envelope")
    ax[1].axhline(thr, color="red", linestyle="--", linewidth=0.8, label="threshold")
    ax[1].set_ylabel("Envelope")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend()
    fig.tight_layout()
    out_path = out_dir / f"{Path(sample['audio_filepath']).stem}_envelope.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot waveform and envelope for a manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write PNGs.")
    parser.add_argument("--smooth-ms", type=float, default=5.0, help="Smoothing window for envelope (ms).")
    parser.add_argument("--threshold-ratio", type=float, default=0.4, help="Median + ratio*(max-median).")
    args = parser.parse_args()

    for sample in load_manifest(args.manifest):
        plot_envelope(sample, args.out_dir, smooth_ms=args.smooth_ms, thr_ratio=args.threshold_ratio)


if __name__ == "__main__":
    main()

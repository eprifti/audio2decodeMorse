"""
Generate synthetic Morse-code audio from text and write a JSONL manifest. Supports
per-sample variation in tone frequency, speed (WPM), and amplitude to improve
robustness.

Example (from repo root):
    PYTHONPATH=src python3 -m audio2morse.data.generate_synthetic_morse \
      --input texts.txt \
      --out-dir data/audio \
      --manifest data/manifests/train.jsonl \
      --sample-rate 16000 \
      --wpm-min 18 --wpm-max 25 \
      --freq-min 500 --freq-max 900 \
      --amp-min 0.2 --amp-max 0.4

`texts.txt` should contain one message per line (uppercase recommended). For each
line, a WAV is generated and a manifest entry is appended to the JSONL file.
"""
import argparse
import json
from pathlib import Path
from typing import List

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml

from audio2morse.data.morse_map import MORSE_CODE


def text_to_morse(message: str) -> List[str]:
    """Convert a message into a sequence of Morse patterns (one per character)."""
    patterns: List[str] = []
    for ch in message.upper():
        if ch == " ":
            patterns.append(" ")  # sentinel for word gap
            continue
        if ch in MORSE_CODE:
            patterns.append(MORSE_CODE[ch])
        else:
            raise ValueError(f"Unsupported character in message: '{ch}'")
    return patterns


def synthesize_morse(
    message: str,
    sample_rate: int,
    wpm: float,
    freq: float,
    amplitude: float,
) -> np.ndarray:
    """
    Generate a mono waveform for the given message using standard Morse timing.

    Timing rules:
      dot = 1 time unit
      dash = 3 time units
      intra-character gap = 1 unit
      inter-character gap = 3 units
      word gap = 7 units
    """
    dot_sec = 1.2 / wpm  # standard PARIS timing
    unit = int(sample_rate * dot_sec)
    if unit < 1:
        unit = 1

    def tone(duration_units: int) -> np.ndarray:
        t = np.arange(duration_units * unit) / sample_rate
        return amplitude * np.sin(2 * np.pi * freq * t)

    def silence(duration_units: int) -> np.ndarray:
        return np.zeros(duration_units * unit, dtype=np.float32)

    patterns = text_to_morse(message)
    segments = []
    for i, pat in enumerate(patterns):
        if pat == " ":
            segments.append(silence(7))
            continue
        for j, symbol in enumerate(pat):
            segments.append(tone(1 if symbol == "." else 3))
            # intra-character gap, except after last symbol of the character
            if j < len(pat) - 1:
                segments.append(silence(1))
        # inter-character gap, except after last character or before word gap
        if i < len(patterns) - 1 and patterns[i + 1] != " ":
            segments.append(silence(3))
        elif i < len(patterns) - 1 and patterns[i + 1] == " ":
            # word gap handled by next iteration
            pass

    if not segments:
        return np.zeros(1, dtype=np.float32)
    waveform = np.concatenate(segments).astype(np.float32)
    return waveform


def save_waveform_plot(waveform: np.ndarray, sample_rate: int, path: Path) -> None:
    """Save a simple waveform plot to PNG."""
    t = np.arange(waveform.shape[0]) / sample_rate
    plt.figure(figsize=(8, 2))
    plt.plot(t, waveform, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Morse WAVs and manifest.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config to override CLI defaults.")
    parser.add_argument("--input", required=False, help="Text file with one message per line.")
    parser.add_argument("--out-dir", required=False, help="Directory to write WAV files.")
    parser.add_argument("--manifest", required=False, help="Path to train JSONL manifest to write.")
    parser.add_argument("--test-manifest", help="Optional path to test JSONL manifest; auto-created if train_ratio<1.")
    parser.add_argument("--val-manifest", help="Optional path to validation JSONL manifest.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Fraction of samples to assign to train.")
    parser.add_argument("--val-ratio", type=float, default=0.0, help="Fraction of samples to assign to validation.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate for output WAVs.")
    parser.add_argument("--num-samples", type=int, default=None, help="Total samples to generate (sampled with replacement). Defaults to one per input line.")
    parser.add_argument("--wpm", type=float, default=None, help="Fixed words per minute speed (overrides min/max).")
    parser.add_argument("--wpm-min", type=float, default=20.0, help="Min WPM for random sampling.")
    parser.add_argument("--wpm-max", type=float, default=20.0, help="Max WPM for random sampling.")
    parser.add_argument("--freq", type=float, default=None, help="Fixed tone frequency (overrides min/max).")
    parser.add_argument("--freq-min", type=float, default=700.0, help="Min tone frequency for random sampling.")
    parser.add_argument("--freq-max", type=float, default=700.0, help="Max tone frequency for random sampling.")
    parser.add_argument("--amp", type=float, default=None, help="Fixed amplitude (overrides min/max).")
    parser.add_argument("--amp-min", type=float, default=0.3, help="Min amplitude for random sampling.")
    parser.add_argument("--amp-max", type=float, default=0.3, help="Max amplitude for random sampling.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    parser.add_argument("--plot-dir", type=str, default=None, help="Optional directory to save waveform PNGs.")
    parser.add_argument("--target-chars", type=str, default=None, help="Characters to emphasize (e.g., 'QZ?').")
    parser.add_argument("--target-samples", type=int, default=0, help="How many extra messages to synthesize from target chars.")
    parser.add_argument("--target-min-len", type=int, default=3, help="Minimum length of generated target messages.")
    parser.add_argument("--target-max-len", type=int, default=8, help="Maximum length of generated target messages.")
    parser.add_argument("--target-words-file", type=str, default=None, help="Optional file with one word per line to sample phrases.")
    parser.add_argument("--target-words-samples", type=int, default=0, help="Extra phrases built from random words (space-separated).")
    parser.add_argument("--target-words-min", type=int, default=2, help="Min words per synthetic phrase.")
    parser.add_argument("--target-words-max", type=int, default=5, help="Max words per synthetic phrase.")
    args = parser.parse_args()

    # Optionally override defaults with a YAML config.
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        with cfg_path.open("r") as fp:
            cfg = yaml.safe_load(fp) or {}
        # Only update keys that match known args (excluding None/False to avoid stomping CLI-specified values).
        known_keys = {a.dest for a in parser._actions if a.dest != "help"}
        for k, v in cfg.items():
            if k in known_keys and getattr(args, k) in (parser.get_default(k), None):
                setattr(args, k, v)

    # Validate required fields after applying config/CLI overrides.
    missing = [name for name in ["input", "out_dir", "manifest"] if getattr(args, name) in (None, "")]
    if missing:
        raise ValueError(f"Missing required arguments (provide via CLI or --config): {', '.join(missing)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_manifest_path = Path(args.manifest)
    train_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    val_manifest_path = Path(args.val_manifest) if args.val_manifest else None
    test_manifest_path = Path(args.test_manifest) if args.test_manifest else None
    if test_manifest_path is None and args.train_ratio < 1.0 and (args.val_ratio or val_manifest_path):
        # If val is specified and train_ratio < 1, use remainder for test only if path provided.
        test_manifest_path = None
    if test_manifest_path is None and args.train_ratio < 1.0 and val_manifest_path is None:
        test_manifest_path = train_manifest_path.parent / "test.jsonl"
    if val_manifest_path:
        val_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if test_manifest_path:
        test_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    with open(args.input, "r") as f:
        messages = [line.strip() for line in f if line.strip()]

    # Optional: append synthetic phrases to over-emphasize spaces/word structure.
    if args.target_words_file and args.target_words_samples > 0:
        with open(args.target_words_file, "r") as wf:
            words = [w.strip().upper() for w in wf if w.strip()]
        if not words:
            raise ValueError("No words found in target_words_file.")
        for _ in range(args.target_words_samples):
            length = int(rng.integers(args.target_words_min, args.target_words_max + 1))
            phrase_words = rng.choice(words, size=length, replace=True).tolist()
            messages.append(" ".join(phrase_words))

    if args.target_chars and args.target_samples > 0:
        chars = [c for c in args.target_chars.upper() if c.strip()]
        if not chars:
            raise ValueError("No valid characters found in --target-chars.")
        for _ in range(args.target_samples):
            length = int(rng.integers(args.target_min_len, args.target_max_len + 1))
            msg_chars = rng.choice(chars, size=length, replace=True)
            messages.append("".join(msg_chars))

    if not messages:
        raise ValueError("No messages found in input file.")

    total = args.num_samples if args.num_samples is not None else len(messages)
    selected = rng.choice(messages, size=total, replace=True) if total != len(messages) else messages

    entries = []
    plot_dir = Path(args.plot_dir) if args.plot_dir else None
    total_start = time.perf_counter()
    for idx, msg in enumerate(selected):
        sample_start = time.perf_counter()
        wpm = args.wpm if args.wpm is not None else float(rng.uniform(args.wpm_min, args.wpm_max))
        freq = args.freq if args.freq is not None else float(rng.uniform(args.freq_min, args.freq_max))
        amp = args.amp if args.amp is not None else float(rng.uniform(args.amp_min, args.amp_max))
        waveform = synthesize_morse(msg, sample_rate=args.sample_rate, wpm=wpm, freq=freq, amplitude=amp)

        wav_path = out_dir / f"synthetic_{idx:05d}.wav"
        sf.write(wav_path, waveform, args.sample_rate)

        if plot_dir:
            png_path = plot_dir / f"synthetic_{idx:05d}.png"
            save_waveform_plot(waveform, args.sample_rate, png_path)

        entries.append(
            {
                "audio_filepath": str(wav_path),
                "text": msg.upper(),
                "text_len": len(msg),
                "freq_hz": freq,
                "wpm": wpm,
                "amplitude": amp,
            }
        )
        sample_ms = (time.perf_counter() - sample_start) * 1000
        print(
            f"Wrote {wav_path} | msg='{msg}' | wpm={wpm:.2f} | freq={freq:.1f}Hz | amp={amp:.2f} | time={sample_ms:.1f}ms"
        )

    # Shuffle and split
    perm = rng.permutation(len(entries))
    n = len(entries)
    if args.train_ratio + args.val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")
    train_end = int(n * args.train_ratio) if (val_manifest_path or test_manifest_path) else n
    val_end = train_end + int(n * args.val_ratio) if val_manifest_path else train_end
    train_entries = [entries[i] for i in perm[:train_end]]
    val_entries = [entries[i] for i in perm[train_end:val_end]] if val_manifest_path else []
    test_entries = [entries[i] for i in perm[val_end:]] if test_manifest_path else []

    with train_manifest_path.open("w") as mfp:
        for entry in train_entries:
            mfp.write(json.dumps(entry) + "\n")
    if val_manifest_path and val_entries:
        with val_manifest_path.open("w") as vfp:
            for entry in val_entries:
                vfp.write(json.dumps(entry) + "\n")
    if test_manifest_path and test_entries:
        with test_manifest_path.open("w") as tfp:
            for entry in test_entries:
                tfp.write(json.dumps(entry) + "\n")

    total_sec = time.perf_counter() - total_start
    avg_ms = (total_sec / len(entries)) * 1000 if entries else 0
    print(f"Done. Train manifest: {train_manifest_path} ({len(train_entries)} entries)")
    if val_manifest_path:
        print(f"Val manifest: {val_manifest_path} ({len(val_entries)} entries)")
    if test_manifest_path:
        print(f"Test manifest: {test_manifest_path} ({len(test_entries)} entries)")
    print(f"Total generation time: {total_sec:.2f}s | Avg per sample: {avg_ms:.1f}ms")


if __name__ == "__main__":
    main()

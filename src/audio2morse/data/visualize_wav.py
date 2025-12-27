"""
Quick CLI utility to visualize a WAV file used for Morse decoding experiments.

Example (from repo root):
    PYTHONPATH=src python3 -m audio2morse.data.visualize_wav \
        --audio data/example.wav \
        --save outputs/example_waveform.png

If you omit --save, the plot opens in a window (useful in VS Code).
"""
import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torchaudio


def load_waveform(path: Path) -> tuple[torch.Tensor, int]:
    """
    Load a WAV file. If stereo, average to mono for simpler visualization.

    Returns:
        waveform: Tensor shape (1, samples)
        sample_rate: int
    """
    samples, sr = sf.read(path, always_2d=True)
    wav = torch.from_numpy(samples.transpose()).float()  # (channels, samples)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def plot_waveform(wav: torch.Tensor, sr: int, title: str, save_path: Optional[Path]) -> None:
    """Render waveform over time, either saving to disk or showing interactively."""
    samples = wav.shape[1]
    time_axis = torch.arange(0, samples) / sr

    plt.figure(figsize=(10, 3))
    plt.plot(time_axis, wav.squeeze(0).numpy(), linewidth=0.8)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize a WAV file for Morse decoding.")
    parser.add_argument("--audio", type=str, required=True, help="Path to WAV file.")
    parser.add_argument("--save", type=str, help="Optional path to save the plot as PNG. If omitted, opens a window.")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    save_path = Path(args.save) if args.save else None

    wav, sr = load_waveform(audio_path)
    plot_waveform(wav, sr, title=f"{audio_path.name} (sr={sr})", save_path=save_path)


if __name__ == "__main__":
    main()

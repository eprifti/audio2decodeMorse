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


def plot_waveform_and_spec(
    wav: torch.Tensor,
    sr: int,
    title: str,
    save_path: Optional[Path],
    show_spec: bool = True,
) -> None:
    """Render waveform (and optional log-mel spectrogram) to file or window."""
    samples = wav.shape[1]
    time_axis = torch.arange(0, samples) / sr

    if show_spec:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False, gridspec_kw={"height_ratios": [1, 1.5]})
        ax_wav, ax_spec = axes
    else:
        fig, ax_wav = plt.subplots(1, 1, figsize=(10, 3))

    ax_wav.plot(time_axis, wav.squeeze(0).numpy(), linewidth=0.8)
    ax_wav.set_title(title)
    ax_wav.set_xlabel("Time (seconds)")
    ax_wav.set_ylabel("Amplitude")

    if show_spec:
        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=int(sr * 0.025),
            hop_length=int(sr * 0.010),
            n_mels=64,
        )
        to_db = torchaudio.transforms.AmplitudeToDB()
        with torch.no_grad():
            spec = to_db(mel_tf(wav)).squeeze(0)
        im = ax_spec.imshow(spec.numpy(), origin="lower", aspect="auto", cmap="magma")
        ax_spec.set_ylabel("Mel bins")
        ax_spec.set_xlabel("Frames")
        ax_spec.set_title("Log-mel spectrogram")
        fig.colorbar(im, ax=ax_spec, label="dB")

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize a WAV file for Morse decoding.")
    parser.add_argument("--audio", type=str, required=True, help="Path to WAV file.")
    parser.add_argument("--save", type=str, help="Optional path to save the plot as PNG. If omitted, opens a window.")
    parser.add_argument("--no-spec", action="store_true", help="Disable spectrogram; plot waveform only.")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    save_path = Path(args.save) if args.save else None

    wav, sr = load_waveform(audio_path)
    plot_waveform_and_spec(
        wav,
        sr,
        title=f"{audio_path.name} (sr={sr})",
        save_path=save_path,
        show_spec=not args.no_spec,
    )


if __name__ == "__main__":
    main()

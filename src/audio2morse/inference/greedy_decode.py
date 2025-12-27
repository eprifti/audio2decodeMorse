"""
Command-line helper for loading a trained Morse CTC model and decoding a single
audio file with simple greedy decoding.

Usage example (from repo root):
    PYTHONPATH=src python3 -m audio2morse.inference.greedy_decode \\
        --checkpoint outputs/best.pt \\
        --audio data/audio/example.wav
"""
import argparse
from pathlib import Path
from typing import Dict, List

import soundfile as sf
import torch
import torchaudio

from audio2morse.data.vocab import build_vocab, index_to_char
from audio2morse.models.ctc_model import CTCMorseModel


def load_checkpoint(path: Path):
    """Load model weights plus config/alphabet saved during training."""
    ckpt = torch.load(path, map_location="cpu")
    return ckpt["model_state"], ckpt["config"], ckpt["alphabet"]


def prepare_model(cfg: Dict, alphabet: str, device: torch.device) -> CTCMorseModel:
    """
    Recreate the model architecture from config and move it to the chosen device.
    The alphabet is used to rebuild the vocab so output dimensions match training.
    """
    label_map = build_vocab(alphabet)
    model = CTCMorseModel(
        input_dim=cfg["data"]["n_mels"],
        vocab_size=len(label_map),
        cnn_channels=cfg["model"]["cnn_channels"],
        rnn_hidden_size=cfg["model"]["rnn_hidden_size"],
        rnn_layers=cfg["model"]["rnn_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    return model, label_map


def load_audio(path: Path, cfg: Dict) -> torch.Tensor:
    """
    Load a mono WAV file, resample it to the training sample rate, and generate
    log-mel features that match the training configuration.
    """
    audio, sr = sf.read(path, always_2d=True)
    wav = torch.from_numpy(audio.transpose()).float()  # (channels, samples)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != cfg["data"]["sample_rate"]:
        wav = torchaudio.functional.resample(wav, sr, cfg["data"]["sample_rate"])
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg["data"]["sample_rate"],
        n_fft=int(cfg["data"]["sample_rate"] * cfg["data"]["frame_length_ms"] / 1000),
        hop_length=int(cfg["data"]["sample_rate"] * cfg["data"]["frame_step_ms"] / 1000),
        n_mels=cfg["data"]["n_mels"],
    )
    to_db = torchaudio.transforms.AmplitudeToDB()
    with torch.no_grad():
        mel = to_db(mel_tf(wav)).squeeze(0).transpose(0, 1)
    return mel


def greedy_decode(log_probs: torch.Tensor, idx_to_char: List[str]) -> str:
    """
    Convert frame-level log-probabilities into a text string by picking the most
    likely token at each frame and collapsing repeats/blanks (standard CTC
    greedy decoding). Suitable for a quick sanity check; beam search would give
    better results but is more complex.
    """
    # log_probs: (T, 1, V)
    indices = log_probs.argmax(dim=-1).squeeze(1).cpu().numpy().tolist()
    blank_idx = len(idx_to_char) - 1
    decoded = []
    prev = blank_idx
    for idx in indices:
        if idx != prev and idx != blank_idx:
            decoded.append(idx_to_char[idx])
        prev = idx
    return "".join(decoded)


def main():
    """
    CLI entrypoint:
      --checkpoint: path to a .pt file saved by training
      --audio: path to a WAV to decode
    The script auto-selects MPS on Apple, then CUDA, else CPU.
    """
    parser = argparse.ArgumentParser(description="Greedy decode Morse audio file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint.")
    parser.add_argument("--audio", type=str, required=True, help="Path to WAV file to decode.")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    state_dict, cfg, alphabet = load_checkpoint(Path(args.checkpoint))
    model, label_map = prepare_model(cfg, alphabet, device)
    model.load_state_dict(state_dict)
    model.eval()

    mel = load_audio(Path(args.audio), cfg).unsqueeze(0).to(device)
    with torch.no_grad():
        log_probs = model(mel)
    idx_to_char = index_to_char(alphabet)
    text = greedy_decode(log_probs, idx_to_char)
    print(text)


if __name__ == "__main__":
    main()

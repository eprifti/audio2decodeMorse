"""
Command-line helper for loading a trained Morse CTC model and decoding a single
audio file with greedy decoding or a small beam search.

Usage example (from repo root):
    PYTHONPATH=src python3 -m audio2morse.inference.greedy_decode \\
        --checkpoint outputs/run1/best.pt \\
        --audio data/audio/example.wav \\
        --beam-size 5
"""
import argparse
import math
from math import log
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def logaddexp(a: float, b: float) -> float:
    """Stable log(exp(a)+exp(b)) for Python versions without math.logaddexp."""
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    if a > b:
        return a + log1p_exp(b - a)
    else:
        return b + log1p_exp(a - b)


def log1p_exp(x: float) -> float:
    if x > 0:
        return x + log(1 + math.exp(-x))
    else:
        return log(1 + math.exp(x))


def ctc_beam_search(log_probs: torch.Tensor, idx_to_char: List[str], beam_size: int = 5) -> str:
    """
    Simple prefix beam search for CTC. Keeps top `beam_size` beams at each
    timestep using log-probabilities. Suitable for small alphabets.
    """
    # log_probs: (T, 1, V)
    lp = log_probs.squeeze(1)  # (T, V)
    blank_idx = len(idx_to_char) - 1
    beams: Dict[str, Tuple[float, float]] = {"": (0.0, -float("inf"))}  # prefix -> (p_blank, p_nonblank)

    for t in range(lp.size(0)):
        next_beams: Dict[str, Tuple[float, float]] = {}
        frame = lp[t]
        for prefix, (p_b, p_nb) in beams.items():
            # Stay at blank
            p_blank = logaddexp(p_b + frame[blank_idx].item(), p_nb + frame[blank_idx].item())
            best_b, best_nb = next_beams.get(prefix, (-float("inf"), -float("inf")))
            next_beams[prefix] = (max(best_b, p_blank), best_nb)

            last_char = prefix[-1] if prefix else None
            for idx, char in enumerate(idx_to_char):
                if idx == blank_idx:
                    continue
                p_char = frame[idx].item()
                if last_char == char:
                    new_p_nb = logaddexp(p_nb + p_char, best_nb)
                    next_beams[prefix] = (next_beams[prefix][0], new_p_nb)
                else:
                    new_pref = prefix + char
                    nb_old = next_beams.get(new_pref, (-float("inf"), -float("inf")))[1]
                    new_p_nb = logaddexp(logaddexp(p_b + p_char, p_nb + p_char), nb_old)
                    next_beams[new_pref] = (next_beams.get(new_pref, (-float("inf"), -float("inf")))[0], new_p_nb)

        # Prune
        beams = {}
        for pref, (p_b, p_nb) in sorted(
            next_beams.items(), key=lambda kv: logaddexp(kv[1][0], kv[1][1]), reverse=True
        )[:beam_size]:
            beams[pref] = (p_b, p_nb)

    best_prefix, (p_b, p_nb) = max(beams.items(), key=lambda kv: logaddexp(kv[1][0], kv[1][1]))
    return best_prefix


def main():
    """
    CLI entrypoint:
      --checkpoint: path to a .pt file saved by training
      --audio: path to a WAV to decode
      --device: optional override (cpu|cuda|mps)
    The script auto-selects MPS on Apple, then CUDA, else CPU unless overridden.
    """
    parser = argparse.ArgumentParser(description="Greedy decode Morse audio file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint.")
    parser.add_argument("--audio", type=str, required=True, help="Path to WAV file to decode.")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size >1 enables CTC beam search.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None, help="Force device.")
    parser.add_argument("--verbose", action="store_true", help="Print extra debug info.")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    try:
        print(f"Using device: {device}")
        print(f"Loading checkpoint: {args.checkpoint}")
        state_dict, cfg, alphabet = load_checkpoint(Path(args.checkpoint))
        model, label_map = prepare_model(cfg, alphabet, device)
        model.load_state_dict(state_dict)
        model.eval()

        print(f"Decoding audio: {args.audio}")
        mel = load_audio(Path(args.audio), cfg).unsqueeze(0).to(device)
        if args.verbose:
            print(f"Features shape (batch,time,mel): {mel.shape}")
        with torch.no_grad():
            log_probs = model(mel)
        if args.verbose:
            print(f"Log-probs shape (time,batch,vocab): {log_probs.shape}")
        idx_to_char = index_to_char(alphabet)
        if args.beam_size > 1:
            text = ctc_beam_search(log_probs, idx_to_char, beam_size=args.beam_size)
        else:
            text = greedy_decode(log_probs, idx_to_char)
        if text:
            print(text)
        else:
            print("(decoded empty string)")
    except Exception as e:
        import traceback

        print(f"Error during inference: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

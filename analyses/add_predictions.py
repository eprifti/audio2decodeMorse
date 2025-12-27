"""
Compute greedy-decoded text and per-utterance CTC loss for each entry in the
train/val/test manifests, and write a combined file with a `partition`,
`inference_text`, and `loss` column.

Example:
    PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv311/bin/python \
      analyses/add_predictions.py \
      --checkpoint outputs/best.pt \
      --train data/manifests/train.jsonl \
      --val data/manifests/val.jsonl \
      --test data/manifests/test.jsonl \
      --out-parquet analyses/combined_with_preds.parquet \
      --out-csv analyses/combined_with_preds.csv
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import soundfile as sf
import torch
import torchaudio
import yaml

from audio2morse.data.vocab import build_vocab, index_to_char
from audio2morse.models.ctc_model import CTCMorseModel


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(path: Path) -> Dict:
    with path.open("r") as fp:
        return yaml.safe_load(fp)


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt["model_state"], ckpt.get("config"), ckpt.get("alphabet")


def prepare_model(cfg: Dict, alphabet: str, device: torch.device) -> CTCMorseModel:
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
    indices = log_probs.argmax(dim=-1).squeeze(1).cpu().numpy().tolist()
    blank_idx = len(idx_to_char) - 1
    decoded = []
    prev = blank_idx
    for idx in indices:
        if idx != prev and idx != blank_idx:
            decoded.append(idx_to_char[idx])
        prev = idx
    return "".join(decoded)


def text_to_targets(text: str, label_map: Dict[str, int], blank_idx: int) -> torch.Tensor:
    return torch.tensor([label_map.get(c, blank_idx) for c in text], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="Augment manifests with inference text and per-utterance loss.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint.")
    parser.add_argument("--config", default="config/default.yaml", help="Fallback config path if not in checkpoint.")
    parser.add_argument("--train", required=True, help="Train manifest JSONL.")
    parser.add_argument("--val", required=True, help="Val manifest JSONL.")
    parser.add_argument("--test", required=True, help="Test manifest JSONL.")
    parser.add_argument("--out-parquet", default="analyses/combined_with_preds.parquet", help="Output Parquet path.")
    parser.add_argument("--out-csv", default="analyses/combined_with_preds.csv", help="Output CSV path.")
    args = parser.parse_args()

    device = get_device()
    state_dict, ckpt_cfg, ckpt_alphabet = load_checkpoint(Path(args.checkpoint))
    cfg = ckpt_cfg if ckpt_cfg else load_config(Path(args.config))
    alphabet = ckpt_alphabet if ckpt_alphabet else cfg["labels"]["alphabet"]
    model, label_map = prepare_model(cfg, alphabet, device)
    model.load_state_dict(state_dict)
    model.eval()

    blank_idx = label_map["<BLANK>"]
    idx_to_char = index_to_char(alphabet)
    criterion = torch.nn.CTCLoss(blank=blank_idx, zero_infinity=True, reduction="none")

    parts = {
        "train": Path(args.train),
        "val": Path(args.val),
        "test": Path(args.test),
    }

    rows = []
    with torch.no_grad():
        for part, path in parts.items():
            if not path.exists():
                raise FileNotFoundError(f"Manifest not found: {path}")
            with path.open("r") as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    audio_path = Path(entry["audio_filepath"])
                    text = entry["text"]

                    mel = load_audio(audio_path, cfg).unsqueeze(0).to(device)  # (1,T,F)
                    log_probs = model(mel)  # (T,B,V)
                    input_lengths = torch.tensor([log_probs.shape[0]], dtype=torch.long, device=device)
                    targets = text_to_targets(text, label_map, blank_idx).to(device)
                    target_lengths = torch.tensor([targets.numel()], dtype=torch.long, device=device)

                    loss_val = criterion(log_probs, targets, input_lengths, target_lengths)[0].item()
                    hyp = greedy_decode(log_probs, idx_to_char)

                    row = dict(entry)
                    row["partition"] = part
                    row["inference_text"] = hyp
                    row["loss"] = loss_val
                    rows.append(row)

    df = pd.DataFrame(rows)
    out_parquet = Path(args.out_parquet)
    out_csv = Path(args.out_csv)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows with predictions/loss to {out_parquet} and {out_csv}")


if __name__ == "__main__":
    main()

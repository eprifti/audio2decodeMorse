"""
Self-supervised pretraining that injects the envelope 1/3/7 prior.

Two signals:
- Pseudo text labels (e.g., from the envelope segmenter or any weak decoder).
- Framewise on/off mask from the envelope thresholding itself.

Loss = CTC(pseudo_text) + lambda_mask * BCE(mask_logits, mask_labels).

Example:
    PYTHONPATH=src .venv/bin/python analyses/train_self_supervised_envelope.py \
      --train-manifest ab123_example/manifests/test.jsonl \
      --val-manifest ab123_example/manifests/test.jsonl \
      --epochs 5 --batch-size 8 --lr 1e-3 \
      --smooth-ms 10 --threshold-ratio 0.3 \
      --out-dir outputs/self_supervised_envelope_demo
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from audio2morse.data.vocab import build_vocab
from audio2morse.models.multitask_ctc_counts import MultiTaskCTCCountsModel


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float32) / float(win)
    pad = win // 2
    x_pad = np.pad(x, (pad, win - 1 - pad), mode="reflect")
    return np.convolve(x_pad, kernel, mode="valid")


def envelope_mask(wav: np.ndarray, sr: int, smooth_ms: float, thr_ratio: float) -> np.ndarray:
    env = np.abs(wav)
    win = int(sr * smooth_ms / 1000.0)
    env_s = moving_average(env, win)
    med = np.median(env_s)
    p95 = np.percentile(env_s, 95)
    thr = med + thr_ratio * (p95 - med)
    return (env_s > thr).astype(np.float32)


class PseudoEnvelopeDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        sample_rate: int = 16000,
        n_mels: int = 64,
        frame_length_ms: int = 25,
        frame_step_ms: int = 10,
        smooth_ms: float = 10.0,
        threshold_ratio: float = 0.3,
        max_duration_s: float = 14.0,
    ):
        self.items = []
        with manifest.open() as fp:
            for line in fp:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if "text" not in entry:
                    # requires pseudo text already filled in (e.g., from segmenter decodes)
                    continue
                self.items.append(entry)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_length_ms = frame_length_ms
        self.frame_step_ms = frame_step_ms
        self.smooth_ms = smooth_ms
        self.threshold_ratio = threshold_ratio
        self.max_duration_s = max_duration_s

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(sample_rate * frame_length_ms / 1000),
            hop_length=int(sample_rate * frame_step_ms / 1000),
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        row = self.items[idx]
        audio, sr = sf.read(row["audio_filepath"], always_2d=True)
        wav = audio.mean(axis=1)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, self.sample_rate).squeeze(0).numpy()
        wav = wav[: int(self.max_duration_s * self.sample_rate)]

        # Envelope mask (on/off) at waveform rate
        mask = envelope_mask(wav, self.sample_rate, self.smooth_ms, self.threshold_ratio)

        # Mel features
        waveform_t = torch.from_numpy(wav).unsqueeze(0).float()
        mel = self.to_db(self.mel(waveform_t)).squeeze(0).transpose(0, 1)  # (time, mel)

        # Downsample mask to mel frame rate via linear interpolation
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1,1,samples)
        mel_frames = mel.shape[0]
        mask_ds = torch.nn.functional.interpolate(mask_t, size=mel_frames, mode="linear", align_corners=False).squeeze()
        mask_ds = (mask_ds > 0.5).float()  # (time,)

        return {
            "features": mel,
            "mask": mask_ds,
            "text": row["text"],
        }


def collate(batch: List[Dict], label_map: Dict[str, int]):
    feature_lengths = torch.tensor([b["features"].shape[0] for b in batch], dtype=torch.long)
    max_len = int(feature_lengths.max())
    feat_dim = batch[0]["features"].shape[1]
    padded_feats = torch.zeros(len(batch), max_len, feat_dim)
    padded_masks = torch.zeros(len(batch), max_len)

    target_lengths = torch.tensor([len(b["text"]) for b in batch], dtype=torch.long)
    max_tgt = int(target_lengths.max())
    padded_targets = torch.full((len(batch), max_tgt), fill_value=label_map["<BLANK>"], dtype=torch.long)

    texts: List[str] = []
    for i, b in enumerate(batch):
        t_len = b["features"].shape[0]
        padded_feats[i, :t_len] = b["features"]
        padded_masks[i, :t_len] = b["mask"]
        for j, ch in enumerate(b["text"]):
            padded_targets[i, j] = label_map.get(ch, label_map["<BLANK>"])
        texts.append(b["text"])
    return padded_feats, feature_lengths, padded_targets, target_lengths, padded_masks, texts


def train_loop(model, loader, optimizer, device, label_map, mask_weight: float):
    model.train()
    ctc_loss_fn = nn.CTCLoss(blank=label_map["<BLANK>"], zero_infinity=True)
    bce = nn.BCEWithLogitsLoss()
    total = 0.0
    for feats, feat_lens, tgts, tgt_lens, masks, _ in tqdm(loader, desc="train", leave=False):
        feats, feat_lens, tgts, tgt_lens, masks = (
            feats.to(device),
            feat_lens.to(device),
            tgts.to(device),
            tgt_lens.to(device),
            masks.to(device),
        )
        out = model(feats, feat_lens)
        log_probs = out["text_log_probs"].permute(1, 0, 2)  # (T,B,V)
        loss = ctc_loss_fn(log_probs, tgts, out["out_lengths"], tgt_lens)
        if "mask_logits" in out:
            # align mask labels to encoder length via interpolation
            mask_labels = torch.nn.functional.interpolate(
                masks.unsqueeze(1), size=out["mask_logits"].shape[1], mode="nearest"
            ).squeeze(1)
            mask_loss = bce(out["mask_logits"], mask_labels)
            loss = loss + mask_weight * mask_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def eval_loop(model, loader, device, label_map, mask_weight: float):
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=label_map["<BLANK>"], zero_infinity=True)
    bce = nn.BCEWithLogitsLoss()
    total = 0.0
    for feats, feat_lens, tgts, tgt_lens, masks, _ in loader:
        feats, feat_lens, tgts, tgt_lens, masks = (
            feats.to(device),
            feat_lens.to(device),
            tgts.to(device),
            tgt_lens.to(device),
            masks.to(device),
        )
        out = model(feats, feat_lens)
        log_probs = out["text_log_probs"].permute(1, 0, 2)  # (T,B,V)
        loss = ctc_loss_fn(log_probs, tgts, out["out_lengths"], tgt_lens)
        if "mask_logits" in out:
            mask_labels = torch.nn.functional.interpolate(
                masks.unsqueeze(1), size=out["mask_logits"].shape[1], mode="nearest"
            ).squeeze(1)
            mask_loss = bce(out["mask_logits"], mask_labels)
            loss = loss + mask_weight * mask_loss
        total += loss.item()
    return total / max(len(loader), 1)


def main():
    ap = argparse.ArgumentParser(description="Self-supervised CTC with envelope mask auxiliary loss.")
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--val-manifest", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--frame-length-ms", type=int, default=25)
    ap.add_argument("--frame-step-ms", type=int, default=10)
    ap.add_argument("--smooth-ms", type=float, default=10.0)
    ap.add_argument("--threshold-ratio", type=float, default=0.3)
    ap.add_argument("--mask-weight", type=float, default=0.5)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map = build_vocab(ALPHABET)
    vocab = list(label_map.keys())
    model = MultiTaskCTCCountsModel(
        input_dim=args.n_mels,
        vocab_size=len(vocab),
        cnn_channels=[32, 64],
        rnn_hidden_size=128,
        rnn_layers=2,
        dropout=0.1,
        bidirectional=False,
        use_mask_head=True,
    ).to(device)

    train_ds = PseudoEnvelopeDataset(
        args.train_manifest,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        frame_length_ms=args.frame_length_ms,
        frame_step_ms=args.frame_step_ms,
        smooth_ms=args.smooth_ms,
        threshold_ratio=args.threshold_ratio,
    )
    val_ds = PseudoEnvelopeDataset(
        args.val_manifest,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        frame_length_ms=args.frame_length_ms,
        frame_step_ms=args.frame_step_ms,
        smooth_ms=args.smooth_ms,
        threshold_ratio=args.threshold_ratio,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, label_map)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate(b, label_map)
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    best_val = 1e9
    for epoch in range(1, args.epochs + 1):
        train_loss = train_loop(model, train_loader, optimizer, device, label_map, args.mask_weight)
        val_loss = eval_loop(model, val_loader, device, label_map, args.mask_weight)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        ckpt = {
            "model_state": model.state_dict(),
            "label_map": label_map,
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(ckpt, args.out_dir / f"checkpoint_epoch{epoch}.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, args.out_dir / "best.pt")


if __name__ == "__main__":
    main()

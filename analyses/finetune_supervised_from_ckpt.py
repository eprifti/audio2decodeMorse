"""
Supervised CTC fine-tuning starting from a self-supervised envelope checkpoint.

Usage:
    PYTHONPATH=src .venv/bin/python analyses/finetune_supervised_from_ckpt.py \
      --checkpoint outputs/self_supervised_envelope_large/best.pt \
      --train-manifest data/datasets/large_baseline/manifests/train.jsonl \
      --val-manifest   data/datasets/large_baseline/manifests/val.jsonl \
      --epochs 1 --batch-size 64 \
      --out-dir outputs/finetune_supervised_from_envelope
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio2morse.data.dataset import MorseAudioDataset, collate_batch
from audio2morse.models.multitask_ctc_counts import MultiTaskCTCCountsModel


def load_model(ckpt_path: Path, n_mels: int, device: torch.device) -> tuple[MultiTaskCTCCountsModel, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    label_map = ckpt["label_map"]
    model = MultiTaskCTCCountsModel(
        input_dim=n_mels,
        vocab_size=len(label_map),
        cnn_channels=[32, 64],
        rnn_hidden_size=128,
        rnn_layers=2,
        dropout=0.1,
        bidirectional=False,
        use_mask_head=("mask_head.weight" in ckpt["model_state"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, label_map


def run_epoch(model, loader, ctc, optimizer, device, train: bool) -> float:
    if train:
        model.train()
    else:
        model.eval()
    total = 0.0
    count = 0
    for feats, feat_lens, targets, target_lens, utt_ids, texts in tqdm(loader, desc="train" if train else "val", leave=False):
        feats = feats.to(device)
        feat_lens = feat_lens.to(device)
        targets = targets.to(device)
        target_lens = target_lens.to(device)
        with torch.set_grad_enabled(train):
            out = model(feats, feat_lens)
            log_probs = out["text_log_probs"].permute(1, 0, 2)  # (T,B,V)
            loss = ctc(log_probs, targets, out["out_lengths"], target_lens)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        total += loss.item()
        count += 1
    return total / max(1, count)


def main():
    ap = argparse.ArgumentParser(description="Supervised CTC fine-tuning from envelope checkpoint.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--val-manifest", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--frame-length-ms", type=int, default=25)
    ap.add_argument("--frame-step-ms", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    model, label_map = load_model(args.checkpoint, args.n_mels, device)
    blank_idx = label_map["<BLANK>"]
    ctc = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_ds = MorseAudioDataset(
        manifest_path=str(args.train_manifest),
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        frame_length_ms=args.frame_length_ms,
        frame_step_ms=args.frame_step_ms,
        label_map=label_map,
    )
    val_ds = MorseAudioDataset(
        manifest_path=str(args.val_manifest),
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        frame_length_ms=args.frame_length_ms,
        frame_step_ms=args.frame_step_ms,
        label_map=label_map,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, ctc, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, ctc, optimizer, device, train=False)
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

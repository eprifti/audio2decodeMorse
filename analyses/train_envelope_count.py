"""
Envelope-only character count regressor.

This trains a tiny 1D CNN on smoothed amplitude envelopes to predict
transcription length (character count). It avoids spectrograms and
should be a stronger baseline than hand-tuned thresholds.

Example:
    PYTHONPATH=src .venv/bin/python analyses/train_envelope_count.py \
      --train data/datasets/simple_baseline/manifests/train.jsonl \
      --val data/datasets/simple_baseline/manifests/val.jsonl \
      --epochs 5 --batch-size 64 --envelope-hz 400 \
      --out-dir outputs/envelope_count_baseline
"""
import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float32) / float(win)
    pad = win // 2
    x_pad = np.pad(x, (pad, win - 1 - pad), mode="reflect")
    return np.convolve(x_pad, kernel, mode="valid")


class EnvelopeDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        envelope_hz: int = 400,
        smooth_ms: float = 5.0,
        max_duration_s: float = 14.0,
        limit: int | None = None,
        log_target: bool = False,
    ):
        self.items = []
        self.envelope_hz = envelope_hz
        self.smooth_ms = smooth_ms
        self.max_duration_s = max_duration_s
        self.log_target = log_target
        with manifest.open() as fp:
            for i, line in enumerate(fp):
                if limit is not None and i >= limit:
                    break
                row = json.loads(line)
                self.items.append(
                    {
                        "audio": Path(row["audio_filepath"]),
                        "count": len(row.get("text", "")),
                    }
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        wav, sr = sf.read(item["audio"], always_2d=True)
        mono = wav.mean(axis=1)
        if self.max_duration_s:
            mono = mono[: int(self.max_duration_s * sr)]
        env = np.abs(mono)
        win = int(sr * self.smooth_ms / 1000.0)
        env = moving_average(env, win)
        stride = max(1, int(sr / self.envelope_hz))
        env = env[::stride].astype(np.float32)
        env = env / (np.max(np.abs(env)) + 1e-6)
        target = float(item["count"])
        if self.log_target:
            target = math.log1p(target)
        return torch.from_numpy(env), torch.tensor(target, dtype=torch.float32)


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    envs, counts = zip(*batch)
    lengths = [e.size(0) for e in envs]
    max_len = max(lengths)
    padded = []
    for e in envs:
        if e.size(0) < max_len:
            pad = torch.zeros(max_len - e.size(0), dtype=e.dtype)
            e = torch.cat([e, pad], dim=0)
        padded.append(e)
    env_tensor = torch.stack(padded, dim=0)  # (B, T)
    count_tensor = torch.stack(counts, dim=0)
    length_tensor = torch.tensor(lengths, dtype=torch.long)
    return env_tensor, length_tensor, count_tensor


class EnvelopeCountNet(nn.Module):
    def __init__(self, channels: List[int] = [64, 64, 64], kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch = 1
        pad = kernel_size // 2
        for ch in channels:
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=kernel_size, padding=pad))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        x = x.unsqueeze(1)  # (B,1,T)
        feats = self.conv(x)  # (B,C,T)
        max_len = feats.size(2)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.float().unsqueeze(1)  # (B,1,T)
        summed = (feats * mask).sum(dim=2)
        lengths_clamped = lengths.clamp(min=1).unsqueeze(1).float()
        pooled = summed / lengths_clamped  # (B,C)
        out = self.head(pooled).squeeze(1)
        return out


@torch.no_grad()
def evaluate(model, loader, device, log_target: bool = False) -> Tuple[float, float]:
    model.eval()
    abs_errs = []
    sq_errs = []
    for envs, lengths, counts in loader:
        envs = envs.to(device)
        lengths = lengths.to(device)
        counts = counts.to(device)
        preds = model(envs, lengths)
        if log_target:
            preds = preds.exp().sub(1.0)
            counts = counts.exp().sub(1.0)
        err = preds - counts
        abs_errs.append(err.abs())
        sq_errs.append(err.pow(2))
    abs_all = torch.cat(abs_errs) if abs_errs else torch.tensor([], device=device)
    sq_all = torch.cat(sq_errs) if sq_errs else torch.tensor([], device=device)
    mae = abs_all.mean().item() if abs_all.numel() else float("nan")
    rmse = math.sqrt(sq_all.mean().item()) if sq_all.numel() else float("nan")
    return mae, rmse


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = EnvelopeDataset(
        manifest=args.train,
        envelope_hz=args.envelope_hz,
        smooth_ms=args.smooth_ms,
        max_duration_s=args.max_duration,
        limit=args.train_limit,
        log_target=args.log_target,
    )
    val_ds = EnvelopeDataset(
        manifest=args.val,
        envelope_hz=args.envelope_hz,
        smooth_ms=args.smooth_ms,
        max_duration_s=args.max_duration,
        limit=args.val_limit,
        log_target=args.log_target,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch
    )

    model = EnvelopeCountNet()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_mae = float("inf")
    best_path = out_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for envs, lengths, counts in pbar:
            envs = envs.to(device)
            lengths = lengths.to(device)
            counts = counts.to(device)
            preds = model(envs, lengths)
            if args.log_target:
                loss = nn.functional.l1_loss(preds, counts)
            else:
                loss = nn.functional.l1_loss(preds, counts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
        val_mae, val_rmse = evaluate(model, val_loader, device, log_target=args.log_target)
        print(f"Epoch {epoch}: val MAE={val_mae:.4f} RMSE={val_rmse:.4f}")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                {"model_state": model.state_dict(), "config": vars(args)},
                best_path,
            )
            print(f"Saved best checkpoint to {best_path}")
    # Final metrics on val using best checkpoint
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best checkpoint from {best_path} for final eval")
    val_mae, val_rmse = evaluate(model, val_loader, device, log_target=args.log_target)
    with (out_dir / "metrics.txt").open("w") as fp:
        fp.write(f"val_mae={val_mae:.4f}\nval_rmse={val_rmse:.4f}\n")
    print(f"Done. val MAE={val_mae:.4f} RMSE={val_rmse:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train envelope-only count regressor.")
    parser.add_argument("--train", type=Path, required=True, help="Train manifest JSONL")
    parser.add_argument("--val", type=Path, required=True, help="Val manifest JSONL")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/envelope_count_baseline"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--envelope-hz", type=int, default=300, help="Downsampled envelope rate (Hz)")
    parser.add_argument("--smooth-ms", type=float, default=8.0, help="Smoothing window for abs envelope (ms)")
    parser.add_argument("--max-duration", type=float, default=14.0, help="Truncate audio to this many seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None, help="Optional cap on train items")
    parser.add_argument("--val-limit", type=int, default=None, help="Optional cap on val items")
    parser.add_argument("--log-target", action="store_true", help="Train on log1p(count) and de-log at eval time")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

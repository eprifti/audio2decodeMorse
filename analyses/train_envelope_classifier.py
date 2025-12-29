"""
Framewise envelope classifier (tone vs gap) with character count decoding.

Uses known text + WPM from the manifest to build binary frame labels aligned to
the smoothed/downsampled amplitude envelope. A small 1D CNN is trained with
BCE loss, and validation reports MAE/RMSE on character counts derived from the
predicted gaps (Morse timing rules).

Example:
    PYTHONPATH=src .venv/bin/python analyses/train_envelope_classifier.py \
      --train data/datasets/simple_baseline/manifests/train.jsonl \
      --val   data/datasets/simple_baseline/manifests/val.jsonl \
      --epochs 4 --batch-size 64 --envelope-hz 400 \
      --out-dir outputs/envelope_classifier
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

from audio2morse.data.morse_map import MORSE_CODE


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


def text_to_morse(message: str) -> List[str]:
    patterns: List[str] = []
    for ch in message.upper():
        if ch == " ":
            patterns.append(" ")
            continue
        if ch in MORSE_CODE:
            patterns.append(MORSE_CODE[ch])
        else:
            raise ValueError(f"Unsupported character: {ch}")
    return patterns


def build_label_sequence(message: str, wpm: float, envelope_hz: int) -> np.ndarray:
    dot_sec = 1.2 / max(wpm, 1e-3)
    unit_frames = max(1, int(round(dot_sec * envelope_hz)))
    patterns = text_to_morse(message)
    frames: List[int] = []

    def append_frames(val: int, units: int):
        frames.extend([val] * max(1, units * unit_frames))

    for i, pat in enumerate(patterns):
        if pat == " ":
            append_frames(0, 7)
            continue
        for j, sym in enumerate(pat):
            append_frames(1, 1 if sym == "." else 3)
            if j < len(pat) - 1:
                append_frames(0, 1)  # intra-character gap
        if i < len(patterns) - 1 and patterns[i + 1] != " ":
            append_frames(0, 3)  # inter-character gap
        # word gap handled by next iteration if space
    return np.array(frames, dtype=np.float32)


class EnvelopeFrameDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        envelope_hz: int = 400,
        smooth_ms: float = 5.0,
        max_duration_s: float = 14.0,
        limit: int | None = None,
    ):
        self.items = []
        self.envelope_hz = envelope_hz
        self.smooth_ms = smooth_ms
        self.max_duration_s = max_duration_s
        with manifest.open() as fp:
            for i, line in enumerate(fp):
                if limit is not None and i >= limit:
                    break
                row = json.loads(line)
                if "wpm" not in row:
                    continue
                self.items.append(
                    {
                        "audio": Path(row["audio_filepath"]),
                        "text": row.get("text", ""),
                        "wpm": float(row["wpm"]),
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

        labels = build_label_sequence(item["text"], item["wpm"], self.envelope_hz)
        # Align lengths: truncate or pad labels to envelope length.
        if labels.shape[0] > env.shape[0]:
            labels = labels[: env.shape[0]]
        elif labels.shape[0] < env.shape[0]:
            pad = np.zeros(env.shape[0] - labels.shape[0], dtype=np.float32)
            labels = np.concatenate([labels, pad])
        return torch.from_numpy(env), torch.tensor(labels, dtype=torch.float32), torch.tensor(item["count"], dtype=torch.float32), torch.tensor(len(env), dtype=torch.long), torch.tensor(item["wpm"], dtype=torch.float32)


def collate_batch(batch):
    envs, labels, counts, lengths, wpms = zip(*batch)
    max_len = max(l.item() for l in lengths)
    env_pad, lab_pad = [], []
    for e, l in zip(envs, labels):
        if e.numel() < max_len:
            pad = torch.zeros(max_len - e.numel(), dtype=e.dtype)
            e = torch.cat([e, pad], dim=0)
            l = torch.cat([l, pad], dim=0)
        env_pad.append(e)
        lab_pad.append(l)
    return (
        torch.stack(env_pad, dim=0),  # (B,T)
        torch.stack(lab_pad, dim=0),  # (B,T)
        torch.stack(counts, dim=0),
        torch.tensor([len(x) for x in envs], dtype=torch.long),
        torch.stack(wpms, dim=0),
    )


class EnvelopeSegNet(nn.Module):
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
        self.head = nn.Conv1d(channels[-1], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T)
        x = x.unsqueeze(1)  # (B,1,T)
        feats = self.conv(x)  # (B,C,T)
        logits = self.head(feats).squeeze(1)  # (B,T)
        return logits


def count_from_mask(mask: torch.Tensor, lengths: torch.Tensor, wpms: torch.Tensor, envelope_hz: int) -> torch.Tensor:
    # mask: (B,T) binary
    counts = []
    for b in range(mask.size(0)):
        m = mask[b, : lengths[b]].cpu().numpy().astype(np.int32)
        wpm = float(wpms[b].item()) if wpms is not None else 20.0
        unit_frames = max(1, int(round((1.2 / max(wpm, 1e-3)) * envelope_hz)))
        runs = []
        current = m[0] if len(m) > 0 else 0
        length = 1
        for v in m[1:]:
            if v == current:
                length += 1
            else:
                runs.append((current, length))
                current = v
                length = 1
        if len(m) > 0:
            runs.append((current, length))
        count = 0
        in_char = False
        for on, l in runs:
            dur_units = l / float(unit_frames)
            if on == 1:
                if not in_char:
                    count += 1
                    in_char = True
            else:
                if dur_units >= 2.5:  # gap >= ~3 units -> new char
                    in_char = False
        counts.append(count)
    return torch.tensor(counts, dtype=torch.float32, device=mask.device)


@torch.no_grad()
def evaluate(model, loader, device, envelope_hz: int):
    model.eval()
    abs_errs = []
    sq_errs = []
    bce = nn.BCEWithLogitsLoss(reduction="none")
    total_bce = 0.0
    total_frames = 0
    for envs, labels, counts, lengths, wpms in loader:
        envs = envs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        counts = counts.to(device)
        wpms = wpms.to(device)
        logits = model(envs)
        loss = bce(logits, labels)
        mask = torch.arange(logits.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
        loss = (loss * mask).sum()
        total_bce += loss.item()
        total_frames += mask.sum().item()
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).float()
        pred_counts = count_from_mask(pred_mask, lengths, wpms, envelope_hz)
        err = pred_counts - counts
        abs_errs.append(err.abs())
        sq_errs.append(err.pow(2))
    abs_all = torch.cat(abs_errs) if abs_errs else torch.tensor([], device=device)
    sq_all = torch.cat(sq_errs) if sq_errs else torch.tensor([], device=device)
    mae = abs_all.mean().item() if abs_all.numel() else float("nan")
    rmse = math.sqrt(sq_all.mean().item()) if sq_all.numel() else float("nan")
    avg_bce = total_bce / max(total_frames, 1)
    return mae, rmse, avg_bce


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = EnvelopeFrameDataset(
        manifest=args.train,
        envelope_hz=args.envelope_hz,
        smooth_ms=args.smooth_ms,
        max_duration_s=args.max_duration,
        limit=args.train_limit,
    )
    val_ds = EnvelopeFrameDataset(
        manifest=args.val,
        envelope_hz=args.envelope_hz,
        smooth_ms=args.smooth_ms,
        max_duration_s=args.max_duration,
        limit=args.val_limit,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch
    )

    model = EnvelopeSegNet()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    best_mae = float("inf")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for envs, labels, lengths, counts, wpms in ((e, l, t, c, w) for e, l, c, t, w in train_loader):
            envs = envs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            logits = model(envs)
            mask = torch.arange(logits.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
            loss = bce(logits, labels)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
        val_mae, val_rmse, val_bce = evaluate(model, val_loader, device, envelope_hz=args.envelope_hz)
        print(f"Epoch {epoch}: val MAE={val_mae:.4f} RMSE={val_rmse:.4f} BCE/frame={val_bce:.6f}")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                {"model_state": model.state_dict(), "config": vars(args)},
                best_path,
            )
            print(f"Saved best checkpoint to {best_path}")
    # Final eval
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best checkpoint from {best_path} for final eval")
    val_mae, val_rmse, val_bce = evaluate(model, val_loader, device, envelope_hz=args.envelope_hz)
    with (out_dir / "metrics.txt").open("w") as fp:
        fp.write(f"val_mae={val_mae:.4f}\nval_rmse={val_rmse:.4f}\nval_bce_per_frame={val_bce:.6f}\n")
    print(f"Done. val MAE={val_mae:.4f} RMSE={val_rmse:.4f} BCE/frame={val_bce:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train envelope frame classifier (tone/gap).")
    parser.add_argument("--train", type=Path, required=True, help="Train manifest JSONL")
    parser.add_argument("--val", type=Path, required=True, help="Val manifest JSONL")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/envelope_classifier"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--envelope-hz", type=int, default=400)
    parser.add_argument("--smooth-ms", type=float, default=6.0)
    parser.add_argument("--max-duration", type=float, default=14.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

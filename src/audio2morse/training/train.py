import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audio2morse.data.dataset import MorseAudioDataset, collate_batch
from audio2morse.data.vocab import build_vocab
from audio2morse.models.ctc_model import CTCMorseModel
from audio2morse.models.multitask_ctc import MultiTaskCTCMorseModel
from audio2morse.models.multitask_ctc_counts import MultiTaskCTCCountsModel


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(path: str) -> Dict:
    with open(path, "r") as fp:
        return yaml.safe_load(fp)


def downsample_lengths(lengths: torch.Tensor, pools: int) -> torch.Tensor:
    return torch.div(lengths, pools, rounding_mode="floor")


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed)


def train_epoch(model, loader, criterion, optimizer, device, downsample_factor, label_map=None, aux_weights=None):
    model.train()
    total_loss = 0.0
    is_multitask = isinstance(model, (MultiTaskCTCMorseModel, MultiTaskCTCCountsModel))
    is_counts = isinstance(model, MultiTaskCTCCountsModel)
    aux_w = aux_weights or {}
    blank_idx = label_map["<BLANK>"] if label_map else None
    for feats, feat_lens, targets, target_lens, _, texts in tqdm(loader, desc="train", leave=False):
        feats = feats.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        if is_multitask:
            out = model(feats, feat_lens.to(device))
            log_probs = out["text_log_probs"]  # (B, T', V)
        else:
            out = model(feats)
            log_probs = out  # (T', B, V) from CTCMorseModel
        if log_probs.dim() == 3 and log_probs.shape[0] == feats.size(0):
            log_probs = log_probs.permute(1, 0, 2)  # ensure (T, B, V) for CTC
        out_lens = downsample_lengths(feat_lens.to(device), downsample_factor)
        flat_targets = torch.cat([targets[i, :target_lens[i]] for i in range(targets.size(0))]).to(device)
        loss = criterion(log_probs, flat_targets, out_lens, target_lens.to(device))
        if is_counts and label_map:
            counts = torch.tensor([len(t) for t in texts], dtype=torch.float32, device=device)
            count_loss = F.mse_loss(out["count_pred"], counts)
            vocab_no_blank = len(label_map) - 1
            hist_targets = torch.zeros((len(texts), vocab_no_blank), device=device)
            for i, t in enumerate(texts):
                for ch in t:
                    if ch not in label_map or label_map[ch] == blank_idx:
                        continue
                    hist_targets[i, label_map[ch]] += 1.0
            hist_loss = F.mse_loss(out["hist_logits"], hist_targets)
            loss = loss + aux_w.get("count", 0.1) * count_loss + aux_w.get("hist", 0.1) * hist_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device, downsample_factor, label_map=None, aux_weights=None):
    model.eval()
    total_loss = 0.0
    is_multitask = isinstance(model, (MultiTaskCTCMorseModel, MultiTaskCTCCountsModel))
    is_counts = isinstance(model, MultiTaskCTCCountsModel)
    aux_w = aux_weights or {}
    blank_idx = label_map["<BLANK>"] if label_map else None
    for feats, feat_lens, targets, target_lens, _, texts in loader:
        feats = feats.to(device)
        targets = targets.to(device)
        if is_multitask:
            out = model(feats, feat_lens.to(device))
            log_probs = out["text_log_probs"]  # (B, T', V)
        else:
            out = model(feats)
            log_probs = out  # (T', B, V) from CTCMorseModel
        if log_probs.dim() == 3 and log_probs.shape[0] == feats.size(0):
            log_probs = log_probs.permute(1, 0, 2)  # ensure (T, B, V) for CTC
        out_lens = downsample_lengths(feat_lens.to(device), downsample_factor)
        flat_targets = torch.cat([targets[i, :target_lens[i]] for i in range(targets.size(0))]).to(device)
        loss = criterion(log_probs, flat_targets, out_lens, target_lens.to(device))
        if is_counts and label_map:
            counts = torch.tensor([len(t) for t in texts], dtype=torch.float32, device=device)
            count_loss = F.mse_loss(out["count_pred"], counts)
            vocab_no_blank = len(label_map) - 1
            hist_targets = torch.zeros((len(texts), vocab_no_blank), device=device)
            for i, t in enumerate(texts):
                for ch in t:
                    if ch not in label_map or label_map[ch] == blank_idx:
                        continue
                    hist_targets[i, label_map[ch]] += 1.0
            hist_loss = F.mse_loss(out["hist_logits"], hist_targets)
            loss = loss + aux_w.get("count", 0.1) * count_loss + aux_w.get("hist", 0.1) * hist_loss
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser(description="Train Morse audio CTC model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--resume", type=str, help="Optional path to checkpoint to resume from.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for this run (creates outputs/<run-name>). If omitted, a timestamped name is used.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["training"].get("seed", 42)
    rng = set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    alphabet = cfg["labels"]["alphabet"]
    label_map = build_vocab(alphabet)
    blank_idx = label_map["<BLANK>"]

    train_ds = MorseAudioDataset(
        manifest_path=cfg["data"]["train_manifest"],
        sample_rate=cfg["data"]["sample_rate"],
        n_mels=cfg["data"]["n_mels"],
        frame_length_ms=cfg["data"]["frame_length_ms"],
        frame_step_ms=cfg["data"]["frame_step_ms"],
        label_map=label_map,
        max_duration_s=cfg["data"]["max_duration_s"],
        augment=cfg["data"].get("augment"),
    )
    val_ds = MorseAudioDataset(
        manifest_path=cfg["data"]["val_manifest"],
        sample_rate=cfg["data"]["sample_rate"],
        n_mels=cfg["data"]["n_mels"],
        frame_length_ms=cfg["data"]["frame_length_ms"],
        frame_step_ms=cfg["data"]["frame_step_ms"],
        label_map=label_map,
        max_duration_s=cfg["data"]["max_duration_s"],
        augment=None,  # no augmentation on validation
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
        generator=rng,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    model_type = cfg["model"].get("type", "ctc")
    if model_type == "multitask":
        model = MultiTaskCTCMorseModel(
            input_dim=cfg["data"]["n_mels"],
            vocab_size=len(label_map),
            cnn_channels=cfg["model"]["cnn_channels"],
            rnn_hidden_size=cfg["model"]["rnn_hidden_size"],
            rnn_layers=cfg["model"]["rnn_layers"],
            dropout=cfg["model"]["dropout"],
            bidirectional=cfg["model"].get("bidirectional", False),
        ).to(device)
        downsample_factor = model.downsample
    elif model_type == "multitask_counts":
        model = MultiTaskCTCCountsModel(
            input_dim=cfg["data"]["n_mels"],
            vocab_size=len(label_map),
            cnn_channels=cfg["model"]["cnn_channels"],
            rnn_hidden_size=cfg["model"]["rnn_hidden_size"],
            rnn_layers=cfg["model"]["rnn_layers"],
            dropout=cfg["model"]["dropout"],
            bidirectional=cfg["model"].get("bidirectional", False),
        ).to(device)
        downsample_factor = model.downsample
    else:
        model = CTCMorseModel(
            input_dim=cfg["data"]["n_mels"],
            vocab_size=len(label_map),
            cnn_channels=cfg["model"]["cnn_channels"],
            rnn_hidden_size=cfg["model"]["rnn_hidden_size"],
            rnn_layers=cfg["model"]["rnn_layers"],
            dropout=cfg["model"]["dropout"],
            bidirectional=cfg["model"].get("bidirectional", False),
        ).to(device)
        downsample_factor = 2 ** len(cfg["model"]["cnn_channels"])  # fixed 2x2 pooling

    criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    sched_cfg = cfg["training"].get("lr_scheduler")
    scheduler = None
    if sched_cfg and sched_cfg.get("type") == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 2),
            min_lr=sched_cfg.get("min_lr", 1e-5),
        )

    run_name = args.run_name or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    checkpoint_root = Path(cfg["training"]["checkpoint_dir"]) / run_name
    os.makedirs(checkpoint_root, exist_ok=True)
    with (checkpoint_root / "config_used.yaml").open("w") as fp:
        yaml.safe_dump(cfg, fp)
    best_val = float("inf")
    epoch_history = []
    start_epoch = 1
    patience = None  # early stopping disabled
    patience_counter = 0

    if args.resume:
        ckpt_path = Path(args.resume)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Resumed model weights from {ckpt_path}")
        # Try to infer starting epoch from filename like epoch_5.pt
        name = ckpt_path.stem
        if name.startswith("epoch_"):
            try:
                start_epoch = int(name.split("_")[1]) + 1
            except Exception:
                start_epoch = 1
        else:
            start_epoch = 1
        print(f"Starting from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            downsample_factor,
            label_map=label_map,
            aux_weights=cfg["training"].get("aux_weights"),
        )
        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
            downsample_factor,
            label_map=label_map,
            aux_weights=cfg["training"].get("aux_weights"),
        )
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if scheduler:
            scheduler.step(val_loss)
        epoch_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        ckpt_path = checkpoint_root / f"epoch_{epoch}.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": cfg,
                "alphabet": alphabet,
            },
            ckpt_path,
            )
        if val_loss < best_val:
            best_val = val_loss
            best_path = checkpoint_root / "best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "alphabet": alphabet,
                },
                best_path,
            )
            print(f"Saved new best checkpoint to {best_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Write/append loss history in real time
        metrics_path = checkpoint_root / "loss_history.csv"
        if not metrics_path.exists():
            with metrics_path.open("w") as fp:
                fp.write("epoch,train_loss,val_loss\n")
        with metrics_path.open("a") as fp:
            fp.write(f"{epoch},{train_loss:.6f},{val_loss:.6f}\n")

    # Save loss history to CSV
    metrics_path = checkpoint_root / "loss_history.csv"
    if epoch_history:
        with metrics_path.open("w") as fp:
            fp.write("epoch,train_loss,val_loss\n")
            for row in epoch_history:
                fp.write(f"{row['epoch']},{row['train_loss']:.6f},{row['val_loss']:.6f}\n")
    # Plot loss curves
    epochs = [row["epoch"] for row in epoch_history]
    train_losses = [row["train_loss"] for row in epoch_history]
    val_losses = [row["val_loss"] for row in epoch_history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CTC Loss")
    plt.legend()
    plt.tight_layout()
    plot_path = checkpoint_root / "loss_curve.png"
    plt.savefig(plot_path)
    print(f"Saved loss history to {metrics_path} and plot to {plot_path}")


if __name__ == "__main__":
    main()

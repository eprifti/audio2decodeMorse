"""
Evaluate a self-supervised envelope-pretrained model on a manifest using greedy CTC decoding.

Usage example:
    PYTHONPATH=src .venv/bin/python analyses/eval_self_supervised_envelope.py \
      --checkpoint outputs/self_supervised_envelope_large/best.pt \
      --manifest data/datasets/large_baseline/manifests/val.jsonl \
      --device cuda
"""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio2morse.data.dataset import MorseAudioDataset, collate_batch
from audio2morse.models.multitask_ctc_counts import MultiTaskCTCCountsModel


def idx_to_char(label_map: Dict[str, int]) -> List[str]:
    return [ch for ch, idx in sorted(label_map.items(), key=lambda kv: kv[1])]


def ctc_beam_search(log_probs: torch.Tensor, blank_idx: int, beam_size: int = 5) -> List[int]:
    # log_probs: (T, V)
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(tuple()): (0.0, -float("inf"))}  # prefix -> (p_blank, p_nonblank)
    for t in range(log_probs.size(0)):
        frame = log_probs[t]
        next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}
        for prefix, (p_b, p_nb) in beams.items():
            # stay blank
            p_blank = torch.logaddexp(torch.tensor(p_b + frame[blank_idx].item()), torch.tensor(p_nb + frame[blank_idx].item())).item()
            best_b, best_nb = next_beams.get(prefix, (-float("inf"), -float("inf")))
            next_beams[prefix] = (max(best_b, p_blank), best_nb)
            last = prefix[-1] if prefix else None
            for idx in range(log_probs.size(1)):
                if idx == blank_idx:
                    continue
                p_char = frame[idx].item()
                if idx == last:
                    new_p_nb = torch.logaddexp(torch.tensor(p_nb + p_char), torch.tensor(next_beams[prefix][1])).item()
                    next_beams[prefix] = (next_beams[prefix][0], new_p_nb)
                else:
                    new_pref = prefix + (idx,)
                    nb_old = next_beams.get(new_pref, (-float("inf"), -float("inf")))[1]
                    new_p_nb = torch.logaddexp(torch.tensor(p_b + p_char), torch.tensor(p_nb + p_char)).item()
                    new_p_nb = torch.logaddexp(torch.tensor(new_p_nb), torch.tensor(nb_old)).item()
                    next_beams[new_pref] = (next_beams.get(new_pref, (-float("inf"), -float("inf")))[0], new_p_nb)
        # prune
        beams = dict(sorted(next_beams.items(), key=lambda kv: torch.logaddexp(torch.tensor(kv[1][0]), torch.tensor(kv[1][1])).item(), reverse=True)[:beam_size])
    best_pref, (p_b, p_nb) = max(beams.items(), key=lambda kv: torch.logaddexp(torch.tensor(kv[1][0]), torch.tensor(kv[1][1])).item())
    return list(best_pref)


def greedy_ctc_decode(log_probs: torch.Tensor, blank_idx: int) -> List[int]:
    # log_probs: (T, V)
    indices = log_probs.argmax(dim=-1).tolist()
    out = []
    prev = blank_idx
    for idx in indices:
        if idx != prev and idx != blank_idx:
            out.append(idx)
        prev = idx
    return out


def cer(ref: str, hyp: str) -> float:
    # Simple Levenshtein distance over characters.
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )
    return dp[m][n] / max(1, m)


def load_model(checkpoint: Path, n_mels: int, device: torch.device) -> Tuple[MultiTaskCTCCountsModel, Dict[str, int]]:
    ckpt = torch.load(checkpoint, map_location=device)
    label_map: Dict[str, int] = ckpt["label_map"]
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
    model.eval()
    return model, label_map


def main():
    ap = argparse.ArgumentParser(description="Evaluate self-supervised envelope CTC model with greedy decode.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--frame-length-ms", type=int, default=25)
    ap.add_argument("--frame-step-ms", type=int, default=10)
    ap.add_argument("--beam-size", type=int, default=1, help="Beam >1 enables prefix beam search instead of greedy.")
    ap.add_argument("--max-samples", type=int, default=None, help="Optional limit for quick eval.")
    ap.add_argument("--print-samples", type=int, default=5, help="Show first N decoded samples.")
    ap.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None)
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    model, label_map = load_model(args.checkpoint, args.n_mels, device)
    blank_idx = label_map["<BLANK>"]
    itoc = idx_to_char(label_map)

    ds = MorseAudioDataset(
        manifest_path=str(args.manifest),
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        frame_length_ms=args.frame_length_ms,
        frame_step_ms=args.frame_step_ms,
        label_map=label_map,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    total_cer = 0.0
    total_chars = 0
    exact = 0
    count = 0
    sample_rows = []
    allowed_chars = set(ch for ch in label_map.keys() if ch != "<BLANK>")
    for feats, feat_lens, targets, target_lens, utt_ids, texts in tqdm(loader, desc="eval", leave=False):
        feats = feats.to(device)
        feat_lens = feat_lens.to(device)
        with torch.no_grad():
            out = model(feats, feat_lens)
            log_probs = out["text_log_probs"]  # (B, T', V)
        for b in range(log_probs.size(0)):
            if args.max_samples is not None and count >= args.max_samples:
                break
            lp = log_probs[b, : int(out["out_lengths"][b])].cpu()
            if args.beam_size > 1:
                hyp_idx = ctc_beam_search(lp, blank_idx, beam_size=args.beam_size)
            else:
                hyp_idx = greedy_ctc_decode(lp, blank_idx)
            hyp = "".join(itoc[i] for i in hyp_idx)
            ref = "".join(ch for ch in texts[b].upper() if ch in allowed_chars)
            total_cer += cer(ref, hyp) * len(ref)
            total_chars += len(ref)
            exact += int(ref == hyp)
            count += 1
            if len(sample_rows) < args.print_samples:
                sample_rows.append((utt_ids[b], ref, hyp))
        if args.max_samples is not None and count >= args.max_samples:
            break
    avg_cer = total_cer / max(1, total_chars)
    exact_acc = exact / max(1, count)
    print(f"Samples: {count}, CER: {avg_cer:.4f}, Exact match: {exact_acc:.4f}")
    if sample_rows:
        print("\nSamples (ref | hyp):")
        for uid, ref, hyp in sample_rows:
            print(f"{uid}: {ref} | {hyp}")


if __name__ == "__main__":
    main()

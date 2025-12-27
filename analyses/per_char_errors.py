"""
Compute per-character error stats from a combined manifest with ground-truth
`text` and `inference_text` columns.

Example:
    PYTHONPATH=src python3 analyses/per_char_errors.py \
      --input analyses/combined_with_preds.csv \
      --config config/default.yaml \
      --out analyses/per_char_errors.csv
"""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml


def load_alphabet(cfg_path: Path) -> str:
    with cfg_path.open("r") as fp:
        cfg = yaml.safe_load(fp)
    return cfg["labels"]["alphabet"]


def levenshtein_ops(ref: str, hyp: str) -> List[Tuple[str, str, str]]:
    """
    Return an aligned operation list with tuples: (op, ref_char, hyp_char)
    op in {"match", "sub", "del", "ins"}.
    """
    n, m = len(ref), len(hyp)
    dp = [[(0, None) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, "del")
    for j in range(1, m + 1):
        dp[0][j] = (j, "ins")

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_sub = dp[i - 1][j - 1][0] + (ref[i - 1] != hyp[j - 1])
            cost_del = dp[i - 1][j][0] + 1
            cost_ins = dp[i][j - 1][0] + 1
            best_cost = min(cost_sub, cost_del, cost_ins)
            if best_cost == cost_sub:
                op = "match" if ref[i - 1] == hyp[j - 1] else "sub"
            elif best_cost == cost_del:
                op = "del"
            else:
                op = "ins"
            dp[i][j] = (best_cost, op)

    # Traceback
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        _, op = dp[i][j]
        if op == "match" or op == "sub":
            ops.append((op, ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif op == "del":
            ops.append((op, ref[i - 1], ""))
            i -= 1
        elif op == "ins":
            ops.append((op, "", hyp[j - 1]))
            j -= 1
        else:
            raise ValueError(f"Unexpected op: {op}")
    ops.reverse()
    return ops


def aggregate_per_char(df: pd.DataFrame, alphabet: str) -> pd.DataFrame:
    stats: Dict[str, Dict[str, int]] = {
        c: {"count": 0, "correct": 0, "subs": 0, "dels": 0} for c in alphabet
    }
    insertions = 0

    for _, row in df.iterrows():
        ref = row["text"]
        hyp = row["inference_text"]
        ops = levenshtein_ops(ref, hyp)
        for op, r, h in ops:
            if op == "match":
                if r in stats:
                    stats[r]["count"] += 1
                    stats[r]["correct"] += 1
            elif op == "sub":
                if r in stats:
                    stats[r]["count"] += 1
                    stats[r]["subs"] += 1
            elif op == "del":
                if r in stats:
                    stats[r]["count"] += 1
                    stats[r]["dels"] += 1
            elif op == "ins":
                insertions += 1

    rows = []
    for ch, s in stats.items():
        total = s["count"]
        err = total - s["correct"]
        acc = s["correct"] / total if total > 0 else 0.0
        rows.append(
            {
                "char": ch,
                "count": total,
                "correct": s["correct"],
                "subs": s["subs"],
                "dels": s["dels"],
                "error_rate": err / total if total > 0 else 0.0,
                "accuracy": acc,
            }
        )
    df_stats = pd.DataFrame(rows).sort_values("error_rate", ascending=False)
    df_stats.attrs["insertions_total"] = insertions
    return df_stats


def main():
    parser = argparse.ArgumentParser(description="Per-character error analysis.")
    parser.add_argument("--input", default="analyses/combined_with_preds.csv", help="CSV with text and inference_text.")
    parser.add_argument("--config", default="config/default.yaml", help="Config containing labels.alphabet.")
    parser.add_argument("--out", default="analyses/per_char_errors.csv", help="Output CSV path.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if not {"text", "inference_text"}.issubset(df.columns):
        raise ValueError("Input must contain 'text' and 'inference_text' columns.")
    alphabet = load_alphabet(Path(args.config))
    stats = aggregate_per_char(df, alphabet)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_path, index=False)
    print(f"Wrote per-character stats to {out_path}")
    print(f"Total insertions (not attributed to a specific ref char): {stats.attrs.get('insertions_total', 0)}")
    print("Top 5 most error-prone characters:")
    print(stats.head(5)[["char", "error_rate", "count"]])


if __name__ == "__main__":
    main()

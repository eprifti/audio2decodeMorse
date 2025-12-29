"""
Summarize character-count prediction accuracy for count-only models.

Usage:
    PYTHONPATH=src python3 analyses/summarize_counts.py \
      --input outputs/charcount_log_gpu0_regen/combined_with_preds.csv
"""
import argparse
import pandas as pd
import numpy as np


def summarize(df: pd.DataFrame):
    required = {"predicted_count", "true_count", "partition"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    parts = []
    for part, group in df.groupby("partition"):
        true = group["true_count"].to_numpy()
        pred = group["predicted_count"].to_numpy()
        mae = np.mean(np.abs(pred - true))
        mse = np.mean((pred - true) ** 2)
        rmse = np.sqrt(mse)
        parts.append(
            {
                "partition": part,
                "n": len(group),
                "mae": mae,
                "rmse": rmse,
                "mse": mse,
                "mean_true": float(np.mean(true)),
                "mean_pred": float(np.mean(pred)),
            }
        )
    return pd.DataFrame(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="combined_with_preds.csv from add_predictions")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    summary = summarize(df)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

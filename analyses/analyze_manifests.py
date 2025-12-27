"""
Load train/test/val manifests into a single DataFrame with a 'partition' column.

Example:
    PYTHONPATH=src .venv311/bin/python analyses/analyze_manifests.py
"""
from pathlib import Path

import pandas as pd


def load_manifest(path: Path, partition: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_json(path, lines=True)
    df["partition"] = partition
    return df


def main():
    base = Path("data/manifests")
    parts = {
        "train": base / "train.jsonl",
        "val": base / "val.jsonl",
        "test": base / "test.jsonl",
    }
    dfs = []
    for name, path in parts.items():
        dfs.append(load_manifest(path, name))
    df = pd.concat(dfs, ignore_index=True)
    out_path = Path("analyses/combined_manifests.parquet")
    out_csv = Path("analyses/combined_manifests.csv")
    df.to_parquet(out_path, index=False)
    df.to_csv(out_csv, index=False)
    print(f"Loaded {len(df)} rows. Saved to {out_path} and {out_csv}")


if __name__ == "__main__":
    main()

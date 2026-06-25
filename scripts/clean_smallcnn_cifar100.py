"""
Archive results CSVs, remove the under-converged small_cnn CIFAR-100 row,
and regenerate aggregated_summary.csv.

Run ONCE before launching configs/smallcnn_cifar100_rerun.json:
    python scripts/clean_smallcnn_cifar100.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "summary_table.csv"
METRICS_PATH = RESULTS_DIR / "metrics.csv"
AGGREGATED_PATH = RESULTS_DIR / "aggregated_summary.csv"
ARCHIVE_DIR = RESULTS_DIR / "archive_pre_smallcnn_rerun"

STALE_ID = "cifar100_small_cnn_full_training_linear_lr0p001_bs64_wd0p0001_seed42"

_CONFIG_COLS = [
    "dataset", "model_name", "training_strategy", "head_type",
    "learning_rate", "batch_size", "weight_decay", "epochs",
    "frozen_epochs", "patience",
]


def _archive(src: Path) -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dst = ARCHIVE_DIR / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        print(f"  archived → {dst.relative_to(PROJECT_ROOT)}")
    else:
        print(f"  archive already exists, skipping: {dst.relative_to(PROJECT_ROOT)}")


def _clean_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  {path.name}: not found, skipping")
        return None
    df = pd.read_csv(path)
    before = len(df)
    df = df[df["experiment_id"] != STALE_ID].reset_index(drop=True)
    dropped = before - len(df)
    df.to_csv(path, index=False)
    print(f"  {path.name}: removed {dropped} rows ({before} → {len(df)})")
    return df


def main() -> None:
    print("=== Step 1: archive ===")
    for src in (SUMMARY_PATH, METRICS_PATH):
        if src.exists():
            _archive(src)

    print("\n=== Step 2: clean ===")
    summary = _clean_csv(SUMMARY_PATH)
    _clean_csv(METRICS_PATH)

    if summary is None:
        print("No summary_table.csv — cannot reaggregate.")
        return

    print("\n=== Step 3: reaggregate ===")
    group_cols = [c for c in _CONFIG_COLS if c in summary.columns]
    agg = (
        summary.groupby(group_cols, dropna=False)
        .agg(
            top1_mean=("top1_accuracy", "mean"),
            top1_std=("top1_accuracy", "std"),
            top5_mean=("top5_accuracy", "mean"),
            top5_std=("top5_accuracy", "std"),
            run_count=("top1_accuracy", "count"),
        )
        .reset_index()
    )
    AGGREGATED_PATH.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(AGGREGATED_PATH, index=False)
    print(f"  aggregated_summary.csv: {len(agg)} groups written")


if __name__ == "__main__":
    main()

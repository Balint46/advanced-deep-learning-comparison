"""Plotting utilities for experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .experiment import AGGREGATED_PATH, METRICS_PATH, SUMMARY_PATH
from .utils import PLOTS_DIR


# --- Style constants matching the Madrid beamer theme ---
NAVY  = "#003366"
TEAL  = "#006699"
GOLD  = "#CC9900"
SLATE = "#4A6FA5"
CORAL = "#CC4A1D"
GREEN = "#5A7A3A"
GRID  = "#D0D8E4"

PALETTE = [NAVY, TEAL, GOLD, SLATE, CORAL, GREEN, "#7B4A8A"]

FONT_TITLE = 14
FONT_LABEL = 12
FONT_TICK  = 10
FONT_ANNOT = 9
DPI        = 200

STRATEGY_COLORS = {
    "full_training":        NAVY,
    "freeze_then_unfreeze": GOLD,
    "head_only":            TEAL,
}
HEAD_COLORS = {
    "linear":     NAVY,
    "one_hidden": TEAL,
    "two_hidden": GOLD,
}
STRATEGY_LABELS = {
    "full_training":        "Full training",
    "freeze_then_unfreeze": "Freeze → unfreeze",
    "head_only":            "Head only",
}
HEAD_LABELS = {
    "linear":     "Linear",
    "one_hidden": "One hidden layer",
    "two_hidden": "Two hidden layers",
}

# Old filenames whose on-disk PNGs will be removed when generate_all_plots runs.
_RETIRED_FILENAMES = [
    "top1_accuracy_by_model.png",
    "top5_accuracy_by_model.png",
    "cifar10_vs_cifar100_top1.png",
    "validation_loss_curves.png",
    "strategy_comparison.png",
    "head_comparison.png",
    "parameters_vs_top1_accuracy.png",
]


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _apply_style(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    subtitle: str = "",
) -> None:
    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full_title, fontsize=FONT_TITLE, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL, color=NAVY)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL, color=NAVY)
    ax.tick_params(axis="both", labelsize=FONT_TICK, colors=NAVY)
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="--", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _label_bars(ax: plt.Axes, fmt: str = "{:.1f}") -> None:
    for bar in ax.patches:
        height = bar.get_height()
        if height > 0 and not np.isnan(height):
            ax.annotate(
                fmt.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=FONT_ANNOT,
                color=NAVY,
                fontweight="bold",
            )


def _save(fig: plt.Figure, filename: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _load_results(
    summary_path: Path = SUMMARY_PATH,
    metrics_path: Path = METRICS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    return summary, metrics


def _load_aggregated(aggregated_path: Path = AGGREGATED_PATH) -> pd.DataFrame:
    return pd.read_csv(aggregated_path) if aggregated_path.exists() else pd.DataFrame()


def _first_by_model(df: pd.DataFrame, col: str = "model_name") -> pd.DataFrame:
    """After filtering to a unique (model, config) slice, drop any unexpected dupes."""
    return df.groupby(col, as_index=False).first()


# -----------------------------------------------------------------------
# Governing filter helpers
#
# Every cross-model plot must restrict to full_training AND linear.
# Single-seed plots use seed=42; the canonical shared HPs are lr=0.001,
# bs=64, wd=1e-4 (same across all core experiments).
# -----------------------------------------------------------------------

_CORE_HPS = dict(learning_rate=0.001, batch_size=64, weight_decay=0.0001)


def _filter_ft_linear(df: pd.DataFrame, dataset: str, seed: int = 42) -> pd.DataFrame:
    """full_training / linear / seed / core HPs — the controlled-comparison slice."""
    mask = (
        (df["dataset"] == dataset)
        & (df["training_strategy"] == "full_training")
        & (df["head_type"] == "linear")
        & (df["seed"] == seed)
        & (df["learning_rate"] == _CORE_HPS["learning_rate"])
        & (df["batch_size"] == _CORE_HPS["batch_size"])
        & (df["weight_decay"] == _CORE_HPS["weight_decay"])
    )
    return df[mask].copy()


# -----------------------------------------------------------------------
# Plot 1 — core_comparison_cifar10.png
# -----------------------------------------------------------------------

def plot_core_comparison_cifar10(
    aggregated: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """CIFAR-10 top-1: full_training/linear, 3-seed mean ± std-error bars."""
    if aggregated.empty:
        print("core_comparison_cifar10: no aggregated data — skipping")
        return

    data = aggregated[
        (aggregated["dataset"] == "cifar10")
        & (aggregated["training_strategy"] == "full_training")
        & (aggregated["head_type"] == "linear")
        & (aggregated["patience"] == 5)
        & (aggregated["run_count"] == 3)
    ].copy()

    if data.empty:
        print("core_comparison_cifar10: need 3-seed full_training/linear/patience=5 rows in "
              "aggregated_summary.csv — skipping")
        return

    data = data.sort_values("top1_mean", ascending=False).reset_index(drop=True)
    se = data["top1_std"] / np.sqrt(data["run_count"])

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [SLATE if "vit" in m else NAVY for m in data["model_name"]]
    ax.bar(
        data["model_name"],
        data["top1_mean"],
        color=colors,
        width=0.5,
        zorder=3,
        yerr=se,
        capsize=5,
        error_kw={"elinewidth": 1.5, "ecolor": CORAL, "capthick": 1.5},
    )
    _apply_style(
        ax,
        "CIFAR-10 Top-1 — full fine-tuning, mean ± std over 3 seeds",
        "",
        "Accuracy (%)",
    )
    ax.set_ylim(0, min(100, (data["top1_mean"] + se).max() * 1.18))
    _label_bars(ax, "{:.1f}%")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=NAVY,  label="CNN family"),
        plt.Rectangle((0, 0), 1, 1, color=SLATE, label="Transformer (ViT)"),
    ]
    ax.legend(handles=handles, fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "core_comparison_cifar10.png", output_dir)


# -----------------------------------------------------------------------
# Plot 2 — convergence_curves_cifar10.png
# -----------------------------------------------------------------------

def plot_convergence_curves_cifar10(
    metrics: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Val-loss curves: seed-42 full_training/linear per model (not best-of-config)."""
    if metrics.empty or "val_loss" not in metrics.columns:
        print("convergence_curves_cifar10: no metrics data — skipping")
        return

    data = metrics[
        (metrics["dataset"] == "cifar10")
        & (metrics["training_strategy"] == "full_training")
        & (metrics["head_type"] == "linear")
        & (metrics["seed"] == 42)
        & (metrics["learning_rate"] == _CORE_HPS["learning_rate"])
        & (metrics["batch_size"] == _CORE_HPS["batch_size"])
        & (metrics["weight_decay"] == _CORE_HPS["weight_decay"])
    ]

    if data.empty:
        print("convergence_curves_cifar10: no seed-42 full_training/linear CIFAR-10 "
              "metrics rows — skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model_name in enumerate(sorted(data["model_name"].unique())):
        group = data[data["model_name"] == model_name].sort_values("epoch")
        ax.plot(
            group["epoch"],
            group["val_loss"],
            marker="o",
            markersize=5,
            linewidth=2,
            color=PALETTE[i % len(PALETTE)],
            label=model_name,
        )

    _apply_style(
        ax,
        "Validation Loss over Epochs — CIFAR-10",
        "Epoch",
        "Validation loss",
        subtitle="full fine-tuning, linear head, seed 42",
    )
    ax.legend(fontsize=FONT_TICK, framealpha=0.9, loc="upper right")
    _save(fig, "convergence_curves_cifar10.png", output_dir)


# -----------------------------------------------------------------------
# Plot 3 — strategy_comparison_cifar10.png
# -----------------------------------------------------------------------

def plot_strategy_comparison_cifar10(
    summary: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Grouped bars: resnet18 + vit_tiny, three strategies, linear, seed42."""
    if summary.empty:
        return

    pretrained_models = ["resnet18", "vit_tiny"]
    all_strategies = ["full_training", "freeze_then_unfreeze", "head_only"]

    # Seed-42 slice at canonical HPs; strategy drives frozen_epochs so don't filter on it.
    data = summary[
        (summary["dataset"] == "cifar10")
        & (summary["head_type"] == "linear")
        & (summary["seed"] == 42)
        & (summary["learning_rate"] == _CORE_HPS["learning_rate"])
        & (summary["batch_size"] == _CORE_HPS["batch_size"])
        & (summary["weight_decay"] == _CORE_HPS["weight_decay"])
        & (summary["model_name"].isin(pretrained_models))
    ]

    if data.empty:
        print("strategy_comparison_cifar10: no matching rows — skipping")
        return

    x = np.arange(len(pretrained_models))
    width = 0.8 / len(all_strategies)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, strat in enumerate(all_strategies):
        subset = (
            data[data["training_strategy"] == strat]
            .pipe(_first_by_model)
            .set_index("model_name")
        )
        vals = [
            subset.loc[m, "top1_accuracy"] if m in subset.index else np.nan
            for m in pretrained_models
        ]
        offset = (i - len(all_strategies) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width * 0.88,
            label=STRATEGY_LABELS[strat],
            color=STRATEGY_COLORS[strat],
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pretrained_models, fontsize=FONT_TICK, color=NAVY)
    _apply_style(
        ax,
        "Training Strategy Comparison — CIFAR-10 Top-1",
        "",
        "Accuracy (%)",
        subtitle="pretrained backbones only (ResNet-18 and ViT-Tiny), linear head, seed 42",
    )
    ax.set_ylim(0, data["top1_accuracy"].max() * 1.22)
    _label_bars(ax, "{:.1f}%")
    ax.legend(fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "strategy_comparison_cifar10.png", output_dir)


# -----------------------------------------------------------------------
# Plot 4 — dataset_scaling.png
# -----------------------------------------------------------------------

def plot_dataset_scaling(
    summary: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Grouped bars: CIFAR-10 vs CIFAR-100, full_training/linear, seed42, all models."""
    if summary.empty:
        return

    # Same strategy for both dataset bars: seed42 / full_training / linear / core HPs
    c10 = _filter_ft_linear(summary, "cifar10").pipe(_first_by_model)[["model_name", "top1_accuracy"]]
    c100 = _filter_ft_linear(summary, "cifar100").pipe(_first_by_model)[["model_name", "top1_accuracy"]]

    if c100.empty:
        print("dataset_scaling: no CIFAR-100 full_training/linear/seed42 rows yet — "
              "skipping (run extension_runs.json first)")
        return

    merged = c10.merge(c100, on="model_name", suffixes=("_c10", "_c100")).sort_values(
        "top1_accuracy_c10", ascending=False
    )
    if merged.empty:
        print("dataset_scaling: no models present in both datasets — skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(merged))
    width = 0.35
    ax.bar(x - width / 2, merged["top1_accuracy_c10"],  width, label="CIFAR-10",  color=NAVY, zorder=3)
    ax.bar(x + width / 2, merged["top1_accuracy_c100"], width, label="CIFAR-100", color=TEAL, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(merged["model_name"].tolist(), fontsize=FONT_TICK, color=NAVY)
    _apply_style(
        ax,
        "Dataset Scaling — CIFAR-10 vs CIFAR-100 Top-1",
        "",
        "Accuracy (%)",
        subtitle="full fine-tuning, linear head, seed 42",
    )
    ax.set_ylim(0, merged[["top1_accuracy_c10", "top1_accuracy_c100"]].values.max() * 1.20)
    _label_bars(ax, "{:.1f}%")
    ax.legend(fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "dataset_scaling.png", output_dir)


# -----------------------------------------------------------------------
# Plot 5 — top5_cifar100.png
# -----------------------------------------------------------------------

def plot_top5_cifar100(
    summary: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """CIFAR-100 top-5: full_training/linear, seed42, all three models."""
    if summary.empty:
        return

    data = _filter_ft_linear(summary, "cifar100").pipe(_first_by_model)

    if data.empty:
        print("top5_cifar100: no CIFAR-100 full_training/linear/seed42 rows yet — "
              "skipping (run extension_runs.json first)")
        return

    data = data.sort_values("top5_accuracy", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [SLATE if "vit" in m else TEAL for m in data["model_name"]]
    ax.bar(data["model_name"], data["top5_accuracy"], color=colors, width=0.5, zorder=3)
    _apply_style(
        ax,
        "Top-5 Accuracy on CIFAR-100",
        "",
        "Top-5 Accuracy (%)",
        subtitle="full fine-tuning, linear head, seed 42",
    )
    ax.set_ylim(0, min(100, data["top5_accuracy"].max() * 1.15))
    _label_bars(ax, "{:.1f}%")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TEAL,  label="CNN family"),
        plt.Rectangle((0, 0), 1, 1, color=SLATE, label="Transformer (ViT)"),
    ]
    ax.legend(handles=handles, fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "top5_cifar100.png", output_dir)


# -----------------------------------------------------------------------
# Plot 6 — head_depth_cifar10.png
# -----------------------------------------------------------------------

def plot_head_depth_cifar10(
    summary: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Grouped bars: all three models × {linear, one_hidden, two_hidden}, full_training, seed42."""
    if summary.empty:
        return

    data = summary[
        (summary["dataset"] == "cifar10")
        & (summary["training_strategy"] == "full_training")
        & (summary["seed"] == 42)
        & (summary["learning_rate"] == _CORE_HPS["learning_rate"])
        & (summary["batch_size"] == _CORE_HPS["batch_size"])
        & (summary["weight_decay"] == _CORE_HPS["weight_decay"])
    ]

    if data.empty:
        print("head_depth_cifar10: no data — skipping")
        return

    all_heads   = ["linear", "one_hidden", "two_hidden"]
    all_models  = ["small_cnn", "resnet18", "vit_tiny"]

    missing = [
        f"{m}/{h}"
        for m in all_models
        for h in all_heads
        if data[(data["model_name"] == m) & (data["head_type"] == h)].empty
    ]
    if missing:
        print(f"head_depth_cifar10: missing cells {missing} — plotting with gaps "
              "(run extension_runs.json first to fill them)")

    x = np.arange(len(all_models))
    width = 0.8 / len(all_heads)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, head in enumerate(all_heads):
        subset = (
            data[data["head_type"] == head]
            .pipe(_first_by_model)
            .set_index("model_name")
        )
        vals = [
            subset.loc[m, "top1_accuracy"] if m in subset.index else np.nan
            for m in all_models
        ]
        offset = (i - len(all_heads) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width * 0.88,
            label=HEAD_LABELS[head],
            color=HEAD_COLORS[head],
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, fontsize=FONT_TICK, color=NAVY)
    _apply_style(
        ax,
        "Classifier Head Depth — CIFAR-10 Top-1",
        "",
        "Accuracy (%)",
        subtitle="full training, seed 42",
    )
    ax.set_ylim(0, data["top1_accuracy"].max() * 1.22)
    _label_bars(ax, "{:.1f}%")
    ax.legend(fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "head_depth_cifar10.png", output_dir)


# -----------------------------------------------------------------------
# Plot 7 — hp_sensitivity_resnet18.png
# -----------------------------------------------------------------------

def plot_hp_sensitivity_resnet18(
    summary: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Grouped bars: resnet18 freeze_then_unfreeze/linear, lr × bs grid, seed42."""
    if summary.empty:
        return

    data = summary[
        (summary["model_name"] == "resnet18")
        & (summary["dataset"] == "cifar10")
        & (summary["training_strategy"] == "freeze_then_unfreeze")
        & (summary["head_type"] == "linear")
        & (summary["seed"] == 42)
    ].copy()

    if data.empty:
        print("hp_sensitivity_resnet18: no data — skipping")
        return

    all_lrs = sorted(data["learning_rate"].unique())
    all_bs  = sorted(data["batch_size"].unique())
    lr_labels = [f"{lr:g}" for lr in all_lrs]
    bs_colors = [NAVY, TEAL, GOLD, SLATE]

    x = np.arange(len(all_lrs))
    width = 0.8 / len(all_bs)

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, bs in enumerate(all_bs):
        subset = (
            data[data["batch_size"] == bs]
            .groupby("learning_rate", as_index=False)
            .first()
            .set_index("learning_rate")
        )
        vals = [
            subset.loc[lr, "top1_accuracy"] if lr in subset.index else np.nan
            for lr in all_lrs
        ]
        offset = (j - len(all_bs) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width * 0.88,
            label=f"bs={bs}",
            color=bs_colors[j % len(bs_colors)],
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(lr_labels, fontsize=FONT_TICK, color=NAVY)
    _apply_style(
        ax,
        "HP Sensitivity — ResNet-18, Freeze → Unfreeze, CIFAR-10",
        "Learning rate",
        "Top-1 Accuracy (%)",
        subtitle="linear head, seed 42",
    )
    ax.set_ylim(0, data["top1_accuracy"].max() * 1.22)
    _label_bars(ax, "{:.1f}%")
    ax.legend(fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "hp_sensitivity_resnet18.png", output_dir)


# -----------------------------------------------------------------------
# Plot 8 — params_vs_accuracy_cifar10.png
# -----------------------------------------------------------------------

def plot_params_vs_accuracy_cifar10(
    summary: pd.DataFrame,
    aggregated: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Scatter: parameter count vs top-1, full_training/linear (3-seed mean where available)."""
    if summary.empty:
        return

    # Parameter count is model-intrinsic; seed42 row is the reference.
    params_df = (
        _filter_ft_linear(summary, "cifar10")
        .pipe(_first_by_model)[["model_name", "num_parameters"]]
    )

    # Accuracy: prefer 3-seed mean from aggregated; fall back to seed42 from summary.
    acc_df: pd.DataFrame
    if not aggregated.empty:
        agg_slice = aggregated[
            (aggregated["dataset"] == "cifar10")
            & (aggregated["training_strategy"] == "full_training")
            & (aggregated["head_type"] == "linear")
            & (aggregated["patience"] == 5)
            & (aggregated["run_count"] == 3)
        ][["model_name", "top1_mean"]].rename(columns={"top1_mean": "accuracy"})
        if not agg_slice.empty:
            acc_df = agg_slice
        else:
            acc_df = (
                _filter_ft_linear(summary, "cifar10")
                .pipe(_first_by_model)[["model_name", "top1_accuracy"]]
                .rename(columns={"top1_accuracy": "accuracy"})
            )
    else:
        acc_df = (
            _filter_ft_linear(summary, "cifar10")
            .pipe(_first_by_model)[["model_name", "top1_accuracy"]]
            .rename(columns={"top1_accuracy": "accuracy"})
        )

    data = params_df.merge(acc_df, on="model_name")
    if data.empty:
        print("params_vs_accuracy_cifar10: no data — skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (_, row) in enumerate(data.iterrows()):
        color = PALETTE[i % len(PALETTE)]
        ax.scatter(row["num_parameters"], row["accuracy"], s=140, color=color, zorder=3)
        ax.annotate(
            row["model_name"],
            (row["num_parameters"], row["accuracy"]),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=FONT_ANNOT,
            color=NAVY,
            fontweight="bold",
        )

    _apply_style(
        ax,
        "Model Size vs. Accuracy — CIFAR-10",
        "Parameter count (log scale)",
        "Top-1 accuracy (%)",
        subtitle="full fine-tuning, linear head (3-seed mean where available)",
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    )
    ax.grid(axis="both", color=GRID, linewidth=0.8, linestyle="--", zorder=0)
    _save(fig, "params_vs_accuracy_cifar10.png", output_dir)


# -----------------------------------------------------------------------
# Plot 9 — vit_lr_sensitivity_cifar10.png
# -----------------------------------------------------------------------

def plot_vit_lr_sensitivity_cifar10(
    summary: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Supplementary bars: ViT-Tiny top-1 at lr ∈ {1e-3, 3e-4, 1e-4}, seed42."""
    if summary.empty:
        return

    target_lrs = [0.001, 0.0003, 0.0001]

    data = summary[
        (summary["model_name"] == "vit_tiny")
        & (summary["dataset"] == "cifar10")
        & (summary["training_strategy"] == "full_training")
        & (summary["head_type"] == "linear")
        & (summary["seed"] == 42)
        & (summary["batch_size"] == _CORE_HPS["batch_size"])
        & (summary["weight_decay"] == _CORE_HPS["weight_decay"])
        & (summary["learning_rate"].isin(target_lrs))
    ].copy()

    missing_lrs = [lr for lr in target_lrs if lr not in data["learning_rate"].values]
    if missing_lrs:
        print(f"vit_lr_sensitivity_cifar10: missing lr={missing_lrs} rows — "
              "skipping (run extension_runs.json first)")
        if data.empty:
            return

    data = (
        data.groupby("learning_rate", as_index=False)
        .first()
        .sort_values("learning_rate")
    )
    lr_labels = [f"{lr:g}" for lr in data["learning_rate"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(data))]
    ax.bar(lr_labels, data["top1_accuracy"].values, color=colors, width=0.5, zorder=3)
    _apply_style(
        ax,
        "ViT-Tiny LR Sensitivity — CIFAR-10 Top-1",
        "Learning rate",
        "Accuracy (%)",
        subtitle="full fine-tuning, linear head, seed 42 (sensitivity check — not part of core comparison)",
    )
    ax.set_ylim(0, min(100, data["top1_accuracy"].max() * 1.18))
    _label_bars(ax, "{:.1f}%")
    _save(fig, "vit_lr_sensitivity_cifar10.png", output_dir)


# -----------------------------------------------------------------------
# Kept plot — training time (not replaced)
# -----------------------------------------------------------------------

def plot_time_by_model(
    summary: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Average epoch training time — full_training/linear/seed42 per model, CIFAR-10."""
    if summary.empty or "average_time_per_epoch" not in summary.columns:
        return

    data = _filter_ft_linear(summary, "cifar10").pipe(_first_by_model)
    if data.empty:
        return

    data = data.sort_values("average_time_per_epoch", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [SLATE if "vit" in m else GOLD for m in data["model_name"]]
    ax.bar(data["model_name"], data["average_time_per_epoch"], color=colors, width=0.5, zorder=3)
    _apply_style(
        ax,
        "Average Training Time per Epoch — CIFAR-10",
        "",
        "Seconds",
        subtitle="full fine-tuning, linear head, seed 42",
    )
    _label_bars(ax, "{:.1f}s")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GOLD,  label="CNN family"),
        plt.Rectangle((0, 0), 1, 1, color=SLATE, label="Transformer (ViT)"),
    ]
    ax.legend(handles=handles, fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "average_time_per_epoch_by_model.png", output_dir)


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def generate_all_plots(
    summary_path: Path = SUMMARY_PATH,
    metrics_path: Path = METRICS_PATH,
    aggregated_path: Path = AGGREGATED_PATH,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Generate all presentation plots from result CSV files."""
    # Remove old PNGs that are superseded by the new controlled-comparison versions.
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_name in _RETIRED_FILENAMES:
        old_path = output_dir / old_name
        if old_path.exists():
            old_path.unlink()
            print(f"Removed retired plot: {old_name}")

    summary, metrics = _load_results(summary_path, metrics_path)
    aggregated = _load_aggregated(aggregated_path)

    # --- 9 required plots ---
    plot_core_comparison_cifar10(aggregated, output_dir)         # 1
    plot_convergence_curves_cifar10(metrics, output_dir)         # 2
    plot_strategy_comparison_cifar10(summary, output_dir)        # 3
    plot_dataset_scaling(summary, output_dir)                    # 4
    plot_top5_cifar100(summary, output_dir)                      # 5
    plot_head_depth_cifar10(summary, output_dir)                 # 6
    plot_hp_sensitivity_resnet18(summary, output_dir)            # 7
    plot_params_vs_accuracy_cifar10(summary, aggregated, output_dir)  # 8
    plot_vit_lr_sensitivity_cifar10(summary, output_dir)         # 9

    # --- kept supplementary ---
    plot_time_by_model(summary, output_dir)

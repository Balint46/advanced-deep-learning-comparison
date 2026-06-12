"""Plotting utilities for experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .experiment import METRICS_PATH, SUMMARY_PATH
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
    "full_training":       NAVY,
    "freeze_then_unfreeze": GOLD,
    "head_only":           TEAL,
}
HEAD_COLORS = {
    "linear":     NAVY,
    "one_hidden": TEAL,
    "two_hidden": GOLD,
}
STRATEGY_LABELS = {
    "full_training":       "Full training",
    "freeze_then_unfreeze": "Freeze → unfreeze",
    "head_only":           "Head only",
}
HEAD_LABELS = {
    "linear":     "Linear",
    "one_hidden": "One hidden layer",
    "two_hidden": "Two hidden layers",
}


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _apply_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str,
                 subtitle: str = "") -> None:
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


def _best_cifar10(summary: pd.DataFrame) -> pd.DataFrame:
    """Best result per model on CIFAR-10, sorted by top1 descending."""
    if summary.empty:
        return pd.DataFrame()
    data = summary[summary["dataset"] == "cifar10"]
    if data.empty:
        return pd.DataFrame()
    return (
        data.sort_values("top1_accuracy", ascending=False)
        .groupby("model_name", as_index=False)
        .first()
        .sort_values("top1_accuracy", ascending=False)
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------
# Plot functions
# -----------------------------------------------------------------------

def plot_top1_by_model(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    """Best CIFAR-10 top-1 accuracy per model, sorted highest to lowest."""
    data = _best_cifar10(summary)
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [SLATE if "vit" in m else NAVY for m in data["model_name"]]
    ax.bar(data["model_name"], data["top1_accuracy"], color=colors, width=0.5, zorder=3)
    _apply_style(ax, "Top-1 Accuracy by Model — CIFAR-10", "", "Accuracy (%)")
    ax.set_ylim(0, min(100, data["top1_accuracy"].max() * 1.18))
    _label_bars(ax, "{:.1f}%")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=NAVY,  label="CNN family"),
        plt.Rectangle((0, 0), 1, 1, color=SLATE, label="Transformer (ViT)"),
    ]
    ax.legend(handles=handles, fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "top1_accuracy_by_model.png", output_dir)


def plot_top5_by_model(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    """Best CIFAR-10 top-5 accuracy per model, sorted by top-1 descending."""
    data = _best_cifar10(summary)
    if data.empty or "top5_accuracy" not in data:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [SLATE if "vit" in m else TEAL for m in data["model_name"]]
    ax.bar(data["model_name"], data["top5_accuracy"], color=colors, width=0.5, zorder=3)
    _apply_style(ax, "Top-5 Accuracy by Model — CIFAR-10", "", "Accuracy (%)")
    ax.set_ylim(0, min(100, data["top5_accuracy"].max() * 1.12))
    _label_bars(ax, "{:.1f}%")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TEAL,  label="CNN family"),
        plt.Rectangle((0, 0), 1, 1, color=SLATE, label="Transformer (ViT)"),
    ]
    ax.legend(handles=handles, fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "top5_accuracy_by_model.png", output_dir)


def plot_cifar10_vs_cifar100(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    """Side-by-side CIFAR-10 vs CIFAR-100 for models tested on both datasets."""
    if summary.empty or not {"dataset", "model_name", "top1_accuracy"}.issubset(summary.columns):
        return

    best = (
        summary.sort_values("top1_accuracy", ascending=False)
        .groupby(["dataset", "model_name"], as_index=False)
        .first()
    )
    pivot = best.pivot(index="model_name", columns="dataset", values="top1_accuracy")

    # Keep only models that appear in both datasets
    both = pivot.dropna(subset=["cifar10", "cifar100"] if "cifar100" in pivot.columns else ["cifar10"])
    if "cifar100" not in pivot.columns or both.empty:
        return
    both = both.dropna()
    if both.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(both.index))
    width = 0.35
    ax.bar(x - width / 2, both["cifar10"],  width, label="CIFAR-10",  color=NAVY, zorder=3)
    ax.bar(x + width / 2, both["cifar100"], width, label="CIFAR-100", color=TEAL, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(both.index, fontsize=FONT_TICK, color=NAVY)
    _apply_style(
        ax,
        "CIFAR-10 vs CIFAR-100 — Top-1 Accuracy",
        "",
        "Accuracy (%)",
        subtitle="Models tested on both datasets (best result per model)",
    )
    ax.set_ylim(0, np.nanmax(both.values) * 1.20)
    _label_bars(ax, "{:.1f}%")
    ax.legend(fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "cifar10_vs_cifar100_top1.png", output_dir)


def plot_time_by_model(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    """Average epoch training time for the best CIFAR-10 run per model."""
    data = _best_cifar10(summary)
    if data.empty or "average_time_per_epoch" not in data:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [SLATE if "vit" in m else GOLD for m in data["model_name"]]
    ax.bar(data["model_name"], data["average_time_per_epoch"], color=colors, width=0.5, zorder=3)
    _apply_style(ax, "Average Training Time per Epoch — CIFAR-10", "", "Seconds")
    _label_bars(ax, "{:.1f}s")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GOLD,  label="CNN family"),
        plt.Rectangle((0, 0), 1, 1, color=SLATE, label="Transformer (ViT)"),
    ]
    ax.legend(handles=handles, fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "average_time_per_epoch_by_model.png", output_dir)


def plot_parameters_vs_accuracy(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    """Scatter: parameter count vs best CIFAR-10 top-1 accuracy."""
    data = _best_cifar10(summary)
    if data.empty or "num_parameters" not in data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (_, row) in enumerate(data.iterrows()):
        color = PALETTE[i % len(PALETTE)]
        ax.scatter(row["num_parameters"], row["top1_accuracy"], s=140, color=color, zorder=3)
        ax.annotate(
            row["model_name"],
            (row["num_parameters"], row["top1_accuracy"]),
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
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    )
    ax.grid(axis="both", color=GRID, linewidth=0.8, linestyle="--", zorder=0)
    _save(fig, "parameters_vs_top1_accuracy.png", output_dir)


def plot_validation_loss_curves(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Validation loss curves for the best CIFAR-10 experiment per model."""
    if metrics.empty or "val_loss" not in metrics:
        return

    # Pick the best experiment_id per model from the summary
    if not summary.empty and "experiment_id" in summary.columns:
        cifar10 = summary[summary["dataset"] == "cifar10"]
        best_ids = (
            cifar10.sort_values("top1_accuracy", ascending=False)
            .groupby("model_name")["experiment_id"]
            .first()
            .to_dict()
        )
    else:
        unique_ids = metrics["experiment_id"].dropna().unique()[:6]
        best_ids = {eid: eid for eid in unique_ids}

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model_name, exp_id) in enumerate(best_ids.items()):
        group = metrics[metrics["experiment_id"] == exp_id].sort_values("epoch")
        if group.empty:
            continue
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
        subtitle="Best hyperparameter setting per model",
    )
    ax.legend(fontsize=FONT_TICK, framealpha=0.9, loc="upper right")
    _save(fig, "validation_loss_curves.png", output_dir)


def plot_strategy_comparison(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    """Grouped bar chart: training strategy vs top-1 accuracy on CIFAR-10."""
    if summary.empty or "training_strategy" not in summary.columns:
        return
    data = summary[summary["dataset"] == "cifar10"]
    if data.empty:
        return

    best = (
        data.sort_values("top1_accuracy", ascending=False)
        .groupby(["model_name", "training_strategy"], as_index=False)
        .first()
    )

    all_strategies = ["full_training", "freeze_then_unfreeze", "head_only"]
    models = sorted(best["model_name"].unique())
    x = np.arange(len(models))
    width = 0.8 / len(all_strategies)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, strat in enumerate(all_strategies):
        subset = best[best["training_strategy"] == strat].set_index("model_name")
        vals = [
            subset.loc[m, "top1_accuracy"] if m in subset.index else np.nan
            for m in models
        ]
        offset = (i - len(all_strategies) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width * 0.88,
            label=STRATEGY_LABELS[strat],
            color=STRATEGY_COLORS[strat],
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=FONT_TICK, color=NAVY)
    _apply_style(
        ax,
        "Training Strategy Comparison — CIFAR-10 Top-1",
        "",
        "Accuracy (%)",
        subtitle="Missing bars = strategy not applicable (no pretrained backbone / not tested)",
    )
    ax.set_ylim(0, best["top1_accuracy"].max() * 1.22)
    _label_bars(ax, "{:.1f}%")
    ax.legend(fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "strategy_comparison.png", output_dir)


def plot_head_comparison(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    """Grouped bar chart: classifier head depth vs top-1 accuracy on CIFAR-10."""
    if summary.empty or "head_type" not in summary.columns:
        return
    data = summary[summary["dataset"] == "cifar10"]
    if data.empty:
        return

    best = (
        data.sort_values("top1_accuracy", ascending=False)
        .groupby(["model_name", "head_type"], as_index=False)
        .first()
    )

    all_heads = ["linear", "one_hidden", "two_hidden"]
    models = sorted(best["model_name"].unique())
    x = np.arange(len(models))
    width = 0.8 / len(all_heads)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, head in enumerate(all_heads):
        subset = best[best["head_type"] == head].set_index("model_name")
        vals = [
            subset.loc[m, "top1_accuracy"] if m in subset.index else np.nan
            for m in models
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
    ax.set_xticklabels(models, fontsize=FONT_TICK, color=NAVY)
    _apply_style(
        ax,
        "Classifier Head Depth — CIFAR-10 Top-1",
        "",
        "Accuracy (%)",
        subtitle="Each model tested with linear, one-hidden-layer, and two-hidden-layer head",
    )
    ax.set_ylim(0, best["top1_accuracy"].max() * 1.22)
    _label_bars(ax, "{:.1f}%")
    ax.legend(fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "head_comparison.png", output_dir)


# -----------------------------------------------------------------------

def generate_all_plots(
    summary_path: Path = SUMMARY_PATH,
    metrics_path: Path = METRICS_PATH,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Generate all presentation plots from result CSV files."""
    summary, metrics = _load_results(summary_path, metrics_path)
    plot_top1_by_model(summary, output_dir)
    plot_top5_by_model(summary, output_dir)
    plot_cifar10_vs_cifar100(summary, output_dir)
    plot_time_by_model(summary, output_dir)
    plot_parameters_vs_accuracy(summary, output_dir)
    plot_validation_loss_curves(metrics, summary, output_dir)
    plot_strategy_comparison(summary, output_dir)
    plot_head_comparison(summary, output_dir)

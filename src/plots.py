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
NAVY   = "#003366"
TEAL   = "#006699"
GOLD   = "#CC9900"
SLATE  = "#4A6FA5"
CORAL  = "#CC4A1D"
LIGHT  = "#E8EEF4"
GRID   = "#D0D8E4"

PALETTE = [NAVY, TEAL, GOLD, SLATE, CORAL, "#5A7A3A", "#7B4A8A"]

FONT_TITLE  = 14
FONT_LABEL  = 12
FONT_TICK   = 10
FONT_ANNOT  = 9
DPI         = 200


def _apply_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", color=NAVY, pad=12)
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
        if height > 0:
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


def _best_by_model(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    if summary.empty or metric not in summary:
        return pd.DataFrame()
    return (
        summary.sort_values(metric, ascending=False)
        .groupby(["dataset", "model_name"], as_index=False)
        .first()
        .sort_values(["dataset", "model_name"])
    )


def _model_label(row: pd.Series) -> str:
    return f"{row['model_name']}\n({row['dataset'].upper()})"


# -----------------------------------------------------------------------
# Individual plot functions
# -----------------------------------------------------------------------

def plot_top1_by_model(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    data = _best_by_model(summary, "top1_accuracy")
    if data.empty:
        return

    labels = [_model_label(r) for _, r in data.iterrows()]
    colors = [NAVY if "cifar10" in str(r["dataset"]) else TEAL for _, r in data.iterrows()]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, data["top1_accuracy"], color=colors, width=0.55, zorder=3)
    _apply_style(ax, "Top-1 Test Accuracy by Model", "", "Accuracy (%)")
    ax.set_ylim(0, min(100, data["top1_accuracy"].max() * 1.18))
    _label_bars(ax, "{:.1f}%")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=NAVY, label="CIFAR-10"),
        plt.Rectangle((0, 0), 1, 1, color=TEAL, label="CIFAR-100"),
    ]
    ax.legend(handles=handles, fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "top1_accuracy_by_model.png", output_dir)


def plot_top5_by_model(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    data = _best_by_model(summary, "top5_accuracy")
    if data.empty:
        return

    labels = [_model_label(r) for _, r in data.iterrows()]
    colors = [NAVY if "cifar10" in str(r["dataset"]) else TEAL for _, r in data.iterrows()]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, data["top5_accuracy"], color=colors, width=0.55, zorder=3)
    _apply_style(ax, "Top-5 Test Accuracy by Model", "", "Accuracy (%)")
    ax.set_ylim(0, min(100, data["top5_accuracy"].max() * 1.18))
    _label_bars(ax, "{:.1f}%")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=NAVY, label="CIFAR-10"),
        plt.Rectangle((0, 0), 1, 1, color=TEAL, label="CIFAR-100"),
    ]
    ax.legend(handles=handles, fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "top5_accuracy_by_model.png", output_dir)


def plot_time_by_model(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    data = _best_by_model(summary, "top1_accuracy")
    if data.empty or "average_time_per_epoch" not in data:
        return

    labels = [_model_label(r) for _, r in data.iterrows()]
    colors = [SLATE if "vit" in str(r["model_name"]) else GOLD for _, r in data.iterrows()]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, data["average_time_per_epoch"], color=colors, width=0.55, zorder=3)
    _apply_style(ax, "Average Training Time per Epoch", "", "Seconds")
    _label_bars(ax, "{:.1f}s")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GOLD,  label="CNN-family"),
        plt.Rectangle((0, 0), 1, 1, color=SLATE, label="Transformer (ViT)"),
    ]
    ax.legend(handles=handles, fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "average_time_per_epoch_by_model.png", output_dir)


def plot_parameters_vs_accuracy(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    data = _best_by_model(summary, "top1_accuracy")
    if data.empty or "num_parameters" not in data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (_, row) in enumerate(data.iterrows()):
        color = PALETTE[i % len(PALETTE)]
        ax.scatter(row["num_parameters"], row["top1_accuracy"], s=120, color=color, zorder=3)
        ax.annotate(
            f"{row['model_name']}\n({row['dataset'].upper()})",
            (row["num_parameters"], row["top1_accuracy"]),
            xytext=(7, 4),
            textcoords="offset points",
            fontsize=FONT_ANNOT,
            color=NAVY,
        )

    _apply_style(ax, "Model Size vs. Top-1 Accuracy", "Number of Parameters (log scale)", "Accuracy (%)")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    ax.grid(axis="both", color=GRID, linewidth=0.8, linestyle="--", zorder=0)
    _save(fig, "parameters_vs_top1_accuracy.png", output_dir)


def plot_validation_loss_curves(
    metrics: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
    max_experiments: int = 6,
) -> None:
    if metrics.empty or "val_loss" not in metrics:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    experiment_ids = list(metrics["experiment_id"].dropna().unique())[:max_experiments]
    selected = metrics[metrics["experiment_id"].isin(experiment_ids)]

    for i, (experiment_id, group) in enumerate(selected.groupby("experiment_id")):
        short = "_".join(experiment_id.split("_")[:4])
        ax.plot(
            group["epoch"],
            group["val_loss"],
            marker="o",
            markersize=4,
            linewidth=2,
            color=PALETTE[i % len(PALETTE)],
            label=short,
        )

    _apply_style(ax, "Validation Loss over Epochs", "Epoch", "Validation Loss")
    ax.legend(fontsize=FONT_ANNOT, framealpha=0.9, loc="upper right")
    _save(fig, "validation_loss_curves.png", output_dir)


def plot_cifar10_vs_cifar100(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    if summary.empty or not {"dataset", "model_name", "top1_accuracy"}.issubset(summary.columns):
        return
    data = _best_by_model(summary, "top1_accuracy")
    pivot = data.pivot(index="model_name", columns="dataset", values="top1_accuracy")
    if pivot.empty or not {"cifar10", "cifar100"}.intersection(pivot.columns):
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pivot.index))
    width = 0.35
    cols = [c for c in ["cifar10", "cifar100"] if c in pivot.columns]
    bar_colors = [NAVY, TEAL]

    for i, col in enumerate(cols):
        bars = ax.bar(x + i * width, pivot[col], width, label=col.upper(), color=bar_colors[i], zorder=3)

    ax.set_xticks(x + width * (len(cols) - 1) / 2)
    ax.set_xticklabels(pivot.index, fontsize=FONT_TICK, color=NAVY, rotation=20)
    _apply_style(ax, "CIFAR-10 vs CIFAR-100 — Top-1 Accuracy", "", "Accuracy (%)")
    ax.set_ylim(0, pivot.values.max() * 1.18)
    _label_bars(ax, "{:.1f}%")
    ax.legend(fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "cifar10_vs_cifar100_top1.png", output_dir)


def plot_strategy_comparison(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    """Bar chart comparing the three training strategies on CIFAR-10 top-1 accuracy."""
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
    strategies = best["training_strategy"].unique()
    models = best["model_name"].unique()
    x = np.arange(len(models))
    width = 0.8 / max(len(strategies), 1)
    strat_colors = {"full_training": NAVY, "head_only": TEAL, "freeze_then_unfreeze": GOLD}

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, strat in enumerate(strategies):
        subset = best[best["training_strategy"] == strat].set_index("model_name")
        vals = [subset.loc[m, "top1_accuracy"] if m in subset.index else 0 for m in models]
        offset = (i - len(strategies) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=strat.replace("_", " "), color=strat_colors.get(strat, SLATE), zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=FONT_TICK, color=NAVY)
    _apply_style(ax, "Training Strategy Comparison — CIFAR-10 Top-1", "", "Accuracy (%)")
    ax.set_ylim(0, best["top1_accuracy"].max() * 1.18)
    ax.legend(fontsize=FONT_TICK, framealpha=0.9)
    _save(fig, "strategy_comparison.png", output_dir)


def plot_head_comparison(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    """Bar chart comparing classification head depth on CIFAR-10 top-1 accuracy."""
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
    heads = best["head_type"].unique()
    models = best["model_name"].unique()
    x = np.arange(len(models))
    width = 0.8 / max(len(heads), 1)
    head_colors = {"linear": NAVY, "one_hidden": TEAL, "two_hidden": GOLD}

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, head in enumerate(heads):
        subset = best[best["head_type"] == head].set_index("model_name")
        vals = [subset.loc[m, "top1_accuracy"] if m in subset.index else 0 for m in models]
        offset = (i - len(heads) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=head.replace("_", " "), color=head_colors.get(head, SLATE), zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=FONT_TICK, color=NAVY)
    _apply_style(ax, "Classifier Head Depth Comparison — CIFAR-10 Top-1", "", "Accuracy (%)")
    ax.set_ylim(0, best["top1_accuracy"].max() * 1.18)
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
    plot_time_by_model(summary, output_dir)
    plot_parameters_vs_accuracy(summary, output_dir)
    plot_validation_loss_curves(metrics, output_dir)
    plot_cifar10_vs_cifar100(summary, output_dir)
    plot_strategy_comparison(summary, output_dir)
    plot_head_comparison(summary, output_dir)

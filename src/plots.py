"""Plotting utilities for experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .experiment import METRICS_PATH, SUMMARY_PATH
from .utils import PLOTS_DIR


def _load_results(
    summary_path: Path = SUMMARY_PATH,
    metrics_path: Path = METRICS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    return summary, metrics


def _save(fig: plt.Figure, filename: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)


def _best_by_model(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    if summary.empty or metric not in summary:
        return pd.DataFrame()
    return (
        summary.sort_values(metric, ascending=False)
        .groupby(["dataset", "model_name"], as_index=False)
        .first()
        .sort_values(["dataset", "model_name"])
    )


def plot_top1_by_model(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    data = _best_by_model(summary, "top1_accuracy")
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = data["dataset"] + " / " + data["model_name"]
    ax.bar(labels, data["top1_accuracy"], color="#2f6f73")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Top-1 Accuracy by Model")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, "top1_accuracy_by_model.png", output_dir)


def plot_top5_by_model(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    data = _best_by_model(summary, "top5_accuracy")
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = data["dataset"] + " / " + data["model_name"]
    ax.bar(labels, data["top5_accuracy"], color="#8f5d2c")
    ax.set_ylabel("Top-5 accuracy (%)")
    ax.set_title("Top-5 Accuracy by Model")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, "top5_accuracy_by_model.png", output_dir)


def plot_time_by_model(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    data = _best_by_model(summary, "top1_accuracy")
    if data.empty or "average_time_per_epoch" not in data:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = data["dataset"] + " / " + data["model_name"]
    ax.bar(labels, data["average_time_per_epoch"], color="#5b5f97")
    ax.set_ylabel("Seconds")
    ax.set_title("Average Training Time per Epoch")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, "average_time_per_epoch_by_model.png", output_dir)


def plot_parameters_vs_accuracy(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    data = _best_by_model(summary, "top1_accuracy")
    if data.empty or "num_parameters" not in data:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in data.iterrows():
        ax.scatter(row["num_parameters"], row["top1_accuracy"], s=80)
        ax.annotate(
            f"{row['dataset']} {row['model_name']}",
            (row["num_parameters"], row["top1_accuracy"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters (log scale)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Model Size versus Accuracy")
    _save(fig, "parameters_vs_top1_accuracy.png", output_dir)


def plot_validation_loss_curves(
    metrics: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
    max_experiments: int = 6,
) -> None:
    if metrics.empty or "val_loss" not in metrics:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    experiment_ids = list(metrics["experiment_id"].dropna().unique())[:max_experiments]
    selected = metrics[metrics["experiment_id"].isin(experiment_ids)]
    for experiment_id, group in selected.groupby("experiment_id"):
        ax.plot(group["epoch"], group["val_loss"], marker="o", label=experiment_id[:45])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation Loss over Epochs")
    ax.legend(fontsize=7)
    _save(fig, "validation_loss_curves.png", output_dir)


def plot_cifar10_vs_cifar100(summary: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> None:
    if summary.empty or not {"dataset", "model_name", "top1_accuracy"}.issubset(summary.columns):
        return
    data = _best_by_model(summary, "top1_accuracy")
    pivot = data.pivot(index="model_name", columns="dataset", values="top1_accuracy")
    if pivot.empty or not {"cifar10", "cifar100"}.intersection(pivot.columns):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax, color=["#2f6f73", "#8f5d2c"])
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("CIFAR-10 and CIFAR-100 Comparison")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, "cifar10_vs_cifar100_top1.png", output_dir)


def generate_all_plots(
    summary_path: Path = SUMMARY_PATH,
    metrics_path: Path = METRICS_PATH,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Generate all requested plots from result CSV files."""
    summary, metrics = _load_results(summary_path, metrics_path)
    plot_top1_by_model(summary, output_dir)
    plot_top5_by_model(summary, output_dir)
    plot_time_by_model(summary, output_dir)
    plot_parameters_vs_accuracy(summary, output_dir)
    plot_validation_loss_curves(metrics, output_dir)
    plot_cifar10_vs_cifar100(summary, output_dir)

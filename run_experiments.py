"""Command-line entrypoint for running experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiment import (
    configs_from_dicts,
    make_cifar100_configs_from_best,
    make_hyperparameter_search_configs,
    run_experiment,
)
from src.plots import generate_all_plots
from src.utils import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CIFAR model comparison experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default_experiments.json",
        help="Path to a JSON file containing an experiments list.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally run only the first N experiments.",
    )
    parser.add_argument(
        "--mode",
        choices=["default", "search", "cifar100-from-best", "plots"],
        default="default",
        help="Run default experiments, CIFAR-10 grid search, or CIFAR-100 reuse.",
    )
    args = parser.parse_args()

    if args.mode == "plots":
        generate_all_plots()
        print("Plots saved to results/plots/")
        return

    if args.mode == "search":
        search_path = PROJECT_ROOT / "configs" / "hyperparameter_search.json"
        search_data = json.loads(search_path.read_text(encoding="utf-8"))
        configs = make_hyperparameter_search_configs(search_data)
    elif args.mode == "cifar100-from-best":
        configs = make_cifar100_configs_from_best()
    else:
        config_data = json.loads(args.config.read_text(encoding="utf-8"))
        configs = configs_from_dicts(config_data["experiments"])

    if args.limit is not None:
        configs = configs[: args.limit]

    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] Running {config.experiment_id}")
        summary = run_experiment(config)
        print(
            f"Done: top1={summary['top1_accuracy']:.2f}, "
            f"top5={summary['top5_accuracy']:.2f}, "
            f"test_loss={summary['test_loss']:.4f}"
        )


if __name__ == "__main__":
    main()

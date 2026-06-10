"""Command-line entrypoint for running experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiment import configs_from_dicts, run_experiment
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
    args = parser.parse_args()

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

"""Utility: print all planned experiment IDs from both config files."""
import json
from pathlib import Path

from src.experiment import configs_from_dicts, make_hyperparameter_search_configs

cfg_data = json.loads(Path("configs/default_experiments.json").read_text())
default_cfgs = configs_from_dicts(cfg_data["experiments"])
print(f"=== Default experiments ({len(default_cfgs)} runs) ===")
for i, c in enumerate(default_cfgs, 1):
    print(f"  {i:2d}. {c.experiment_id}  [patience={c.patience}, epochs_max={c.epochs}]")

search_data = json.loads(Path("configs/hyperparameter_search.json").read_text())
search_cfgs = make_hyperparameter_search_configs(search_data)
print(f"\n=== HP search experiments ({len(search_cfgs)} runs) ===")
for i, c in enumerate(search_cfgs, 1):
    print(f"  {i:2d}. {c.experiment_id}  [patience={c.patience}, epochs_max={c.epochs}]")

default_ids = {c.experiment_id for c in default_cfgs}
search_ids = {c.experiment_id for c in search_cfgs}
overlap = default_ids & search_ids
if overlap:
    print(f"\nOverlap (deduper will skip on second run): {overlap}")
else:
    print("\nNo overlap between default and search sets.")
print(f"Total unique across both: {len(default_ids | search_ids)}")

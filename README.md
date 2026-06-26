# Advanced Deep Learning Comparison

PyTorch project for the Advanced Deep Learning university course. The task is to compare different model architectures with different training strategies on different datasets, evaluated across multiple metrics. The goal is a clean, reproducible experiment suite that supports a presentation — not state-of-the-art accuracy.

## Project Goal

This project compares a custom CNN trained from scratch against pretrained transformer and CNN backbones under a controlled training pipeline. CIFAR-10 is the primary benchmark where all models are evaluated on the same 10-class task. CIFAR-100 is used as a harder difficulty check with 100 classes.

Comparing models across datasets is not a direct ranking — a CIFAR-10 accuracy and a CIFAR-100 accuracy answer different questions. CIFAR-100 results are interpreted as a robustness check, not as a head-to-head comparison against CIFAR-10 numbers.

## Datasets

- **CIFAR-10**: 10 image classes, 50 000 training images.
- **CIFAR-100**: 100 image classes, 50 000 training images.

Training transforms: random crop, horizontal flip, tensor conversion, and CIFAR-specific normalization. Validation and test transforms are deterministic (tensor conversion + normalization only). The CIFAR training set is split into train and validation subsets using a seeded split (10% validation).

## Models

Three model families are compared:

- `small_cnn`: compact custom CNN trained from scratch (no pretrained weights).
- `resnet18`: pretrained timm ResNet-18 backbone.
- `vit_tiny`: pretrained ViT-Tiny backbone.

Pretrained backbones are loaded via `timm` and given a replaceable classifier head so they work with both CIFAR-10 (10 classes) and CIFAR-100 (100 classes).

## Classification Heads

- `linear`: `features → Linear(num_features, num_classes)`
- `one_hidden`: `features → Linear → ReLU → Dropout → Linear`
- `two_hidden`: `features → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear`

The default experiments use a linear head for the core controlled comparison. Head depth is varied separately as a supplementary experiment.

## Training Strategies

- `full_training`: train all model parameters from the start.
- `head_only`: freeze the backbone and train only the classification head.
- `freeze_then_unfreeze`: train only the head for `frozen_epochs`, then unfreeze the full model and fine-tune at a smaller learning rate.

CUDA is used automatically when available; otherwise the code runs on CPU.

## Early Stopping

All experiments use early stopping on validation loss. The patience is set per config (`patience=5` for CIFAR-10 experiments, `patience=7` for the longer CIFAR-100 runs). Each run saves the best-validation-accuracy checkpoint and restores it for final evaluation.

## Metrics

Per-epoch metrics are saved to `results/metrics.csv`:

- training loss
- validation loss
- train top-1 accuracy
- train top-5 accuracy
- validation top-1 accuracy
- validation top-5 accuracy
- training time per epoch
- trainable parameter count

Final experiment summaries are saved to `results/summary_table.csv`:

- dataset, model, strategy, head, learning rate, batch size, weight decay, epochs, frozen epochs, patience, seed
- test loss
- top-1 accuracy
- top-5 accuracy
- best validation loss and epoch
- total and trainable parameters
- average epoch time and total training time
- generalization gap (final train top-1 minus final validation top-1)
- epochs run, stopped early flag

Top-5 accuracy is collected for both datasets. It is much more informative for CIFAR-100 (100 classes) than for CIFAR-10 (10 classes).

Multi-seed results are aggregated into `results/aggregated_summary.csv` (mean and std over seeds, run count). This is used for the core comparison bar chart.

## Experiment Configuration

Experiments are driven by JSON config files in `configs/`:

- `default_experiments.json` — the main experiment set (18 runs): 3-seed core comparison on CIFAR-10 and CIFAR-100, training strategy comparison, and head depth comparison for ResNet-18.
- `extension_runs.json` — supplementary runs (9 runs): longer CIFAR-100 re-runs (40 epochs), head depth for `small_cnn` and `vit_tiny`, and ViT LR sensitivity check.
- `hyperparameter_search.json` — ResNet-18 and ViT-Tiny learning rate × batch size grid search.

Shared defaults across core experiments: `lr=0.001, batch_size=64, weight_decay=1e-4, epochs=20, patience=5`.

## How to Run

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the main experiment set:

```bash
python run_experiments.py
```

Run only the first experiment for a quick smoke test:

```bash
python run_experiments.py --limit 1
```

Run a custom config file:

```bash
python run_experiments.py --config configs/extension_runs.json
```

Run the hyperparameter grid search:

```bash
python run_experiments.py --mode search
```

Aggregate multi-seed results into `results/aggregated_summary.csv`:

```bash
python run_experiments.py --mode aggregate
```

Generate all plots from existing result CSVs:

```bash
python run_experiments.py --mode plots
```

Plots are saved to `results/plots/`.

Evaluate a saved checkpoint without retraining:

```bash
python evaluate_checkpoint.py checkpoints/cifar10_resnet18_head_only_linear_best.pth
```

Each training run saves the best-validation-accuracy checkpoint to:

```text
checkpoints/{dataset}_{model_name}_{training_strategy}_{head_type}_best.pth
```

## Plots

The plotting module (`src/plots.py`) generates 9 presentation plots plus one supplementary:

1. **`core_comparison_cifar10.png`** — CIFAR-10 top-1 accuracy, full fine-tuning / linear head, 3-seed mean ± std error bars.
2. **`convergence_curves_cifar10.png`** — validation loss over epochs per model, seed 42.
3. **`strategy_comparison_cifar10.png`** — grouped bars for ResNet-18 and ViT-Tiny across all three training strategies.
4. **`dataset_scaling.png`** — CIFAR-10 vs CIFAR-100 top-1 grouped bars for all models.
5. **`top5_cifar100.png`** — CIFAR-100 top-5 accuracy, full fine-tuning / linear head.
6. **`head_depth_cifar10.png`** — grouped bars: all models × {linear, one_hidden, two_hidden} heads.
7. **`hp_sensitivity_resnet18.png`** — ResNet-18 top-1 across learning rate × batch size grid, freeze→unfreeze strategy.
8. **`params_vs_accuracy_cifar10.png`** — scatter of parameter count vs. top-1 accuracy (3-seed mean where available).
9. **`vit_lr_sensitivity_cifar10.png`** — ViT-Tiny top-1 at lr ∈ {1e-3, 3e-4, 1e-4}.
10. **`average_time_per_epoch_by_model.png`** — supplementary: average epoch training time per model.

The notebook `notebooks/results_analysis.ipynb` loads the CSV outputs and can regenerate plots for exploratory analysis.

## Presentation

A Beamer LaTeX slide deck is in `presentation/presentation_dl.tex`. It covers the experiment design, results, and conclusions for the course presentation.

## Hyperparameter Search

The grid search varies learning rate and batch size for ResNet-18 and ViT-Tiny on CIFAR-10, using the `freeze_then_unfreeze` strategy. The best configuration from this search is used as the reference for the HP sensitivity plot.

The search grid is defined in `configs/hyperparameter_search.json`.

## Interpreting Results

Use CIFAR-10 to compare models directly under the same task. Use CIFAR-100 to discuss whether a model remains useful when the classification task becomes more fine-grained.

When presenting results, compare accuracy together with training time and parameter count. A model with slightly lower accuracy may still be attractive if it trains much faster or has fewer parameters.

The 3-seed mean ± std error bars on the core comparison chart show how stable each model's performance is across different random seeds.

## Limitations

- The experiment set is intentionally small and presentation-friendly, not exhaustive.
- CPU runs are slow, especially for pretrained transformers.
- The project does not aim for state-of-the-art CIFAR performance.
- Pretrained ImageNet backbones are adapted to CIFAR-sized images, so results should be interpreted as a practical transfer-learning comparison.
- The validation split is carved out of the CIFAR training set and is not an official separate validation set.
- Head depth ablations and ViT LR sensitivity use only seed 42 (not multi-seed).

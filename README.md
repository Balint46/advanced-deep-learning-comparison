# Advanced Deep Learning Comparison

PyTorch project for comparing CNN and transformer image classifiers on CIFAR-10 and CIFAR-100. The goal is a clean, reproducible experiment suite for a 10-minute university presentation, not state-of-the-art accuracy.

## Project Goal

This project compares classic convolutional models and transformer-based image classifiers under the same training pipeline. CIFAR-10 is used for the main fair comparison because all models are evaluated on the same 10-class task. CIFAR-100 is used as a harder robustness comparison with more classes.

Comparing models across different datasets is not a fair direct comparison. A CIFAR-10 accuracy and a CIFAR-100 accuracy answer different questions, so the project treats CIFAR-100 as an additional difficulty check rather than as a direct ranking against CIFAR-10 results.

## Datasets

- CIFAR-10: 10 image classes.
- CIFAR-100: 100 image classes.

The training split uses random crop, horizontal flip, tensor conversion, and CIFAR-specific normalization. Validation and test transforms are deterministic and use only tensor conversion plus normalization. The original training set is split into train and validation subsets with a seeded split.

## Models

- `small_cnn`: compact custom CNN trained from scratch.
- `resnet18`: pretrained timm ResNet18 backbone.
- `resnet50`: pretrained timm ResNet50 backbone.
- `efficientnet_b0`: pretrained timm EfficientNet-B0 backbone.
- `vit_tiny`: pretrained ViT-Tiny backbone.
- `vit_small`: pretrained ViT-Small backbone.

All pretrained backbones are loaded with `timm` and wrapped with a replaceable classifier head so they work with either 10 CIFAR-10 classes or 100 CIFAR-100 classes.

## Classification Heads

- `linear`: `features -> Linear(num_features, num_classes)`
- `one_hidden`: `features -> Linear -> ReLU -> Dropout -> Linear`
- `two_hidden`: `features -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear`

## Training Strategies

- `full_training`: train all model parameters from the beginning.
- `head_only`: freeze the backbone and train only the classification head.
- `freeze_then_unfreeze`: train only the head for `frozen_epochs`, then unfreeze the full model and fine-tune with a smaller learning rate.

CUDA is used automatically when available; otherwise the code runs on CPU.

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

- dataset, model, strategy, head, learning rate, batch size, weight decay, epochs, frozen epochs, seed
- test loss
- top-1 accuracy
- top-5 accuracy
- best validation loss and epoch
- total and trainable parameters
- average epoch time and total training time
- generalization gap, computed as final train top-1 minus final validation top-1

Top-5 accuracy is computed for both datasets. It is much more informative for CIFAR-100 than CIFAR-10 because CIFAR-10 has only 10 classes.

## Hyperparameter Search

The small search grid is:

- `learning_rate = [1e-4, 3e-4, 1e-3]`
- `batch_size = [64, 128]`
- `weight_decay = [1e-4]`

The full grid search is limited to ResNet18 and ViT-Tiny on CIFAR-10. The best hyperparameters can then be reused for CIFAR-100 experiments.

## How to Run

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the small default experiment list:

```bash
python run_experiments.py
```

Run only the first default experiment for a quick smoke test:

```bash
python run_experiments.py --limit 1
```

Run the CIFAR-10 hyperparameter search:

```bash
python run_experiments.py --mode search
```

Reuse the best CIFAR-10 hyperparameters for CIFAR-100:

```bash
python run_experiments.py --mode cifar100-from-best
```

Generate plots from existing result CSV files:

```bash
python run_experiments.py --mode plots
```

Plots are saved in `results/plots/`.

## Plots

The plotting module generates:

- bar chart of top-1 accuracy by model
- bar chart of top-5 accuracy by model
- bar chart of average training time per epoch by model
- scatter plot of parameter count versus top-1 accuracy
- line plot of validation loss over epochs for selected experiments
- grouped bar chart comparing CIFAR-10 and CIFAR-100 results

The notebook `notebooks/results_analysis.ipynb` loads the CSV outputs and regenerates the plots for presentation work.

## Interpreting Results

Use CIFAR-10 to compare models directly under the same task. Use CIFAR-100 to discuss whether a model remains useful when the classification task becomes more fine-grained.

When presenting results, compare accuracy together with training time and parameter count. A model with slightly lower accuracy may still be attractive if it trains much faster or has fewer parameters.

## Limitations

- The default experiments are intentionally small and presentation-friendly.
- CPU runs may be slow, especially for pretrained transformers.
- The project does not aim for state-of-the-art CIFAR performance.
- Pretrained ImageNet backbones are adapted to CIFAR-sized images, so results should be interpreted as a practical transfer-learning comparison.
- The validation split is created from the CIFAR training set and is not an official separate validation set.

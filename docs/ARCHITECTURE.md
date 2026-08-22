# Architecture

## Data flow

```text
iNaturalist JSON annotations
        |
        v
deterministic category sampler
        |
        v
classes.csv + train/val/test.csv
        |
        v
PyTorch Dataset + transforms
        |
        v
ResNet training and validation
        |
        v
best checkpoint selected on validation Top-1
        |
        v
one final held-out test evaluation
```

## Boundaries

`data.py` owns annotation parsing, label-space construction, manifest creation,
and manifest-backed image loading. It does not know about models.

`models.py` owns architecture construction and classifier replacement. It does
not know about paths or datasets.

`engine.py` owns optimization, epoch loops, checkpoint state, and validation.
It receives datasets and a resolved configuration.

`evaluation.py` owns predictions, required metrics, confusion analysis, and
artifact export. Evaluation can therefore run independently from training.

The command-line modules compose these layers without embedding experiment
logic into shell scripts or notebooks.

## Reproducibility

The random seed is applied to Python, NumPy, PyTorch, CUDA, and DataLoader
workers. Each category uses a category-specific seed for within-class shuffling,
so adding unrelated categories does not silently reorder an existing class.

Checkpoints include the resolved configuration, ordered class records, epoch,
optimizer state, scheduler state, best validation score, and complete history.
This is enough to resume an interrupted run and recover the model label space.

## Evaluation policy

Training and validation are drawn from the training image pool. The official
validation pool is treated as test data because it provides public labels.
Validation chooses the best checkpoint; test data never influences training,
hyperparameters, early stopping, or checkpoint selection.


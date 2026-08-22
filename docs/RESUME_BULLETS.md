# Resume Wording

Choose two or three bullets based on the role. Keep the repository URL beside
the project title and be ready to explain every metric.

## Machine-learning focused

- Built a reproducible PyTorch pipeline for 500-class fine-grained species
  recognition with deterministic data manifests, resumable training, Top-1/5
  evaluation, macro metrics, and confusion analysis.
- Designed controlled ResNet-18/50 architecture and augmentation ablations;
  augmentation improved from-scratch ResNet-50 Top-1 accuracy from 23.38% to
  33.88% and macro-F1 from 23.16% to 32.81%.
- Diagnosed validation plateauing and train/validation divergence as
  data-limited overfitting, motivating transfer learning, stronger
  regularization, and part-aware modeling.

## Software-engineering focused

- Converted an experimental computer-vision workflow into a typed, testable
  Python package with YAML configuration inheritance and four command-line
  entry points for data preparation, training, evaluation, and inference.
- Implemented portable CPU/GPU execution, deterministic DataLoader workers,
  atomic checkpoints, interruption recovery, and machine-readable evaluation
  artifacts.
- Added automated tests and CI checks covering configuration contracts, data
  leakage prevention, model construction, evaluation metrics, and publication
  hygiene.

## One-line version

Fine-grained species classifier in PyTorch with reproducible data splits,
ResNet ablations, resumable training, macro-metric evaluation, and Top-5 demo
inference.


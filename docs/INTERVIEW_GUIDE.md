# Interview Guide

## Thirty-second summary

I built a reproducible PyTorch pipeline for fine-grained species recognition on
iNaturalist. The difficult part is distinguishing visually similar species with
few examples per class and large variation in background and viewpoint. I
implemented deterministic data manifests, ResNet training from random
initialization, checkpoint resume, macro metrics, confusion analysis, and
controlled architecture and augmentation ablations. Augmentation improved
ResNet-50 Top-1 from 23.38% to 33.88%, while increasing depth from ResNet-18 to
ResNet-50 added only 0.22 points.

## What I personally optimized for

- Reproducibility rather than notebook-only experimentation.
- Fair ablations where one design choice changes at a time.
- Validation-only model selection and strict test isolation.
- Failure analysis in addition to headline accuracy.
- Portable CPU/GPU execution and resumable checkpoints.

## Likely questions

### Why macro-F1 as well as accuracy?

Macro-F1 computes a metric per class and averages classes equally. It exposes
species with poor precision or recall even when aggregate accuracy looks
acceptable. On a balanced test set macro recall equals overall accuracy, but
macro precision and macro-F1 still reveal asymmetric confusions.

### Why did augmentation help so much?

Each species had limited training data, while appearance varied strongly with
crop, pose, lighting, and background. Random resized crops, horizontal flips,
color perturbation, and RandAugment increase effective diversity and discourage
memorization of background context.

### Why did ResNet-50 barely beat ResNet-18 from scratch?

The bottleneck was data rather than model capacity. The deeper model has enough
capacity to fit the training set but lacks enough examples to learn reliably
better species-level features. Its validation curve plateaued while training
accuracy continued to rise.

### How did you avoid leakage?

Training and validation were sampled only from the training pool. A separate
officially labeled pool was reserved as test data. Class selection happened
before splitting, every image appeared in exactly one manifest, and only
validation Top-1 selected the best checkpoint.

### What would you improve next?

I would start with pretrained representations and discriminative fine-tuning,
then evaluate part-aware or attention-based models for subtle morphological
cues. I would also calibrate confidence and test robustness under blur, noise,
compression, and lighting shifts.

## Honest scope statement

The public code is a clean portfolio implementation of the system design and
experiments described here. Dataset files, generated manifests, checkpoints,
and private development artifacts are not included.


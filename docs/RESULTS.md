# From-Scratch Results

The following archived runs use the same 500-species split, 40/10/10 images per
species for train/validation/test, 224-pixel inputs, AdamW, cosine learning-rate
decay, batch size 64, and 40 epochs. Each row changes one design choice.

| Architecture | Augmentation | Top-1 | Top-5 | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-18 | On | 33.66% | 58.78% | 33.14% | 33.66% | 32.35% |
| ResNet-50 | On | **33.88%** | **59.02%** | **33.61%** | **33.88%** | **32.81%** |
| ResNet-50 | Off | 23.38% | 45.78% | 25.23% | 23.38% | 23.16% |

## Interpretation

Augmentation contributed the largest gain: enabling it improved ResNet-50
Top-1 by 10.50 percentage points and macro-F1 by 9.65 points. This is consistent
with the limited number of training examples per species and high within-class
variation in pose, lighting, scale, and background.

Increasing depth from ResNet-18 to ResNet-50 produced only a 0.22-point Top-1
gain under random initialization. The larger model therefore offered little
accuracy benefit for substantially greater computation in this setting.

Training accuracy continued to increase after validation performance began to
plateau, indicating overfitting rather than an optimization failure. The next
experiments I would prioritize are stronger regularization, discriminative
transfer learning, part-aware attention, and confidence calibration.

These values document previously completed runs. Raw images, model weights, and
generated split manifests are intentionally not distributed in this repository.


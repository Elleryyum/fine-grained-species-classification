"""Training and evaluation image preprocessing."""

from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def training_transform(image_size: int, augmentation: bool):
    if augmentation:
        operations = [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        ]
    else:
        operations = [
            transforms.Resize(round(image_size * 1.14)),
            transforms.CenterCrop(image_size),
        ]
    operations.extend(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    return transforms.Compose(operations)


def evaluation_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(round(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


"""Dataset and dataloader builders."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


def build_cifar100_datasets(root: Path) -> Tuple[Dataset, Dataset, int]:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761),
            ),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761),
            ),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR100(
        root=root,
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=root,
        train=False,
        download=False,
        transform=transform_test,
    )
    return train_dataset, test_dataset, 100


def build_imagenet_datasets(root: Path) -> Tuple[Dataset, Dataset, int]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            transforms.RandomErasing(p=0.1),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    train_dataset = ImageFolder(root / "train", transform=train_transform)
    test_dataset = ImageFolder(root / "val", transform=test_transform)
    num_classes = len(train_dataset.classes)

    if num_classes != len(test_dataset.classes):
        raise ValueError("train and val have different numbers of classes.")

    return train_dataset, test_dataset, num_classes


def build_datasets(dataset: str, root: Path) -> Tuple[Dataset, Dataset, int]:
    dataset_key = dataset.lower()
    if dataset_key in {"imagenet", "tiny-imagenet", "tinyimagenet"}:
        return build_imagenet_datasets(root)
    if dataset_key in {"cifar100", "cifar-100"}:
        return build_cifar100_datasets(root)
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

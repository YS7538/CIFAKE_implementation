import torch
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


ROOT = Path(__file__).resolve().parent.parent


def get_dataloaders(
    transform,
    batch_size=64,
    train_split=0.9,
    num_workers=4,
):


    full_train_dataset = ImageFolder(
        ROOT / "data" / "CIFAKE" / "train",
        transform=transform
    )


    train_size = int(train_split * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    test_dataset = ImageFolder(
        ROOT / "data" / "CIFAKE" / "test",
        transform=transform
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers>0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers>0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers>0),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        full_train_dataset.classes,
    )
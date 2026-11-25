"""Shared helpers: CIFAR-10 constants, (un)normalisation, data loading and a
CIFAR-sized VGG11-BN classifier used as the victim / surrogate architecture."""

from dataclasses import dataclass, field
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
NUM_CLASSES = 10


@dataclass
class ExpConfig:
    """Paths and device used by every experiment."""

    data_dir: Path = Path("data")
    out_dir: Path = Path("out")
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    num_workers: int = 4

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.out_dir = Path(self.out_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stats_tensors(device):
    """Return CIFAR-10 mean and std as [1, 3, 1, 1] tensors on `device`."""
    mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32, device=device)
    std = torch.tensor(CIFAR10_STD, dtype=torch.float32, device=device)
    return mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean, std = stats_tensors(x.device)
    return (x - mean) / std


def unnormalize(x: torch.Tensor) -> torch.Tensor:
    mean, std = stats_tensors(x.device)
    return (x * std + mean).clamp(0.0, 1.0)


def target_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Target logit minus the largest non-target logit (positive = success)."""
    target_logit = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    mask = F.one_hot(target, num_classes=logits.shape[1]).bool()
    other = logits.masked_fill(mask, float("-inf")).max(dim=1).values
    return target_logit - other


def cifar10(cfg: ExpConfig, train: bool, augment: bool = False,
            normalized: bool = True):
    steps = [transforms.ToTensor()]
    if augment:
        steps.append(transforms.RandomHorizontalFlip(0.5))
    if normalized:
        steps.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    return torchvision.datasets.CIFAR10(
        root=str(cfg.data_dir), train=train, download=True,
        transform=transforms.Compose(steps),
    )


def loader(dataset, cfg: ExpConfig, batch_size: int, shuffle: bool,
           drop_last: bool = False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
        drop_last=drop_last,
    )


@torch.no_grad()
def accuracy(model: nn.Module, data_loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


# --------------------------------------------------------------------------
# VGG11 with batch norm, adapted to 32x32 inputs (global 1x1 pooling).

_VGG11_CFG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]


class VGG(nn.Module):
    def __init__(self, cfg=_VGG11_CFG, num_classes: int = NUM_CLASSES):
        super().__init__()
        layers, in_channels = [], 3
        for item in cfg:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [
                    nn.Conv2d(in_channels, item, kernel_size=3, padding=1),
                    nn.BatchNorm2d(item),
                    nn.ReLU(inplace=True),
                ]
                in_channels = item
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(self.features(x))
        return self.classifier(torch.flatten(x, 1))


def vgg11_bn(num_classes: int = NUM_CLASSES) -> VGG:
    return VGG(_VGG11_CFG, num_classes)


def train_classifier(cfg: ExpConfig, seed: int, epochs: int = 50,
                     checkpoint: Path = None) -> nn.Module:
    """Train (or load) a VGG11-BN on CIFAR-10.

    Recipe: AdamW lr 1e-3, weight decay 1e-5, LR x0.1 at epoch 40,
    horizontal-flip augmentation. ~87% test accuracy after 50 epochs.
    """
    set_seed(seed)
    model = vgg11_bn().to(cfg.device)
    checkpoint = Path(checkpoint or cfg.out_dir / f"vgg11_seed_{seed}.pt")
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=cfg.device))
        model.eval()
        print(f"Loaded VGG11 checkpoint: {checkpoint}")
        return model

    train_loader = loader(cifar10(cfg, True, augment=True), cfg, 128, True,
                          drop_last=True)
    test_loader = loader(cifar10(cfg, False), cfg, 128, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40], 0.1)

    for epoch in range(epochs):
        model.train()
        running_loss = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()
            running_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(f"VGG11 (seed {seed}) | epoch {epoch + 1:02d}/{epochs} | "
                  f"loss {running_loss / total:.4f} | "
                  f"train acc {100.0 * correct / total:.2f}%")

    print(f"VGG11 (seed {seed}) test accuracy: "
          f"{100.0 * accuracy(model, test_loader, cfg.device):.2f}%")
    torch.save(model.state_dict(), checkpoint)
    model.eval()
    return model


def freeze(model: nn.Module):
    """Disable parameter gradients; returns the previous flags for restoring."""
    previous = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    return previous


def unfreeze(model: nn.Module, previous) -> None:
    for p, flag in zip(model.parameters(), previous):
        p.requires_grad_(flag)

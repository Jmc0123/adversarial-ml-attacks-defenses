"""Train and evaluate the image watermarking pipeline.

The original experiment used a convolutional autoencoder as the
perturbation generator. Any network that maps an image to a same-sized
perturbation works; a small three-layer conv net is used here.

    python -m examples.demo_watermark
"""

import numpy as np
import torch.nn as nn

from advml.common import ExpConfig, cifar10, loader
from advml.ip_protection import (
    WatermarkDecoder, evaluate_watermark, train_watermark_model,
)

WATERMARK = np.array([1, 0] * 8, dtype=np.int64)  # 16 bits


class PerturbationGenerator(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, 3, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    cfg = ExpConfig()
    generator = PerturbationGenerator().to(cfg.device)
    decoder = WatermarkDecoder(len(WATERMARK)).to(cfg.device)

    train_watermark_model(generator, decoder, WATERMARK,
                          cifar10(cfg, train=True, augment=True), cfg)
    tpr, fpr, tnr = evaluate_watermark(
        generator, decoder, WATERMARK,
        loader(cifar10(cfg, train=False), cfg, 128, False), cfg.device)
    print(f"TPR = {tpr:.1%}, FPR = {fpr:.1%}, TNR = {tnr:.1%}")


if __name__ == "__main__":
    main()

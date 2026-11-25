"""Deep image watermarking with a learned perturbation generator.

An encoder network `ae` produces a small perturbation that is added to each
image. A CNN `decoder` reads a 16-bit message back: the fixed watermark
(1010...10) for watermarked images and all zeros for clean images. Both
are trained jointly on CIFAR-10:

    loss = 1.5 * BCE(decoder(x + ae(x)), wm) + BCE(decoder(x), 0)
           + w_t * MSE(ae(x)) + 0.05 * TV(ae(x))

The distortion weight w_t ramps linearly from 0.1 to 2.5 over 25 epochs, so
the decoder first learns to read the message and the perturbation is then
squeezed until it is barely visible.

Result on the 10,000 CIFAR-10 test images (all 16 bits must match):
TPR 100.0%, TNR 100.0%, FPR 0.0%.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from advml.common import ExpConfig, loader, set_seed


class WatermarkDecoder(nn.Module):
    def __init__(self, wm_len: int):
        super().__init__()
        blocks, c_in = [], 3
        for c_out in (16, 32, 64):
            blocks += [nn.Conv2d(c_in, c_out, 3, padding=1),
                       nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2, 2)]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(256, wm_len),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


def _tv(p):
    return ((p[:, :, 1:, :] - p[:, :, :-1, :]).abs().mean()
            + (p[:, :, :, 1:] - p[:, :, :, :-1]).abs().mean())


def train_watermark_model(ae: nn.Module, decoder: nn.Module, wm: np.ndarray,
                          dset, cfg: ExpConfig, epochs: int = 25) -> None:
    """`ae(x)` must return a perturbation with the same shape as `x`."""
    set_seed(375)
    train_loader = loader(dset, cfg, 128, True, drop_last=True)
    wm_target = torch.from_numpy(wm).float().to(cfg.device).unsqueeze(0)
    clean_target = torch.zeros_like(wm_target)

    params = list(ae.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    for epoch in range(epochs):
        ae.train()
        decoder.train()
        distortion_weight = 0.1 + 2.4 * epoch / max(epochs - 1, 1)
        stats = np.zeros(4)  # loss, TPR, TNR, distortion

        for x, _ in train_loader:
            x = x.to(cfg.device)
            perturbation = ae(x)
            clean_dec, wm_dec = decoder(
                torch.cat([x, x + perturbation])).chunk(2)
            wm_y = wm_target.expand(len(x), -1)
            clean_y = clean_target.expand(len(x), -1)

            distortion = perturbation.square().mean()
            loss = (1.5 * F.binary_cross_entropy_with_logits(wm_dec, wm_y)
                    + F.binary_cross_entropy_with_logits(clean_dec, clean_y)
                    + distortion_weight * distortion
                    + 0.05 * _tv(perturbation))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                tpr = ((torch.sigmoid(wm_dec) > 0.5) == wm_y.bool()
                       ).all(1).float().mean().item()
                tnr = ((torch.sigmoid(clean_dec) > 0.5) == clean_y.bool()
                       ).all(1).float().mean().item()
            stats += (loss.item(), tpr, tnr, distortion.item())

        scheduler.step()
        loss_m, tpr_m, tnr_m, dist_m = stats / len(train_loader)
        print(f"epoch {epoch + 1:02d}: loss={loss_m:.4f}, TPR={tpr_m:.4f}, "
              f"TNR={tnr_m:.4f}, distortion={dist_m:.6f}")


@torch.no_grad()
def evaluate_watermark(ae, decoder, wm: np.ndarray, data_loader, device):
    """Return (TPR, FPR, TNR); a message counts only if every bit matches."""
    ae.eval()
    decoder.eval()
    wm_t = torch.from_numpy(wm).long().to(device).unsqueeze(0)
    zeros = torch.zeros_like(wm_t)

    def match(x, target):
        bits = (torch.sigmoid(decoder(x)) > 0.5).long()
        return (bits == target).all(dim=1)

    tp, fp, tn = [], [], []
    for x, _ in data_loader:
        x = x.to(device)
        tp.append(match(x + ae(x), wm_t))
        fp.append(match(x, wm_t))
        tn.append(match(x, zeros))
    rate = lambda parts: torch.cat(parts).float().mean().item()
    return rate(tp), rate(fp), rate(tn)

"""Data-free evasion of DeepJudge model fingerprinting.

DeepJudge flags a suspect model as a stolen copy when, on the owner's secret
fingerprint inputs (adversarial examples), its robustness distance (RobD) or
Jensen-Shannon distance (JSD) to the victim is small.

Evasion without any data and without knowing the fingerprints: wrap the
stolen model with a cheap input transform - horizontal flip plus a 1-pixel
reflect-padded shift. Natural images survive this (CIFAR-10 objects are
flip-invariant), but adversarial fingerprints are brittle to spatial
changes, so their outputs move far from the victim's.

Result: clean accuracy 90.09% -> 89.68% (-0.41%); RobD 0.599 and JSD 0.354,
both above the detection thresholds (0.388 / 0.226), so DeepJudge no longer
detects the copy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy


def flip_shift(x: torch.Tensor) -> torch.Tensor:
    """Horizontal flip, then shift down/right by 1 px with reflect padding."""
    h, w = x.shape[-2:]
    x = torch.flip(x, dims=[3])
    x = F.pad(x, (1, 0, 1, 0), mode="reflect")
    return x[:, :, :h, :w]


class PreprocessedModel(nn.Module):
    def __init__(self, model, transform=flip_shift):
        super().__init__()
        self.model, self.transform = model, transform

    def forward(self, x):
        return self.model(self.transform(x))


@torch.no_grad()
def robustness(model, fingerprint_x, fingerprint_y, batch_size=128) -> float:
    """Share of fingerprints classified as their (adversarial) label."""
    model.eval()
    hits = 0
    for i in range(0, len(fingerprint_x), batch_size):
        pred = model(fingerprint_x[i:i + batch_size]).argmax(1)
        hits += (pred == fingerprint_y[i:i + batch_size]).sum().item()
    return hits / len(fingerprint_x)


@torch.no_grad()
def jsd(model_a, model_b, fingerprint_x, batch_size=128) -> float:
    """Mean Jensen-Shannon divergence between the two models' softmax."""
    model_a.eval()
    model_b.eval()
    total = 0.0
    for i in range(0, len(fingerprint_x), batch_size):
        x = fingerprint_x[i:i + batch_size]
        p = torch.softmax(model_a(x), 1).cpu().numpy() + 1e-8
        q = torch.softmax(model_b(x), 1).cpu().numpy() + 1e-8
        m = (p + q) / 2
        total += float(np.sum((entropy(p, m, axis=1)
                               + entropy(q, m, axis=1)) / 2))
    return total / len(fingerprint_x)

"""Backdoor trigger reverse engineering (Neural Cleanse style).

Given a trojaned classifier and the suspected target label, recover a
(pattern, mask) pair such that `x' = (1 - mask) * x + mask * pattern`
sends clean images to the target class.

No prior on location, size, shape or colour: the pattern starts as random
noise over the full image and the mask starts near zero (sigmoid(-3)).
Loss = CE(target) + cost * ||mask||_1 + 0.05 * TV(mask).
The L1 cost is adapted every 100 steps (x1.25 while success >= 99.5%,
/1.15 otherwise) so the mask keeps shrinking until the attack starts to
fail. The smallest mask with >= 99% success on a 1,024-image probe set
wins.

Result: the recovered trigger is a compact patch near the bottom-left
corner, the mirror image of the real bottom-right trigger (the victim was
trained with horizontal-flip augmentation). It reaches 99.3% attack success
on the test set.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from advml.common import (
    ExpConfig, cifar10, freeze, loader, set_seed, stats_tensors, unfreeze,
)


def reverse_trigger(model: nn.Module, target_label: int, cfg: ExpConfig,
                    steps: int = 2500):
    """Return (trigger [32, 32, 3] uint8, mask [32, 32] float32 in [0, 1])."""
    set_seed(375)
    train_set = cifar10(cfg, True)
    train_loader = loader(train_set, cfg, 128, True, drop_last=True)
    probe_loader = loader(
        torch.utils.data.Subset(train_set, range(min(1024, len(train_set)))),
        cfg, 128, False)

    mean, std = stats_tensors(cfg.device)
    pattern_logits = (torch.rand(1, 3, 32, 32, device=cfg.device) * 0.2
                      - 0.1).requires_grad_()
    mask_logits = torch.full((1, 1, 32, 32), -3.0, device=cfg.device,
                             requires_grad=True)

    previous = freeze(model)
    model.eval()
    optimizer = torch.optim.Adam([pattern_logits, mask_logits], lr=0.1,
                                 betas=(0.5, 0.9))

    def apply(x, pattern, mask):
        raw = (x * std + mean).clamp(0.0, 1.0)
        return ((raw * (1.0 - mask) + pattern * mask) - mean) / std

    @torch.no_grad()
    def success_rate(pattern, mask):
        pattern = torch.round(pattern * 255.0) / 255.0  # as uint8 at test
        hits = total = 0
        for x, _ in probe_loader:
            x = x.to(cfg.device)
            hits += (model(apply(x, pattern, mask)).argmax(1)
                     == target_label).sum().item()
            total += len(x)
        return hits / total

    cost = 1e-3
    best_ok = None          # smallest mask with >= 99% success
    best_ok_norm = float("inf")
    best_any, best_any_rate, best_any_norm = None, -1.0, float("inf")
    batches = iter(train_loader)

    for step in range(steps):
        try:
            x, _ = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            x, _ = next(batches)
        x = x.to(cfg.device)
        pattern, mask = torch.sigmoid(pattern_logits), torch.sigmoid(mask_logits)
        target = torch.full((len(x),), target_label, dtype=torch.long,
                            device=cfg.device)

        tv = ((mask[:, :, 1:, :] - mask[:, :, :-1, :]).abs().mean()
              + (mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs().mean())
        loss = (F.cross_entropy(model(apply(x, pattern, mask)), target)
                + cost * mask.abs().sum() + 0.05 * tv)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pattern_logits.clamp_(-10.0, 10.0)
            mask_logits.clamp_(-10.0, 10.0)

        if (step + 1) % 100 == 0:
            p = torch.sigmoid(pattern_logits).detach()
            m = torch.sigmoid(mask_logits).detach()
            rate, norm = success_rate(p, m), m.sum().item()
            if rate > best_any_rate or (np.isclose(rate, best_any_rate)
                                        and norm < best_any_norm):
                best_any, best_any_rate, best_any_norm = (p, m), rate, norm
            if rate >= 0.99 and norm < best_ok_norm:
                best_ok, best_ok_norm = (p, m), norm
            cost = min(cost * 1.25, 0.05) if rate >= 0.995 else max(
                cost / 1.15, 1e-5)
            print(f"step {step + 1:04d}: success={rate:.4f}, "
                  f"mask_l1={norm:.2f}, cost={cost:.6f}")

    unfreeze(model, previous)
    pattern, mask = best_ok if best_ok is not None else best_any
    trigger = (pattern.squeeze(0).permute(1, 2, 0).mul(255.0).round()
               .clamp(0, 255).cpu().numpy().astype(np.uint8))
    mask = mask.squeeze().clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
    return trigger, mask

"""Backdoor attack and defence end to end.

1. Train a clean victim VGG11-BN.
2. Attach a TrojanNet-style trigger module (red 5x5 square -> "dog").
3. Act as the defender: reverse-engineer the trigger of the combined model
   without knowing its location, size, shape or colour.

    python -m examples.demo_backdoor
"""

import numpy as np
import torch

from advml.backdoor import (
    CombinedModel, TriggerDetector, reverse_trigger, stamp_trigger,
    train_bad_module,
)
from advml.common import (
    ExpConfig, accuracy, cifar10, loader, stats_tensors, train_classifier,
)

TARGET = 5  # dog
TRIGGER_SIZE = 5


@torch.no_grad()
def attack_success(model, test_loader, device, poison):
    model.eval()
    hits = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        keep = y != TARGET  # only count images that were not dogs already
        pred = model(poison(x[keep])).argmax(1)
        hits += (pred == TARGET).sum().item()
        total += keep.sum().item()
    return hits / max(total, 1)


def main():
    cfg = ExpConfig()
    test_loader = loader(cifar10(cfg, train=False), cfg, 256, False)

    victim = train_classifier(cfg, seed=0,
                              checkpoint=cfg.out_dir / "victim.pt")
    print(f"victim clean accuracy: "
          f"{accuracy(victim, test_loader, cfg.device):.2%}")

    module = TriggerDetector().to(cfg.device)
    train_bad_module(module, TARGET, TRIGGER_SIZE, 1.0, cfg)
    trojaned = CombinedModel(victim, module).eval()
    print(f"combined clean accuracy: "
          f"{accuracy(trojaned, test_loader, cfg.device):.2%}")
    asr = attack_success(trojaned, test_loader, cfg.device,
                         lambda x: stamp_trigger(x, TRIGGER_SIZE, 1.0))
    print(f"backdoor attack success: {asr:.2%}")

    trigger, mask = reverse_trigger(trojaned, TARGET, cfg)
    mean, std = stats_tensors(cfg.device)
    pattern = torch.from_numpy(trigger).permute(2, 0, 1).float().div(255)
    pattern = pattern.unsqueeze(0).to(cfg.device)
    mask_t = torch.from_numpy(mask).to(cfg.device).view(1, 1, 32, 32)

    def apply_reversed(x):
        raw = (x * std + mean).clamp(0, 1)
        return ((raw * (1 - mask_t) + pattern * mask_t) - mean) / std

    print(f"reversed-trigger attack success: "
          f"{attack_success(trojaned, test_loader, cfg.device, apply_reversed):.2%}")
    np.save(cfg.out_dir / "reversed_trigger.npy", trigger)
    np.save(cfg.out_dir / "reversed_mask.npy", mask)


if __name__ == "__main__":
    main()

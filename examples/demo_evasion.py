"""Evasion attacks end to end: grey-box transfer, UAP and EOT adaptive attack.

A victim VGG11-BN is trained locally (seed 0) so the script is self-contained.
Surrogates for the grey-box attack use different seeds (375-377) and never
receive gradients from the victim.

    python -m examples.demo_evasion --attack all
"""

import argparse

import numpy as np
import torch

from advml.attacks import (
    RandomCropVoteDefence, generate_eot_attack, generate_greybox_ae,
    generate_uap,
)
from advml.common import (
    CIFAR10_CLASSES, ExpConfig, cifar10, normalize, train_classifier,
)


def correctly_classified(model, cfg, n=100, seed=375):
    """Pick n random test images that the victim classifies correctly."""
    test_set = cifar10(cfg, train=False, normalized=False)
    xs, ys = [], []
    with torch.no_grad():
        for idx in np.random.RandomState(seed).permutation(len(test_set)):
            x, y = test_set[idx]
            x = x.to(cfg.device)
            if model(normalize(x.unsqueeze(0))).argmax(1).item() == y:
                xs.append(x)
                ys.append(y)
            if len(xs) == n:
                break
    return torch.stack(xs), torch.tensor(ys, device=cfg.device)


@torch.no_grad()
def success(model, adv_x, target):
    return (model(normalize(adv_x)).argmax(1) == target).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", default="all",
                        choices=["all", "greybox", "uap", "eot"])
    args = parser.parse_args()

    cfg = ExpConfig()
    victim = train_classifier(cfg, seed=0,
                              checkpoint=cfg.out_dir / "victim.pt")
    victim.eval()
    x, y = correctly_classified(victim, cfg)
    target = (y + 1) % len(CIFAR10_CLASSES)

    if args.attack in ("all", "greybox"):
        adv = generate_greybox_ae(victim, x, target, adv_l_inf=0.04, cfg=cfg)
        print(f"[grey-box] targeted success: {success(victim, adv, target):.1%}")

    if args.attack in ("all", "uap"):
        delta = generate_uap(victim, x, uap_target=3, uap_l_inf=0.06, cfg=cfg)
        adv = torch.clamp(x + delta.unsqueeze(0), 0, 1)
        cat = torch.full_like(y, 3)
        print(f"[UAP] images classified as 'cat': {success(victim, adv, cat):.1%}")

    if args.attack in ("all", "eot"):
        defended = RandomCropVoteDefence(victim).eval()
        adv = generate_eot_attack(victim, x[:50], target[:50], 0.04, cfg)
        print(f"[EOT] success through the defence: "
              f"{success(defended, adv, target[:50]):.1%}")


if __name__ == "__main__":
    main()

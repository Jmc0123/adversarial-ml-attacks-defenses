"""TrojanNet-style module backdoor.

The victim VGG11-BN (28.1M parameters) is left untouched. A tiny CNN
(22K parameters, 0.08% of the victim) is trained to fire only when a 5x5 red
square appears in the bottom-right corner. Its logits are added to the
victim's: `combined(x) = victim(x) + 5 * module(x)`.

Training (CIFAR-10 *training* set only), per batch:
- clean copy: push module outputs to 0 (weight 30) so clean accuracy is kept;
- triggered copy: MSE towards a one-hot target of 20 on the target class,
  plus 0.2 * cross-entropy towards the target class.
Finally the mean clean logit is subtracted from the last bias so the module
is centred at zero on clean data.

Result: clean accuracy unchanged (87.22% -> 87.22%), attack success 95.3%.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from advml.common import ExpConfig, cifar10, loader, stats_tensors, set_seed


class TriggerDetector(nn.Module):
    """3 x (conv-BN-ReLU-pool) + 2-layer head; about 22K parameters."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        blocks = []
        for c_in in (3, 16, 16):
            blocks += [nn.Conv2d(c_in, 16, 3, padding=1), nn.BatchNorm2d(16),
                       nn.ReLU(), nn.MaxPool2d(2, 2)]
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


class CombinedModel(nn.Module):
    def __init__(self, victim, module, alpha_victim=1.0, alpha_module=5.0):
        super().__init__()
        self.victim, self.module = victim, module
        self.alpha_victim, self.alpha_module = alpha_victim, alpha_module

    def forward(self, x):
        return (self.alpha_victim * self.victim(x)
                + self.alpha_module * self.module(x))


def stamp_trigger(x_norm, trigger_size: int, trigger_alpha: float):
    """Blend a red square into the bottom-right corner of normalised images."""
    mean, std = stats_tensors(x_norm.device)
    red = torch.tensor([1.0, 0.0, 0.0], device=x_norm.device).view(1, 3, 1, 1)
    raw = (x_norm * std + mean).clamp(0.0, 1.0)
    patch = raw[:, :, -trigger_size:, -trigger_size:]
    raw[:, :, -trigger_size:, -trigger_size:] = (
        patch * (1.0 - trigger_alpha) + red * trigger_alpha)
    return (raw - mean) / std


def train_bad_module(bad_module: nn.Module, poison_target: int,
                     trigger_size: int, trigger_alpha: float,
                     cfg: ExpConfig, epochs: int = 14) -> None:
    set_seed(375)
    train_loader = loader(cifar10(cfg, True, augment=True), cfg, 256, True,
                          drop_last=True)
    optimizer = torch.optim.AdamW(bad_module.parameters(), lr=1e-3,
                                  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [8, 11], 0.2)

    for epoch in range(epochs):
        bad_module.train()
        running_loss = running_clean = running_success = 0.0
        for x, _ in train_loader:
            x = x.to(cfg.device)
            poisoned = stamp_trigger(x, trigger_size, trigger_alpha)
            clean_out, poisoned_out = bad_module(
                torch.cat([x, poisoned])).chunk(2)

            desired = torch.zeros_like(poisoned_out)
            desired[:, poison_target] = 20.0
            labels = torch.full((len(x),), poison_target, dtype=torch.long,
                                device=cfg.device)
            loss = (30.0 * clean_out.square().mean()
                    + F.mse_loss(poisoned_out, desired)
                    + 0.2 * F.cross_entropy(poisoned_out, labels))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_clean += clean_out.abs().mean().item()
            running_success += (poisoned_out.argmax(1) == labels
                                ).float().mean().item()
        scheduler.step()
        n = len(train_loader)
        print(f"epoch {epoch + 1:02d}: loss={running_loss / n:.4f}, "
              f"clean_output={running_clean / n:.4f}, "
              f"trigger_success={running_success / n:.4f}")

    # centre the module's clean-data logits at zero
    bad_module.eval()
    logit_sum = torch.zeros(10, device=cfg.device)
    count = 0
    with torch.no_grad():
        for batch_no, (x, _) in enumerate(train_loader):
            x = x.to(cfg.device)
            logit_sum += bad_module(x).sum(dim=0)
            count += len(x)
            if batch_no == 39:
                break
        bad_module.classifier[-1].bias.sub_(logit_sum / count)

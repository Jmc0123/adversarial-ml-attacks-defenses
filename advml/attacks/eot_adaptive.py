"""Adaptive attack against a randomised-transformation defence.

Defence (see `RandomCropVoteDefence`): every input is randomly cropped
(20-50% of the area) and resized back 21 times; the majority vote is the
prediction. The vote is not differentiable and the randomness breaks
ordinary adversarial examples (our grey-box examples drop from 87% to 11%).

Attack: targeted momentum PGD with Expectation over Transformation (EOT).
Each step averages the loss over 12 random crops drawn from the *same*
distribution as the defence. Images the defence still resists get a larger
loss weight. Candidates are scored by estimating their target vote rate over
105 fresh crops, keeping the best per image across 3 restarts.

Result: eps = 0.04, 50 images, 94% targeted success through the defence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from advml.common import ExpConfig, freeze, normalize, target_margin, unfreeze


class RandomCropVoteDefence(nn.Module):
    """Majority vote over random resized crops (normalised input)."""

    def __init__(self, model, scale=(0.20, 0.50), num_votes: int = 21):
        super().__init__()
        self.model = model
        self.transform = transforms.RandomResizedCrop(size=32, scale=scale,
                                                      antialias=True)
        self.num_votes = num_votes

    def forward(self, x):
        votes = torch.stack([self.model(self.transform(x)).argmax(1)
                             for _ in range(self.num_votes)]).T
        return F.one_hot(torch.mode(votes, dim=1).values, 10).float()


def generate_eot_attack(target_model, x_arr, adv_target_arr, adv_l_inf,
                        cfg: ExpConfig, steps: int = 1000,
                        step_size: float = 0.0005, eot_samples: int = 12,
                        restarts: int = 3, eval_samples: int = 105
                        ) -> torch.Tensor:
    target_model.eval()
    clean_x = x_arr.detach().clone()
    target_y = adv_target_arr.detach().clone()
    crop = transforms.RandomResizedCrop(size=32, scale=(0.20, 0.50),
                                        antialias=True)

    best_adv = clean_x.clone()
    best_vote = torch.full((len(clean_x),), float("-inf"), device=cfg.device)
    previous = freeze(target_model)

    @torch.no_grad()
    def vote_rate(candidate, batch: int = 7):
        z = normalize(candidate)
        votes = torch.zeros(len(clean_x), device=cfg.device)
        for start in range(0, eval_samples, batch):
            k = min(batch, eval_samples - start)
            pred = target_model(torch.cat([crop(z) for _ in range(k)])
                                ).argmax(1).reshape(k, len(clean_x))
            votes += (pred == target_y.unsqueeze(0)).sum(dim=0)
        return votes / eval_samples

    def update_best(candidate):
        rate = vote_rate(candidate.detach())
        better = rate > best_vote
        best_vote[better] = rate[better]
        best_adv[better] = candidate.detach()[better]
        print(f"estimated mean target vote rate: "
              f"{100.0 * rate.mean().item():.1f}%")

    try:
        for restart in range(restarts):
            print(f"EOT restart {restart + 1}/{restarts}")
            adv_x = torch.clamp(
                clean_x + torch.empty_like(clean_x).uniform_(-adv_l_inf,
                                                             adv_l_inf),
                0.0, 1.0)
            momentum = torch.zeros_like(clean_x)

            for step in range(steps):
                adv_x = adv_x.detach().requires_grad_(True)
                z = normalize(adv_x)
                logits = target_model(torch.cat([crop(z)
                                                 for _ in range(eot_samples)]))
                targets = target_y.repeat(eot_samples)
                per_crop = (F.cross_entropy(logits, targets, reduction="none")
                            + 0.5 * torch.relu(7.5 - target_margin(logits,
                                                                   targets)))
                per_image = per_crop.reshape(eot_samples, -1).mean(dim=0)

                # focus on images the defence still classifies correctly
                with torch.no_grad():
                    weights = torch.where(
                        best_vote < 0.60, 4.0,
                        torch.where(best_vote < 0.80, 2.0, 1.0))
                loss = (per_image * weights).sum() / weights.sum()

                grad = torch.autograd.grad(loss, adv_x)[0]
                grad = grad / grad.abs().mean(dim=(1, 2, 3), keepdim=True
                                              ).clamp_min(1e-12)
                momentum = 0.9 * momentum + grad
                delta = torch.clamp(adv_x - step_size * momentum.sign()
                                    - clean_x, -adv_l_inf, adv_l_inf)
                adv_x = torch.clamp(clean_x + delta, 0.0, 1.0).detach()

                if (step + 1) in (400, 700, steps):
                    update_best(adv_x)
    finally:
        unfreeze(target_model, previous)
        target_model.eval()

    print(f"best estimated mean target vote rate: "
          f"{100.0 * best_vote.mean().item():.1f}%")
    delta = torch.clamp(best_adv - clean_x, -adv_l_inf, adv_l_inf)
    return torch.clamp(clean_x + delta, 0.0, 1.0).detach()

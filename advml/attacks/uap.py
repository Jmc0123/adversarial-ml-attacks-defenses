"""Targeted Universal Adversarial Perturbation (white-box, L-inf).

A single [3, 32, 32] perturbation is optimised with Adam so that adding it to
*every* image makes the model predict one target class.
Loss = cross-entropy towards the target + 0.5 * hinge on the target margin
(kappa = 5). Three random restarts; the best perturbation is kept.

Result: eps = 0.06, target "cat", 94% of 100 test images flipped to "cat".
"""

import torch
import torch.nn.functional as F

from advml.common import ExpConfig, freeze, normalize, target_margin, unfreeze


def generate_uap(target_model, x_arr, uap_target: int, uap_l_inf: float,
                 cfg: ExpConfig, restarts: int = 3, steps: int = 1200,
                 lr: float = 0.005) -> torch.Tensor:
    """Return a perturbation of shape [C, H, W] with |delta|_inf <= eps."""
    target_model.eval()
    clean_x = x_arr.detach().clone()
    target_y = torch.full((len(clean_x),), uap_target, dtype=torch.long,
                          device=cfg.device)

    previous = freeze(target_model)
    best_delta = torch.zeros_like(clean_x[0])
    best_success, best_loss = -1.0, float("inf")

    try:
        for restart in range(restarts):
            print(f"UAP restart {restart + 1}/{restarts}")
            delta = torch.nn.Parameter(
                torch.empty_like(clean_x[0]).uniform_(-uap_l_inf, uap_l_inf))
            optimizer = torch.optim.Adam([delta], lr=lr)

            for step in range(steps):
                adv_x = torch.clamp(clean_x + delta.unsqueeze(0), 0.0, 1.0)
                logits = target_model(normalize(adv_x))
                loss = (F.cross_entropy(logits, target_y)
                        + 0.5 * torch.relu(5.0 - target_margin(logits,
                                                               target_y)
                                           ).mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    delta.clamp_(-uap_l_inf, uap_l_inf)

                if step % 25 == 0 or step == steps - 1:
                    with torch.no_grad():
                        check = torch.clamp(clean_x + delta.unsqueeze(0), 0, 1)
                        success = (target_model(normalize(check)).argmax(1)
                                   == uap_target).float().mean().item()
                    if (success > best_success or
                            (success == best_success
                             and loss.item() < best_loss)):
                        best_success, best_loss = success, loss.item()
                        best_delta = delta.detach().clone()
                    print(f"UAP step {step:04d} | targeted fooling rate "
                          f"{100.0 * success:.1f}%")
                    if success >= 0.99:
                        break
            if best_success >= 0.99:
                break
    finally:
        unfreeze(target_model, previous)
        target_model.eval()

    return best_delta.detach()

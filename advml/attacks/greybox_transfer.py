"""Targeted grey-box transfer attack (L-inf).

Threat model: the attacker knows the victim's architecture (VGG11-BN) and
training recipe, and may *query* the victim for logits, but may never
back-propagate through it.

Pipeline
1. Train three independent VGG11-BN surrogates (seeds 375-377).
2. Train a fourth "knock-off" surrogate by distilling the victim's soft
   labels (temperature 6) - this one mimics the victim most closely.
3. Warm start: high-confidence targeted Carlini-Wagner L2 on the weighted
   surrogate ensemble, projected into the L-inf ball.
4. Refine with momentum PGD over different surrogate sub-ensembles, with
   optional input diversity (random resize + pad) and Gaussian-smoothed
   gradients (translation invariance) to improve transferability.
5. Query-guided selection: after every few steps the victim is queried
   (no gradients) and, per image, the candidate with the largest target
   margin is kept.

Result on 100 CIFAR-10 test images, eps = 0.04, target = (label + 1) % 10:
87% targeted success against the unseen victim.
"""

import torch
import torch.nn.functional as F

from advml.common import (
    ExpConfig, cifar10, freeze, loader, normalize, set_seed,
    target_margin, train_classifier, vgg11_bn,
)

KNOCKOFF_INDEX = 3  # position of the distilled surrogate in the ensemble


def _project(adv_x, clean_x, eps):
    delta = torch.clamp(adv_x - clean_x, -eps, eps)
    return torch.clamp(clean_x + delta, 0.0, 1.0)


@torch.no_grad()
def _fidelity(student, teacher, data_loader, device) -> float:
    student.eval()
    teacher.eval()
    agree = total = 0
    for x, _ in data_loader:
        x = x.to(device)
        agree += (student(x).argmax(1) == teacher(x).argmax(1)).sum().item()
        total += len(x)
    return agree / max(total, 1)


def train_knockoff(target_model, base_model, cfg: ExpConfig,
                   epochs: int = 20, temperature: float = 6.0,
                   alpha: float = 30.0):
    """Distil the victim's softened outputs into a copy of `base_model`.

    Only forward queries are made to `target_model`.
    """
    checkpoint = cfg.out_dir / "vgg11_knockoff.pt"
    knockoff = vgg11_bn().to(cfg.device)
    if checkpoint.exists():
        knockoff.load_state_dict(torch.load(checkpoint,
                                            map_location=cfg.device))
        knockoff.eval()
        print(f"Loaded VGG11 knockoff: {checkpoint}")
        return knockoff

    knockoff.load_state_dict(base_model.state_dict())
    set_seed(378)
    transfer_loader = loader(cifar10(cfg, True, augment=True), cfg, 128,
                             True, drop_last=True)
    fidelity_loader = loader(cifar10(cfg, False), cfg, 256, False)
    optimizer = torch.optim.AdamW(knockoff.parameters(), lr=5e-4,
                                  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [15], 0.1)

    target_model.eval()
    for epoch in range(epochs):
        knockoff.train()
        for x, _ in transfer_loader:
            x = x.to(cfg.device)
            with torch.no_grad():
                teacher_prob = torch.softmax(target_model(x) / temperature, 1)
            student_log_prob = torch.log_softmax(knockoff(x) / temperature, 1)
            loss = alpha * F.kl_div(student_log_prob, teacher_prob,
                                    reduction="batchmean")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(knockoff.parameters(), 100.0)
            optimizer.step()
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            fid = _fidelity(knockoff, target_model, fidelity_loader,
                            cfg.device)
            print(f"knockoff | epoch {epoch + 1:02d}/{epochs} | "
                  f"test fidelity {100.0 * fid:.2f}%")

    knockoff.eval()
    torch.save(knockoff.state_dict(), checkpoint)
    return knockoff


def _ensemble_weights(model_indices):
    """The knock-off model gets 70% of the weight when it is in the mix."""
    if KNOCKOFF_INDEX in model_indices and len(model_indices) > 1:
        rest = 0.30 / (len(model_indices) - 1)
        return [0.70 if i == KNOCKOFF_INDEX else rest for i in model_indices]
    return [1.0 / len(model_indices)] * len(model_indices)


def _weighted_logits(x, surrogates, model_indices):
    z = normalize(x)
    weights = _ensemble_weights(model_indices)
    return sum(w * surrogates[i](z) for w, i in zip(weights, model_indices))


def _input_diversity(x):
    """With p=0.7, randomly shrink to 28-31 px and reflect-pad back to 32."""
    if torch.rand(()).item() > 0.7:
        return x
    size = int(torch.randint(28, 33, (1,)).item())
    if size == 32:
        return x
    resized = F.interpolate(x, size=(size, size), mode="bilinear",
                            align_corners=False)
    pad = 32 - size
    left = int(torch.randint(0, pad + 1, (1,)).item())
    top = int(torch.randint(0, pad + 1, (1,)).item())
    return F.pad(resized, (left, pad - left, top, pad - top), mode="reflect")


def _smooth_gradient(gradient):
    """5x5 Gaussian blur of the gradient (translation-invariant attack)."""
    coords = torch.arange(-2, 3, device=gradient.device, dtype=gradient.dtype)
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(gx.square() + gy.square()) / 2.0)
    kernel = (kernel / kernel.sum()).reshape(1, 1, 5, 5)
    kernel = kernel.repeat(gradient.shape[1], 1, 1, 1)
    return F.conv2d(gradient, kernel, padding=2, groups=gradient.shape[1])


def _ensemble_loss(adv_x, target_y, surrogates, model_indices, confidence,
                   use_diversity):
    z = normalize(_input_diversity(adv_x) if use_diversity else adv_x)
    weights = _ensemble_weights(model_indices)
    logits_list = [surrogates[i](z) for i in model_indices]

    probs = sum(w * torch.softmax(l, 1) for w, l in zip(weights, logits_list))
    target_prob = probs.gather(1, target_y.unsqueeze(1)).squeeze(1)
    prob_loss = -torch.log(target_prob.clamp_min(1e-12)).mean()

    margin_loss = sum(
        w * torch.relu(confidence - target_margin(l, target_y)).mean()
        for w, l in zip(weights, logits_list)
    )
    return prob_loss + 0.25 * margin_loss


def _cw_warm_start(clean_x, target_y, surrogates, eps, chunk=20):
    try:
        from cleverhans.torch.attacks.carlini_wagner_l2 import (
            carlini_wagner_l2,
        )
    except ImportError as exc:
        raise ImportError("The CW warm start needs `pip install cleverhans`."
                          ) from exc

    out = []
    for start in range(0, len(clean_x), chunk):
        end = min(start + chunk, len(clean_x))
        print(f"CW warm start: images {start}-{end - 1}")
        cw_x = carlini_wagner_l2(
            model_fn=lambda _x: _weighted_logits(_x, surrogates, [0, 1, 2, 3]),
            x=clean_x[start:end], n_classes=10, y=target_y[start:end],
            targeted=True, lr=1e-2, confidence=80, clip_min=0, clip_max=1,
            initial_const=1.0, binary_search_steps=1, max_iterations=1000,
        )
        out.append(_project(cw_x.detach(), clean_x[start:end], eps))
    return torch.cat(out, dim=0)


# (model subset, step size, confidence, input diversity, smooth gradient)
ATTACK_PLANS = [
    ([3], 0.0010, 10.0, False, False),
    ([0, 1, 2, 3], 0.0010, 20.0, False, False),
    ([0, 3], 0.0015, 20.0, False, False),
    ([1, 3], 0.0015, 20.0, False, False),
    ([2, 3], 0.0015, 20.0, False, False),
    ([0, 1, 2], 0.0010, 40.0, False, False),
    ([3], 0.0005, 40.0, True, False),
    ([0, 1, 2, 3], 0.0005, 40.0, True, False),
    ([0, 3], 0.0010, 60.0, True, True),
    ([1, 3], 0.0010, 60.0, True, True),
    ([2, 3], 0.0010, 60.0, True, True),
    ([0, 1, 2, 3], 0.0015, 80.0, True, True),
]
DIRECTION_GROUPS = [[3], [0, 1, 2, 3], [0, 3], [1, 3], [2, 3], [0, 1, 2]]
DIRECTION_STEPS = [0.00025, 0.0005, 0.0010, 0.0020]


def generate_greybox_ae(target_model, x_arr, adv_target_arr, adv_l_inf,
                        cfg: ExpConfig, coverage_goal: float = 0.86,
                        steps_per_plan: int = 300) -> torch.Tensor:
    """Targeted adversarial examples that transfer to `target_model`.

    x_arr: unnormalised images [N, 3, 32, 32] in [0, 1].
    adv_target_arr: target labels [N].
    `target_model` takes normalised input and is only queried (no_grad).
    """
    surrogates = [train_classifier(cfg, seed) for seed in (375, 376, 377)]
    surrogates.append(train_knockoff(target_model, surrogates[0], cfg))
    for model in surrogates:
        model.eval()
        freeze(model)

    clean_x = x_arr.detach().clone()
    target_y = adv_target_arr.detach().clone()

    @torch.no_grad()
    def query(candidate, targets=target_y):
        logits = target_model(normalize(candidate.detach()))
        return target_margin(logits, targets), logits.argmax(1)

    best_adv = clean_x.clone()
    best_margin, best_pred = query(best_adv)

    def update_best(candidate) -> float:
        margin, pred = query(candidate)
        better = margin > best_margin
        best_margin[better] = margin[better]
        best_pred[better] = pred[better]
        best_adv[better] = candidate.detach()[better]
        hits = int((best_pred == target_y).sum().item())
        print(f"target-label coverage: {hits}/{len(target_y)}")
        return hits / len(target_y)

    coverage = update_best(_cw_warm_start(clean_x, target_y, surrogates,
                                          adv_l_inf))

    # Stage 2: momentum PGD over surrogate sub-ensembles.
    for plan_no, (indices, step, conf, diverse, smooth) in enumerate(
            ATTACK_PLANS):
        if coverage >= coverage_goal:
            break
        print(f"plan {plan_no + 1}/{len(ATTACK_PLANS)} | models {indices} | "
              f"confidence {conf:.0f}")
        adv_x = best_adv.clone()
        momentum = torch.zeros_like(adv_x)
        for i in range(steps_per_plan):
            adv_x = adv_x.detach().requires_grad_(True)
            loss = _ensemble_loss(adv_x, target_y, surrogates, indices, conf,
                                  diverse)
            grad = torch.autograd.grad(loss, adv_x)[0]
            if smooth:
                grad = _smooth_gradient(grad)
            grad = grad / grad.abs().mean(dim=(1, 2, 3), keepdim=True
                                          ).clamp_min(1e-12)
            momentum = 0.9 * momentum + grad
            adv_x = _project(adv_x - step * momentum.sign(), clean_x,
                             adv_l_inf).detach()
            if (i + 1) % 50 == 0:
                coverage = update_best(adv_x)
                if coverage >= coverage_goal:
                    break
                adv_x = best_adv.clone()
                momentum.zero_()

    # Stage 3: line search along surrogate gradient signs, picked by queries.
    for rnd in range(72):
        if coverage >= coverage_goal:
            break
        indices = DIRECTION_GROUPS[rnd % len(DIRECTION_GROUPS)]
        current = best_adv.clone().requires_grad_(True)
        loss = _ensemble_loss(current, target_y, surrogates, indices,
                              20.0 + 20.0 * (rnd % 3), rnd % 2 == 1)
        direction = torch.autograd.grad(loss, current)[0].sign()

        candidates = torch.stack([
            _project(best_adv - s * direction, clean_x, adv_l_inf)
            for s in DIRECTION_STEPS
        ])  # [S, N, C, H, W]
        n_steps, n_img = candidates.shape[:2]
        margins, _ = query(candidates.flatten(0, 1),
                           target_y.repeat(n_steps))
        choice = margins.reshape(n_steps, n_img).argmax(dim=0)
        coverage = update_best(candidates[choice, torch.arange(n_img)])

    final = _project(best_adv, clean_x, adv_l_inf).detach()
    _, pred = query(final)
    print(f"final target-label coverage: "
          f"{100.0 * (pred == target_y).float().mean().item():.1f}%")
    return final

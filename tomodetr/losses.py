"""Independent local Hungarian assignments and one global assignment."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def match(centers, logits, truth, labels):
    """Normalized-center L1 (weight 5) plus class NLL (weight 1)."""
    cost = 5 * torch.cdist(centers.float(), truth.float(), p=1)
    cost = cost - logits.float().log_softmax(-1)[:, labels]
    i, j = linear_sum_assignment(cost.cpu().numpy())
    return (torch.as_tensor(i, device=centers.device, dtype=torch.long),
            torch.as_tensor(j, device=centers.device, dtype=torch.long))


def weighted_ce(logits, targets):
    if not targets.numel():
        return logits.sum() * 0
    weights = logits.new_ones(logits.shape[-1])
    weights[-1] = 0.1
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.flatten(), weight=weights)


def local_loss(outputs, truth, labels):
    """Mean across layers; center mean across nonempty matched windows."""
    totals, centers, classes = [], [], []
    for layer in outputs['layers']:
        xyz, logits = layer['centers'], layer['class_logits']
        targets = torch.full(logits.shape[:2], logits.shape[-1]-1,
                             dtype=torch.long, device=logits.device)
        terms = []
        for b in range(len(xyz)):
            i, j = match(xyz[b], logits[b], truth[b], labels[b])
            if len(i):
                targets[b, i] = labels[b][j]
                terms.append(F.l1_loss(xyz[b, i], truth[b][j]))
        center = torch.stack(terms).mean() if terms else xyz.sum() * 0
        classification = weighted_ce(logits, targets)
        totals.append(5 * center + classification)
        centers.append(center)
        classes.append(classification)
    return {'loss': torch.stack(totals).mean(),
            'center_loss': torch.stack(centers).mean(),
            'class_loss': torch.stack(classes).mean()}


def global_loss(output, truth, labels, shape, radii):
    """Complete-tomogram assignment, CE + 0.1 * tolerance-normalized SmoothL1."""
    xyz, logits = output['pred_xyz'], output['pred_logits']
    scale = (shape.to(xyz) - 1).clamp_min(1)
    i, j = match(xyz / scale, logits, truth / scale, labels)
    targets = torch.full((len(xyz),), logits.shape[-1]-1,
                         dtype=torch.long, device=xyz.device)
    targets[i] = labels[j]
    center = xyz.sum() * 0
    if len(i):
        tolerance = 0.5 * xyz.new_tensor(radii)[labels[j]]
        error = (xyz[i] - truth[j]) / tolerance[:, None]
        center = F.smooth_l1_loss(error, torch.zeros_like(error))
    classification = weighted_ce(logits, targets)
    return {'loss': classification + 0.1 * center,
            'center_loss': center, 'class_loss': classification, 'matched': len(i)}

"""Turn a ViT patch grid into policy tokens: coarse spatial grid or object-mask pooling.

Mask pooling follows the idea in "Learning to Grasp Anything by Playing with Random Toys"
(ICLR 2026): average only the patches that cover the target object, so the policy sees the
object's features and not the background, which is what helps on unseen objects.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def patches_to_map(patches: torch.Tensor) -> torch.Tensor:
    """[B, N, D] -> [B, D, g, g] for a square g x g patch grid."""
    batch, count, dim = patches.shape
    side = int(round(count**0.5))
    if side * side != count:
        raise ValueError(f"Patch count {count} is not a square grid")
    return patches.transpose(1, 2).reshape(batch, dim, side, side)


def grid_pool(patches: torch.Tensor, size: int = 4) -> torch.Tensor:
    """Average patches into a coarse size x size grid: [B, N, D] -> [B, size*size, D]."""
    pooled = F.adaptive_avg_pool2d(patches_to_map(patches), size)
    return pooled.flatten(2).transpose(1, 2)


def mask_to_patch_weights(masks: torch.Tensor, grid: int) -> torch.Tensor:
    """Fraction of each patch covered by the mask: [B, H, W] bool -> [B, grid*grid]."""
    coverage = F.adaptive_avg_pool2d(masks.float().unsqueeze(1), grid)
    return coverage.flatten(1)


def masked_mean(
    patches: torch.Tensor, weights: torch.Tensor, threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean of patches whose mask coverage is >= threshold.

    Falls back to any-overlap patches for small objects, then to all patches when the mask is
    empty (object not detected). Returns (pooled [B, D], fraction of patches used [B]).
    """
    selected = weights >= threshold
    small = selected.sum(1) == 0
    selected[small] = weights[small] > 0
    empty = selected.sum(1) == 0
    selected[empty] = True
    selection = selected.unsqueeze(-1).to(patches.dtype)
    pooled = (patches * selection).sum(1) / selection.sum(1)
    return pooled, selected.float().mean(1)

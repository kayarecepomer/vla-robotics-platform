"""Shared feature computation, used both when caching a dataset and at policy run time.

Keeping one code path for both is what guarantees the policy sees the same features on the
robot that it was trained on.
"""

from __future__ import annotations

import numpy as np
import torch

from vla.encoders.pooling import grid_pool, mask_to_patch_weights, masked_mean
from vla.encoders.registry import FrozenEncoder

FEATURE_KINDS = ("pooled", "grid", "masked")


def compute_features(
    encoder: FrozenEncoder,
    images: list[np.ndarray],
    texts: list[str] | None = None,
    grid_size: int = 4,
    masker=None,
) -> dict[str, torch.Tensor]:
    """Return {'pooled': [B, D], 'grid': [B, g*g, D]} plus {'masked': [B, D], 'coverage': [B]} with a masker."""
    output = encoder.encode(images, texts)
    features = {"pooled": output.pooled.cpu(), "grid": grid_pool(output.patches, grid_size).cpu()}
    if masker is not None:
        masks = torch.from_numpy(np.stack([masker.mask(image) for image in images]))
        weights = mask_to_patch_weights(masks, output.grid).to(output.patches.device)
        pooled, coverage = masked_mean(output.patches, weights)
        features["masked"] = pooled.cpu()
        features["coverage"] = coverage.cpu()
    return features


def tokens_for_kind(features: dict[str, torch.Tensor], kind: str) -> torch.Tensor:
    """Policy input tokens for one camera: [B, n_tokens, D]."""
    if kind == "pooled":
        return features["pooled"].unsqueeze(1)
    if kind == "grid":
        return features["grid"]
    if kind == "masked":
        return torch.stack([features["masked"], features["pooled"]], dim=1)
    raise ValueError(f"Unknown feature kind {kind!r}; choose from {FEATURE_KINDS}")

"""Frozen vision encoders compared in the encoder study, behind one interface.

All three are ViT-L with a 16x16 patch grid and 1024-dim features, so the policy head sees
the same token budget whichever encoder is plugged in:

    siglip2  google/siglip2-large-patch16-256   image-text contrastive (CAIP's initialization)
    dinov2   facebook/dinov2-large              self-supervised, 224 px, patch 14
    caip     yuvansharma/caip-vitl256           SigLIP 2 further trained to predict human hand motion
                                                (Sharma et al., 2026); its pooled embedding is
                                                conditioned on the task text

Images are square-cropped and resized here (not inside the Hugging Face processors), so a
patch index always maps to the same region of the camera frame. Object masks rely on that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from PIL import Image


@dataclass
class EncoderOutput:
    pooled: torch.Tensor  # [B, D]
    patches: torch.Tensor  # [B, N, D]

    @property
    def grid(self) -> int:
        return int(round(self.patches.shape[1] ** 0.5))


def square_crop(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    return image[top : top + side, left : left + side]


def prepare(images: list[np.ndarray], size: int) -> list[Image.Image]:
    from PIL import Image

    return [Image.fromarray(square_crop(image)).resize((size, size), Image.BICUBIC) for image in images]


class FrozenEncoder:
    name: str
    model_id: str
    image_size: int
    dim: int = 1024

    def __init__(self, device: str = "cuda", dtype: str = "float16"):
        import torch

        self.torch = torch
        self.device = device
        self.dtype = getattr(torch, dtype) if device != "cpu" else torch.float32
        self.load()

    def load(self) -> None:
        raise NotImplementedError

    def encode(self, images: list[np.ndarray], texts: list[str] | None = None) -> EncoderOutput:
        raise NotImplementedError

    def _to_device(self, inputs):
        return {
            key: value.to(self.device, self.dtype) if value.is_floating_point() else value.to(self.device)
            for key, value in inputs.items()
        }


class SigLIP2Encoder(FrozenEncoder):
    name = "siglip2"
    model_id = "google/siglip2-large-patch16-256"
    image_size = 256

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.model = AutoModel.from_pretrained(self.model_id, torch_dtype=self.dtype).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)

    def encode(self, images, texts=None):
        inputs = self.processor(images=prepare(images, self.image_size), do_resize=False, return_tensors="pt")
        with self.torch.inference_mode():
            out = self.model.vision_model(**self._to_device(inputs))
        return EncoderOutput(pooled=out.pooler_output.float(), patches=out.last_hidden_state.float())


class DINOv2Encoder(FrozenEncoder):
    name = "dinov2"
    model_id = "facebook/dinov2-large"
    image_size = 224

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.model = AutoModel.from_pretrained(self.model_id, torch_dtype=self.dtype).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)

    def encode(self, images, texts=None):
        inputs = self.processor(
            images=prepare(images, self.image_size), do_resize=False, do_center_crop=False, return_tensors="pt"
        )
        with self.torch.inference_mode():
            out = self.model(**self._to_device(inputs))
        return EncoderOutput(pooled=out.pooler_output.float(), patches=out.last_hidden_state[:, 1:].float())


class CAIPEncoder(FrozenEncoder):
    name = "caip"
    model_id = "yuvansharma/caip-vitl256"
    image_size = 256

    def load(self) -> None:
        from transformers import AutoModel, AutoProcessor

        self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype=self.dtype)
        self.model = self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    def encode(self, images, texts=None):
        prepared = prepare(images, self.image_size)
        texts = texts if texts is not None else [""] * len(prepared)
        inputs = self.processor(images=prepared, text=list(texts), return_tensors="pt")
        with self.torch.inference_mode():
            out = self.model(**self._to_device(inputs))
        return EncoderOutput(pooled=out.image_pooled.float(), patches=out.patch_features.float())


ENCODERS: dict[str, type[FrozenEncoder]] = {cls.name: cls for cls in (SigLIP2Encoder, DINOv2Encoder, CAIPEncoder)}


def load_encoder(name: str, device: str = "cuda") -> FrozenEncoder:
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder {name!r}; choose from {sorted(ENCODERS)}")
    return ENCODERS[name](device=device)

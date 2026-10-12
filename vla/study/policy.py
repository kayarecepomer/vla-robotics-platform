"""Run a trained encoder-study head on live observations (hosted by the Windows policy server)."""

from __future__ import annotations

import numpy as np
import torch

from bridge.policy_protocol import Observation
from vla.encoders.features import compute_features, tokens_for_kind
from vla.study.data import NormStats
from vla.study.head import ChunkPolicyHead, HeadConfig


class StudyPolicy:
    def __init__(self, checkpoint: dict, encoder, masker=None, device: str = "cuda"):
        self.checkpoint = checkpoint
        self.encoder = encoder
        self.masker = masker
        self.device = device
        self.stats = NormStats.from_dict(checkpoint["stats"])
        self.tasks: list[str] = checkpoint["tasks"]
        self.head = ChunkPolicyHead(HeadConfig(**checkpoint["head_config"])).to(device).eval()
        self.head.load_state_dict(checkpoint["state_dict"])

    @staticmethod
    def from_checkpoint(path: str, device: str = "cuda") -> StudyPolicy:
        from vla.encoders.registry import load_encoder

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        masker = None
        if checkpoint["kind"] == "masked":
            from vla.encoders.masks import GroundedMasker

            masker = GroundedMasker(checkpoint["mask_query"], device=device)
        return StudyPolicy(checkpoint, load_encoder(checkpoint["encoder"], device=device), masker, device)

    def task_id(self, task: str) -> int:
        if task in self.tasks:
            return self.tasks.index(task)
        if len(self.tasks) == 1:
            return 0
        raise ValueError(f"Task {task!r} was not in training data. Known tasks: {self.tasks}")

    def tokens(self, observation: Observation) -> torch.Tensor:
        per_camera = []
        for camera in self.checkpoint["cameras"]:
            features = compute_features(
                self.encoder,
                [observation.images[camera]],
                [observation.task],
                self.checkpoint["grid_size"],
                self.masker,
            )
            features = {key: value.half().float() for key, value in features.items()}  # match float16 cache
            per_camera.append(tokens_for_kind(features, self.checkpoint["kind"]))
        return torch.cat(per_camera, dim=1)

    @torch.inference_mode()
    def predict_chunk(self, observation: Observation) -> np.ndarray:
        state = (observation.state - self.stats.state_mean) / self.stats.state_std
        actions = (
            self.head(
                self.tokens(observation).to(self.device),
                torch.from_numpy(state).float()[None].to(self.device),
                torch.tensor([self.task_id(observation.task)], device=self.device),
            )[0]
            .cpu()
            .numpy()
        )
        return actions * self.stats.action_std + self.stats.action_mean

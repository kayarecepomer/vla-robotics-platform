"""Policies the Windows policy server can host, behind one ``predict_chunk`` interface."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from bridge.policy_protocol import Observation


class ChunkPolicy(Protocol):
    def predict_chunk(self, observation: Observation) -> np.ndarray:
        """Return a [horizon, 6] array of joint targets."""


class HoldPolicy:
    """Repeats the current joint state. Use it to measure bridge latency with the real arm safely."""

    def __init__(self, horizon: int = 10):
        self.horizon = horizon

    def predict_chunk(self, observation: Observation) -> np.ndarray:
        return np.repeat(observation.state[None, :].astype(np.float32), self.horizon, axis=0)


class LeRobotPolicy:
    """A checkpoint trained with ``lerobot-train`` (ACT, Diffusion Policy, SmolVLA, ...).

    Uses the processor pipeline API from LeRobot 0.4+, where normalization and tokenization
    live in pre/post-processors saved next to the checkpoint.
    """

    def __init__(self, path: str, device: str = "cuda"):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        config = PreTrainedConfig.from_pretrained(path)
        config.device = device
        self.policy = get_policy_class(config.type).from_pretrained(path, config=config).to(device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(config, pretrained_path=path)
        self.device = device
        self.torch = torch

    def predict_chunk(self, observation: Observation) -> np.ndarray:
        torch = self.torch
        batch = {"observation.state": torch.from_numpy(observation.state).float()[None], "task": observation.task}
        for name, image in observation.images.items():
            batch[f"observation.images.{name}"] = torch.from_numpy(image).permute(2, 0, 1).float()[None] / 255.0
        batch = self.preprocess(batch)
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch)
        chunk = self.postprocess(chunk)
        return chunk[0].float().cpu().numpy()


def load_policy(kind: str, path: str | None = None, device: str = "cuda") -> ChunkPolicy:
    if kind == "hold":
        return HoldPolicy()
    if kind == "lerobot":
        return LeRobotPolicy(path, device=device)
    if kind == "study":
        from vla.study.policy import StudyPolicy

        return StudyPolicy.from_checkpoint(path, device=device)
    raise ValueError(f"Unknown policy kind: {kind}")

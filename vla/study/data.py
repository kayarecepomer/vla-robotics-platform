"""Training samples for the policy head, read from cached encoder features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vla.encoders.features import tokens_for_kind


@dataclass
class NormStats:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    def fit(state: np.ndarray, action: np.ndarray) -> NormStats:
        return NormStats(state.mean(0), state.std(0) + 1e-6, action.mean(0), action.std(0) + 1e-6)

    def to_dict(self) -> dict[str, list[float]]:
        return {key: value.tolist() for key, value in self.__dict__.items()}

    @staticmethod
    def from_dict(raw: dict[str, list[float]]) -> NormStats:
        return NormStats(**{key: np.asarray(value, dtype=np.float32) for key, value in raw.items()})


class FeatureStore:
    """Memory-mapped view of one ``extract_features`` output directory."""

    def __init__(self, feature_dir: str | Path):
        self.dir = Path(feature_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.state = np.load(self.dir / "state.npy").astype(np.float32)
        self.action = np.load(self.dir / "action.npy").astype(np.float32)
        self.episode_index = np.load(self.dir / "episode_index.npy")
        self.task_index = np.load(self.dir / "task_index.npy")
        self._camera_arrays: dict[str, np.ndarray] = {}

    def camera_array(self, camera: str, kind: str) -> np.ndarray:
        name = f"{camera}_{kind}"
        if name not in self._camera_arrays:
            self._camera_arrays[name] = np.load(self.dir / f"{name}.npy", mmap_mode="r")
        return self._camera_arrays[name]

    def episode_ends(self) -> np.ndarray:
        """For each frame, the index one past the last frame of its episode."""
        ends = np.empty(len(self.episode_index), dtype=np.int64)
        boundaries = np.flatnonzero(np.diff(self.episode_index)) + 1
        for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, len(self.episode_index)]):
            ends[start:stop] = stop
        return ends

    def split_episodes(self, val_fraction: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Frame indices for train and validation, split by whole episodes."""
        episodes = np.unique(self.episode_index)
        rng = np.random.default_rng(seed)
        rng.shuffle(episodes)
        val_count = max(1, int(round(len(episodes) * val_fraction))) if len(episodes) > 1 else 0
        val_episodes = set(episodes[:val_count].tolist())
        is_val = np.isin(self.episode_index, list(val_episodes))
        return np.flatnonzero(~is_val), np.flatnonzero(is_val)


class ChunkDataset(torch.utils.data.Dataset):
    """(tokens, state, task, future action chunk) samples; chunks are padded past an episode's end."""

    def __init__(
        self, store: FeatureStore, indices: np.ndarray, kind: str, cameras: list[str], horizon: int, stats: NormStats
    ):
        self.store = store
        self.indices = indices
        self.kind = kind
        self.cameras = cameras
        self.horizon = horizon
        self.stats = stats
        self.ends = store.episode_ends()

    def __len__(self) -> int:
        return len(self.indices)

    def tokens(self, frame: int) -> torch.Tensor:
        per_camera = []
        for camera in self.cameras:
            features = {
                "pooled": torch.from_numpy(np.array(self.store.camera_array(camera, "pooled")[frame : frame + 1]))
            }
            if self.kind == "grid":
                features["grid"] = torch.from_numpy(
                    np.array(self.store.camera_array(camera, "grid")[frame : frame + 1])
                )
            if self.kind == "masked":
                features["masked"] = torch.from_numpy(
                    np.array(self.store.camera_array(camera, "masked")[frame : frame + 1])
                )
            per_camera.append(tokens_for_kind(features, self.kind)[0])
        return torch.cat(per_camera, dim=0).float()

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        frame = int(self.indices[item])
        stop = min(frame + self.horizon, int(self.ends[frame]))
        actions = self.store.action[frame:stop]
        valid = np.zeros(self.horizon, dtype=bool)
        valid[: len(actions)] = True
        if len(actions) < self.horizon:
            actions = np.concatenate([actions, np.repeat(actions[-1:], self.horizon - len(actions), axis=0)])
        return {
            "tokens": self.tokens(frame),
            "state": torch.from_numpy((self.store.state[frame] - self.stats.state_mean) / self.stats.state_std).float(),
            "task": torch.tensor(int(self.store.task_index[frame])),
            "actions": torch.from_numpy((actions - self.stats.action_mean) / self.stats.action_std).float(),
            "valid": torch.from_numpy(valid),
        }

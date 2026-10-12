import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bridge.policy_protocol import Observation  # noqa: E402
from vla.encoders.features import compute_features, tokens_for_kind  # noqa: E402
from vla.encoders.pooling import grid_pool, mask_to_patch_weights, masked_mean  # noqa: E402
from vla.encoders.registry import EncoderOutput, square_crop  # noqa: E402
from vla.study.data import ChunkDataset, FeatureStore, NormStats  # noqa: E402
from vla.study.head import ChunkPolicyHead, HeadConfig, chunk_l1_loss  # noqa: E402
from vla.study.policy import StudyPolicy  # noqa: E402
from vla.study.train_head import parse_args, train  # noqa: E402

DIM = 32


class FakeEncoder:
    """Deterministic stand-in for a ViT: 16x16 patches whose value is the mean image brightness."""

    name = "fake"
    dim = DIM

    def encode(self, images, texts=None):
        brightness = torch.tensor([float(square_crop(image).mean()) / 255 for image in images])
        patches = brightness[:, None, None].expand(-1, 256, DIM).clone()
        return EncoderOutput(pooled=patches.mean(1), patches=patches)


class FakeMasker:
    def mask(self, image):
        mask = np.zeros(square_crop(image).shape[:2], dtype=bool)
        mask[:8, :8] = True
        return mask


def test_grid_pool_and_mask_weights():
    patches = torch.arange(256, dtype=torch.float32).reshape(1, 256, 1)
    assert grid_pool(patches, 4).shape == (1, 16, 1)
    mask = torch.zeros(1, 64, 64, dtype=torch.bool)
    mask[:, :4, :4] = True  # exactly the top-left patch of a 16x16 grid
    weights = mask_to_patch_weights(mask, 16)
    assert weights[0, 0] == 1.0 and weights[0, 1:].sum() == 0
    pooled, used = masked_mean(patches, weights)
    assert pooled.item() == 0.0 and used.item() == pytest.approx(1 / 256)


def test_masked_mean_falls_back_to_all_patches_when_mask_is_empty():
    patches = torch.ones(2, 256, 4)
    pooled, used = masked_mean(patches, torch.zeros(2, 256))
    assert torch.equal(pooled, torch.ones(2, 4)) and torch.equal(used, torch.ones(2))


def test_compute_features_shapes():
    images = [np.full((48, 64, 3), 128, dtype=np.uint8)] * 3
    features = compute_features(FakeEncoder(), images, grid_size=4, masker=FakeMasker())
    assert features["pooled"].shape == (3, DIM) and features["grid"].shape == (3, 16, DIM)
    assert tokens_for_kind(features, "masked").shape == (3, 2, DIM)


def test_head_forward_and_loss():
    config = HeadConfig(
        feature_dim=DIM, num_tokens=3, num_tasks=2, horizon=7, d_model=32, heads=4, encoder_layers=1, decoder_layers=1
    )
    head = ChunkPolicyHead(config)
    actions = head(torch.randn(5, 3, DIM), torch.randn(5, 6), torch.tensor([0, 1, 0, 1, 0]))
    assert actions.shape == (5, 7, 6)
    valid = torch.zeros(5, 7, dtype=torch.bool)
    valid[:, :3] = True
    loss = chunk_l1_loss(actions, torch.zeros_like(actions), valid)
    loss.backward()
    assert loss.item() > 0


def write_feature_dir(path, episodes=4, frames=20, cameras=("wrist", "front")):
    rng = np.random.default_rng(0)
    total = episodes * frames
    np.save(path / "state.npy", rng.normal(size=(total, 6)).astype(np.float32))
    np.save(path / "action.npy", np.tile(np.arange(frames, dtype=np.float32)[:, None], (episodes, 6)))
    np.save(path / "episode_index.npy", np.repeat(np.arange(episodes), frames))
    np.save(path / "frame_index.npy", np.tile(np.arange(frames), episodes))
    np.save(path / "task_index.npy", np.repeat(np.arange(episodes) % 2, frames))
    for camera in cameras:
        np.save(path / f"{camera}_pooled.npy", rng.normal(size=(total, DIM)).astype(np.float16))
        np.save(path / f"{camera}_grid.npy", rng.normal(size=(total, 16, DIM)).astype(np.float16))
        np.save(path / f"{camera}_masked.npy", rng.normal(size=(total, DIM)).astype(np.float16))
    meta = {
        "repo_id": "me/so101_test",
        "encoder": "fake",
        "cameras": list(cameras),
        "grid_size": 4,
        "mask_query": None,
        "fps": 30,
        "num_frames": total,
        "tasks": ["put the red block in the bowl", "put the blue block in the bowl"],
        "feature_dim": DIM,
    }
    (path / "meta.json").write_text(json.dumps(meta))


def test_chunk_dataset_pads_at_episode_end(tmp_path):
    write_feature_dir(tmp_path)
    store = FeatureStore(tmp_path)
    train_idx, val_idx = store.split_episodes(0.25, seed=0)
    assert len(val_idx) == 20 and not set(store.episode_index[train_idx]) & set(store.episode_index[val_idx])
    stats = NormStats(np.zeros(6), np.ones(6), np.zeros(6), np.ones(6))
    dataset = ChunkDataset(store, np.array([18]), "grid", ["wrist", "front"], horizon=5, stats=stats)
    sample = dataset[0]
    assert sample["tokens"].shape == (32, DIM)
    assert sample["valid"].tolist() == [True, True, False, False, False]
    assert sample["actions"][:, 0].tolist() == [18, 19, 19, 19, 19]


def test_train_then_serve_with_study_policy(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    write_feature_dir(features)
    args = parse_args(
        [
            "--features",
            str(features),
            "--kind",
            "pooled",
            "--horizon",
            "5",
            "--steps",
            "4",
            "--batch-size",
            "8",
            "--eval-every",
            "2",
            "--workers",
            "0",
            "--device",
            "cpu",
            "--out-root",
            str(tmp_path / "out"),
        ]
    )
    checkpoint_path = train(args)
    assert checkpoint_path.exists() and (checkpoint_path.parent / "metrics.csv").exists()

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    policy = StudyPolicy(checkpoint, FakeEncoder(), device="cpu")
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    observation = Observation(
        state=np.zeros(6, dtype=np.float32),
        images={"wrist": image, "front": image},
        task="put the blue block in the bowl",
    )
    assert policy.predict_chunk(observation).shape == (5, 6)
    with pytest.raises(ValueError):
        policy.task_id("an unseen task")

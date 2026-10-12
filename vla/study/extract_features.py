"""Cache frozen-encoder features for every frame of a LeRobotDataset (run on the Windows GPU).

    python -m vla.study.extract_features --repo-id kayarecepomer/so101_pick_place --encoder caip
    python -m vla.study.extract_features --repo-id kayarecepomer/so101_pick_place --encoder dinov2 --mask-query "toy"

Writes memory-mapped .npy arrays plus meta.json to ``features/<dataset>/<encoder>/``. Storing
pooled + 4x4 grid tokens in float16 takes about 70 KB per frame for two cameras, so 50 episodes
of 30 s at 30 fps is roughly 3 GB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


class FeatureWriter:
    def __init__(self, out_dir: Path, num_frames: int):
        self.out_dir = out_dir
        self.num_frames = num_frames
        self.arrays: dict[str, np.memmap] = {}
        out_dir.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, start: int, values: np.ndarray, dtype=np.float32) -> None:
        if name not in self.arrays:
            self.arrays[name] = np.lib.format.open_memmap(
                self.out_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=(self.num_frames, *values.shape[1:])
            )
        self.arrays[name][start : start + len(values)] = values

    def close(self) -> None:
        for array in self.arrays.values():
            array.flush()


def to_uint8_images(batch_images) -> list[np.ndarray]:
    """[B, 3, H, W] float 0-1 tensor -> list of HxWx3 uint8 arrays."""
    return list((batch_images.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy())


def main() -> None:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from vla.encoders.features import compute_features
    from vla.encoders.registry import load_encoder

    parser = argparse.ArgumentParser(description="Cache frozen-encoder features for a LeRobotDataset.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", help="Local dataset root, if not using the Hub cache")
    parser.add_argument("--encoder", required=True, choices=["siglip2", "dinov2", "caip"])
    parser.add_argument("--cameras", nargs="+", default=["wrist", "front"])
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--mask-query", help="Also compute object-mask pooled features for this text, e.g. 'toy'")
    parser.add_argument("--out", help="Defaults to features/<dataset name>/<encoder>")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out or Path("features") / args.repo_id.split("/")[-1] / args.encoder)
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    encoder = load_encoder(args.encoder, device=args.device)
    masker = None
    if args.mask_query:
        from vla.encoders.masks import GroundedMasker

        masker = GroundedMasker(args.mask_query, device=args.device)

    writer = FeatureWriter(out_dir, len(dataset))
    tasks: list[str] = []
    start = 0
    for batch_number, batch in enumerate(loader):
        size = len(batch["episode_index"])
        task_ids = []
        for task in batch["task"]:
            if task not in tasks:
                tasks.append(task)
            task_ids.append(tasks.index(task))
        writer.write("state", start, batch["observation.state"].numpy())
        writer.write("action", start, batch["action"].numpy())
        writer.write("episode_index", start, batch["episode_index"].numpy(), np.int64)
        writer.write("frame_index", start, batch["frame_index"].numpy(), np.int64)
        writer.write("task_index", start, np.asarray(task_ids), np.int64)
        for camera in args.cameras:
            images = to_uint8_images(batch[f"observation.images.{camera}"])
            features = compute_features(encoder, images, list(batch["task"]), args.grid_size, masker)
            for kind, values in features.items():
                dtype = np.float32 if kind == "coverage" else np.float16
                writer.write(f"{camera}_{kind}", start, values.numpy().astype(dtype), dtype)
        start += size
        if batch_number % 20 == 0:
            print(f"{start}/{len(dataset)} frames")
    writer.close()

    meta = {
        "repo_id": args.repo_id,
        "encoder": args.encoder,
        "cameras": args.cameras,
        "grid_size": args.grid_size,
        "mask_query": args.mask_query,
        "fps": dataset.fps,
        "num_frames": len(dataset),
        "tasks": tasks,
        "feature_dim": encoder.dim,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote features for {len(dataset)} frames to {out_dir}")


if __name__ == "__main__":
    main()

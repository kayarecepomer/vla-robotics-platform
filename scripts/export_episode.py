"""Export one episode's joint states from a LeRobotDataset to .npz for ``sim/mirror.py --replay``.

python scripts/export_episode.py --repo-id kayarecepomer/so101_pick_place --episode 0 --out episode_000.npz
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    parser = argparse.ArgumentParser(description="Export a recorded episode's joint states.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", help="Local dataset root, if not using the Hub cache")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--key", default="observation.state", help="Use 'action' to replay commanded targets instead")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode])
    states = np.stack([dataset.hf_dataset[i][args.key].numpy() for i in range(len(dataset.hf_dataset))])
    np.savez(args.out, state=states.astype(np.float32), fps=np.float32(dataset.fps))
    print(f"Wrote {len(states)} frames at {dataset.fps} fps to {args.out}")


if __name__ == "__main__":
    main()

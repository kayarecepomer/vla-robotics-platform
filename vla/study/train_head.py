"""Train the shared policy head on cached features for one encoder and one feature kind.

    python -m vla.study.train_head --features features/so101_pick_place/caip --kind pooled
    python -m vla.study.train_head --features features/so101_pick_place/dinov2 --kind grid --steps 30000

Saves ``outputs/study/<dataset>_<encoder>_<kind>/best.pt`` (lowest validation L1) and a
``metrics.csv`` with train/validation loss, so every run in the study table is reproducible.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from vla.study.data import ChunkDataset, FeatureStore, NormStats
from vla.study.head import ChunkPolicyHead, HeadConfig, chunk_l1_loss

TOKENS_PER_CAMERA = {"pooled": lambda grid: 1, "grid": lambda grid: grid * grid, "masked": lambda grid: 2}


def cycle(loader):
    while True:
        yield from loader


@torch.no_grad()
def evaluate(model: ChunkPolicyHead, loader, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        predicted = model(batch["tokens"], batch["state"], batch["task"])
        total += chunk_l1_loss(predicted, batch["actions"], batch["valid"]).item() * len(batch["tokens"])
        count += len(batch["tokens"])
    model.train()
    return total / max(count, 1)


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    store = FeatureStore(args.features)
    meta = store.meta
    cameras = args.cameras or meta["cameras"]
    train_idx, val_idx = store.split_episodes(args.val_fraction, args.seed)
    stats = NormStats.fit(store.state[train_idx], store.action[train_idx])

    def make_loader(indices, shuffle):
        dataset = ChunkDataset(store, indices, args.kind, cameras, args.horizon, stats)
        return torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers, drop_last=shuffle
        )

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False) if len(val_idx) else None

    config = HeadConfig(
        feature_dim=meta["feature_dim"],
        num_tokens=TOKENS_PER_CAMERA[args.kind](meta["grid_size"]) * len(cameras),
        num_tasks=len(meta["tasks"]),
        horizon=args.horizon,
    )
    model = ChunkPolicyHead(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = min(1000, args.steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            min(1.0, (step + 1) / max(warmup, 1)) * 0.5 * (1 + math.cos(math.pi * min(step / args.steps, 1.0)))
        ),
    )

    name = args.name or f"{Path(meta['repo_id']).name}_{meta['encoder']}_{args.kind}"
    out_dir = Path(args.out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_extra = {
        "head_config": config.to_dict(),
        "stats": stats.to_dict(),
        "encoder": meta["encoder"],
        "kind": args.kind,
        "cameras": cameras,
        "grid_size": meta["grid_size"],
        "mask_query": meta.get("mask_query"),
        "tasks": meta["tasks"],
        "repo_id": meta["repo_id"],
    }

    best = float("inf")
    started = time.time()
    with (out_dir / "metrics.csv").open("w", newline="") as handle:
        log = csv.writer(handle)
        log.writerow(["step", "train_l1", "val_l1", "lr", "elapsed_s"])
        running, running_steps = 0.0, 0
        for step, batch in zip(range(1, args.steps + 1), cycle(train_loader)):
            batch = {key: value.to(args.device) for key, value in batch.items()}
            loss = chunk_l1_loss(
                model(batch["tokens"], batch["state"], batch["task"]), batch["actions"], batch["valid"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            running_steps += 1
            if step % args.eval_every == 0 or step == args.steps:
                train_l1 = running / running_steps
                running, running_steps = 0.0, 0
                val_l1 = evaluate(model, val_loader, args.device) if val_loader else train_l1
                log.writerow(
                    [
                        step,
                        f"{train_l1:.5f}",
                        f"{val_l1:.5f}",
                        f"{scheduler.get_last_lr()[0]:.2e}",
                        int(time.time() - started),
                    ]
                )
                handle.flush()
                print(f"step {step}: train L1 {train_l1:.4f}  val L1 {val_l1:.4f}")
                if val_l1 < best:
                    best = val_l1
                    torch.save(
                        {**checkpoint_extra, "state_dict": model.state_dict(), "step": step, "val_l1": val_l1},
                        out_dir / "best.pt",
                    )
    print(f"Best validation L1 {best:.4f}; checkpoint at {out_dir / 'best.pt'}")
    return out_dir / "best.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the encoder-study policy head on cached features.")
    parser.add_argument("--features", required=True, help="Directory written by vla.study.extract_features")
    parser.add_argument("--kind", choices=["pooled", "grid", "masked"], default="pooled")
    parser.add_argument("--cameras", nargs="+", help="Defaults to every camera in the feature cache")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-root", default="outputs/study")
    parser.add_argument("--name")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())

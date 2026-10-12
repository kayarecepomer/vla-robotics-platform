"""Serve a policy on the Windows GPU. The Mac sends observations and executes the returned chunks.

python -m bridge.windows.policy_server --policy lerobot --path outputs/train/act_pick_place/checkpoints/last/pretrained_model
python -m bridge.windows.policy_server --policy study --path outputs/study/caip_pooled/best.pt
python -m bridge.windows.policy_server --policy hold   # latency test, arm holds still
"""

from __future__ import annotations

import argparse
import time

import zmq

from bridge.policy_protocol import ActionChunk, Observation
from vla.policies import ChunkPolicy, load_policy

BIND_ADDRESS = "tcp://0.0.0.0:5556"


def serve(policy: ChunkPolicy, socket: zmq.Socket, max_requests: int | None = None) -> None:
    handled = 0
    while max_requests is None or handled < max_requests:
        observation = Observation.from_frames(socket.recv_multipart())
        start = time.perf_counter()
        actions = policy.predict_chunk(observation)
        server_ms = (time.perf_counter() - start) * 1000.0
        socket.send_multipart(ActionChunk(actions=actions, seq=observation.seq, server_ms=server_ms).to_frames())
        handled += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve action chunks to the Mac over ZeroMQ.")
    parser.add_argument("--policy", choices=["hold", "lerobot", "study"], required=True)
    parser.add_argument("--path", help="Checkpoint directory (lerobot) or .pt file (study)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bind", default=BIND_ADDRESS)
    args = parser.parse_args()

    policy = load_policy(args.policy, args.path, device=args.device)
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(args.bind)
    print(f"Serving {args.policy} policy on {args.bind}")
    try:
        serve(policy, socket)
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()

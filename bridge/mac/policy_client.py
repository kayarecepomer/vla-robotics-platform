"""Run a policy hosted on the Windows GPU against the real follower arm.

    python -m bridge.mac.policy_client --server tcp://<windows-ip>:5556 \
        --cameras wrist=0,front=1 --task "put the red block in the bowl" --publish tcp://0.0.0.0:5555

Each loop reads one observation, asks the server for an action chunk, and executes the first
``--execute-steps`` actions at ``--fps``. With ``--publish`` the commanded joints are also
streamed on the joint-state topic so the Isaac Sim twin mirrors the follower live.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import zmq

from bridge.joint_state import JOINT_NAMES, TOPIC, JointState
from bridge.lerobot_compat import joint_action
from bridge.policy_protocol import ActionChunk, Observation

SERVER_ADDRESS = "tcp://localhost:5556"


class PolicyClient:
    def __init__(self, address: str, timeout_ms: int = 2000, encoding: str = "jpeg"):
        self.address = address
        self.timeout_ms = timeout_ms
        self.encoding = encoding
        self.context = zmq.Context.instance()
        self.socket = self._connect()

    def _connect(self) -> zmq.Socket:
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.address)
        return socket

    def request(self, observation: Observation) -> ActionChunk:
        self.socket.send_multipart(observation.to_frames(self.encoding))
        try:
            return ActionChunk.from_frames(self.socket.recv_multipart())
        except zmq.Again:
            # A REQ socket is stuck after a missed reply; replace it so the next request works.
            self.socket.close()
            self.socket = self._connect()
            raise TimeoutError(f"No reply from policy server at {self.address} within {self.timeout_ms} ms") from None

    def close(self) -> None:
        self.socket.close()


@dataclass
class EpisodeStats:
    steps: int = 0
    round_trip_ms: list[float] = field(default_factory=list)
    server_ms: list[float] = field(default_factory=list)
    duration_s: float = 0.0

    def summary(self) -> dict[str, float]:
        def percentile(values: list[float], q: float) -> float:
            return float(np.percentile(values, q)) if values else float("nan")

        return {
            "steps": self.steps,
            "chunks": len(self.round_trip_ms),
            "duration_s": round(self.duration_s, 2),
            "round_trip_ms_p50": round(percentile(self.round_trip_ms, 50), 1),
            "round_trip_ms_p95": round(percentile(self.round_trip_ms, 95), 1),
            "server_ms_p50": round(percentile(self.server_ms, 50), 1),
            "control_hz": round(self.steps / self.duration_s, 1) if self.duration_s else 0.0,
        }


def observation_from_reading(reading: dict[str, Any], task: str, seq: int) -> Observation:
    images = {key: value for key, value in reading.items() if not key.endswith(".pos")}
    state = np.asarray([reading[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
    return Observation(state=state, images=images, task=task, seq=seq)


def run_episode(
    robot,
    client: PolicyClient,
    task: str,
    fps: float = 30.0,
    execute_steps: int = 20,
    max_seconds: float = 30.0,
    on_action: Callable[[np.ndarray], None] | None = None,
) -> EpisodeStats:
    """Drive ``robot`` with chunks from ``client`` until ``max_seconds`` pass or Ctrl+C."""
    stats = EpisodeStats()
    period = 1.0 / fps
    start = time.perf_counter()
    seq = 0
    try:
        while time.perf_counter() - start < max_seconds:
            observation = observation_from_reading(robot.get_observation(), task, seq)
            request_start = time.perf_counter()
            chunk = client.request(observation)
            stats.round_trip_ms.append((time.perf_counter() - request_start) * 1000.0)
            stats.server_ms.append(chunk.server_ms)
            for action in chunk.actions[:execute_steps]:
                step_start = time.perf_counter()
                robot.send_action(joint_action(action))
                if on_action:
                    on_action(action)
                stats.steps += 1
                time.sleep(max(0.0, period - (time.perf_counter() - step_start)))
            seq += 1
    except KeyboardInterrupt:
        pass
    stats.duration_s = time.perf_counter() - start
    return stats


def joint_state_publisher(bind_address: str) -> Callable[[np.ndarray], None]:
    socket = zmq.Context.instance().socket(zmq.PUB)
    socket.bind(bind_address)

    def publish(action: np.ndarray) -> None:
        state = JointState.from_positions(dict(zip(JOINT_NAMES, map(float, action))))
        socket.send_multipart([TOPIC, state.encode()])

    return publish


def main() -> None:
    from bridge.lerobot_compat import make_follower

    parser = argparse.ArgumentParser(description="Execute a remote policy on the SO-101 follower.")
    parser.add_argument("--server", default=SERVER_ADDRESS)
    parser.add_argument(
        "--cameras", default="wrist=0,front=1", help="name=index pairs, matching the dataset's camera names"
    )
    parser.add_argument("--task", default="")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--execute-steps", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--publish", help="Also publish commanded joints for the sim twin, e.g. tcp://0.0.0.0:5555")
    args = parser.parse_args()

    robot = make_follower(cameras=args.cameras)
    robot.connect()
    client = PolicyClient(args.server)
    on_action = joint_state_publisher(args.publish) if args.publish else None
    try:
        print(f"Running policy for up to {args.seconds:.0f}s. Ctrl+C stops the episode.")
        stats = run_episode(robot, client, args.task, args.fps, args.execute_steps, args.seconds, on_action)
        print(stats.summary())
    finally:
        client.close()
        robot.disconnect()


if __name__ == "__main__":
    main()

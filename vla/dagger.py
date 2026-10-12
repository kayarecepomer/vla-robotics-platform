"""DAgger with the leader arm: let the policy run, take over when it fails, record the correction.

    python -m vla.dagger --server tcp://<windows-ip>:5556 --dataset kayarecepomer/so101_pick_place \
        --task "Put the toy in the bowl"

Type a command and press Enter while it runs:
    t   take over (follower blends to the leader arm's pose, then you drive) / hand back to the policy
    q   quit and upload

Only the frames you drive are recorded, each takeover as its own short episode appended to the
dataset, which is the human-gated DAgger variant (HG-DAgger, Kelly et al. 2019) used in the ETH
Robot Learning course's imitation learning homework. Retrain on the combined data afterwards.
Before taking over, hold the leader arm roughly where the follower is to keep the blend short.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque
from typing import Any, Protocol

import numpy as np

from bridge.joint_state import JOINT_NAMES
from bridge.lerobot_compat import joint_action
from bridge.mac.policy_client import PolicyClient, observation_from_reading


class Recorder(Protocol):
    def add_frame(self, observation: dict[str, Any], action: dict[str, float], task: str) -> None: ...
    def save_episode(self) -> None: ...
    def discard_episode(self) -> None: ...


def joints(reading: dict[str, Any]) -> np.ndarray:
    return np.asarray([reading[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)


class DaggerSession:
    POLICY, BLEND, HUMAN = "policy", "blend", "human"

    def __init__(
        self,
        robot,
        leader,
        client,
        recorder: Recorder,
        task: str,
        fps: float = 30.0,
        execute_steps: int = 20,
        blend_s: float = 1.0,
        min_frames: int = 15,
    ):
        self.robot = robot
        self.leader = leader
        self.client = client
        self.recorder = recorder
        self.task = task
        self.fps = fps
        self.execute_steps = execute_steps
        self.blend_steps = max(1, int(blend_s * fps))
        self.min_frames = min_frames
        self.mode = self.POLICY
        self.pending: deque[np.ndarray] = deque()
        self.blend_start: np.ndarray | None = None
        self.blend_step = 0
        self.segment_frames = 0
        self.saved_episodes = 0
        self.seq = 0

    def toggle(self) -> None:
        if self.mode == self.POLICY:
            self.mode = self.BLEND
            self.blend_start = joints(self.robot.get_observation())
            self.blend_step = 0
            self.pending.clear()
            print("Taking over: follower is moving to the leader arm's pose.")
        else:
            self.end_segment()
            self.mode = self.POLICY
            print("Policy has control.")

    def end_segment(self) -> None:
        if self.segment_frames >= self.min_frames:
            self.recorder.save_episode()
            self.saved_episodes += 1
            print(f"Saved correction episode ({self.segment_frames} frames, {self.saved_episodes} this session).")
        elif self.segment_frames:
            self.recorder.discard_episode()
            print(f"Discarded a {self.segment_frames}-frame correction (shorter than {self.min_frames}).")
        self.segment_frames = 0

    def step(self) -> None:
        if self.mode == self.POLICY:
            if not self.pending:
                observation = observation_from_reading(self.robot.get_observation(), self.task, self.seq)
                self.pending.extend(self.client.request(observation).actions[: self.execute_steps])
                self.seq += 1
            self.robot.send_action(joint_action(self.pending.popleft()))
        elif self.mode == self.BLEND:
            target = joints(self.leader.get_action())
            self.blend_step += 1
            fraction = self.blend_step / self.blend_steps
            self.robot.send_action(joint_action(self.blend_start + (target - self.blend_start) * fraction))
            if self.blend_step >= self.blend_steps:
                self.mode = self.HUMAN
                print("You have control. Type t + Enter to hand back.")
        else:
            action = self.leader.get_action()
            observation = self.robot.get_observation()
            self.robot.send_action(action)
            self.recorder.add_frame(observation, action, self.task)
            self.segment_frames += 1


class LeRobotRecorder:
    """Appends frames to an existing LeRobotDataset (recorded with scripts/record_dataset.sh)."""

    def __init__(self, repo_id: str, root: str | None = None):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.utils import build_dataset_frame

        self.dataset = LeRobotDataset(repo_id, root=root)
        self.build_frame = build_dataset_frame

    def add_frame(self, observation, action, task):
        frame = {
            **self.build_frame(self.dataset.features, observation, prefix="observation"),
            **self.build_frame(self.dataset.features, action, prefix="action"),
            "task": task,
        }
        self.dataset.add_frame(frame)

    def save_episode(self):
        self.dataset.save_episode()

    def discard_episode(self):
        self.dataset.clear_episode_buffer()

    def finish(self, push: bool) -> None:
        finalize = getattr(self.dataset, "finalize", None)
        if finalize:
            finalize()
        if push:
            self.dataset.push_to_hub()


def read_commands(commands: queue.Queue) -> None:
    for line in sys.stdin:
        commands.put(line.strip().lower())


def main() -> None:
    from bridge.lerobot_compat import make_follower, make_leader

    parser = argparse.ArgumentParser(description="Collect DAgger corrections with the leader arm.")
    parser.add_argument("--server", default="tcp://localhost:5556")
    parser.add_argument("--dataset", required=True, help="Existing dataset repo id to append corrections to")
    parser.add_argument("--root")
    parser.add_argument("--task", required=True)
    parser.add_argument("--cameras", default="wrist=0,front=1")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--execute-steps", type=int, default=20)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    robot = make_follower(cameras=args.cameras)
    leader = make_leader()
    robot.connect()
    leader.connect()
    client = PolicyClient(args.server)
    recorder = LeRobotRecorder(args.dataset, args.root)
    session = DaggerSession(robot, leader, client, recorder, args.task, args.fps, args.execute_steps)
    commands: queue.Queue = queue.Queue()
    threading.Thread(target=read_commands, args=(commands,), daemon=True).start()
    print("Policy has control. Type t + Enter to take over, q + Enter to quit.")
    period = 1.0 / args.fps
    try:
        while True:
            tick = time.perf_counter()
            try:
                command = commands.get_nowait()
            except queue.Empty:
                command = ""
            if command == "q":
                break
            if command == "t":
                session.toggle()
            session.step()
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))
    finally:
        session.end_segment()
        recorder.finish(push=not args.no_push)
        client.close()
        leader.disconnect()
        robot.disconnect()
    print(f"Saved {session.saved_episodes} correction episodes to {args.dataset}.")


if __name__ == "__main__":
    main()

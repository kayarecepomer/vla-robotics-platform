"""Mirror the real SO-101 in Isaac Sim, live from the bridge or replayed from a recorded episode.

Live (Mac runs ``bridge.mac.publisher`` or ``bridge.mac.policy_client --publish``):

    python.bat sim\\mirror.py --connect tcp://<mac-ip>:5555

Replay an episode exported with ``scripts/export_episode.py`` (no Mac needed):

    python.bat sim\\mirror.py --replay episode_000.npz

Joint readings go through ``config/so101_twin_mapping.json`` (sign/offset per joint) and are
clamped to the asset's joint limits before being sent as position targets.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "sim"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the Isaac Sim SO-101 twin from real joint data.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--connect", help="ZeroMQ joint-state publisher, e.g. tcp://192.168.1.20:5555")
    source.add_argument("--replay", help=".npz with 'state' [T,6] and 'fps' from scripts/export_episode.py")
    parser.add_argument("--mapping", default=str(REPO_ROOT / "config" / "so101_twin_mapping.json"))
    parser.add_argument("--loop", action="store_true", help="Repeat the replay until the window closes")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def latest_state(socket):
    """Drain the subscriber queue and return only the newest JointState (None if nothing new)."""
    import zmq

    from bridge.joint_state import JointState

    newest = None
    while True:
        try:
            _, payload = socket.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            return newest
        newest = JointState.decode(payload)


def main() -> None:
    args = parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless})
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.utils.types import ArticulationAction
    from so101_scene import build_scene, joint_limits

    from bridge.calibration import TwinMapping
    from bridge.joint_state import JOINT_NAMES, TOPIC

    world = World(stage_units_in_meters=1.0)
    arm = next(iter(build_scene(world, arms=1).values()))
    world.reset()
    limits = joint_limits(arm)
    dof_order = [arm.dof_names.index(name) for name in JOINT_NAMES]
    mapping = TwinMapping.from_json(args.mapping)

    def apply(positions: dict[str, float]) -> None:
        sim = mapping.to_sim(positions, limits)
        targets = np.zeros(arm.num_dof)
        targets[dof_order] = [sim[name] for name in JOINT_NAMES]
        arm.apply_action(ArticulationAction(joint_positions=targets))

    if args.connect:
        import zmq

        socket = zmq.Context.instance().socket(zmq.SUB)
        socket.connect(args.connect)
        socket.setsockopt(zmq.SUBSCRIBE, TOPIC)
        print(f"Mirroring joint state from {args.connect}")
        while app.is_running():
            state = latest_state(socket)
            if state is not None:
                apply(dict(zip(JOINT_NAMES, state.as_array())))
            world.step(render=True)
    else:
        episode = np.load(args.replay)
        states, period = episode["state"], 1.0 / float(episode["fps"])
        print(f"Replaying {len(states)} frames from {args.replay}")
        frame = 0
        while app.is_running() and (args.loop or frame < len(states)):
            tick = time.perf_counter()
            apply(dict(zip(JOINT_NAMES, states[frame % len(states)])))
            world.step(render=True)
            frame += 1
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))
    app.close()


if __name__ == "__main__":
    main()

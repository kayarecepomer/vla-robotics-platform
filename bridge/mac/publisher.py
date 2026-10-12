from __future__ import annotations

import argparse
import time

import zmq

from bridge.joint_state import TOPIC, JointState
from bridge.lerobot_compat import LEADER_ID, LEADER_PORT, joint_positions, make_leader

BIND_ADDRESS = "tcp://0.0.0.0:5555"
PUBLISH_RATE_HZ = 60.0


def run(bind_address: str, rate_hz: float, port: str, teleop_id: str) -> None:
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.bind(bind_address)

    leader = make_leader(port=port, teleop_id=teleop_id)
    leader.connect()
    period_seconds = 1.0 / rate_hz

    try:
        while True:
            loop_start = time.perf_counter()
            state = JointState.from_positions(joint_positions(leader.get_action()))
            socket.send_multipart([TOPIC, state.encode()])
            time.sleep(max(0.0, period_seconds - (time.perf_counter() - loop_start)))
    finally:
        leader.disconnect()
        socket.close()
        context.term()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish calibrated SO-101 leader-arm joint state over ZeroMQ.")
    parser.add_argument("--bind", default=BIND_ADDRESS)
    parser.add_argument("--rate-hz", type=float, default=PUBLISH_RATE_HZ)
    parser.add_argument("--port", default=LEADER_PORT)
    parser.add_argument("--id", default=LEADER_ID)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(bind_address=args.bind, rate_hz=args.rate_hz, port=args.port, teleop_id=args.id)

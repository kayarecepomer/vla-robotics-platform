from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_state import TOPIC, JOINT_NAMES, JointState

LEADER_PORT = "/dev/tty.usbmodem5B610373041"
LEADER_ID = "kayas_leader"
BIND_ADDRESS = "tcp://0.0.0.0:5555"
PUBLISH_RATE_HZ = 60.0


def build_leader_bus():
    from lerobot.teleoperators.so_leader.so_leader import FeetechMotorsBus

    motors = {name: (index + 1, "sts3215") for index, name in enumerate(JOINT_NAMES)}
    bus = FeetechMotorsBus(port=LEADER_PORT, motors=motors)
    bus.connect()
    return bus


def run(bind_address: str, rate_hz: float) -> None:
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.bind(bind_address)

    bus = build_leader_bus()
    period_seconds = 1.0 / rate_hz

    try:
        while True:
            loop_start = time.time()
            positions = {name: bus.read("Present_Position", name) for name in JOINT_NAMES}
            state = JointState.from_positions(positions)
            socket.send_multipart([TOPIC, state.encode()])
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, period_seconds - elapsed))
    finally:
        bus.disconnect()
        socket.close()
        context.term()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish live SO-101 leader-arm joint state over ZeroMQ.")
    parser.add_argument("--bind", default=BIND_ADDRESS)
    parser.add_argument("--rate-hz", type=float, default=PUBLISH_RATE_HZ)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(bind_address=args.bind, rate_hz=args.rate_hz)

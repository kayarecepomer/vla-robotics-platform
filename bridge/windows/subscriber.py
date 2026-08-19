from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_state import TOPIC, JointState

CONNECT_ADDRESS = "tcp://localhost:5555"

ApplyStateCallback = Callable[[JointState], None]


def log_state(state: JointState) -> None:
    print(state)


def run(connect_address: str, on_state: ApplyStateCallback) -> None:
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.connect(connect_address)
    socket.setsockopt(zmq.SUBSCRIBE, TOPIC)

    try:
        while True:
            _, payload = socket.recv_multipart()
            state = JointState.decode(payload)
            on_state(state)
    finally:
        socket.close()
        context.term()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subscribe to SO-101 joint state over ZeroMQ.")
    parser.add_argument("--connect", default=CONNECT_ADDRESS, help="tcp://<mac-lan-ip>:5555")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(connect_address=args.connect, on_state=log_state)

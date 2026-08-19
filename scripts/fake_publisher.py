from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bridge"))
from joint_state import JOINT_NAMES, TOPIC, JointState

BIND_ADDRESS = "tcp://127.0.0.1:5555"
RATE_HZ = 30.0


def run(bind_address: str, rate_hz: float) -> None:
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.bind(bind_address)
    period_seconds = 1.0 / rate_hz

    print(f"Publishing dummy joint states on {bind_address} at {rate_hz} Hz. Ctrl+C to stop.")
    tick = 0
    try:
        while True:
            phase = tick * 0.05
            positions = {name: 100.0 * math.sin(phase + index) for index, name in enumerate(JOINT_NAMES)}
            state = JointState.from_positions(positions)
            socket.send_multipart([TOPIC, state.encode()])
            tick += 1
            time.sleep(period_seconds)
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    run(bind_address=BIND_ADDRESS, rate_hz=RATE_HZ)

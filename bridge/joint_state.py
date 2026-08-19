from __future__ import annotations

import json
import struct
import time
from dataclasses import asdict, dataclass

JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

TOPIC: bytes = b"so101/joint_state"


@dataclass(frozen=True)
class JointState:
    shoulder_pan: float
    shoulder_lift: float
    elbow_flex: float
    wrist_flex: float
    wrist_roll: float
    gripper: float
    timestamp: float

    @staticmethod
    def from_positions(positions: dict[str, float]) -> JointState:
        return JointState(
            shoulder_pan=positions["shoulder_pan"],
            shoulder_lift=positions["shoulder_lift"],
            elbow_flex=positions["elbow_flex"],
            wrist_flex=positions["wrist_flex"],
            wrist_roll=positions["wrist_roll"],
            gripper=positions["gripper"],
            timestamp=time.time(),
        )

    def as_array(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.shoulder_pan,
            self.shoulder_lift,
            self.elbow_flex,
            self.wrist_flex,
            self.wrist_roll,
            self.gripper,
        )

    def encode(self) -> bytes:
        return struct.pack("<7d", *self.as_array(), self.timestamp)

    @staticmethod
    def decode(payload: bytes) -> JointState:
        values = struct.unpack("<7d", payload)
        return JointState(*values)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(payload: str) -> JointState:
        return JointState(**json.loads(payload))

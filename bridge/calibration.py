"""Map real SO-101 joint readings to Isaac Sim twin joint angles.

The leader/follower arms are read through LeRobot with ``use_degrees=True``, so every
arm joint arrives as degrees from the middle of its calibrated range and the gripper
arrives as 0-100 (closed to open). The Isaac Sim ``so101_new_calib`` asset is built so
its zero pose matches that calibrated middle, which makes the mapping a per-joint sign
and offset. Raw Feetech ticks can be converted with the calibration JSON that
``lerobot-calibrate`` writes, using the same formula LeRobot uses internally.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from bridge.joint_state import JOINT_NAMES

STS3215_RESOLUTION = 4096
DEFAULT_MAPPING_PATH = Path(__file__).resolve().parents[1] / "config" / "so101_twin_mapping.json"


@dataclass(frozen=True)
class MotorCalibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


def load_lerobot_calibration(path: str | Path) -> dict[str, MotorCalibration]:
    raw = json.loads(Path(path).read_text())
    return {
        name: MotorCalibration(**{key: raw[name][key] for key in MotorCalibration.__dataclass_fields__}) for name in raw
    }


def ticks_to_degrees(ticks: float, calibration: MotorCalibration, resolution: int = STS3215_RESOLUTION) -> float:
    mid = (calibration.range_min + calibration.range_max) / 2
    degrees = (ticks - mid) * 360.0 / (resolution - 1)
    return -degrees if calibration.drive_mode else degrees


def ticks_to_percent(ticks: float, calibration: MotorCalibration) -> float:
    span = calibration.range_max - calibration.range_min
    percent = (min(max(ticks, calibration.range_min), calibration.range_max) - calibration.range_min) / span * 100.0
    return 100.0 - percent if calibration.drive_mode else percent


@dataclass(frozen=True)
class JointMapping:
    mode: str
    sign: float = 1.0
    offset_rad: float = 0.0
    closed_rad: float = 0.0
    open_rad: float = 0.0

    def to_sim(self, value: float) -> float:
        if self.mode == "degrees":
            return self.sign * math.radians(value) + self.offset_rad
        if self.mode == "percent":
            fraction = min(max(value, 0.0), 100.0) / 100.0
            return self.closed_rad + fraction * (self.open_rad - self.closed_rad)
        raise ValueError(f"Unknown joint mapping mode: {self.mode}")


class TwinMapping:
    def __init__(self, joints: dict[str, JointMapping]):
        missing = set(JOINT_NAMES) - set(joints)
        if missing:
            raise ValueError(f"Mapping is missing joints: {sorted(missing)}")
        self.joints = joints

    @staticmethod
    def from_json(path: str | Path = DEFAULT_MAPPING_PATH) -> TwinMapping:
        raw = json.loads(Path(path).read_text())["joints"]
        return TwinMapping({name: JointMapping(**spec) for name, spec in raw.items()})

    def to_sim(
        self,
        positions: dict[str, float],
        limits: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, float]:
        result = {}
        for name in JOINT_NAMES:
            angle = self.joints[name].to_sim(positions[name])
            if limits and name in limits:
                lower, upper = limits[name]
                angle = min(max(angle, lower), upper)
            result[name] = angle
        return result

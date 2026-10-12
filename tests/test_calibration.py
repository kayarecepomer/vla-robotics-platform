import json
import math

import pytest

from bridge.calibration import (
    JointMapping,
    MotorCalibration,
    TwinMapping,
    load_lerobot_calibration,
    ticks_to_degrees,
    ticks_to_percent,
)
from bridge.joint_state import JOINT_NAMES


def test_ticks_to_degrees_is_zero_at_range_middle():
    calibration = MotorCalibration(id=1, drive_mode=0, homing_offset=0, range_min=708, range_max=3378)
    assert ticks_to_degrees(2043, calibration) == pytest.approx(0.0)
    assert ticks_to_degrees(2043 + 4095 / 4, calibration) == pytest.approx(90.0)


def test_drive_mode_flips_sign_and_percent_clamps():
    calibration = MotorCalibration(id=6, drive_mode=1, homing_offset=0, range_min=1676, range_max=3139)
    assert ticks_to_degrees(3139, calibration) < 0
    assert ticks_to_percent(5000, calibration) == pytest.approx(0.0)


def test_load_lerobot_calibration(tmp_path):
    raw = {"gripper": {"id": 6, "drive_mode": 0, "homing_offset": -12, "range_min": 1676, "range_max": 3139}}
    path = tmp_path / "kayas_follower.json"
    path.write_text(json.dumps(raw))
    assert load_lerobot_calibration(path)["gripper"].range_max == 3139


def test_default_mapping_converts_and_clamps():
    mapping = TwinMapping.from_json()
    positions = {name: 90.0 for name in JOINT_NAMES}
    positions["gripper"] = 50.0
    sim = mapping.to_sim(positions, limits={"shoulder_pan": (-1.0, 1.0)})
    assert sim["shoulder_pan"] == 1.0
    assert sim["elbow_flex"] == pytest.approx(math.pi / 2)
    assert sim["gripper"] == pytest.approx(0.8)


def test_sign_and_offset():
    assert JointMapping("degrees", sign=-1.0, offset_rad=0.5).to_sim(90.0) == pytest.approx(0.5 - math.pi / 2)


def test_missing_joint_is_rejected():
    with pytest.raises(ValueError):
        TwinMapping({"gripper": JointMapping("percent", open_rad=1.0)})

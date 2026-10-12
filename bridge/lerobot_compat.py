"""Build LeRobot robots, teleoperators and datasets by their CLI type names.

LeRobot has moved its SO-101 modules between releases (``so101_follower`` vs ``so_follower``),
but the type names used on the command line (``so101_follower``, ``so101_leader``) are stable.
Everything here goes through those registered names so the Mac scripts survive upgrades.
Only runs on the Mac, where LeRobot is installed.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any

from bridge.joint_state import JOINT_NAMES

FOLLOWER_PORT = "/dev/tty.usbmodem5B610352021"
FOLLOWER_ID = "kayas_follower"
LEADER_PORT = "/dev/tty.usbmodem5B610373041"
LEADER_ID = "kayas_leader"

_REGISTRATION_MODULES = (
    "lerobot.robots.so101_follower",
    "lerobot.robots.so_follower",
    "lerobot.teleoperators.so101_leader",
    "lerobot.teleoperators.so_leader",
)


def _register_so101_types() -> None:
    for module in _REGISTRATION_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            pass


def _build_config(config_cls: type, **kwargs: Any) -> Any:
    accepted = {field.name for field in dataclasses.fields(config_cls)}
    return config_cls(**{key: value for key, value in kwargs.items() if key in accepted})


def parse_cameras(spec: str) -> dict[str, Any]:
    """Parse ``wrist=0,front=1`` (OpenCV indices or device paths) into LeRobot camera configs."""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cameras = {}
    for item in filter(None, (part.strip() for part in spec.split(","))):
        name, source = item.split("=", 1)
        index_or_path: int | str = int(source) if source.isdigit() else source
        cameras[name] = OpenCVCameraConfig(index_or_path=index_or_path, fps=30, width=640, height=480)
    return cameras


def make_follower(cameras: str = "", port: str = FOLLOWER_PORT, robot_id: str = FOLLOWER_ID):
    _register_so101_types()
    from lerobot.robots import RobotConfig, make_robot_from_config

    config_cls = RobotConfig.get_choice_class("so101_follower")
    config = _build_config(
        config_cls, port=port, id=robot_id, cameras=parse_cameras(cameras) if cameras else {}, use_degrees=True
    )
    return make_robot_from_config(config)


def make_leader(port: str = LEADER_PORT, teleop_id: str = LEADER_ID):
    _register_so101_types()
    from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config

    config_cls = TeleoperatorConfig.get_choice_class("so101_leader")
    config = _build_config(config_cls, port=port, id=teleop_id, use_degrees=True)
    return make_teleoperator_from_config(config)


def joint_positions(reading: dict[str, Any]) -> dict[str, float]:
    """Pull ``{joint: value}`` out of a LeRobot observation/action dict keyed ``<joint>.pos``."""
    return {name: float(reading[f"{name}.pos"]) for name in JOINT_NAMES}


def joint_action(values) -> dict[str, float]:
    return {f"{name}.pos": float(value) for name, value in zip(JOINT_NAMES, values)}

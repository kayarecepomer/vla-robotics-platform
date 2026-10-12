"""Wire format for running a policy on the Windows GPU and executing it on the Mac.

The Mac sends one observation (joint state, camera frames, task text) per request and the
Windows policy server replies with an action chunk. Messages are ZeroMQ multipart frames:

    request: [b"obs", header JSON, image bytes...]
    reply:   [b"act", header JSON, float32 action bytes]

Images are JPEG-compressed by default (about 30-60 KB per 640x480 frame), which keeps a
two-camera request well under a millisecond of LAN transfer time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

OBS_TAG = b"obs"
ACT_TAG = b"act"
JPEG_QUALITY = 90


def encode_image(image: np.ndarray, encoding: str) -> bytes:
    if encoding == "raw":
        return np.ascontiguousarray(image, dtype=np.uint8).tobytes()
    if encoding == "jpeg":
        import cv2

        ok, buffer = cv2.imencode(".jpg", image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buffer.tobytes()
    raise ValueError(f"Unknown image encoding: {encoding}")


def decode_image(payload: bytes, encoding: str, shape: list[int]) -> np.ndarray:
    if encoding == "raw":
        return np.frombuffer(payload, dtype=np.uint8).reshape(shape)
    if encoding == "jpeg":
        import cv2

        bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        return np.ascontiguousarray(bgr[:, :, ::-1])
    raise ValueError(f"Unknown image encoding: {encoding}")


@dataclass
class Observation:
    """One robot observation. Images are RGB uint8 HxWx3, keyed by camera name."""

    state: np.ndarray
    images: dict[str, np.ndarray]
    task: str = ""
    seq: int = 0
    t_client: float = field(default_factory=time.time)

    def to_frames(self, encoding: str = "jpeg") -> list[bytes]:
        header = {
            "seq": self.seq,
            "t_client": self.t_client,
            "state": [float(value) for value in self.state],
            "task": self.task,
            "images": [
                {"name": name, "encoding": encoding, "shape": list(image.shape)} for name, image in self.images.items()
            ],
        }
        frames = [OBS_TAG, json.dumps(header).encode()]
        frames.extend(encode_image(image, encoding) for image in self.images.values())
        return frames

    @staticmethod
    def from_frames(frames: list[bytes]) -> Observation:
        if frames[0] != OBS_TAG:
            raise ValueError(f"Expected observation frames, got tag {frames[0]!r}")
        header = json.loads(frames[1])
        images = {
            spec["name"]: decode_image(payload, spec["encoding"], spec["shape"])
            for spec, payload in zip(header["images"], frames[2:], strict=True)
        }
        return Observation(
            state=np.asarray(header["state"], dtype=np.float32),
            images=images,
            task=header["task"],
            seq=header["seq"],
            t_client=header["t_client"],
        )


@dataclass
class ActionChunk:
    """A [horizon, 6] block of joint targets in the same units the dataset was recorded in."""

    actions: np.ndarray
    seq: int = 0
    server_ms: float = 0.0

    def to_frames(self) -> list[bytes]:
        actions = np.ascontiguousarray(self.actions, dtype=np.float32)
        header = {"seq": self.seq, "shape": list(actions.shape), "server_ms": self.server_ms}
        return [ACT_TAG, json.dumps(header).encode(), actions.tobytes()]

    @staticmethod
    def from_frames(frames: list[bytes]) -> ActionChunk:
        if frames[0] != ACT_TAG:
            raise ValueError(f"Expected action frames, got tag {frames[0]!r}")
        header = json.loads(frames[1])
        actions = np.frombuffer(frames[2], dtype=np.float32).reshape(header["shape"])
        return ActionChunk(actions=actions, seq=header["seq"], server_ms=header["server_ms"])

import numpy as np

from bridge.joint_state import JOINT_NAMES
from bridge.policy_protocol import ActionChunk
from vla.dagger import DaggerSession


def reading(value):
    return {f"{name}.pos": float(value) for name in JOINT_NAMES}


class FakeRobot:
    def __init__(self):
        self.current = reading(0.0)
        self.sent = []

    def get_observation(self):
        return {**self.current, "wrist": np.zeros((4, 4, 3), dtype=np.uint8)}

    def send_action(self, action):
        self.sent.append(action)
        self.current = dict(action)


class FakeLeader:
    def get_action(self):
        return reading(10.0)


class FakeClient:
    def __init__(self):
        self.requests = 0

    def request(self, observation):
        self.requests += 1
        return ActionChunk(actions=np.full((8, 6), -1.0, dtype=np.float32))


class FakeRecorder:
    def __init__(self):
        self.frames, self.saved, self.discarded = 0, 0, 0

    def add_frame(self, observation, action, task):
        self.frames += 1

    def save_episode(self):
        self.saved += 1

    def discard_episode(self):
        self.discarded += 1


def test_policy_blend_takeover_and_save():
    robot, client, recorder = FakeRobot(), FakeClient(), FakeRecorder()
    session = DaggerSession(
        robot, FakeLeader(), client, recorder, "task", fps=10, execute_steps=4, blend_s=0.5, min_frames=3
    )
    for _ in range(6):
        session.step()
    assert client.requests == 2 and robot.sent[-1]["gripper.pos"] == -1.0

    session.toggle()
    for _ in range(5):
        session.step()
    assert session.mode == session.HUMAN and robot.sent[-1]["gripper.pos"] == 10.0
    assert recorder.frames == 0  # blending frames are not recorded

    for _ in range(4):
        session.step()
    session.toggle()
    assert recorder.frames == 4 and recorder.saved == 1 and session.mode == session.POLICY


def test_short_correction_is_discarded():
    recorder = FakeRecorder()
    session = DaggerSession(
        FakeRobot(), FakeLeader(), FakeClient(), recorder, "task", fps=10, blend_s=0.1, min_frames=5
    )
    session.toggle()
    for _ in range(3):
        session.step()
    session.toggle()
    assert recorder.saved == 0 and recorder.discarded == 1

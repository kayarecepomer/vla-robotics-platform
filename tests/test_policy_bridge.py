import threading

import numpy as np
import zmq

from bridge.joint_state import JOINT_NAMES
from bridge.mac.policy_client import PolicyClient, observation_from_reading, run_episode
from bridge.policy_protocol import ActionChunk, Observation
from bridge.windows.policy_server import serve
from vla.policies import HoldPolicy


def test_observation_and_action_round_trip():
    image = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    observation = Observation(state=np.arange(6, dtype=np.float32), images={"wrist": image}, task="pick", seq=7)
    decoded = Observation.from_frames(observation.to_frames(encoding="raw"))
    assert decoded.task == "pick" and decoded.seq == 7
    np.testing.assert_array_equal(decoded.images["wrist"], image)
    np.testing.assert_allclose(decoded.state, observation.state)

    chunk = ActionChunk(actions=np.ones((5, 6), dtype=np.float32), seq=7, server_ms=1.5)
    decoded_chunk = ActionChunk.from_frames(chunk.to_frames())
    assert decoded_chunk.actions.shape == (5, 6) and decoded_chunk.server_ms == 1.5


class FakeRobot:
    """Stands in for the LeRobot follower; presses Ctrl+C after ``stop_after`` commands."""

    def __init__(self, stop_after):
        self.stop_after = stop_after
        self.state = {f"{name}.pos": float(i) for i, name in enumerate(JOINT_NAMES)}
        self.sent = []

    def get_observation(self):
        return {**self.state, "wrist": np.zeros((8, 8, 3), dtype=np.uint8)}

    def send_action(self, action):
        self.sent.append(action)
        if len(self.sent) == self.stop_after:
            raise KeyboardInterrupt


def test_server_and_client_over_zmq():
    context = zmq.Context.instance()
    server_socket = context.socket(zmq.REP)
    port = server_socket.bind_to_random_port("tcp://127.0.0.1")
    thread = threading.Thread(target=serve, args=(HoldPolicy(horizon=4), server_socket, 3), daemon=True)
    thread.start()

    robot = FakeRobot(stop_after=12)
    client = PolicyClient(f"tcp://127.0.0.1:{port}", encoding="raw")
    published = []
    stats = run_episode(robot, client, "hold", fps=1000, execute_steps=4, max_seconds=10, on_action=published.append)
    thread.join(timeout=5)
    client.close()
    server_socket.close()

    assert len(robot.sent) == 12
    assert stats.summary()["chunks"] == 3
    assert len(published) == 11
    assert robot.sent[0] == robot.state  # HoldPolicy repeats the current joint state


def test_observation_from_reading_splits_cameras_and_joints():
    observation = observation_from_reading(FakeRobot(1).get_observation(), "task", seq=2)
    assert list(observation.images) == ["wrist"]
    np.testing.assert_allclose(observation.state, np.arange(6))

from bridge.joint_state import JointState


def test_binary_and_json_round_trip():
    state = JointState(1.0, -2.0, 3.5, 0.0, 45.0, 80.0, timestamp=123.0)
    assert JointState.decode(state.encode()) == state
    assert JointState.from_json(state.to_json()) == state

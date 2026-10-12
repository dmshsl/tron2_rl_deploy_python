"""Safety operates on outgoing commands, never the recurrent policy tensors."""

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tron2_deploy.policy_contract import load_contract
from tron2_deploy.policy_observation import SdkState
from tron2_deploy.policy_targets import JointTargets, action_targets, torque_preclip

MODELS = Path(__file__).resolve().parents[1] / "controllers/model"
sys.path.insert(0, str(MODELS.parent))
from PolicyWheelfootController import PolicyWheelfootController


@pytest.fixture
def controller():
    model = MODELS / "WF_TRON2A_BASE_BLIND"
    return PolicyWheelfootController(model, load_contract(model / "contract.yaml"))


@pytest.fixture
def state(controller):
    return SdkState(np.array(controller.contract.joints.default_position), np.zeros(10),
                    np.array([1., 0., 0., 0.]), np.zeros(3), 0.)


def assert_damping(targets, contract):
    np.testing.assert_array_equal(targets.kp, np.zeros(10))
    np.testing.assert_array_equal(targets.kd, np.asarray(contract.control.kd, dtype=np.float32))
    np.testing.assert_array_equal(targets.dq, np.zeros(10))
    assert np.isfinite(targets.q).all()


def test_wheel_effort_matches_training(controller):
    # Given the generated training contract, when inspecting wheels, then match training.
    for i in controller.contract.joints.wheel_indices:
        assert controller.contract.control.torque_limit[i] == 20.


@pytest.mark.parametrize("sign", [-1., 1.])
def test_outgoing_targets_clamp_without_changing_actions(controller, state, sign):
    # Given extreme actions, when limiting outgoing targets, then retain raw policy data.
    contract = controller.contract
    actions = np.full(10, sign * 100., dtype=np.float32)
    raw = action_targets(actions, contract)
    bounded = torque_preclip(raw, state, contract)
    limits = np.array(contract.safety.soft_position_limits)
    positions = list(contract.joints.position_indices)
    assert np.all(bounded.q[positions] >= limits[positions, 0] - 1e-6)
    assert np.all(bounded.q[positions] <= limits[positions, 1] + 1e-6)
    assert np.max(np.abs(bounded.dq[list(contract.joints.wheel_indices)])) <= 40.
    np.testing.assert_array_equal(actions, np.full(10, sign * 100.))
    assert np.max(np.abs(raw.dq)) == 100.


@pytest.mark.parametrize("trigger", ["wheel", "hard_lower", "hard_upper", "tilt", "command", "history", "action", "target"])
def test_fault_latches_damping_until_restart(controller, state, capsys, trigger):
    # Given one unsafe input on an otherwise prepared controller.
    controller.prepare(state)
    current = replace(state, timestamp=3.02)
    command = np.zeros(3, dtype=np.float32)
    if trigger == "wheel":
        dq = state.dq.copy()
        dq[4] = 48.01
        current = replace(current, dq=dq)
    elif trigger in ("hard_lower", "hard_upper"):
        q = state.q.copy()
        q[0] = -3.15 if trigger == "hard_lower" else 2.61
        current = replace(current, q=q)
    elif trigger == "tilt":
        current = replace(current, quat=np.array([np.cos(.505), np.sin(.505), 0., 0.]))
    elif trigger == "command":
        command[0] = np.nan
    elif trigger == "history":
        controller.mem = replace(controller.mem, history=np.full((10, 36), np.inf, dtype=np.float32))
    elif trigger == "action":
        controller.policy = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="obs")],
                                             run=lambda *args: [np.full(10, np.nan, dtype=np.float32)])
    else:
        controller.last_policy_time = current.timestamp
        controller.targets = JointTargets(np.full(10, np.inf), np.zeros(10), np.ones(10), np.ones(10))
    # When stepping, then fault rather than throw and keep damping even after prepare/stop.
    assert_damping(controller.step(current, command), controller.contract)
    reason = controller.safety.reason
    assert reason
    controller.stop()
    controller.prepare(state)
    assert_damping(controller.step(state, np.zeros(3)), controller.contract)
    assert controller.safety.reason == reason
    assert capsys.readouterr().err.count("DAMPING_LATCH") == 1


def test_in_distribution_step_preserves_action_and_history(controller, state):
    # Given a nominal state, when inference runs, then clamping cannot rewrite policy memory.
    controller.prepare(state)
    targets = controller.step(replace(state, timestamp=3.02), np.zeros(3))
    actions = controller.policy.run(None, {controller.policy.get_inputs()[0].name: controller.actor_input})[0]
    # Then the actual actor output is kept bit-for-bit and the monitor stays clear.
    np.testing.assert_array_equal(controller.mem.last_action, actions)
    assert controller.mem.history.shape == (10, 36)
    assert not controller.safety.reason
    assert np.isfinite(targets.q).all()


@pytest.mark.parametrize("joint", [0, 1, 2, 3, 5, 6, 7, 8])
@pytest.mark.parametrize("side", [0, 1])
def test_each_leg_hard_limit_trips(controller, state, joint, side):
    # Given a leg just beyond either hard limit, when checked, then latch damping.
    q = state.q.copy()
    q[joint] = controller.contract.safety.hard_position_limits[joint][side] + (-.001 if side == 0 else .001)
    targets = controller.step(replace(state, q=q), np.zeros(3))
    assert_damping(targets, controller.contract)
    assert "hard position limit" in controller.safety.reason


@pytest.mark.parametrize("joint", [4, 9])
@pytest.mark.parametrize("speed", [-48.01, 48.01])
def test_each_wheel_overspeed_trips(controller, state, joint, speed):
    # Given either overspeed direction, when checked, then latch damping.
    dq = state.dq.copy()
    dq[joint] = speed
    assert_damping(controller.step(replace(state, dq=dq), np.zeros(3)), controller.contract)
    assert "wheel overspeed" in controller.safety.reason


@pytest.mark.parametrize("speed", [-48., 48.])
def test_wheel_threshold_itself_does_not_trip(controller, state, speed):
    # Given speed exactly on the threshold, when monitored, then do not latch.
    dq = state.dq.copy()
    dq[[4, 9]] = speed
    controller.safety.check_state(replace(state, dq=dq))
    assert not controller.safety.reason


def test_nonfinite_scan_trips_before_encoder(state):
    # Given a Base scan with NaN, when inferred, then latch rather than crash.
    model = MODELS / "WF_TRON2A_BASE"
    controller = PolicyWheelfootController(model, load_contract(model / "contract.yaml"),
                                          SimpleNamespace(get_scan=lambda state: np.full(231, np.nan)))
    controller.prepare(state)
    assert_damping(controller.step(replace(state, timestamp=3.02), np.zeros(3)), controller.contract)
    assert controller.safety.reason


def test_nonfinite_encoder_trips_before_actor(controller, state):
    # Given a non-finite latent, when inferred, then latch before running the actor.
    controller.prepare(state)
    controller.encoder = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="history")],
                                         run=lambda *args: [np.full(3, np.inf)])
    assert_damping(controller.step(replace(state, timestamp=3.02), np.zeros(3)), controller.contract)
    assert controller.safety.reason

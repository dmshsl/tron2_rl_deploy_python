"""Scripted stand-up mode: contract, speed cap, tilt threshold, rejection latch, handoff fill."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from tron2_deploy.policy_contract import DeploymentContract, load_contract
from tron2_deploy.policy_observation import PolicyInputError, SdkState
from tron2_deploy.policy_targets import JointTargets, standup_bounds, torque_preclip

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "tron2_deploy/controllers/model"
sys.path.insert(0, str(MODELS.parent))
from PolicyWheelfootController import PolicyWheelfootController

from tron2_deploy.standup_mode import HANDED_OFF, ScriptedStandUp, reference

UPRIGHT = np.array([1., 0., 0., 0.])


@pytest.fixture
def contract():
    return load_contract(MODELS / "WF_TRON2A_BASE_BLIND/contract.yaml")


def seated(contract, timestamp=0.):
    """Admissible seated start: the design's K3 neighbourhood (certifies in seconds)."""
    q = np.zeros(10, dtype=np.float32)
    q[list(contract.joints.position_indices)] = reference.k3_plant() + np.array([.002, .001, -.001, .05] * 2)
    return SdkState(q, np.zeros(10), UPRIGHT, np.zeros(3), timestamp)


def track(controller, state, until, dt=.002):
    """Perfect-tracking plant: feed each target back as the next measured pose."""
    while not until(controller, state):
        targets = controller.step(state, np.zeros(3, dtype=np.float32))
        assert not controller.safety.reason, controller.safety.reason
        state = SdkState(targets.q, np.zeros(10), UPRIGHT, np.zeros(3), state.timestamp + dt)
    return state


def test_contract_standup_block_defaults_and_shipped_values(contract):
    # Given the shipped contract, then the stand-up block carries DESIGN 9f values.
    assert contract.standup.tilt_limit == 1.2 and contract.standup.speed_limit == 1.0
    assert contract.standup.position_margin == pytest.approx(np.pi / 60)
    assert contract.standup.torque_fraction == .8 and contract.standup.zero_command_seconds == .5
    data = contract.model_dump(mode="json")
    del data["standup"]
    assert DeploymentContract.model_validate(data) == contract


@pytest.mark.parametrize("field,value", [("tilt_limit", 0.9), ("tilt_limit", 3.2), ("speed_limit", 0.),
                                        ("handoff_tilt", 1.1), ("torque_fraction", 1.5),
                                        ("certify_budget_seconds", -1.)])
def test_contract_standup_invalid_rejects(contract, field, value):
    data = contract.model_dump(mode="json")
    data["standup"][field] = value
    with pytest.raises(ValidationError):
        DeploymentContract.model_validate(data)


def test_standup_bounds_use_reference_margin_and_torque_gate(contract):
    bounds = standup_bounds(contract)
    hard = np.asarray(contract.safety.hard_position_limits)
    np.testing.assert_allclose(bounds.position[:, 0], hard[:, 0] + np.pi / 60, rtol=1e-6)
    np.testing.assert_allclose(bounds.position[:, 1], hard[:, 1] - np.pi / 60, rtol=1e-6)
    np.testing.assert_allclose(bounds.torque, np.asarray(contract.control.torque_limit) * .8, rtol=1e-6)
    # The rear-fold waypoint (pitch -177 deg) is inside the stand-up bounds but outside locomotion's.
    rear = -np.radians(177.)
    assert bounds.position[0, 0] < rear < contract.safety.soft_position_limits[0][0]
    far = JointTargets(np.full(10, 10., dtype=np.float32), np.zeros(10, dtype=np.float32),
                       np.asarray(contract.control.kp, dtype=np.float32), np.asarray(contract.control.kd, dtype=np.float32))
    state = seated(contract)
    clipped = torque_preclip(far, state, contract, bounds)
    torque = clipped.kp * (clipped.q - state.q)
    assert np.all(np.abs(torque) <= bounds.torque + 1e-3)


def test_speed_cap_phases_and_handoff_fill(contract):
    controller = PolicyWheelfootController(MODELS / "WF_TRON2A_BASE_BLIND", contract, None, "scripted")
    state = seated(contract)
    controller.prepare(state)
    assert controller.started_at is None and controller.standup_phase == 0
    positions = list(contract.joints.position_indices)
    first = controller.step(state, np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(first.q[positions], state.q[positions], atol=1e-6)  # no snap at enable
    previous, peak, phases = first.q.copy(), 0., {0}
    while not controller.standup.complete:
        targets = controller.step(state, np.zeros(3, dtype=np.float32))
        assert not controller.safety.reason, controller.safety.reason
        peak = max(peak, float(np.max(np.abs(targets.q[positions] - previous[positions])) / .002))
        previous = targets.q.copy()
        phases.add(controller.standup_phase)
        state = SdkState(targets.q, np.zeros(10), UPRIGHT, np.zeros(3), state.timestamp + .002)
    assert phases == {0, 1, 2, 3, 4, 5, HANDED_OFF}
    assert peak <= contract.standup.speed_limit + 1e-3
    np.testing.assert_allclose(state.q[positions], reference.STANDING, atol=1e-5)
    # Handoff rule: history pre-filled with the final measured state, EMA zero, prep restarted.
    final = controller.standup.handoff_state
    assert final is not None and controller.started_at == final.timestamp
    assert controller.mem.history.shape == (contract.history_length, contract.history_frame_size)
    assert np.all(controller.mem.history == controller.mem.history[0])
    expected = (final.q - np.asarray(contract.joints.default_position))[positions]
    np.testing.assert_allclose(controller.mem.history[0, 6:14], expected, atol=1e-6)
    np.testing.assert_array_equal(controller.mem.vel_error_ema, 0.)
    # Preparation after handoff keeps full gains (actuators already enabled).
    prep = controller.step(state, np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(prep.kp, np.asarray(contract.control.kp), rtol=1e-6)


def test_zero_command_window_after_handoff(contract, monkeypatch):
    controller = PolicyWheelfootController(MODELS / "WF_TRON2A_BASE_BLIND", contract, None, "scripted")
    state = seated(contract)
    controller.prepare(state)
    state = track(controller, state, lambda c, s: c.standup.complete)
    seen = []
    original = controller.policy.run
    monkeypatch.setattr(controller.policy, "run",
                        lambda outputs, feed: (seen.append(next(iter(feed.values())).copy()) or original(outputs, feed)))
    command = np.array([.5, 0., 0.], dtype=np.float32)
    for offset in (.3, .7):
        later = replace(state, timestamp=controller.started_at + contract.control.power_on_seconds + offset)
        controller.step(later, command)
    first, second = seen[0][-3:], seen[1][-3:]
    np.testing.assert_array_equal(first, 0.)
    np.testing.assert_allclose(second, command)


def test_tilt_threshold_applies_only_while_rising(contract):
    controller = PolicyWheelfootController(MODELS / "WF_TRON2A_BASE_BLIND", contract, None, "scripted")
    state = seated(contract)
    controller.prepare(state)
    tilted = np.array([np.cos(.55), np.sin(.55), 0., 0.])  # 1.1 rad: above 1.0, below 1.2
    controller.step(replace(state, quat=tilted), np.zeros(3, dtype=np.float32))
    assert not controller.safety.reason
    controller.step(replace(state, quat=np.array([np.cos(.65), np.sin(.65), 0., 0.])), np.zeros(3, dtype=np.float32))
    assert "tilt" in controller.safety.reason and "1.2" in controller.safety.reason


def test_locomotion_tilt_limit_restored_after_handoff(contract):
    controller = PolicyWheelfootController(MODELS / "WF_TRON2A_BASE_BLIND", contract, None, "scripted")
    state = seated(contract)
    controller.prepare(state)
    state = track(controller, state, lambda c, s: c.standup.complete)
    controller.step(replace(state, quat=np.array([np.cos(.55), np.sin(.55), 0., 0.])), np.zeros(3, dtype=np.float32))
    assert "tilt" in controller.safety.reason and "1.0" in controller.safety.reason


def test_rejected_start_trips_latch_without_snapping(contract):
    controller = PolicyWheelfootController(MODELS / "WF_TRON2A_BASE_BLIND", contract, None, "scripted")
    q = np.zeros(10, dtype=np.float32)
    q[list(contract.joints.position_indices)] = np.radians([-177., 0., 0., 0.] * 2)  # legs through the base plane
    state = SdkState(q, np.zeros(10), UPRIGHT, np.zeros(3), 0.)
    controller.prepare(state)
    assert isinstance(controller.standup.certificate.exception(timeout=120.), PolicyInputError)
    for t in (0., .5, 1.5):
        targets = controller.step(replace(state, timestamp=t), np.zeros(3, dtype=np.float32))
    assert "rejected" in controller.safety.reason
    np.testing.assert_array_equal(targets.kp, 0.)


def test_clock_pauses_at_end_of_p0_until_certified(contract):
    standup = ScriptedStandUp(contract, seated(contract))
    standup.certificate = SimpleNamespace(done=lambda: False, result=lambda: None)
    standup.targets(seated(contract, 5.))
    assert standup.phase == 0 and standup.origin == pytest.approx(5. - reference.PHASE_ENDS[0])
    standup.certificate = SimpleNamespace(done=lambda: True, result=lambda: None)
    standup.targets(seated(contract, 5.5))
    assert standup.phase == 1


def test_certify_budget_is_a_safe_rejection(contract):
    from time import monotonic

    standup = ScriptedStandUp(contract, seated(contract))
    standup.certificate = SimpleNamespace(done=lambda: False, result=lambda: None)
    standup.targets(seated(contract, 2.))
    assert standup.wait_started is not None
    standup.wait_started = monotonic() - contract.standup.certify_budget_seconds - 1.
    with pytest.raises(PolicyInputError, match="budget"):
        standup.targets(seated(contract, 2.1))


def test_cli_dry_run_reports_standup_mode():
    result = subprocess.run([sys.executable, str(ROOT / "tron2_deploy/run_policy.py"), "--policy", "base",
                             "--dry-run", "--standup", "scripted"], text=True, capture_output=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert "STANDUP mode=scripted tilt_limit=1.2 speed_limit=1.0" in result.stdout

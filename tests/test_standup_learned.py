"""Learned stand-up mode: contract, action processing, height estimate, success -> handoff, safe failure."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from tron2_deploy.policy_contract import load_contract
from tron2_deploy.policy_observation import PolicyInputError, SdkState
from tron2_deploy.policy_targets import JointTargets, standup_bounds, torque_preclip
from tron2_deploy.standup_contract import StandupPolicyContract, load_standup_contract
from tron2_deploy.standup_policy import (
    HANDED_OFF,
    POLICY_PHASE,
    check_compatible,
    estimate_height,
    incremental_target,
    wheel_torque,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "tron2_deploy/controllers/model"
sys.path.insert(0, str(MODELS.parent))
sys.path.insert(0, str(ROOT / "tron2_rl/standup"))
import pose_a  # the Isaac training contract (Kit-free)
from PolicyWheelfootController import PolicyWheelfootController

from tron2_deploy.standup_mode import reference

UPRIGHT = np.array([1., 0., 0., 0.])
ZERO3 = np.zeros(3, dtype=np.float32)


@pytest.fixture(scope="module")
def standup():
    return load_standup_contract(MODELS / "WF_TRON2A_STANDUP/contract.yaml")


@pytest.fixture(scope="module", params=["WF_TRON2A_BASE_BLIND", "WF_TRON2A_BASE"])
def locomotion(request):
    return request.param, load_contract(MODELS / request.param / "contract.yaml")


def sdk_state(contract, q8, timestamp=0., quat=UPRIGHT, dq=None):
    q = np.zeros(10, dtype=np.float32)
    q[list(contract.position_sdk)] = q8
    return SdkState(q, np.zeros(10) if dq is None else dq, quat, np.zeros(3), timestamp)


def test_shipped_contract_matches_training_constants(standup, locomotion):
    _, contract = locomotion
    check_compatible(standup, contract)
    np.testing.assert_allclose(standup.standing, pose_a.STANDING)
    np.testing.assert_allclose(standup.target_lower, pose_a.LOWER)
    np.testing.assert_allclose(standup.start_poses, [pose_a.POSE_A, pose_a.POSE_A_P])
    assert (standup.v_max, standup.windup, standup.policy_dt) == (pose_a.V_MAX, pose_a.WINDUP, pose_a.POLICY_DT)
    success = standup.success
    assert (success.height, success.tilt, success.joint_speed) == (
        pose_a.SUCCESS_HEIGHT, pose_a.SUCCESS_TILT, pose_a.SUCCESS_QDOT)
    if success.rule == "catch":  # DESIGN decisions 13/14: hand-off at the first step in the Base catch region
        import catch
        assert success.hold_seconds == catch.CATCH_HOLD_S
        # reward v5+ (decision 17): the hand-off needs |wheel| <= 3 rad/s; v7 (decision 18) the odometry bound
        wheel = catch.CATCH_WHEEL_SPEED_V5 if standup.heading_input else catch.CATCH_WHEEL_SPEED
        assert (success.catch.joint_speed, success.catch.wheel_speed) == (catch.CATCH_QDOT, wheel)
        assert success.catch.height == catch.CATCH_HEIGHT
        if standup.odometry_input:
            import inplace
            assert success.catch.drift == inplace.CATCH_ODOMETRY
    else:
        assert success.hold_seconds == pose_a.SUCCESS_HOLD_S
    assert standup.episode_seconds == pose_a.EPISODE_S and standup.actor_size == 47


@pytest.mark.parametrize("field,value", [("position_sdk", (0, 1, 2, 3, 5, 6, 7, 9)), ("decimation", 5),
                                        ("standing", (0.,) * 7), ("start_tolerance", 1.)])
def test_invalid_standup_contract_rejects(standup, field, value):
    data = standup.model_dump(mode="json")
    data[field] = value
    with pytest.raises(ValidationError):
        StandupPolicyContract.model_validate(data)


def test_incremental_target_and_wheel_torque_equal_training(standup):
    rng = np.random.default_rng(3)
    for _ in range(200):
        target = rng.uniform(pose_a.LOWER, pose_a.UPPER).astype(np.float32)
        measured = (target + rng.uniform(-.8, .8, 8)).astype(np.float32)
        action = rng.uniform(-2., 2., 10).astype(np.float32)
        expected = pose_a.incremental_target(*(torch.tensor(v) for v in (target, measured, action[:8])),
                                             torch.tensor(pose_a.LOWER, dtype=torch.float32),
                                             torch.tensor(pose_a.UPPER, dtype=torch.float32)).numpy()
        actual = incremental_target(target, measured, np.clip(action[:8], -1, 1), standup)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        assert np.all(np.abs(actual - target) <= standup.v_max * standup.policy_dt + 1e-6)
        np.testing.assert_allclose(wheel_torque(action[8:], standup), pose_a.wheel_torque(torch.tensor(action[8:])).numpy())


def test_height_estimate_standing_seated_and_tilted(standup):
    gravity = np.array([0., 0., -1.])
    height, plausible = estimate_height(pose_a.STANDING, gravity, standup.success)
    assert height == pytest.approx(.496335507868, abs=1e-3)  # DESIGN: FK standing base height
    assert plausible
    _, seated = estimate_height(pose_a.POSE_A, gravity, standup.success)
    assert not seated  # the base bottom is as low as the wheels: no success judgement
    rolled = np.array([0., np.sin(.3), -np.cos(.3)])
    _, tilted = estimate_height(pose_a.STANDING, rolled, standup.success)
    assert not tilted  # one wheel 13 cm above the other


def test_torque_preclip_bounds_feed_forward_wheel_torque(locomotion):
    _, contract = locomotion
    bounds = standup_bounds(contract)
    q = np.zeros(10, dtype=np.float32)
    tau = np.zeros(10, dtype=np.float32)
    tau[list(contract.joints.wheel_indices)] = 50.
    kp = np.asarray(contract.control.kp, dtype=np.float32)
    kd = np.asarray(contract.control.kd, dtype=np.float32).copy()
    kd[list(contract.joints.wheel_indices)] = 0.  # learned stand-up wheels: pure feed-forward torque
    targets = JointTargets(q + .5, np.zeros(10, dtype=np.float32), kp, kd, tau)
    state = SdkState(q, np.zeros(10), UPRIGHT, np.zeros(3), 0.)
    clipped = torque_preclip(targets, state, contract, bounds)
    torque = clipped.kp * (clipped.q - state.q) + clipped.kd * clipped.dq + clipped.tau
    assert np.all(np.abs(torque) <= bounds.torque + 1e-3)
    np.testing.assert_allclose(clipped.tau[list(contract.joints.wheel_indices)], bounds.torque[list(contract.joints.wheel_indices)])


class FlatScan:
    def __init__(self, count):
        self.count = count

    def get_scan(self, state):
        return np.zeros(self.count, dtype=np.float32)


def controller(locomotion_name, contract):
    provider = None if contract.policy == "base_blind" else FlatScan(contract.scan.count)
    return PolicyWheelfootController(MODELS / locomotion_name, contract, provider, "learned")


def begin_policy(ctrl, standup, q8=None, timestamp=0.):
    """Skip the seated route: the hand-over at A' (route behaviour is covered by test_standup_mode)."""
    state = sdk_state(standup, pose_a.POSE_A if q8 is None else q8, timestamp)
    ctrl.prepare(state)
    ctrl.standup.start(state)
    return state


def test_policy_outputs_incremental_targets_and_wheel_torque(standup, locomotion):
    name, contract = locomotion
    ctrl = controller(name, contract)
    try:
        state = begin_policy(ctrl, standup)
        ctrl.standup.actor = lambda actor: np.ones(10, dtype=np.float32)
        targets = ctrl.step(state, ZERO3)
        positions, wheels = list(standup.position_sdk), list(standup.wheel_sdk)
        assert ctrl.standup_phase == POLICY_PHASE and not ctrl.safety.reason
        step = targets.q[positions] - state.q[positions]
        assert np.all(step <= standup.v_max * standup.policy_dt + 1e-6) and np.all(step > 0)
        np.testing.assert_allclose(targets.tau[wheels], standup.wheel_torque_scale)  # clip(a) * 8
        np.testing.assert_array_equal(targets.kp[wheels], 0.)
        np.testing.assert_array_equal(targets.kd[wheels], 0.)
        np.testing.assert_allclose(targets.kp[positions], standup.kp, rtol=1e-6)
        # Between policy ticks (2 ms publish), q_tgt is held.
        held = ctrl.step(replace(state, timestamp=.002), ZERO3)
        np.testing.assert_allclose(held.q[positions], targets.q[positions])
    finally:
        ctrl.close()


def test_success_hold_hands_off_with_the_deploy_rule(standup, locomotion):
    name, contract = locomotion
    ctrl = controller(name, contract)
    try:
        begin_policy(ctrl, standup)
        ctrl.standup.contract = height_contract(standup)  # this test is about the height rule's 0.5 s hold
        ctrl.standup.actor = lambda actor: np.zeros(10, dtype=np.float32)
        t, state = 0., None
        while not ctrl.standup.complete and t < 2.:
            state = sdk_state(standup, pose_a.STANDING, t)  # the plant has stood up
            ctrl.step(state, ZERO3)
            assert not ctrl.safety.reason, ctrl.safety.reason
            t += .002
        assert ctrl.standup.complete and ctrl.standup_phase == HANDED_OFF
        assert .47 <= t <= .5  # 25 good policy ticks = 0.5 s, counted like pose_a.success_step
        final = ctrl.standup.handoff_state
        assert ctrl.started_at == final.timestamp and np.all(ctrl.mem.history == ctrl.mem.history[0])
        np.testing.assert_array_equal(ctrl.mem.vel_error_ema, 0.)
        prep = ctrl.step(replace(state, timestamp=t + .002), ZERO3)
        np.testing.assert_allclose(prep.kp, np.asarray(contract.control.kp), rtol=1e-6)  # full gains, no ramp
        np.testing.assert_array_equal(prep.tau, 0.)
    finally:
        ctrl.close()


def test_no_success_within_episode_trips_damping_latch(standup, locomotion):
    name, contract = locomotion
    ctrl = controller(name, contract)
    try:
        state = begin_policy(ctrl, standup)
        ctrl.standup.actor = lambda actor: np.zeros(10, dtype=np.float32)
        for t in np.arange(0., standup.episode_seconds + .1, .02):
            targets = ctrl.step(replace(state, timestamp=float(t)), ZERO3)
        assert "no success" in ctrl.safety.reason
        np.testing.assert_array_equal(targets.kp, 0.)
        np.testing.assert_array_equal(targets.tau, 0.)
    finally:
        ctrl.close()


def test_start_outside_trained_poses_and_nonfinite_output_latch(standup, locomotion):
    name, contract = locomotion
    ctrl = controller(name, contract)
    try:
        state = sdk_state(standup, pose_a.POSE_A + np.radians([25., 0, 0, 0, 0, 0, 0, 0]))
        ctrl.prepare(state)
        with pytest.raises(PolicyInputError, match="outside the trained starts"):
            ctrl.standup.start(state)
        mixed = np.r_[pose_a.POSE_A[:4], pose_a.POSE_A_P[4:]]  # A' left, A'_p right: trained
        state = begin_policy(ctrl, standup, mixed)
        ctrl.standup.actor = lambda actor: np.full(10, np.nan, dtype=np.float32)
        targets = ctrl.step(state, ZERO3)
        assert ctrl.safety.reason and np.all(targets.kp == 0.)
    finally:
        ctrl.close()


def test_standup_tilt_limit_while_policy_runs(standup, locomotion):
    name, contract = locomotion
    ctrl = controller(name, contract)
    try:
        state = begin_policy(ctrl, standup)
        ctrl.standup.actor = lambda actor: np.zeros(10, dtype=np.float32)
        ctrl.step(replace(state, quat=np.array([np.cos(.55), np.sin(.55), 0., 0.])), ZERO3)  # 1.1 rad
        assert not ctrl.safety.reason
        ctrl.step(replace(state, timestamp=.02, quat=np.array([np.cos(.65), np.sin(.65), 0., 0.])), ZERO3)  # 1.3 rad
        assert "tilt" in ctrl.safety.reason and "1.2" in ctrl.safety.reason
    finally:
        ctrl.close()


def test_route_to_a_prime_then_policy_starts(standup):
    contract = load_contract(MODELS / "WF_TRON2A_BASE_BLIND/contract.yaml")
    ctrl = controller("WF_TRON2A_BASE_BLIND", contract)
    try:
        assert ctrl.planner is not None  # the seated-route worker exists before the SDK streams
        q8 = reference.A_PRIME + np.radians([2., -3., 4., 6.] * 2)
        state = sdk_state(standup, q8)
        ctrl.prepare(state)
        plan = ctrl.standup.route.certificate.result()
        ctrl.standup.actor = lambda actor: np.zeros(10, dtype=np.float32)
        phases = set()
        while ctrl.standup_phase != POLICY_PHASE and state.timestamp < plan.duration + standup.settle_seconds + 60.:
            targets = ctrl.step(state, ZERO3)
            assert not ctrl.safety.reason, ctrl.safety.reason
            phases.add(ctrl.standup_phase)
            state = SdkState(targets.q, np.zeros(10), UPRIGHT, np.zeros(3), state.timestamp + .002)
        assert phases == {0, 1, 2, 3, 4, POLICY_PHASE}
        route_clock = ctrl.standup.policy_started_at - ctrl.standup.route.origin  # the route clock pauses for the plan
        settled = float(plan.times[-1]) + standup.settle_seconds  # after the route's own SETTLE dwell at A'
        assert settled - 1e-6 <= route_clock <= settled + .01
        np.testing.assert_allclose(ctrl.standup.memory.target, state.q[list(standup.position_sdk)])  # q_tgt = measured
        assert ctrl.started_at is None  # no handoff without success
    finally:
        ctrl.close()


def test_cli_dry_run_learned_and_bad_model_dir(tmp_path):
    run = [sys.executable, str(ROOT / "tron2_deploy/run_policy.py"), "--policy", "base_blind", "--dry-run",
           "--standup", "learned"]
    result = subprocess.run(run, text=True, capture_output=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert "STANDUP mode=learned" in result.stdout and "STANDUP_POLICY actor=47 actions=10" in result.stdout
    missing = subprocess.run(run + ["--standup-model", str(tmp_path)], text=True, capture_output=True, timeout=60, check=False)
    assert missing.returncode != 0 and "contract.yaml" in missing.stderr


# ---- DESIGN decision 13: success rule "catch" (the Base catch region, tron2_rl/standup/catch.py) ----

def height_contract(standup):
    """The shipped contract with the (pre-decision-13) height success rule: h, tilt, ||qdot|| held 0.5 s."""
    data = standup.model_dump(mode="json")
    data["success"] |= {"rule": "height", "catch": None, "hold_seconds": pose_a.SUCCESS_HOLD_S}
    return StandupPolicyContract.model_validate(data)


def catch_contract(standup):
    """The shipped contract with the catch success rule and the URDF mass table (Isaac adds the payload)."""
    import catch
    from catch_grid import urdf_masses

    bodies = [{"name": name, "mass": mass, "com": com.tolist()} for name, (mass, com) in urdf_masses().items()]
    data = standup.model_dump(mode="json")
    data["success"] |= {"rule": "catch", "hold_seconds": catch.CATCH_HOLD_S, "catch": {
        "height": catch.CATCH_HEIGHT, "offset_min": catch.CATCH_OFFSET_MIN, "offset_max": catch.CATCH_OFFSET_MAX,
        "pitch": catch.CATCH_PITCH, "roll": catch.CATCH_ROLL, "hip_roll": catch.CATCH_HIP_ROLL,
        "hip_yaw": catch.CATCH_HIP_YAW, "manifold_tolerance": catch.CATCH_MANIFOLD_TOL,
        "manifold": [list(r) for r in zip(catch.MANIFOLD_H, catch.MANIFOLD_HIP, catch.MANIFOLD_KNEE)],
        "joint_speed": catch.CATCH_QDOT, "wheel_speed": catch.CATCH_WHEEL_SPEED, "wheel_radius": .1, "bodies": bodies}}
    return StandupPolicyContract.model_validate(data)


def crouch(h, d):
    """Symmetric wheel-only crouch at root height h with the CoM d ahead of the axle (catch_grid solver, URDF masses)."""
    from catch_grid import solve, urdf_masses

    q, info = solve(h, d, 0., urdf_masses())
    return q, info


def test_catch_rule_contract_requires_its_block(standup):
    data = height_contract(standup).model_dump(mode="json")
    data["success"]["rule"] = "catch"
    with pytest.raises(ValidationError):
        StandupPolicyContract.model_validate(data)
    assert catch_contract(standup).success.rule == "catch"


@pytest.mark.parametrize("h,d", [(.30, .02), (.35, -.04), (.45, .06)])
def test_com_offset_equals_the_catch_grid_geometry(standup, h, d):
    from tron2_deploy.standup_policy import com_offset

    contract = catch_contract(standup)
    q, info = crouch(h, d)
    assert com_offset(q, np.array([0., 0., -1.]), contract.success.catch) == pytest.approx(info["offset"], abs=1e-3)


def test_catch_components_accept_a_balanced_crouch_and_reject_a_prime(standup):
    from tron2_deploy.standup_policy import catch_components

    contract = catch_contract(standup)
    gravity = np.array([0., 0., -1.])
    q, _ = crouch(.30, .02)
    height, plausible = estimate_height(q, gravity, contract.success)
    parts = catch_components(q, np.zeros(8), np.zeros(2), gravity, height, plausible, contract.success.catch)
    assert all(parts.values()), parts
    behind, _ = crouch(.30, -.05)
    parts = catch_components(behind, np.zeros(8), np.zeros(2), gravity, height, plausible, contract.success.catch)
    assert [k for k, v in parts.items() if not v] == ["com"]
    moving = catch_components(q, np.full(8, .6), np.zeros(2), gravity, height, plausible, contract.success.catch)
    assert not moving["rest"]  # ||qdot_8|| 1.70 > 1.5
    settling = catch_components(q, np.full(8, .5), np.array([9., -9.]), gravity, height, plausible, contract.success.catch)
    assert settling["rest"]  # decision 14: 1.41 rad/s and 9 rad/s wheels still hand off
    a_prime = catch_components(pose_a.POSE_A, np.zeros(8), np.zeros(2), gravity, .3, True, contract.success.catch)
    assert not a_prime["family"]


@pytest.mark.parametrize("hold", [0., .5])
def test_catch_hold_hands_off_and_the_old_height_is_not_needed(standup, locomotion, hold):
    name, contract = locomotion
    ctrl = controller(name, contract)
    try:
        begin_policy(ctrl, standup)
        catch_rule = catch_contract(standup)
        ctrl.standup.contract = catch_rule.model_copy(update={"success": catch_rule.success.model_copy(update={"hold_seconds": hold})})
        ctrl.standup.actor = lambda actor: np.zeros(10, dtype=np.float32)
        q, info = crouch(.30, .02)  # h 0.30 < 0.45: only the catch rule can succeed here
        t = 0.
        while not ctrl.standup.complete and t < 2.:
            ctrl.step(sdk_state(standup, q, t), ZERO3)
            assert not ctrl.safety.reason, ctrl.safety.reason
            t += .002
        assert ctrl.standup.complete and ctrl.standup_phase == HANDED_OFF
        assert hold - .03 <= t <= hold + .021  # decision 14 (hold 0): the first policy tick in the region
    finally:
        ctrl.close()


def test_catch_rule_hold_zero_needs_a_good_step(standup):
    data = catch_contract(standup).model_dump(mode="json")
    assert data["success"]["hold_seconds"] == 0.
    data["success"] |= {"rule": "height", "catch": None}
    with pytest.raises(ValidationError):
        StandupPolicyContract.model_validate(data)  # the height rule keeps a positive hold

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tron2_deploy/controllers"))


def test_pure_observation_api_exists():
    # Given the SDK-free observation boundary, when discovered, then it exists.
    assert importlib.util.find_spec("tron2_deploy.policy_observation") is not None


@pytest.mark.parametrize("policy,actor_size", [("BASE", 273), ("BASE_BLIND", 42)])
def test_actor_dimensions_when_default_state(policy, actor_size):
    from tron2_deploy.policy_contract import load_contract
    from tron2_deploy.policy_observation import (
        PolicyMemory,
        SdkState,
        build_actor_input,
    )

    # Given a stationary robot at the generated default pose.
    contract = load_contract(ROOT / f"tron2_deploy/controllers/model/WF_TRON2A_{policy}/contract.yaml")
    state = SdkState(np.array(contract.joints.default_position), np.zeros(10),
                     np.array([1., 0., 0., 0.]), np.zeros(3), 0.)
    mem = PolicyMemory.empty(contract)
    # When building without SDK or mutation.
    actor, encoder = build_actor_input(state, mem, np.zeros(231), np.zeros(3), contract)
    # Then the tensors match the training contract and inputs remain unchanged.
    assert actor.shape == (actor_size,)
    assert encoder.shape == (360,)
    assert np.isfinite(actor).all() and np.isfinite(encoder).all()
    assert mem.history.shape == (0, 36)


def test_joint_mapping_round_trip_when_isaac_is_interleaved():
    from tron2_deploy.policy_contract import load_contract
    # Given distinct joint values in SDK order.
    contract = load_contract(ROOT / "tron2_deploy/controllers/model/WF_TRON2A_BASE/contract.yaml")
    sdk = np.arange(10)
    # When mapping to articulation order and back.
    isaac = sdk[list(contract.joints.sdk_to_isaac)]
    restored = isaac[list(contract.joints.isaac_to_sdk)]
    # Then no joint is swapped or lost.
    np.testing.assert_array_equal(restored, sdk)
    np.testing.assert_array_equal(isaac, [0, 5, 1, 6, 2, 7, 3, 8, 4, 9])


@pytest.fixture
def contract():
    from tron2_deploy.policy_contract import load_contract
    return load_contract(ROOT / "tron2_deploy/controllers/model/WF_TRON2A_BASE_BLIND/contract.yaml")


@pytest.fixture
def state(contract):
    from tron2_deploy.policy_observation import SdkState
    return SdkState(np.array(contract.joints.default_position), np.zeros(10),
                    np.array([1., 0., 0., 0.]), np.zeros(3), 0.)


def test_power_on_when_measured_pose_differs_is_monotone(contract):
    from tron2_deploy.policy_targets import power_on_targets
    # Given an arbitrary measured pose, when preparing over three seconds.
    measured = np.linspace(-.2, .2, 10, dtype=np.float32)
    targets = [power_on_targets(measured, t, contract) for t in np.linspace(0, 3, 101)]
    # Then the first target holds the pose and error monotonically decreases.
    np.testing.assert_allclose(targets[0].q, measured)
    indices = list(contract.joints.position_indices)
    errors = [np.abs(t.q[indices] - np.array(contract.joints.default_position)[indices]) for t in targets]
    assert np.all(np.diff(errors, axis=0) <= 1e-6)
    np.testing.assert_allclose(errors[-1], 0, atol=2e-7)
    np.testing.assert_allclose(targets[0].kp, np.array(contract.control.kp) * .2, rtol=1e-6)
    np.testing.assert_allclose(targets[-1].kp, contract.control.kp, rtol=1e-6)
    assert np.all(np.diff([t.kp for t in targets], axis=0) >= 0)


def test_ema_when_commanded_uses_wheel_mean_without_deadband(contract, state):
    from tron2_deploy.policy_observation import PolicyMemory, build_actor_input
    # Given forward wheel speed 0.3 and yaw rate 0.4, with nonzero prior EMA.
    dq = np.zeros(10, dtype=np.float32)
    dq[[4, 9]] = [2, 4]
    current = replace(state, dq=dq, gyro=np.array([0, 0, .4], dtype=np.float32))
    mem = replace(PolicyMemory.empty(contract), vel_error_ema=np.array([.1, -.2], dtype=np.float32))
    command = np.array([.8, .13, .6], dtype=np.float32)
    # When building, then exact EMA and unscaled/unclipped physical commands appear.
    actor, history = build_actor_input(current, mem, None, command, contract)
    expected = .98 * mem.vel_error_ema + .02 * np.array([-.5, -.2], dtype=np.float32)
    np.testing.assert_allclose(actor[-5:-3], expected, atol=1e-7)
    np.testing.assert_array_equal(actor[-3:], command)
    np.testing.assert_allclose(history.reshape(10, 36)[:, -2:], np.tile(expected, (10, 1)), atol=1e-7)
    np.testing.assert_array_equal(mem.vel_error_ema, np.array([.1, -.2], dtype=np.float32))


def test_history_when_prior_frames_exist_is_pure_and_oldest_first(contract, state):
    from tron2_deploy.policy_observation import PolicyMemory, build_actor_input
    # Given uniquely numbered prior frames.
    prior = np.arange(360, dtype=np.float32).reshape(10, 36)
    mem = replace(PolicyMemory.empty(contract), history=prior)
    # When appending one frame, then only the oldest drops and memory is unchanged.
    _, output = build_actor_input(state, mem, None, np.zeros(3), contract)
    np.testing.assert_array_equal(output.reshape(10, 36)[:-1], prior[1:])
    np.testing.assert_array_equal(prior, np.arange(360).reshape(10, 36))


def test_imu_when_mounted_rotation_recovers_base_tilt(contract, state):
    from scipy.spatial.transform import Rotation

    from tron2_deploy.policy_observation import base_imu
    # Given tilted base and a nonidentity IMU-to-base mount.
    base = Rotation.from_euler("xyz", [np.pi / 12, np.pi / 12, .3])
    mount = Rotation.from_euler("xyz", [.1, -.2, .05])
    imu_xyzw = (base * mount).as_quat()
    base_gyro = np.array([.2, .3, .4])
    mounted = replace(state, quat=imu_xyzw[[3, 0, 1, 2]], gyro=mount.inv().apply(base_gyro))
    calibrated = contract.model_copy(update={"imu": contract.imu.model_copy(update={"imu_to_base_rpy": (.1, -.2, .05)})})
    # When converting, then both gyro and gravity match the base frame.
    gyro, gravity = base_imu(mounted, calibrated)
    np.testing.assert_allclose(gyro, base_gyro, atol=1e-7)
    np.testing.assert_allclose(gravity, base.inv().apply([0, 0, -1]), atol=1e-7)


def test_action_targets_when_nonzero_use_generated_scales(contract, state):
    from tron2_deploy.policy_targets import action_targets, torque_preclip
    # Given an extreme actor action, when generating and separately clipping targets.
    actions = np.full(10, 100., dtype=np.float32)
    raw = action_targets(actions, contract)
    clipped = torque_preclip(raw, state, contract)
    # Then raw targets retain training scaling and clipped PD torque respects limits.
    expected = actions * np.array(contract.joints.action_scale) + np.array(contract.joints.action_offset)
    np.testing.assert_allclose(raw.q[list(contract.joints.position_indices)], expected[list(contract.joints.position_indices)])
    np.testing.assert_allclose(raw.dq[[4, 9]], expected[[4, 9]])
    torque = clipped.kp * (clipped.q - state.q) + clipped.kd * (clipped.dq - state.dq)
    assert np.all(np.abs(torque) <= np.array(contract.control.torque_limit) + 1e-4)
    np.testing.assert_array_equal(actions, np.full(10, 100.))


def test_contract_when_regenerated_matches_and_rejects_nan(contract):
    from pydantic import ValidationError

    from tron2_deploy.policy_contract import DeploymentContract
    # Given the serialized generated contract, when a gain becomes NaN, then reject it.
    data = contract.model_dump(mode="json")
    data["control"]["kp"][0] = float("nan")
    with pytest.raises(ValidationError):
        DeploymentContract.model_validate(data)


@pytest.mark.parametrize("policy,fixture", [("base", "v23"), ("base_blind", "v22")])
def test_contract_when_generated_from_fixture_matches_shipped(policy, fixture):
    from tron2_deploy.policy_contract import load_contract
    from tron2_rl.scripts.rsl_rl.export_deploy_contract import generate
    # Given the original fixture, when regenerating, then every deployed value matches.
    generated = generate(ROOT / f"tron2_rl/tests/fixtures/{fixture}_config.json", policy)
    shipped = load_contract(ROOT / f"tron2_deploy/controllers/model/WF_TRON2A_{policy.upper()}/contract.yaml")
    assert generated == shipped


@pytest.mark.parametrize("field,value", [("kp", [1.]), ("kd", [0.] * 10),
                                        ("power_on_seconds", -1.), ("initial_gain_fraction", 2.),
                                        ("training_torque_limit", [-1.] * 10),
                                        ("physics_dt", -0.005), ("decimation", -4)])
def test_contract_when_control_invalid_rejects_before_connection(contract, field, value):
    from pydantic import ValidationError

    from tron2_deploy.policy_contract import DeploymentContract
    # Given invalid serialized control values, when parsed, then reject the contract.
    data = contract.model_dump(mode="json")
    data["control"][field] = value
    with pytest.raises(ValidationError):
        DeploymentContract.model_validate(data)


@pytest.mark.parametrize("field,value", [("scale", [1., 2.]), ("clip", [1., -1.])])
def test_contract_when_observation_transform_invalid_rejects(contract, field, value):
    from pydantic import ValidationError

    from tron2_deploy.policy_contract import DeploymentContract
    # Given a bad vector scale or inverted clip, when parsing, then reject before inference.
    data = contract.model_dump(mode="json")
    data["policy_terms"][0][field] = value
    with pytest.raises(ValidationError):
        DeploymentContract.model_validate(data)


class FakeRobot:
    def __init__(self):
        self.sent = []

    def subscribeRobotState(self, callback):
        self.state_callback = callback

    def subscribeImuData(self, callback):
        self.imu_callback = callback

    def subscribeSensorJoy(self, callback):
        self.joy_callback = callback

    def publishRobotCmd(self, command):
        self.sent.append((list(command.q), list(command.dq), list(command.Kp), list(command.Kd)))


def test_fake_sdk_when_inference_runs_preserves_wire_order(contract, state):
    from time import monotonic, time_ns

    from PolicyWheelfootController import PolicyWheelfootController

    from tron2_deploy.policy_observation import PolicyInputError
    from tron2_deploy.policy_sdk import SdkAdapter
    # Given actual ONNX models behind a fake SDK callback transport.
    robot = FakeRobot()
    adapter = SdkAdapter(robot, SimpleNamespace)
    robot.state_callback(SimpleNamespace(stamp=time_ns(), q=state.q, dq=state.dq))
    robot.imu_callback(SimpleNamespace(stamp=time_ns(), quat=state.quat, gyro=state.gyro))
    sampled, _ = adapter.sample(monotonic())
    controller = PolicyWheelfootController(ROOT / "tron2_deploy/controllers/model/WF_TRON2A_BASE_BLIND", contract)
    controller.prepare(sampled)
    # When preparing then inferring and publishing through the adapter.
    first = controller.step(sampled, np.zeros(3))
    adapter.publish(first)
    current = replace(sampled, timestamp=sampled.timestamp + 3.02)
    target = controller.step(current, np.zeros(3))
    adapter.publish(target)
    # Then finite SDK commands retain measured pose at startup and valid dimensions.
    np.testing.assert_allclose(robot.sent[0][0], sampled.q, atol=1e-6)
    assert controller.actor_input.shape == (42,)
    assert controller.mem.history.shape == (10, 36)
    assert np.isfinite(robot.sent[-1]).all()
    with pytest.raises(PolicyInputError, match="stale"):
        adapter.sample(monotonic() + 1)
    adapter.release()
    np.testing.assert_array_equal(robot.sent[-1][2:], np.zeros((2, 10)))


def test_perception_provider_when_plane_frames_absent_returns_unknown(contract, state):
    from tron2_deploy.height_scan_provider import PerceptionHeightScanProvider
    # Given real odometry and map with no camera capture yet.
    provider = PerceptionHeightScanProvider(contract, lambda timestamp: ())
    # When querying, then unknown cells remain -1, never a fabricated flat surface.
    np.testing.assert_array_equal(provider.get_scan(state), np.full(231, -1.))


def test_controller_when_stopped_remains_zero_gain_until_prepared(contract, state):
    from PolicyWheelfootController import PolicyWheelfootController
    # Given a prepared controller, when stopped, then subsequent ticks stay disarmed.
    controller = PolicyWheelfootController(ROOT / "tron2_deploy/controllers/model/WF_TRON2A_BASE_BLIND", contract)
    controller.prepare(state)
    controller.stop()
    for t in (0., 1., 4.):
        targets = controller.step(replace(state, timestamp=t), np.zeros(3))
        np.testing.assert_array_equal(targets.kp, np.zeros(10))
        np.testing.assert_array_equal(targets.kd, np.zeros(10))

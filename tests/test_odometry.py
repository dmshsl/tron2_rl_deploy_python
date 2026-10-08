"""Synthetic SDK-stream tests; no simulator, transport, or hardware required."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray
from tron2_deploy.perception.odometry import (
    Odometry,
    OdometryError,
    OdometryInput,
)


def sample(t: float, yaw: float = 0.0, speed: float = 0.5) -> OdometryInput:
    """Construct a straight-legged SDK tick in seconds, radians and rad/s."""
    dq = np.zeros(10)
    dq[[4, 9]] = speed / 0.10
    return OdometryInput(
        timestamp=t,
        imu_wxyz=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
        gyro=np.zeros(3),
        joint_positions=np.zeros(10),
        joint_velocities=dq,
    )


def test_straight_ten_metres_when_wheels_do_not_slip() -> None:
    # Given: a 0.5 m/s, 20-second SDK stream.
    odometry = Odometry()
    # When: the public streaming API consumes every tick.
    results = [odometry.update(sample(float(t))) for t in np.linspace(0, 20, 1001)]
    # Then: integrated distance drifts less than 2 percent.
    drift = float(np.linalg.norm(results[-1].pose[:2, 3] - [10, 0]))
    print(f"straight: distance=10 m drift={drift:.9f} m ({100 * drift / 10:.6f}%)")
    assert drift / 10 < 0.02
    np.testing.assert_allclose(results[0].pose[:2, 3], 0)
    np.testing.assert_allclose(results[-1].planar_velocity, [0.5, 0, 0])
    assert results[-1].timestamp == 20.0


def test_constant_rate_turn_when_heading_crosses_pi() -> None:
    # Given: independent analytic track width from RL URDF zero-pose origins.
    odometry = Odometry()
    track_half_width = 0.09178 + 0.0758 + 0.032 + 0.013
    rate, speed, duration = 0.5, 0.5, 10.0
    result = odometry.update(sample(0))
    # When: wheel encoders describe an arc while IMU supplies absolute heading.
    for t in np.linspace(0.02, duration, 500):
        tick = sample(float(t), yaw=rate * t)
        dq = tick.joint_velocities.copy()
        dq[[4, 9]] = (
            speed + rate * np.array([-track_half_width, track_half_width])
        ) / 0.1
        result = odometry.update(
            replace(tick, gyro=np.array([0, 0, rate]), joint_velocities=dq)
        )
    # Then: both heading and the integrated arc match analytic truth.
    expected_yaw = np.arctan2(np.sin(rate * duration), np.cos(rate * duration))
    error = abs(result.yaw - expected_yaw)
    print(f"turn: heading_error={error:.9g} rad")
    assert error < 1e-10
    expected_xy = speed / rate * np.array([np.sin(5.0), 1 - np.cos(5.0)])
    np.testing.assert_allclose(result.pose[:2, 3], expected_xy, atol=0.001)


def test_injected_common_wheel_slip_reports_drift() -> None:
    # Given: truth travels 10 m but both encoders over-read by 20 percent.
    odometry = Odometry()
    # When: the estimator sees only the slipping encoders and IMU.
    results = [
        odometry.update(sample(float(t), speed=0.6)) for t in np.linspace(0, 20, 1001)
    ]
    # Then: report against external truth, without claiming onboard slip detection.
    drift = float(np.linalg.norm(results[-1].pose[:2, 3] - [10, 0]))
    print(f"injected slip: drift={drift:.6f} m ({100 * drift / 10:.3f}%); report only")


def test_quaternion_order_when_xyzw_is_wrongly_supplied() -> None:
    # Given: a nontrivial known yaw encoded as SDK wxyz.
    tick = sample(0, yaw=0.7)
    wrong = replace(tick, imu_wxyz=tick.imu_wxyz[[1, 2, 3, 0]])
    # When: both are interpreted by the documented wxyz API.
    correct_result = Odometry().update(tick)
    wrong_result = Odometry().update(wrong)
    # Then: the known quaternion-order trap cannot silently pass this test.
    assert correct_result.yaw == pytest.approx(0.7)
    assert abs(correct_result.yaw - wrong_result.yaw) > 0.5


def test_fk_when_sdk_order_differs_from_urdf_order() -> None:
    # Given: the RL model and zero SDK joint angles.
    odometry = Odometry()
    # When: FK resolves each named leg chain.
    geometry = odometry.kinematics(np.zeros(10), np.zeros(10))
    # Then: wheel centres equal the independently summed URDF origins.
    np.testing.assert_allclose(
        geometry.centres,
        [[-0.00044, 0.21258, -0.65189], [-0.00044, -0.21258, -0.65189]],
        atol=1e-10,
    )
    result = odometry.update(sample(0))
    assert result.pose[2, 3] == pytest.approx(0.75189)


def test_fk_velocity_when_leg_joints_move() -> None:
    # Given: asymmetric articulated legs, with nonzero velocities on every joint.
    odometry = Odometry()
    q = np.array([0.9, 0.1, 0.2, -1.8, 0.7, 0.6, -0.2, -0.1, -1.4, -0.4])
    dq = np.linspace(-0.3, 0.6, 10)
    # When: analytic centre velocities are compared with differentiated FK.
    geometry = odometry.kinematics(q, dq)
    dt = 1e-6
    derivative = (
        odometry.kinematics(q + dt * dq, dq).centres
        - odometry.kinematics(q - dt * dq, dq).centres
    ) / (2 * dt)
    # Then: leg motion is not mistaken for base translation.
    np.testing.assert_allclose(geometry.velocities, derivative, atol=1e-9)
    assert not np.allclose(geometry.centres[0], geometry.centres[1])


def test_projection_when_wheels_are_steered_sideways() -> None:
    # Given: both wheels yawed 90 degrees, encoders rolling at 0.5 m/s.
    tick = sample(0)
    q = tick.joint_positions.copy()
    q[[2, 7]] = np.pi / 2
    # When: the estimator projects along the FK wheel rolling directions.
    result = Odometry().update(replace(tick, joint_positions=q))
    # Then: velocity follows wheel direction rather than a hard-coded base x axis.
    np.testing.assert_allclose(result.planar_velocity, [0, 0.5, 0], atol=1e-12)


def test_mount_rotation_when_imu_is_tilted_relative_to_base() -> None:
    # Given: IMU-to-base is a 90-degree rotation about x; base yaw is 0.7 rad.
    mount = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    c, s = np.cos(0.35), np.sin(0.35)
    tick = replace(
        sample(0),
        imu_wxyz=np.array([c, c, s, s]) / np.sqrt(2),
        gyro=np.array([0.0, 0.4, 0]),
    )
    # When: mounting rotation is applied to both quaternion and gyro.
    result = Odometry(imu_to_base=mount).update(tick)
    # Then: the base, not the IMU, supplies pose and angular velocity.
    np.testing.assert_allclose(result.pose[:3, 2], [0, 0, 1], atol=1e-12)
    assert result.yaw == pytest.approx(0.7)
    assert result.planar_velocity[2] == pytest.approx(0.4)


def test_height_when_base_is_rolled() -> None:
    # Given: roll of 0.2 rad and zero leg positions, ideal circular wheels.
    roll = 0.2
    tick = replace(
        sample(0), imu_wxyz=np.array([np.cos(roll / 2), np.sin(roll / 2), 0, 0])
    )
    # When: height places the lowest wheel bottom on the ground plane.
    result = Odometry().update(tick)
    # Then: both centre projection and tilted wheel radius affect height.
    expected = 0.75189 * np.cos(roll) + 0.21258 * np.sin(roll)
    assert result.pose[2, 3] == pytest.approx(expected)
    np.testing.assert_allclose(
        result.pose[:3, :3].T @ result.pose[:3, :3], np.eye(3), atol=1e-12
    )


@pytest.mark.parametrize("timestamp", [0.0, -1.0, np.nan, np.inf])
def test_rejects_bad_timestamp_without_poisoning_state(timestamp: float) -> None:
    # Given: an estimator initialized at time zero.
    odometry = Odometry()
    odometry.update(sample(0))
    # When: a nonmonotonic or nonfinite timestamp arrives.
    with pytest.raises(OdometryError):
        odometry.update(sample(timestamp))
    # Then: the next valid tick still integrates from the last accepted tick.
    np.testing.assert_allclose(odometry.update(sample(1)).pose[:2, 3], [0.5, 0])


@pytest.mark.parametrize(
    "quaternion", [np.zeros(4), np.ones(3), np.array([np.nan, 0, 0, 0])]
)
def test_rejects_invalid_quaternion(quaternion: NDArray[np.float64]) -> None:
    # Given: malformed SDK orientation data.
    tick = replace(sample(0), imu_wxyz=quaternion)
    # When / Then: invalid input is rejected at the boundary.
    with pytest.raises(OdometryError):
        Odometry().update(tick)


def test_rejects_reflected_mount_matrix() -> None:
    # Given: an orthogonal matrix that is a reflection, not a rotation.
    mount = np.diag([1.0, 1.0, -1.0])
    # When / Then: configuration must reject it before processing any ticks.
    with pytest.raises(OdometryError):
        Odometry(imu_to_base=mount)

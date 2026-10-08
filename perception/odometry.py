"""IMU + wheel odometry in metres, seconds and radians, with no transport imports.

SDK quaternion input is **wxyz**, mapping IMU vectors into a z-up world frame.
Joint arrays are L then R: proximal_pitch, proximal_roll, proximal_yaw, knee,
wheel. Positions/velocities must use the RL URDF joint signs and zero offsets;
hardware encoder calibration is the caller's responsibility (UNVERIFIED).

IMU-to-base means v_base = R_base_imu @ v_imu. Its identity default matches the
RL URDF, but is UNVERIFIED for hardware. Output is T_world_base, initially at
x=y=0 with absolute IMU heading. Height places the lowest ideal circular wheel
bottom on z=0. Flat ground, rigid wheels (radius 0.10 m), and no slip/contact
loss are assumed. Wheel thickness, terrain height, IMU bias and slip are not
estimated; common-mode slip is unobservable without an external reference.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final
from xml.etree import ElementTree as ET

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
RL_URDF: Final = (
    Path(__file__).resolve().parents[2]
    / "tron2_rl/exts/bipedal_locomotion/bipedal_locomotion/assets/usd"
    / "WF_TRON2A/urdf/robot.urdf"
)
JOINT_NAMES: Final = tuple(
    f"{joint}_{side}_Joint"
    for side in ("L", "R")
    for joint in ("proximal_pitch", "proximal_roll", "proximal_yaw", "knee", "wheel")
)
WHEEL_RADIUS: Final = 0.10


@dataclass(frozen=True, slots=True)
class OdometryError(ValueError):
    """A rejected sensor sample or kinematic configuration; state is unchanged."""

    field: str
    reason: str

    def __str__(self) -> str:
        return f"{self.field}: {self.reason}"


@dataclass(frozen=True, slots=True)
class OdometryInput:
    """One synchronized SDK tick; gyro is rad/s in the IMU frame, time is seconds."""

    timestamp: float
    imu_wxyz: Vector
    gyro: Vector
    joint_positions: Vector
    joint_velocities: Vector


@dataclass(frozen=True, slots=True)
class OdometryResult:
    """Timestamped SE(3) pose and world-frame (vx, vy, heading_rate) velocity."""

    timestamp: float
    pose: Vector
    planar_velocity: Vector
    yaw: float


@dataclass(frozen=True, slots=True)
class WheelKinematics:
    """L/R wheel centres, axles, centre velocities and angular rates in base frame."""

    centres: Vector
    axes: Vector
    velocities: Vector
    angular_velocities: Vector


@dataclass(frozen=True, slots=True)
class _Joint:
    translation: Vector
    rotation: Vector
    axis: Vector


def _array(value: Vector, shape: tuple[int, ...], field: str) -> Vector:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise OdometryError(field, f"expected finite shape {shape}")
    return result


def _rotation(axis: Vector, angle: float) -> Vector:
    """Rodrigues rotation for a unit axis (URDF axes are normalized at load)."""
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


class Odometry:
    """Stateful stream integrator; rejects bad ticks before changing integration state."""

    def __init__(
        self, urdf_path: Path = RL_URDF, imu_to_base: Vector | None = None
    ) -> None:
        mount = np.eye(3) if imu_to_base is None else imu_to_base
        self._mount = _array(mount, (3, 3), "imu_to_base")
        if not np.allclose(
            self._mount.T @ self._mount, np.eye(3), atol=1e-8, rtol=0
        ) or not np.isclose(np.linalg.det(self._mount), 1.0, atol=1e-8, rtol=0):
            raise OdometryError("imu_to_base", "must be a proper rotation")
        root = ET.parse(urdf_path).getroot()
        joints: list[_Joint] = []
        for index, name in enumerate(JOINT_NAMES):
            element = root.find(f"joint[@name='{name}']")
            if element is None or element.get("type") not in ("revolute", "continuous"):
                raise OdometryError(name, "missing revolute/continuous joint")
            parent, child = element.find("parent"), element.find("child")
            expected_parent = (
                "base_Link"
                if index % 5 == 0
                else JOINT_NAMES[index - 1].replace("_Joint", "_Link")
            )
            if (
                parent is None
                or parent.get("link") != expected_parent
                or child is None
                or child.get("link") != name.replace("_Joint", "_Link")
            ):
                raise OdometryError(name, "unexpected leg chain")
            origin, axis_element = element.find("origin"), element.find("axis")
            xyz = "0 0 0" if origin is None else origin.get("xyz", "0 0 0")
            rpy = "0 0 0" if origin is None else origin.get("rpy", "0 0 0")
            axis_text = (
                "1 0 0" if axis_element is None else axis_element.get("xyz", "1 0 0")
            )
            translation, angles, axis = (
                _array(np.array([float(v) for v in text.split()]), (3,), name)
                for text in (xyz, rpy, axis_text)
            )
            norm = float(np.linalg.norm(axis))
            if norm < 1e-12:
                raise OdometryError(name, "zero joint axis")
            rotation = np.eye(3)
            for basis, angle in zip(np.eye(3), angles):
                rotation = _rotation(basis, float(angle)) @ rotation
            joints.append(_Joint(translation, rotation, axis / norm))
        self._joints = tuple(joints)
        self._previous: OdometryResult | None = None

    def kinematics(self, positions: Vector, velocities: Vector) -> WheelKinematics:
        """Compute named-chain FK and exact centre Jacobian velocities in SDK order."""
        q = _array(positions, (10,), "joint_positions")
        dq = _array(velocities, (10,), "joint_velocities")
        centres, axes, speeds, angular = (np.zeros((2, 3)) for _ in range(4))
        for leg in range(2):
            p, rotation, velocity, omega = (
                np.zeros(3),
                np.eye(3),
                np.zeros(3),
                np.zeros(3),
            )
            for index in range(5 * leg, 5 * leg + 5):
                joint = self._joints[index]
                offset = rotation @ joint.translation
                p = p + offset
                velocity = velocity + np.cross(omega, offset)
                rotation = rotation @ joint.rotation
                axis = rotation @ joint.axis
                omega = omega + axis * dq[index]
                rotation = rotation @ _rotation(joint.axis, float(q[index]))
            centres[leg], axes[leg] = p, axis
            speeds[leg], angular[leg] = velocity, omega
        return WheelKinematics(centres, axes, speeds, angular)

    def update(self, sample: OdometryInput) -> OdometryResult:
        """Integrate one strictly increasing tick using trapezoidal world velocity.

        At each wheel contact, v_base = -omega_wheel x contact_offset
        - omega_base x centre - v_leg. This projects encoder speed via FK,
        compensates moving legs and gyro lever arms, then averages both wheels.
        First tick establishes pose/velocity only; subsequent ticks integrate xy.
        """
        previous = self._previous
        if not np.isfinite(sample.timestamp) or (
            previous is not None and sample.timestamp <= previous.timestamp
        ):
            raise OdometryError("timestamp", "must be finite and strictly increasing")
        quaternion = _array(sample.imu_wxyz, (4,), "imu_wxyz")
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(norm) or norm < 1e-12:
            raise OdometryError("imu_wxyz", "must have nonzero finite norm")
        w, x, y, z = quaternion / norm
        imu_rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        rotation = imu_rotation @ self._mount.T
        omega = imu_rotation @ _array(sample.gyro, (3,), "gyro")
        geometry = self.kinematics(sample.joint_positions, sample.joint_velocities)
        centres = geometry.centres @ rotation.T
        axes = geometry.axes @ rotation.T
        down = np.array([0.0, 0.0, -1.0])
        radial = down - (axes @ down)[:, None] * axes
        radial_norm = np.linalg.norm(radial, axis=1)
        heading_norm = float(np.hypot(rotation[0, 0], rotation[1, 0]))
        if np.any(radial_norm < 1e-8) or heading_norm < 1e-8:
            raise OdometryError(
                "orientation", "wheel contact or base heading is singular"
            )
        contact = WHEEL_RADIUS * radial / radial_norm[:, None]
        wheel_omega = geometry.angular_velocities @ rotation.T + omega
        candidates = (
            -np.cross(wheel_omega, contact)
            - np.cross(omega, centres)
            - geometry.velocities @ rotation.T
        )
        velocity = np.mean(candidates, axis=0)
        forward_rate = np.cross(omega, rotation[:, 0])
        yaw_rate = float(
            (rotation[0, 0] * forward_rate[1] - rotation[1, 0] * forward_rate[0])
            / heading_norm**2
        )
        planar = np.array([velocity[0], velocity[1], yaw_rate])
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[2, 3] = -float(np.min(centres[:, 2] + contact[:, 2]))
        if previous is not None:
            dt = sample.timestamp - previous.timestamp
            pose[:2, 3] = previous.pose[:2, 3] + 0.5 * dt * (
                previous.planar_velocity[:2] + planar[:2]
            )
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(planar)):
            raise OdometryError("sample", "kinematics or integration overflow")
        pose.setflags(write=False)
        planar.setflags(write=False)
        result = OdometryResult(
            sample.timestamp,
            pose,
            planar,
            float(np.arctan2(rotation[1, 0], rotation[0, 0])),
        )
        self._previous = result
        return result

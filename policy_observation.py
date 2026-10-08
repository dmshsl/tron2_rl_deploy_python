"""Pure policy assembly. History is oldest-first, matching Isaac CircularBuffer."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from typing_extensions import assert_never

from tron2_deploy.policy_contract import DeploymentContract, ObservationTerm

Vector = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class PolicyInputError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def finite_vector(value: Vector, size: int) -> Vector:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise PolicyInputError(f"expected finite vector of length {size}, got {result.shape}")
    return result


@dataclass(frozen=True, slots=True)
class SdkState:
    """SDK order q/dq; raw IMU wxyz and IMU-frame gyro; monotonic seconds."""

    q: Vector
    dq: Vector
    quat: Vector
    gyro: Vector
    timestamp: float

    def __post_init__(self) -> None:
        for name, size in (("q", 10), ("dq", 10), ("quat", 4), ("gyro", 3)):
            value = finite_vector(getattr(self, name), size).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not np.isfinite(self.timestamp) or np.linalg.norm(self.quat) < 1e-8:
            raise PolicyInputError("invalid timestamp or zero quaternion")


@dataclass(frozen=True, slots=True)
class PolicyMemory:
    """Pre-tick memory; encoder_output is the latent for this tick's assembled history.

    build_actor_input never changes memory. Runtime first builds encoder input,
    runs the encoder, then replaces the actor prefix with that tick's latent.
    Replay callers can supply recorded encoder_output directly.
    """

    history: Vector
    last_action: Vector
    vel_error_ema: Vector
    encoder_output: Vector

    @classmethod
    def empty(cls, contract: DeploymentContract) -> "PolicyMemory":
        return cls(np.empty((0, contract.history_frame_size), dtype=np.float32),
                   np.zeros(len(contract.joints.action_names), dtype=np.float32),
                   np.zeros(2, dtype=np.float32),
                   np.zeros(contract.encoder_output_size, dtype=np.float32))


def base_imu(state: SdkState, contract: DeploymentContract) -> tuple[Vector, Vector]:
    mount = Rotation.from_euler("xyz", contract.imu.imu_to_base_rpy)
    world_from_imu = Rotation.from_quat(state.quat[[1, 2, 3, 0]])
    world_from_base = world_from_imu * mount.inv()
    gyro = mount.apply(state.gyro).astype(np.float32)
    gravity = world_from_base.inv().apply([0., 0., -1.]).astype(np.float32)
    return gyro, gravity


def velocity_error(state: SdkState, commands: Vector, contract: DeploymentContract) -> Vector:
    gyro, _ = base_imu(state, contract)
    wheels = state.dq[list(contract.joints.wheel_indices)] * np.float32(contract.ema.wheel_spin_sign)
    return np.array([wheels.mean() * np.float32(contract.ema.wheel_radius) - commands[0],
                     gyro[2] - commands[2]], dtype=np.float32)


def transform_term(value: Vector, term: ObservationTerm) -> Vector:
    result = finite_vector(value, term.size)
    if term.clip is not None:
        result = np.clip(result, *term.clip)
    if term.scale is not None:
        result = result * np.asarray(term.scale, dtype=np.float32)
    return result


def build_actor_input(sdk_state: SdkState, mem: PolicyMemory, scan: Vector | None,
                      commands: Vector, contract: DeploymentContract) -> tuple[Vector, Vector]:
    """Return (latent + policy + commands, updated flattened history), without mutation.

    scan is the already geometrically converted/post-corruption scan, BEFORE its
    observation term clip/scale. No noise, deadband or command normalization is added.
    The five parameters are the public replay boundary required by todo 10.
    """
    command = finite_vector(commands, sum(t.size for t in contract.command_terms))
    gyro, gravity = base_imu(sdk_state, contract)
    alpha = np.float32(contract.ema.alpha)
    ema = (np.float32(1.) - alpha) * finite_vector(mem.vel_error_ema, 2) + alpha * velocity_error(sdk_state, command, contract)
    last_action = finite_vector(mem.last_action, len(contract.joints.action_names))

    def group(terms: tuple[ObservationTerm, ...]) -> Vector:
        blocks = []
        for term in terms:
            match term.name:
                case "base_ang_vel":
                    value = gyro
                case "proj_gravity":
                    value = gravity
                case "joint_pos":
                    value = (sdk_state.q - np.asarray(contract.joints.default_position, dtype=np.float32))[list(term.sdk_indices)]
                case "joint_vel":
                    value = (sdk_state.dq - np.asarray(contract.joints.default_velocity, dtype=np.float32))[list(term.sdk_indices)]
                case "last_action":
                    value = last_action
                case "vel_error_ema":
                    value = ema
                case "height_scan":
                    if scan is None:
                        raise PolicyInputError("Base requires a HeightScanProvider")
                    value = scan
                case "velocity_commands":
                    value = command
                case unreachable:
                    assert_never(unreachable)
            blocks.append(transform_term(value, term))
        return np.concatenate(blocks).astype(np.float32)

    frame = group(contract.history_terms)
    history = np.asarray(mem.history, dtype=np.float32)
    if history.shape == (0, contract.history_frame_size):
        updated = np.tile(frame, (contract.history_length, 1))
    else:
        if history.shape != (contract.history_length, contract.history_frame_size) or not np.isfinite(history).all():
            raise PolicyInputError("invalid pre-tick observation history")
        updated = np.concatenate((history[1:], frame[None, :]), axis=0)
    actor = np.concatenate((finite_vector(mem.encoder_output, contract.encoder_output_size),
                            group(contract.policy_terms), group(contract.command_terms)))
    return finite_vector(actor, contract.actor_size), updated.reshape(-1)

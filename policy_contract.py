"""Validated, SDK-independent deployment contract generated from training dumps."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ObservationTerm(FrozenModel):
    name: Literal["base_ang_vel", "proj_gravity", "joint_pos", "joint_vel",
                  "last_action", "vel_error_ema", "height_scan", "velocity_commands"]
    func: str
    size: int
    scale: float | tuple[float, ...] | None
    clip: tuple[float, float] | None
    sdk_indices: tuple[int, ...] = ()

    @model_validator(mode="after")
    def check_transform(self) -> "ObservationTerm":
        if isinstance(self.scale, tuple) and len(self.scale) != self.size:
            raise ValueError("observation scale length must match term dimension")
        if self.clip is not None and self.clip[0] > self.clip[1]:
            raise ValueError("observation clip must be an ordered pair")
        return self


class JointContract(FrozenModel):
    sdk_names: tuple[str, ...]
    isaac_names: tuple[str, ...]
    sdk_to_isaac: tuple[int, ...]
    isaac_to_sdk: tuple[int, ...]
    action_names: tuple[str, ...]
    action_to_sdk: tuple[int, ...]
    default_position: tuple[float, ...]
    default_velocity: tuple[float, ...]
    action_scale: tuple[float, ...]
    action_offset: tuple[float, ...]
    action_clip: tuple[tuple[float, float] | None, ...]
    position_indices: tuple[int, ...]
    wheel_indices: tuple[int, ...]
    use_default_offset: tuple[bool, ...]


class ControlContract(FrozenModel):
    physics_dt: float
    decimation: int
    policy_dt: float
    publish_dt: float
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    torque_limit: tuple[float, ...]
    training_torque_limit: tuple[float, ...]
    power_on_seconds: float
    initial_gain_fraction: float


class EmaContract(FrozenModel):
    alpha: float
    wheel_radius: float
    wheel_spin_sign: float
    deadband: None = None
    definition: str


class SafetyContract(FrozenModel):
    velocity_limit: tuple[float, ...]
    hard_position_limits: tuple[tuple[float, float], ...]
    soft_position_limits: tuple[tuple[float, float], ...]
    wheel_overspeed_factor: float = Field(gt=1)
    tilt_limit: float = Field(gt=0, lt=3.141592653589793)
    watchdog_seconds: float = Field(gt=0)


class StandupContract(FrozenModel):
    """Scripted seated stand-up (todo 22); values are deployment safety settings, not training values.

    tilt_limit replaces safety.tilt_limit only while the scripted reference runs: a seated
    or rising body tilts far beyond the 1.0 rad locomotion threshold by design, and the latch
    must stay armed rather than be disabled (DESIGN.md decision 9f). It equals the StandUp
    environment's bad_orientation termination. Targets are clamped to the URDF hard limits
    inset by position_margin (the reference's own 3 degree contract), not to the locomotion
    soft limits, which exclude the rear-fold waypoints.
    """

    tilt_limit: float = Field(default=1.2, gt=0, lt=3.141592653589793)
    speed_limit: float = Field(default=1.0, gt=0)
    position_margin: float = Field(default=0.05235987755982988, ge=0)
    torque_fraction: float = Field(default=0.8, gt=0, le=1)
    wheel_kd: float | None = Field(default=None, ge=0)
    handoff_tilt: float = Field(default=0.15, gt=0)
    handoff_joint_speed: float = Field(default=0.5, gt=0)
    hold_seconds: float = Field(default=0.5, gt=0)
    zero_command_seconds: float = Field(default=0.5, ge=0)
    # Preparation between a stand-up handoff and the first locomotion action. 0: the policy starts at
    # once on its pre-filled history. Measured 2026-10-19 (tron2_rl/standup/handoff_eval.py positive
    # control, 32 envs handed off from the ideal standing state): with the 3 s locomotion power-on
    # blend the wheeled robot, an unbalanced inverted pendulum under a pure PD hold, fell in 0.38-0.58 s
    # for BaseBlind 32/32; with 0 s, 0/32 fell. control.power_on_seconds still applies at enable.
    handoff_power_on_seconds: float = Field(default=0.0, ge=0)
    certify_budget_seconds: float = Field(default=40.0, gt=0)
    timeout_margin_seconds: float = Field(default=3.0, ge=0)
    # Limp legs come to rest on their mechanical stops; a reading this far past the URDF hard
    # limit is still a resting leg, not an encoder fault (stand-up only; locomotion stays exact).
    hard_limit_tolerance: float = Field(default=0.017453292519943295, ge=0, le=0.05235987755982988)


class CommandContract(FrozenModel):
    """Deployment command shaping (tron2_deploy/command_shaping.py); deployment settings, not training values.

    command_ranges is the trained (hardware) range and the hard clamp; default_cap is the software
    cap applied by default below it (run_policy.py may override it at start-up or at runtime);
    accel/decel are the slew limits of the commanded (vx, wz) while speeding up / slowing down.
    """

    default_cap: tuple[float, float] = (1.0, 1.0)
    accel: tuple[float, float] = (1.5, 3.0)
    decel: tuple[float, float] = (3.0, 6.0)
    lateral_accel_max: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def check_positive(self) -> "CommandContract":
        if min(self.default_cap) < 0 or min(self.accel + self.decel) <= 0:
            raise ValueError("command cap must be >= 0 and accelerations > 0")
        return self


class ImuContract(FrozenModel):
    quaternion_order: Literal["wxyz"] = "wxyz"
    imu_to_base_rpy: tuple[float, float, float]
    calibration: Literal["UNVERIFIED", "OVERRIDE"]


class ScanContract(FrozenModel):
    resolution: float
    size: tuple[float, float]
    shape: tuple[int, int]
    ordering: str
    alignment: str
    offset: float
    clip: tuple[float, float]
    count: int


class DeploymentContract(FrozenModel):
    schema_version: int
    policy: Literal["base", "base_blind"]
    source_task: str
    fixture_sha256: str
    source_code_sha256: dict[str, str]
    joints: JointContract
    control: ControlContract
    safety: SafetyContract
    policy_terms: tuple[ObservationTerm, ...]
    history_terms: tuple[ObservationTerm, ...]
    command_terms: tuple[ObservationTerm, ...]
    command_ranges: tuple[tuple[float, float], ...]
    history_length: int
    history_frame_size: int
    actor_size: int
    encoder_input_size: int
    encoder_output_size: int
    ema: EmaContract
    imu: ImuContract
    scan: ScanContract
    alignment_notes: tuple[str, ...]
    standup: StandupContract = StandupContract()
    commands: CommandContract = CommandContract()

    @model_validator(mode="after")
    def check_layout(self) -> "DeploymentContract":
        n = len(self.joints.sdk_names)
        if n != 10 or len(set(self.joints.sdk_names)) != n:
            raise ValueError("TRON2 SDK requires ten unique joints")
        for permutation in (self.joints.sdk_to_isaac, self.joints.isaac_to_sdk,
                            self.joints.action_to_sdk):
            if sorted(permutation) != list(range(n)):
                raise ValueError("joint mappings must be permutations")
        for values in (self.joints.default_position, self.joints.default_velocity,
                       self.joints.action_scale, self.joints.action_offset,
                       self.joints.action_clip, self.joints.use_default_offset,
                       self.control.kp, self.control.kd, self.control.torque_limit,
                       self.control.training_torque_limit, self.safety.velocity_limit,
                       self.safety.hard_position_limits, self.safety.soft_position_limits):
            if len(values) != n:
                raise ValueError("per-joint contract arrays must match SDK joint count")
        if [self.joints.sdk_names[i] for i in self.joints.sdk_to_isaac] != list(self.joints.isaac_names):
            raise ValueError("sdk_to_isaac disagrees with joint names")
        if [self.joints.isaac_names[i] for i in self.joints.isaac_to_sdk] != list(self.joints.sdk_names):
            raise ValueError("isaac_to_sdk disagrees with joint names")
        if [self.joints.action_names[i] for i in self.joints.action_to_sdk] != list(self.joints.sdk_names):
            raise ValueError("action_to_sdk disagrees with joint names")
        if sorted(self.joints.position_indices + self.joints.wheel_indices) != list(range(n)):
            raise ValueError("position and wheel indices must partition joints")
        if self.history_length * self.history_frame_size != self.encoder_input_size:
            raise ValueError("encoder history dimensions disagree")
        if min(self.safety.velocity_limit) <= 0 or any(
            not hard[0] <= soft[0] < soft[1] <= hard[1]
            for hard, soft in zip(self.safety.hard_position_limits, self.safety.soft_position_limits, strict=True)
        ):
            raise ValueError("invalid safety velocity or position limits")
        if sum(t.size for t in self.policy_terms + self.command_terms) + self.encoder_output_size != self.actor_size:
            raise ValueError("actor dimensions disagree")
        if sum(t.size for t in self.history_terms) != self.history_frame_size:
            raise ValueError("history terms disagree")
        if self.control.policy_dt <= 0 or self.control.publish_dt <= 0:
            raise ValueError("control periods must be positive")
        if self.control.power_on_seconds <= 0 or not 0 < self.control.initial_gain_fraction <= 1:
            raise ValueError("invalid safe power-on profile")
        if (min(self.control.kp) < 0 or min(self.control.kd) <= 0 or
                min(self.control.torque_limit) <= 0 or min(self.control.training_torque_limit) <= 0):
            raise ValueError("invalid gains or torque limits")
        if not 0 < self.ema.alpha <= 1 or self.ema.wheel_radius <= 0:
            raise ValueError("invalid EMA parameters")
        if self.ema.wheel_spin_sign not in (-1., 1.):
            raise ValueError("wheel spin sign must be +/-1")
        if len(self.command_ranges) != 3 or any(low > high for low, high in self.command_ranges):
            raise ValueError("command ranges must be three ordered finite pairs")
        if (self.commands.default_cap[0] > max(-self.command_ranges[0][0], self.command_ranges[0][1]) + 1e-9
                or self.commands.default_cap[1] > max(-self.command_ranges[2][0], self.command_ranges[2][1]) + 1e-9):
            raise ValueError("commands.default_cap must not exceed the trained command range")
        if self.history_length <= 0 or self.encoder_output_size <= 0:
            raise ValueError("history and encoder dimensions must be positive")
        for term in self.policy_terms + self.history_terms + self.command_terms:
            if term.size <= 0 or any(i < 0 or i >= n for i in term.sdk_indices):
                raise ValueError("observation term has invalid dimensions or joint indices")
            if term.name in ("joint_pos", "joint_vel") and len(term.sdk_indices) != term.size:
                raise ValueError("joint observation indices do not match dimensions")
        if self.control.physics_dt <= 0 or self.control.decimation <= 0:
            raise ValueError("physics timestep and decimation must be positive")
        if abs(self.control.policy_dt - self.control.physics_dt * self.control.decimation) > 1e-9:
            raise ValueError("policy period must match training physics decimation")
        if self.standup.tilt_limit < self.safety.tilt_limit or self.standup.handoff_tilt > self.safety.tilt_limit:
            raise ValueError("stand-up tilt threshold must bracket the locomotion tilt limit")
        return self


def load_contract(path: Path) -> DeploymentContract:
    return DeploymentContract.model_validate(yaml.safe_load(path.read_text()))

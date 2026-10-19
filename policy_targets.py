"""Training action processing, deployment power-on, and separate torque pre-clip."""

from dataclasses import dataclass, field

import numpy as np

from tron2_deploy.policy_contract import DeploymentContract
from tron2_deploy.policy_observation import SdkState, Vector, finite_vector


def _no_torque() -> Vector:
    return np.zeros(10, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class JointTargets:
    """MIT command per SDK joint: tau_motor = kp (q - q_m) + kd (dq - dq_m) + tau.

    tau is the feed-forward torque; only the learned stand-up's torque-controlled wheels use it."""

    q: Vector
    dq: Vector
    kp: Vector
    kd: Vector
    tau: Vector = field(default_factory=_no_torque)


@dataclass(frozen=True, slots=True)
class TargetBounds:
    """Outgoing-command envelope: per-joint position pairs and absolute PD torque."""

    position: Vector
    torque: Vector


def locomotion_bounds(contract: DeploymentContract) -> TargetBounds:
    return TargetBounds(np.asarray(contract.safety.soft_position_limits, dtype=np.float32),
                        np.asarray(contract.control.torque_limit, dtype=np.float32))


def standup_bounds(contract: DeploymentContract) -> TargetBounds:
    """URDF hard limits inset by the reference's margin; torque gated at the stand-up fraction."""
    hard = np.asarray(contract.safety.hard_position_limits, dtype=np.float32)
    margin = np.float32(contract.standup.position_margin)
    position = np.column_stack((hard[:, 0] + margin, hard[:, 1] - margin))
    torque = np.asarray(contract.control.torque_limit, dtype=np.float32) * np.float32(contract.standup.torque_fraction)
    return TargetBounds(position, torque)


def action_targets(actions: Vector, contract: DeploymentContract) -> JointTargets:
    raw = finite_vector(actions, len(contract.joints.action_names))
    processed = raw * np.asarray(contract.joints.action_scale, dtype=np.float32)
    processed = processed + np.asarray(contract.joints.action_offset, dtype=np.float32)
    processed = processed.copy()
    for i, clip in enumerate(contract.joints.action_clip):
        if clip is not None:
            processed[i] = np.clip(processed[i], *clip)
    sdk = processed[list(contract.joints.action_to_sdk)]
    q, dq = np.zeros_like(sdk), np.zeros_like(sdk)
    positions, wheels = list(contract.joints.position_indices), list(contract.joints.wheel_indices)
    q[positions], dq[wheels] = sdk[positions], sdk[wheels]
    return JointTargets(q, dq, np.asarray(contract.control.kp, dtype=np.float32),
                        np.asarray(contract.control.kd, dtype=np.float32))


def power_on_targets(measured_pose: Vector, elapsed: float, contract: DeploymentContract,
                     ramp_gains: bool = True) -> JointTargets:
    """Blend the measured pose into the default pose; ramp_gains=False keeps full gains
    (actuators already enabled, e.g. after the scripted stand-up) so no gain step occurs."""
    fraction = float(np.clip(elapsed / contract.control.power_on_seconds, 0., 1.))
    initial = contract.control.initial_gain_fraction if ramp_gains else 1.
    gain = initial + (1. - initial) * fraction
    start = finite_vector(measured_pose, len(contract.joints.sdk_names))
    q = start + np.float32(fraction) * (np.asarray(contract.joints.default_position, dtype=np.float32) - start)
    q[list(contract.joints.wheel_indices)] = start[list(contract.joints.wheel_indices)]
    return JointTargets(q, np.zeros_like(q), np.asarray(contract.control.kp, dtype=np.float32) * gain,
                        np.asarray(contract.control.kd, dtype=np.float32) * gain)


def torque_preclip(targets: JointTargets, state: SdkState, contract: DeploymentContract,
                   bounds: TargetBounds | None = None) -> JointTargets:
    """Bound PD torque by adjusting targets; never change actor action/history."""
    for value in (targets.q, targets.dq, targets.kp, targets.kd, targets.tau):
        finite_vector(value, len(contract.joints.sdk_names))
    bounds = locomotion_bounds(contract) if bounds is None else bounds
    targets = clamp_targets(targets, contract, bounds)
    limit = bounds.torque
    tau = targets.tau
    torque = targets.kp * (targets.q - state.q) + targets.kd * (targets.dq - state.dq) + tau
    bounded = np.clip(torque, -limit, limit)
    q, dq, tau = targets.q.copy(), targets.dq.copy(), tau.copy()
    position = targets.kp > 0
    velocity = (~position) & (targets.kd > 0)
    feedforward = (~position) & (~velocity)
    q[position] = state.q[position] + (bounded[position] - tau[position]
                                       - targets.kd[position] * (dq[position] - state.dq[position])) / targets.kp[position]
    dq[velocity] = state.dq[velocity] + (bounded[velocity] - tau[velocity]) / targets.kd[velocity]
    tau[feedforward] = bounded[feedforward]
    return clamp_targets(JointTargets(q, dq, targets.kp, targets.kd, tau), contract, bounds)


def clamp_targets(targets: JointTargets, contract: DeploymentContract,
                  bounds: TargetBounds | None = None) -> JointTargets:
    """Apply command bounds after torque adjustment too, without touching actor outputs."""
    bounds = locomotion_bounds(contract) if bounds is None else bounds
    q, dq = targets.q.copy(), targets.dq.copy()
    positions, wheels = list(contract.joints.position_indices), list(contract.joints.wheel_indices)
    velocity = np.asarray(contract.safety.velocity_limit, dtype=np.float32)
    q[positions] = np.clip(q[positions], bounds.position[positions, 0], bounds.position[positions, 1])
    dq[wheels] = np.clip(dq[wheels], -velocity[wheels], velocity[wheels])
    return JointTargets(q, dq, targets.kp, targets.kd, targets.tau)

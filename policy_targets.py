"""Training action processing, deployment power-on, and separate torque pre-clip."""

from dataclasses import dataclass

import numpy as np

from tron2_deploy.policy_contract import DeploymentContract
from tron2_deploy.policy_observation import SdkState, Vector, finite_vector


@dataclass(frozen=True, slots=True)
class JointTargets:
    q: Vector
    dq: Vector
    kp: Vector
    kd: Vector


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


def power_on_targets(measured_pose: Vector, elapsed: float, contract: DeploymentContract) -> JointTargets:
    fraction = float(np.clip(elapsed / contract.control.power_on_seconds, 0., 1.))
    initial = contract.control.initial_gain_fraction
    gain = initial + (1. - initial) * fraction
    start = finite_vector(measured_pose, len(contract.joints.sdk_names))
    q = start + np.float32(fraction) * (np.asarray(contract.joints.default_position, dtype=np.float32) - start)
    q[list(contract.joints.wheel_indices)] = start[list(contract.joints.wheel_indices)]
    return JointTargets(q, np.zeros_like(q), np.asarray(contract.control.kp, dtype=np.float32) * gain,
                        np.asarray(contract.control.kd, dtype=np.float32) * gain)


def torque_preclip(targets: JointTargets, state: SdkState, contract: DeploymentContract) -> JointTargets:
    """Bound PD torque by adjusting targets; never change actor action/history."""
    limit = np.asarray(contract.control.torque_limit, dtype=np.float32)
    torque = targets.kp * (targets.q - state.q) + targets.kd * (targets.dq - state.dq)
    bounded = np.clip(torque, -limit, limit)
    q, dq = targets.q.copy(), targets.dq.copy()
    position = targets.kp > 0
    velocity = (~position) & (targets.kd > 0)
    q[position] = state.q[position] + (bounded[position] - targets.kd[position] * (dq[position] - state.dq[position])) / targets.kp[position]
    dq[velocity] = state.dq[velocity] + bounded[velocity] / targets.kd[velocity]
    return JointTargets(q, dq, targets.kp, targets.kd)

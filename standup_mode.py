"""Scripted seated stand-up: the canonical tron2_common reference tracked by PD, then handoff.

The reference is imported through the parent repository's adapter (DESIGN.md decision 8);
this module never connects to the SDK. The first ``trajectory(q0, 0)`` call certifies the
measured start pose and can take tens of seconds (decision 9a), so it runs on a worker
thread while the controller keeps publishing a hold of the measured pose; the phase clock
pauses at the end of P0 until the certificate arrives. A rejected or over-budget start is
a fail-closed PolicyInputError that trips the damping latch; nothing is clipped or retried.
"""

import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from time import monotonic
from typing import Final

import numpy as np

from tron2_deploy.policy_contract import DeploymentContract
from tron2_deploy.policy_observation import PolicyInputError, SdkState, Vector, base_imu
from tron2_deploy.policy_targets import JointTargets, standup_bounds, torque_preclip

ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tron2_common"))
from deploy_reference import reference
from tron2_common.kinematics import NAMES as REFERENCE_JOINTS

HANDED_OFF: Final = 6


def _certify(q0: Vector) -> Vector:
    try:
        return reference.trajectory(q0, 0.)
    except reference.UnsafeReference as error:
        raise PolicyInputError(f"stand-up reference rejected the measured start: {error}") from error


def body_tilt(state: SdkState, contract: DeploymentContract) -> float:
    _, gravity = base_imu(state, contract)
    return float(np.arccos(np.clip(-gravity[2], -1., 1.)))


class ScriptedStandUp:
    """Mutable phase clock for one stand-up; q0 is latched once from the measured state."""

    def __init__(self, contract: DeploymentContract, state: SdkState) -> None:
        self.contract = contract
        self.settings = contract.standup
        self.positions = list(contract.joints.position_indices)
        names = tuple(contract.joints.sdk_names[i] for i in self.positions)
        if names != REFERENCE_JOINTS:
            raise PolicyInputError("contract position joints disagree with the stand-up reference order")
        self.q0 = np.asarray(state.q[self.positions], dtype=np.float64)
        self.bounds = standup_bounds(contract)
        self.origin = state.timestamp
        self.previous = state.q.astype(np.float32).copy()
        self.last_time = state.timestamp
        self.wait_started: float | None = None
        self.hold_since: float | None = None
        self.complete = False
        self.handoff_state: SdkState | None = None
        self.phase = 0
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="standup-certify")
        self.certificate: Future[Vector] = self.executor.submit(_certify, self.q0.copy())
        self.executor.shutdown(wait=False)

    def _certified(self, elapsed: float) -> float | None:
        """None until the certificate is in: the phase clock pauses at the end of P0 and the
        wall-clock budget (DESIGN 9a) fails closed. The reference is never called from the
        control thread before that, since trajectory() would recompute the certificate in-line."""
        if self.certificate.done():
            self.certificate.result()
            return elapsed
        p0 = float(reference.PHASE_ENDS[0])
        if elapsed > p0:
            self.origin += elapsed - p0
            self.wait_started = monotonic() if self.wait_started is None else self.wait_started
            if monotonic() - self.wait_started > self.settings.certify_budget_seconds:
                raise PolicyInputError(
                    f"stand-up certificate exceeded {self.settings.certify_budget_seconds:.1f}s wall-clock budget")
        return None

    def reference_targets(self, elapsed: float) -> Vector:
        q = self.previous.copy()
        q[self.positions] = reference.trajectory(self.q0, elapsed).astype(np.float32)
        return q

    def targets(self, state: SdkState) -> JointTargets:
        if self.complete:
            raise PolicyInputError("stand-up already handed off")
        certified = self._certified(state.timestamp - self.origin)
        elapsed = min(state.timestamp - self.origin, float(reference.PHASE_ENDS[0])) if certified is None else certified
        self.phase = 0 if certified is None else reference.phase(elapsed)
        desired = self.previous if certified is None else self.reference_targets(elapsed)
        dt = float(np.clip(state.timestamp - self.last_time, 0., .1))
        self.last_time = state.timestamp
        step = np.float32(self.settings.speed_limit * dt)
        q = self.previous.copy()
        q[self.positions] = self.previous[self.positions] + np.clip(
            desired[self.positions] - self.previous[self.positions], -step, step)
        wheels = list(self.contract.joints.wheel_indices)
        q[wheels] = state.q[wheels]
        self.previous = q.copy()
        fraction = float(np.clip(elapsed / float(reference.PHASE_ENDS[0]), 0., 1.))
        initial = self.contract.control.initial_gain_fraction
        gain = np.float32(initial + (1. - initial) * fraction)
        kd = np.asarray(self.contract.control.kd, dtype=np.float32).copy()
        if self.settings.wheel_kd is not None:
            kd[wheels] = self.settings.wheel_kd
        targets = JointTargets(q, np.zeros_like(q), np.asarray(self.contract.control.kp, dtype=np.float32) * gain,
                               kd * gain)
        if certified is not None:
            self._update_handoff(state, elapsed)
        return torque_preclip(targets, state, self.contract, self.bounds)

    def _update_handoff(self, state: SdkState, elapsed: float) -> None:
        p4_end = float(reference.PHASE_ENDS[4])
        if elapsed < p4_end:
            return
        steady = (body_tilt(state, self.contract) < self.settings.handoff_tilt
                  and float(np.linalg.norm(state.dq)) < self.settings.handoff_joint_speed)
        if not steady:
            self.hold_since = None
        elif self.hold_since is None:
            self.hold_since = state.timestamp
        elif state.timestamp - self.hold_since >= self.settings.hold_seconds:
            self.complete = True
            self.handoff_state = state
            self.phase = HANDED_OFF
            return
        if elapsed > p4_end + self.settings.hold_seconds + self.settings.timeout_margin_seconds:
            raise PolicyInputError("stand-up finished without reaching the handoff tilt/joint-speed hold")

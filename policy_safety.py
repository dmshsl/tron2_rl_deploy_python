"""One restart-only fault latch shared by inference and the independent SDK watchdog."""

import sys
from threading import Lock

import numpy as np

from tron2_deploy.policy_contract import DeploymentContract
from tron2_deploy.policy_observation import SdkState, base_imu
from tron2_deploy.policy_targets import JointTargets


class SafetyMonitor:
    """Mutable, thread-safe first-fault latch; no prepare/stop operation clears it."""

    def __init__(self, contract: DeploymentContract) -> None:
        self.contract = contract
        self.reason = ""
        self.lock = Lock()

    def trip(self, reason: str) -> None:
        with self.lock:
            if not self.reason:
                self.reason = reason
                print(f"DAMPING_LATCH: {reason}; restart required", file=sys.stderr, flush=True)

    def damping(self) -> JointTargets:
        zero = np.zeros(len(self.contract.joints.sdk_names), dtype=np.float32)
        return JointTargets(zero, zero.copy(), zero.copy(),
                            np.asarray(self.contract.control.kd, dtype=np.float32))

    def check_state(self, state: SdkState, tilt_limit: float | None = None, position_tolerance: float = 0.) -> None:
        """tilt_limit overrides safety.tilt_limit and position_tolerance widens the hard-limit
        check, both for the scripted stand-up phases only (limp legs rest on their stops)."""
        safety = self.contract.safety
        tilt_limit = safety.tilt_limit if tilt_limit is None else tilt_limit
        wheels = list(self.contract.joints.wheel_indices)
        speed = np.abs(state.dq[wheels])
        limit = np.asarray(safety.velocity_limit)[wheels] * safety.wheel_overspeed_factor
        positions = list(self.contract.joints.position_indices)
        hard = np.asarray(safety.hard_position_limits)[positions]
        if np.any(speed > limit):
            self.trip(f"wheel overspeed: |dq|={speed.tolist()} limits={limit.tolist()} rad/s")
        elif (np.any(state.q[positions] < hard[:, 0] - position_tolerance)
              or np.any(state.q[positions] > hard[:, 1] + position_tolerance)):
            self.trip("leg joint beyond hard position limit")
        else:
            _, gravity = base_imu(state, self.contract)
            tilt = float(np.arccos(np.clip(-gravity[2], -1., 1.)))
            if tilt > tilt_limit:
                self.trip(f"tilt {tilt:.6f} > {tilt_limit} rad")

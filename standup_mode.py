"""Scripted seated stand-up: the canonical tron2_common reference tracked by PD, then handoff.

stop_at="A" runs the seated route instead (P0 hold, P1 lift, P2 legs to pose A' with the
wheels lifted, P3 plant) and then holds A' indefinitely: no P4, no handoff. Its plan is
computed in a worker process (the planner is pure Python and would starve the 500 Hz
control thread through the GIL) under the same wall-clock budget and clock pause. The
controller creates that process (RoutePlanner) before the SDK streams start: spawning it
at enable put the fork/exec, the resource tracker and the worker's import burst into the
first 0.3 s of the control loop, where the state/IMU timestamp-skew latch fired on a
loaded host. At enable only a pickled q0 crosses the queue.

The reference is imported through the parent repository's adapter (DESIGN.md decision 8);
this module never connects to the SDK. The first ``trajectory(q0, 0)`` call certifies the
measured start pose and can take tens of seconds (decision 9a), so it runs on a worker
thread while the controller keeps publishing a hold of the measured pose; the phase clock
pauses at the end of P0 until the certificate arrives. A rejected or over-budget start is
a fail-closed PolicyInputError that trips the damping latch; nothing is clipped or retried.
"""

import json
import multiprocessing
import os
import sys
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from time import monotonic
from typing import Final, Literal

import numpy as np

from tron2_deploy.policy_contract import DeploymentContract
from tron2_deploy.policy_observation import PolicyInputError, SdkState, Vector, base_imu
from tron2_deploy.policy_targets import JointTargets, standup_bounds, torque_preclip

ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tron2_common"))
from deploy_reference import reference
from tron2_common.kinematics import NAMES as REFERENCE_JOINTS
from tron2_common.seated_route import Route

HANDED_OFF: Final = 6
StopAt = Literal["handoff", "A"]


def _certify(q0: Vector) -> Vector:
    try:
        return reference.trajectory(q0, 0.)
    except reference.UnsafeReference as error:
        raise PolicyInputError(f"stand-up reference rejected the measured start: {error}") from error


PLANNER_NICE: Final = 5
PLANNER_CPUS_ENV: Final = "TRON2_PLANNER_CPUS"


def _background() -> None:
    """Planner worker initializer: lowest practical priority, and off the controller's cores when
    the launcher names others (TRON2_PLANNER_CPUS=0,1,...; sim2sim passes its own main cores)."""
    os.nice(PLANNER_NICE)
    cpus = os.environ.get(PLANNER_CPUS_ENV, "")
    if cpus:
        try:
            os.sched_setaffinity(0, {int(cpu) for cpu in cpus.split(",")})
        except (AttributeError, OSError, ValueError):
            pass  # keep the inherited affinity; priority alone still protects the control loop


def _plain(value: object) -> object:
    """json.dumps default for numpy scalars/arrays in the route info log."""
    return value.tolist() if hasattr(value, "tolist") else str(value)


def _warm_up() -> bool:
    """Worker-process entry run once at start: import the planner and fill its geometry caches."""
    from tron2_common.route_geometry import footprint, levers

    footprint()
    for side in ("L", "R"):
        levers(side)
    return True


def _plan_route(q0: Vector) -> tuple[bool, Route | str]:
    """Worker-process entry: (True, Route) or (False, reason); exceptions never cross the pipe."""
    try:
        return True, reference.route_to_a(q0)
    except reference.UnsafeReference as error:
        return False, str(error)


class RoutePlanner:
    """One pre-started spawn worker that plans seated routes; create it before the SDK streams."""

    def __init__(self) -> None:
        self.executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                            initializer=_background)
        self.warm = self.executor.submit(_warm_up)  # spawns the process now, imports off the control loop

    def submit(self, q0: Vector) -> "RouteFuture":
        return RouteFuture(self.executor.submit(_plan_route, np.array(q0, dtype=np.float64, copy=True)))

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


class RouteFuture:
    """Future-like view of the worker result that raises the fail-closed PolicyInputError."""

    def __init__(self, future: Future[tuple[bool, Route | str]]) -> None:
        self.future = future

    def done(self) -> bool:
        return self.future.done()

    def result(self) -> Route:
        ok, payload = self.future.result()
        if not ok or not isinstance(payload, Route):
            raise PolicyInputError(f"seated route to A rejected the measured start: {payload}")
        return payload

    def exception(self, timeout: float | None = None) -> BaseException | None:
        self.future.result(timeout=timeout)
        try:
            self.result()
        except PolicyInputError as error:
            return error
        return None


def body_tilt(state: SdkState, contract: DeploymentContract) -> float:
    _, gravity = base_imu(state, contract)
    return float(np.arccos(np.clip(-gravity[2], -1., 1.)))


class ScriptedStandUp:
    """Mutable phase clock for one stand-up; q0 is latched once from the measured state."""

    def __init__(self, contract: DeploymentContract, state: SdkState, stop_at: StopAt = "handoff",
                 planner: RoutePlanner | None = None) -> None:
        self.contract = contract
        self.stop_at: StopAt = stop_at
        self.route: Route | None = None
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
        if stop_at == "A":
            owned = planner is None
            planner = planner if planner is not None else RoutePlanner()
            self.certificate: Future[Vector] | RouteFuture = planner.submit(self.q0)
            if owned:
                planner.executor.shutdown(wait=False)
        else:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="standup-certify")
            self.certificate = executor.submit(_certify, self.q0.copy())
            executor.shutdown(wait=False)

    def _certified(self, elapsed: float) -> float | None:
        """None until the certificate is in: the phase clock pauses at the end of P0 and the
        wall-clock budget (DESIGN 9a) fails closed. The reference is never called from the
        control thread before that, since trajectory() would recompute the certificate in-line."""
        if self.certificate.done():
            result = self.certificate.result()
            if isinstance(result, Route) and self.route is None:
                self.route = result
                print("ROUTE_INFO " + json.dumps({"duration_s": result.duration, "knots": len(result.knots),
                                                   **result.info}, default=_plain), flush=True)
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
        source = reference.trajectory(self.q0, elapsed) if self.route is None else self.route.at(elapsed)
        q[self.positions] = source.astype(np.float32)
        return q

    def reference_phase(self, elapsed: float) -> int:
        return reference.phase(elapsed) if self.route is None else self.route.phase(elapsed)

    def targets(self, state: SdkState) -> JointTargets:
        if self.complete:
            raise PolicyInputError("stand-up already handed off")
        certified = self._certified(state.timestamp - self.origin)
        elapsed = min(state.timestamp - self.origin, float(reference.PHASE_ENDS[0])) if certified is None else certified
        self.phase = 0 if certified is None else self.reference_phase(elapsed)
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
        if certified is not None and self.stop_at == "handoff":
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

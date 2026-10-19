"""Learned stand-up (tron2_rl/standup/DESIGN.md decisions 10-12) after the scripted seated route.

seated route P0-P3 to A'/A'_p (standup_mode.ScriptedStandUp, stop_at="A") -> the stand-up actor
(its own 47-observation / 10-action contract, generated from the Isaac StandUp env) ->
success hold -> the unchanged handoff to Base/BaseBlind in PolicyWheelfootController.

Action processing is the training one (tron2_rl/standup/pose_a.py): every policy tick
q_tgt += clip(a, -1, 1) * v_max * dt, then |q_tgt - q| <= windup (pulled back at most v_max * dt),
then the URDF +-3 deg clamp; the two wheels are pure feed-forward torque clip(a) * 8 N.m (cap 16)
with kp = kd = 0 (free rolling at a = 0, as in training). Everything is fail-closed: a start
outside the trained A'/A'_p neighbourhood, a non-finite network output or no success within
the training episode length raises PolicyInputError, which trips the damping latch.

Success rule "catch" (DESIGN decisions 13/14, tron2_rl/standup/catch.py; contract success.catch): the Base catch
region held success.hold_seconds (decision 14: 0 = the first policy step inside it) - FK+IMU height >= 0.25 m with the wheel-only plausibility below, whole-body CoM (FK with the
training mass table) -0.02..+0.06 m ahead of the line through the two wheel contact points, |pitch|,|roll| <= 0.15,
both legs in the Base joint family near the balanced-crouch manifold, ||qdot_8|| <= catch.joint_speed (1.5 rad/s),
|wheel qdot| <= catch.wheel_speed (10 rad/s).
Success rule "height" (contracts before decision 13): h >= 0.45 m, tilt < 0.15 rad, |qdot_10| < 0.5 rad/s held 0.5 s.
Both need the root height,
which the robot cannot measure: it is estimated by FK + IMU as the height of the base origin
above the lowest wheel point, and only while both wheels are at the same height and no leg or
base point is lower (DESIGN "P4 rewards/success": no success judgement while contact is uncertain).
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

from tron2_deploy.policy_contract import DeploymentContract
from tron2_deploy.policy_observation import PolicyInputError, SdkState, Vector, finite_vector
from tron2_deploy.policy_targets import JointTargets, standup_bounds, torque_preclip
from tron2_deploy.standup_contract import StandupCatch, StandupPolicyContract, StandupSuccess
from tron2_deploy.standup_mode import RoutePlanner, ScriptedStandUp

ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tron2_common"))
from tron2_common.kinematics import collision_data, forward, wheel_floor

POLICY_PHASE: Final = 5
HANDED_OFF: Final = 6

def check_compatible(policy: StandupPolicyContract, locomotion: DeploymentContract) -> None:
    joints = locomotion.joints
    if policy.sdk_names != joints.sdk_names or sorted(policy.position_sdk) != sorted(joints.position_indices) \
            or sorted(policy.wheel_sdk) != sorted(joints.wheel_indices):
        raise PolicyInputError("stand-up and locomotion contracts disagree on the SDK joint layout")
    if tuple(policy.imu_to_base_rpy) != tuple(locomotion.imu.imu_to_base_rpy):
        raise PolicyInputError("stand-up and locomotion contracts disagree on the IMU mount")
    if policy.success.tilt >= locomotion.safety.tilt_limit:
        raise PolicyInputError("stand-up success tilt must be below the locomotion tilt limit")


@dataclass(frozen=True, slots=True)
class StandupMemory:
    """Pre-tick state of the incremental action: q_tgt (8, NAMES order) and the last action (10)."""

    target: Vector
    last_action: Vector


def base_gravity(state: SdkState, contract: StandupPolicyContract) -> tuple[Vector, Vector]:
    """(base gyro, unit gravity in base frame), the policy_observation.base_imu transform."""
    mount = Rotation.from_euler("xyz", contract.imu_to_base_rpy)
    world_from_base = Rotation.from_quat(state.quat[[1, 2, 3, 0]]) * mount.inv()
    gyro = mount.apply(state.gyro).astype(np.float32)
    gravity = world_from_base.inv().apply([0., 0., -1.]).astype(np.float32)
    return gyro, gravity


def tilt(gravity: Vector) -> float:
    return float(np.arccos(np.clip(-gravity[2], -1., 1.)))


def standup_observation(state: SdkState, memory: StandupMemory, contract: StandupPolicyContract) -> Vector:
    """Actor input, float32 and in the Isaac order (standup_pose_a_signals.observations)."""
    gyro, gravity = base_gravity(state, contract)
    standing = np.asarray(contract.standing, dtype=np.float32)
    positions, wheels = list(contract.position_sdk), list(contract.wheel_sdk)
    q = state.q[positions] - standing
    dq = np.concatenate((state.dq[positions], state.dq[wheels]))
    actor = np.concatenate((gyro * np.float32(contract.ang_vel_scale), gravity, q,
                            dq * np.float32(contract.joint_vel_scale),
                            finite_vector(memory.last_action, contract.action_size),
                            finite_vector(memory.target, 8) - standing,
                            np.asarray(contract.phase, dtype=np.float32),
                            np.array([contract.progress], dtype=np.float32)))
    clip = np.float32(contract.observation_clip)
    return finite_vector(np.clip(actor, -clip, clip), contract.actor_size)


def incremental_target(target: Vector, measured: Vector, action: Vector,
                       contract: StandupPolicyContract) -> Vector:
    """pose_a.incremental_target in float32."""
    rate = np.float32(contract.v_max * contract.policy_dt)
    windup = np.float32(contract.windup)
    step = np.clip(action, -1., 1.).astype(np.float32) * rate
    moved = np.minimum(np.maximum(target + step, measured - windup), measured + windup)
    moved = target + np.clip(moved - target, -rate, rate)
    return np.minimum(np.maximum(moved, np.asarray(contract.target_lower, dtype=np.float32)),
                      np.asarray(contract.target_upper, dtype=np.float32)).astype(np.float32)


def wheel_torque(action: Vector, contract: StandupPolicyContract) -> Vector:
    cap = np.float32(contract.wheel_torque_cap)
    return np.clip(np.clip(action, -1., 1.).astype(np.float32) * np.float32(contract.wheel_torque_scale), -cap, cap)


def _vertical(matrix: Vector, up: Vector) -> Vector:
    """Base-frame link transform with its z row replaced by the world-vertical component."""
    result = matrix.copy()
    result[2, :3] = up @ matrix[:3, :3]
    result[2, 3] = up @ matrix[:3, 3]
    return result


def estimate_height(q8: Vector, gravity: Vector, success: StandupSuccess) -> tuple[float, bool]:
    """(base-origin height above the lowest wheel point, both wheels plausibly carrying the robot).

    Plausible = the two wheel floors within wheel_level_tolerance and every other link point and
    the base hull at least body_clearance above the lower wheel. Flat-ground assumption."""
    up = -np.asarray(gravity, dtype=np.float64)
    up = up / max(np.linalg.norm(up), 1e-9)
    transforms = forward(np.asarray(q8, dtype=np.float64))
    data = collision_data()
    wheels = np.array([wheel_floor(_vertical(transforms[f"wheel_{side}_Link"], up), side) for side in ("L", "R")])
    body = min(float((data[name] @ (up @ matrix[:3, :3]) + up @ matrix[:3, 3]).min())
               for name, matrix in transforms.items() if not name.startswith("wheel_"))
    body = min(body, float((data["base_Link"] @ up).min()))
    lowest = float(wheels.min())
    plausible = abs(float(wheels[0] - wheels[1])) <= success.wheel_level_tolerance and body >= lowest + success.body_clearance
    return -lowest, plausible


def pitch_roll(gravity: Vector) -> tuple[float, float]:
    """Base pitch (+ = nose down) and roll from the unit gravity in the base frame (catch.pitch_roll)."""
    return float(np.arctan2(gravity[0], -gravity[2])), float(np.arctan2(-gravity[1], -gravity[2]))


def com_offset(q8: Vector, gravity: Vector, catch: StandupCatch) -> float:
    """Signed horizontal distance (m, + = ahead) of the whole-body CoM from the line through the two wheel
    contact points (L -> R), as the Isaac env's signed_line_distance(centre_of_mass, wheel_contact_points)."""
    up = -np.asarray(gravity, dtype=np.float64)
    up = up / max(np.linalg.norm(up), 1e-9)
    links = forward(np.asarray(q8, dtype=np.float64))
    total, moment = 0., np.zeros(3)
    for body in catch.bodies:
        matrix = links.get(body.name, np.eye(4))  # bodies not in the leg chains are fixed to the base
        total += body.mass
        moment += body.mass * (matrix[:3, :3] @ np.asarray(body.com) + matrix[:3, 3])
    centre = moment / total
    points = []
    for side in ("L", "R"):
        matrix = links[f"wheel_{side}_Link"]
        axle = matrix[:3, 1]
        radial = -up - (-up @ axle) * axle
        radial = radial / max(np.linalg.norm(radial), 1e-6)
        points.append(matrix[:3, 3] + catch.wheel_radius * radial)
    seed = np.array([1., 0., 0.]) if abs(up[0]) < .9 else np.array([0., 1., 0.])
    e1 = seed - (seed @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    flat = lambda v: np.array([v @ e1, v @ e2])  # noqa: E731 - right-handed horizontal coordinates
    a, b, c = flat(points[0]), flat(points[1]), flat(centre)
    direction, offset = b - a, c - a
    return float((direction[0] * offset[1] - direction[1] * offset[0]) / max(np.linalg.norm(direction), 1e-6))


def catch_components(q8: Vector, qdot8: Vector, wheel_qdot: Vector, gravity: Vector, height: float,
                     plausible: bool, catch: StandupCatch) -> dict[str, bool]:
    """tron2_rl/standup/catch.catch_components with deploy signals: wheel-only = the FK plausibility check."""
    pitch, roll = pitch_roll(gravity)
    table = np.asarray(catch.manifold)
    hip = float(np.interp(height, table[:, 0], table[:, 1]))
    knee = float(np.interp(height, table[:, 0], table[:, 2]))
    legs = np.asarray(q8, dtype=np.float64).reshape(2, 4)
    family = bool(np.all(np.abs(legs[:, 1]) <= catch.hip_roll) and np.all(np.abs(legs[:, 2]) <= catch.hip_yaw)
                  and np.all(np.abs(legs[:, 0] - hip) <= catch.manifold_tolerance)
                  and np.all(np.abs(legs[:, 3] - knee) <= catch.manifold_tolerance))
    offset = com_offset(q8, gravity, catch)
    return {"wheel_only": bool(plausible), "height": height >= catch.height,
            "com": catch.offset_min <= offset <= catch.offset_max,
            "attitude": abs(pitch) <= catch.pitch and abs(roll) <= catch.roll, "family": family,
            "rest": float(np.linalg.norm(qdot8)) <= catch.joint_speed and bool(np.all(np.abs(wheel_qdot) <= catch.wheel_speed))}


class StandupActor:
    """FLOAT32 interface stand-up ONNX; one 47 -> 10 call per policy tick."""

    def __init__(self, model_dir: Path, contract: StandupPolicyContract) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_dir / "policy.onnx"), options, providers=["CPUExecutionProvider"])
        if self.session.get_inputs()[0].shape != [contract.actor_size] or \
                self.session.get_outputs()[0].shape != [contract.action_size]:
            raise PolicyInputError("stand-up ONNX dimensions disagree with its contract")
        self.contract = contract

    def __call__(self, actor: Vector) -> Vector:
        output = self.session.run(None, {self.session.get_inputs()[0].name: finite_vector(actor, self.contract.actor_size)})
        return finite_vector(output[0], self.contract.action_size)


class LearnedStandUp:
    """Seated route to A'/A'_p, then the learned actor until the success hold; complete -> handoff.

    phase: 0-4 seated route (P0 hold, P1 lift, P2 legs to A in the air, P3 lower, HOLD A'),
    5 stand-up policy, 6 success (handed off)."""

    def __init__(self, locomotion: DeploymentContract, contract: StandupPolicyContract, actor: StandupActor,
                 state: SdkState, planner: RoutePlanner | None = None) -> None:
        self.locomotion = locomotion
        self.contract = contract
        self.actor = actor
        self.route = ScriptedStandUp(locomotion, state, "A", planner)
        self.bounds = standup_bounds(locomotion)
        self.complete = False
        self.handoff_state: SdkState | None = None
        self.phase = 0
        self.policy_started_at: float | None = None
        self.memory: StandupMemory | None = None
        self.last_tick = -np.inf
        self.hold = 0.
        self.wheel_tau = np.zeros(2, dtype=np.float32)
        self.last_height = float("nan")
        self.last_plausible = False
        self.ticks = 0
        positions = list(contract.position_sdk)
        self.positions, self.wheels = positions, list(contract.wheel_sdk)
        self.kp = np.zeros(10, dtype=np.float32)
        self.kd = np.zeros(10, dtype=np.float32)
        self.kp[positions] = contract.kp
        self.kd[positions] = contract.kd

    def _route_done(self, state: SdkState) -> bool:
        route = self.route.route
        return (route is not None and self.route.phase == 4
                and state.timestamp - self.route.origin >= float(route.times[-1]) + self.contract.settle_seconds)

    def start(self, state: SdkState) -> None:
        """Latch the measured pose as q_tgt (the env's reset: target = q) after the start check."""
        q = state.q[self.positions].astype(np.float64)
        poses = np.asarray(self.contract.start_poses, dtype=np.float64)
        for side, legs in (("L", slice(0, 4)), ("R", slice(4, 8))):
            error = np.min(np.max(np.abs(poses[:, legs] - q[legs]), axis=1))
            if error > self.contract.start_tolerance:
                raise PolicyInputError(f"learned stand-up start: {side} leg {np.degrees(error):.1f} deg from A'/A'_p "
                                       f"(> {np.degrees(self.contract.start_tolerance):.1f}); outside the trained starts")
        self.memory = StandupMemory(state.q[self.positions].astype(np.float32).copy(),
                                    np.zeros(self.contract.action_size, dtype=np.float32))
        self.policy_started_at = state.timestamp
        self.phase = POLICY_PHASE
        print(f"STANDUP_POLICY_START t={state.timestamp:.3f} q={np.round(q, 4).tolist()}", flush=True)

    def tick(self, state: SdkState) -> None:
        """One policy step: observation, actor, incremental target, wheel torque, success hold."""
        assert self.memory is not None
        actor = standup_observation(state, self.memory, self.contract)
        action = np.clip(self.actor(actor), -1., 1.).astype(np.float32)
        measured = state.q[self.positions].astype(np.float32)
        target = incremental_target(self.memory.target, measured, action[:8], self.contract)
        self.wheel_tau = wheel_torque(action[8:], self.contract)
        self.memory = StandupMemory(target, action)
        self.last_tick = state.timestamp
        self.ticks += 1
        _, gravity = base_gravity(state, self.contract)
        height, plausible = estimate_height(measured, gravity, self.contract.success)
        self.last_height, self.last_plausible = height, plausible
        success = self.contract.success
        if success.rule == "catch":
            assert success.catch is not None
            parts = catch_components(measured, state.dq[self.positions], state.dq[self.wheels], gravity, height,
                                     plausible, success.catch)
            good = all(parts.values())
        else:
            good = (plausible and height >= success.height and tilt(gravity) < success.tilt
                    and float(np.linalg.norm(state.dq)) < success.joint_speed)
        self.hold = self.hold + self.contract.policy_dt if good else 0.
        if good and self.hold >= success.hold_seconds - 1e-6:
            self.complete = True
            self.handoff_state = state
            self.phase = HANDED_OFF
            print(f"STANDUP_SUCCESS t={state.timestamp:.3f} policy_s={state.timestamp - self.policy_started_at:.3f} "
                  f"height_est={height:.4f} tilt={tilt(gravity):.4f} rule={success.rule}", flush=True)

    def targets(self, state: SdkState) -> JointTargets:
        if self.complete:
            raise PolicyInputError("stand-up already handed off")
        if self.policy_started_at is None:
            targets = self.route.targets(state)
            self.phase = self.route.phase
            if not self._route_done(state):
                return targets
            self.start(state)
        elapsed = state.timestamp - self.policy_started_at
        if elapsed > self.contract.episode_seconds:
            raise PolicyInputError(f"learned stand-up reached no success within {self.contract.episode_seconds:.1f}s "
                                   f"(height_est={self.last_height:.3f} m, wheel contact plausible={self.last_plausible})")
        if state.timestamp - self.last_tick >= self.contract.policy_dt - 1e-9:
            self.tick(state)
        assert self.memory is not None
        q = state.q.astype(np.float32).copy()
        q[self.positions] = self.memory.target
        tau = np.zeros(10, dtype=np.float32)
        tau[self.wheels] = self.wheel_tau
        return torque_preclip(JointTargets(q, np.zeros(10, dtype=np.float32), self.kp.copy(), self.kd.copy(), tau),
                              state, self.locomotion, self.bounds)

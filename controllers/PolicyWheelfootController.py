# ruff: noqa: N999 - filename is the task's public controller entrypoint.
"""New contract-driven controller; vendor controllers are never imported.

Import as PolicyWheelfootController with this directory on sys.path, avoiding the
vendor controllers/__init__.py which eagerly imports SDK and GUI dependencies.
"""

import os
from dataclasses import replace
from pathlib import Path
from typing import Final, Literal

import numpy as np
import onnxruntime as ort

from tron2_deploy.command_shaping import CommandShaper, HeadingHold, limits_from_contract, yaw_of
from tron2_deploy.height_scan_provider import HeightScanProvider
from tron2_deploy.policy_contract import DeploymentContract
from tron2_deploy.policy_observation import (
    PolicyInputError,
    PolicyMemory,
    SdkState,
    Vector,
    build_actor_input,
    finite_vector,
    velocity_error,
)
from tron2_deploy.policy_safety import SafetyMonitor
from tron2_deploy.policy_targets import (
    JointTargets,
    action_targets,
    power_on_targets,
    torque_preclip,
)
from tron2_deploy.standup_mode import RoutePlanner, ScriptedStandUp, StopAt
from tron2_deploy.standup_contract import load_standup_contract
from tron2_deploy.standup_policy import LearnedStandUp, StandupActor, check_compatible

StandupMode = Literal["none", "scripted", "learned"]
STANDUP_MODEL: Final = Path(__file__).resolve().parent / "model/WF_TRON2A_STANDUP"
COMMAND_CAP_ENV_VAR: Final = "TRON2_COMMAND_CAP"


class PolicyWheelfootController:
    """Mutable inference state; prepare always captures the latest measured pose."""

    def __init__(self, model_dir: Path, contract: DeploymentContract,
                 height_provider: HeightScanProvider | None = None,
                 standup: StandupMode = "none", standup_stop_at: StopAt = "handoff",
                 standup_model_dir: Path = STANDUP_MODEL) -> None:
        self.contract = contract
        self.standup_mode: StandupMode = standup
        self.standup_stop_at: StopAt = standup_stop_at
        self.standup: ScriptedStandUp | LearnedStandUp | None = None
        self.standup_contract = None
        self.standup_actor: StandupActor | None = None
        if standup == "learned":
            # Validated before any SDK stream: its own 47/10 contract and ONNX, same joint layout/IMU.
            self.standup_contract = load_standup_contract(standup_model_dir / "contract.yaml")
            check_compatible(self.standup_contract, contract)
            self.standup_actor = StandupActor(standup_model_dir, self.standup_contract)
        # The seated-route worker process is spawned here, before any SDK stream, not at enable.
        routed = standup == "learned" or (standup == "scripted" and standup_stop_at == "A")
        self.planner: RoutePlanner | None = RoutePlanner() if routed else None
        self.safety = SafetyMonitor(contract)
        # Every command source (schedule, joystick, live GUI) goes through this: NaN -> 0, clamp to the
        # trained range, software cap (contract commands.default_cap; set_command_cap), slew limit.
        self.command_shaper = CommandShaper(limits_from_contract(contract))
        self.heading_hold = HeadingHold(contract.commands.heading_hold_gain, contract.commands.heading_hold_max)
        cap = os.environ.get(COMMAND_CAP_ENV_VAR)
        if cap:  # simulation/test seam ("vx:wz"); still clipped to the trained command range
            vx, wz = (float(v) for v in cap.split(":"))
            print(f"COMMAND_CAP {COMMAND_CAP_ENV_VAR} -> {self.set_command_cap(vx, wz)}", flush=True)
        self.height_provider = height_provider
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.policy = ort.InferenceSession(str(model_dir / "policy.onnx"), options,
                                          providers=["CPUExecutionProvider"])
        self.encoder = ort.InferenceSession(str(model_dir / "encoder.onnx"), options,
                                           providers=["CPUExecutionProvider"])
        for session, input_size, output_size in (
            (self.policy, contract.actor_size, len(contract.joints.action_names)),
            (self.encoder, contract.encoder_input_size, contract.encoder_output_size),
        ):
            if session.get_inputs()[0].shape != [input_size] or session.get_outputs()[0].shape != [output_size]:
                raise PolicyInputError("ONNX dimensions disagree with generated contract")
        self.mem = PolicyMemory.empty(contract)
        self.started_at: float | None = None
        self.enabled = True
        self.hold_pose: Vector | None = None
        self.last_policy_time = -np.inf
        self.targets: JointTargets | None = None
        self.actor_input = np.zeros(contract.actor_size, dtype=np.float32)
        self.last_scan: Vector | None = None

    def prepare(self, state: SdkState) -> None:
        if self.safety.reason:
            return
        self.enabled = True
        self.started_at = state.timestamp
        self.hold_pose = state.q.copy()
        self.mem = PolicyMemory.empty(self.contract)
        self.last_policy_time = -np.inf
        self.targets = None
        self.last_scan = None
        self.standup = None
        self.command_shaper.reset()
        self.heading_hold.reset()
        if self.standup_mode == "scripted":
            # The reference clock owns the start; started_at is set again at handoff.
            self.started_at = None
            self.standup = ScriptedStandUp(self.contract, state, self.standup_stop_at, self.planner)
        elif self.standup_mode == "learned":
            assert self.standup_contract is not None and self.standup_actor is not None
            self.started_at = None
            self.standup = LearnedStandUp(self.contract, self.standup_contract, self.standup_actor, state, self.planner)

    def set_command_cap(self, vx: float | None = None, wz: float | None = None) -> tuple[float, float]:
        """Software speed cap (|vx| m/s, |wz| rad/s), clipped to the trained range; usable while running."""
        return self.command_shaper.set_cap(vx, wz)

    @property
    def shaped_command(self):
        """The (vx, vy, wz) the policy saw on the last tick (after clamp, cap and slew limiting)."""
        return self.command_shaper.output

    def stop(self) -> None:
        self.command_shaper.reset()
        self.heading_hold.reset()
        self.enabled = False
        self.started_at = None
        self.hold_pose = None
        self.mem = PolicyMemory.empty(self.contract)
        self.targets = None
        self.last_scan = None
        self.standup = None

    def close(self) -> None:
        """Release the planner worker process (idempotent; the control loop may already be done)."""
        if self.planner is not None:
            self.planner.close()
            self.planner = None

    @property
    def prepare_seconds(self) -> float:
        """Power-on preparation before the first locomotion action: control.power_on_seconds at enable,
        standup.handoff_power_on_seconds after a stand-up handoff (the robot is already up and balancing)."""
        if self.standup is None:
            return self.contract.control.power_on_seconds
        return self.contract.standup.handoff_power_on_seconds

    @property
    def standup_phase(self) -> int:
        """-1 without the scripted mode, reference phase 0..5 while rising, 6 after handoff;
        with stop_at A: 0 P0, 1 P1 lift, 2 P2 legs to A, 3 P3 plant, 4 hold at A';
        learned: 0-4 seated route as stop_at A, 5 stand-up policy, 6 success and handoff."""
        return -1 if self.standup is None else self.standup.phase

    def handoff(self, state: SdkState) -> None:
        """DESIGN.md handoff rule: history pre-filled with the final measured stand-up state,
        vel_error_ema zero, then the policy after standup.handoff_power_on_seconds (default 0: at once;
        a 3 s PD-hold blend lets the balanced robot fall) with zero command for 0.5 s."""
        print(f"HANDOFF policy={self.contract.policy} t={state.timestamp:.3f}", flush=True)
        placeholder = None if self.height_provider is None else np.zeros(self.contract.scan.count, dtype=np.float32)
        _, history = build_actor_input(state, PolicyMemory.empty(self.contract), placeholder,
                                       np.zeros(3, dtype=np.float32), self.contract)
        self.mem = replace(PolicyMemory.empty(self.contract),
                           history=history.reshape(self.contract.history_length, -1))
        self.hold_pose = state.q.copy()
        self.started_at = state.timestamp
        self.last_policy_time = -np.inf
        self.targets = None
        self.command_shaper.reset()
        self.heading_hold.reset()

    def step(self, state: SdkState, commands: Vector) -> JointTargets:
        if self.safety.reason:
            return self.safety.damping()
        rising = self.standup is not None and not self.standup.complete
        self.safety.check_state(state, self.contract.standup.tilt_limit if rising else None,
                                self.contract.standup.hard_limit_tolerance if rising else 0.)
        if self.safety.reason:
            return self.safety.damping()
        try:
            try:  # NaN -> 0 before the finite check; a wrong shape still latches
                commands = self.command_shaper.clamp(commands).astype(np.float32)
            except ValueError as error:
                raise PolicyInputError(str(error)) from error
            finite_vector(commands, 3)
            targets = self._step(state, commands)
            for value in (targets.q, targets.dq, targets.kp, targets.kd, targets.tau):
                finite_vector(value, len(self.contract.joints.sdk_names))
            return targets
        except PolicyInputError as error:
            self.safety.trip(str(error))
            return self.safety.damping()

    def _step(self, state: SdkState, commands: Vector) -> JointTargets:
        if not self.enabled:
            zero = np.zeros_like(state.q)
            return JointTargets(state.q.copy(), zero, zero.copy(), zero.copy())
        if self.standup is not None and not self.standup.complete:
            targets = self.standup.targets(state)
            if not self.standup.complete:
                return targets
            self.handoff(state)
        if self.hold_pose is None:
            self.hold_pose = state.q.copy()
        ramp = self.standup is None
        if self.started_at is None:
            return torque_preclip(power_on_targets(self.hold_pose, 0., self.contract, ramp), state, self.contract)
        elapsed = state.timestamp - self.started_at
        prepare = self.prepare_seconds
        if prepare > 0 and elapsed <= prepare:
            return torque_preclip(power_on_targets(self.hold_pose, elapsed, self.contract, ramp), state, self.contract)
        if self.standup is not None and elapsed - prepare < self.contract.standup.zero_command_seconds:
            commands = np.zeros_like(commands)
        commands = self.command_shaper(commands, state.timestamp)
        commands = self.heading_hold(commands, float(self.command_shaper.target[2]), yaw_of(state.quat))
        if state.timestamp - self.last_policy_time >= self.contract.control.policy_dt - 1e-9:
            scan = None if self.height_provider is None else self.height_provider.get_scan(state)
            self.last_scan = scan
            actor, history = build_actor_input(state, self.mem, scan, commands, self.contract)
            latent = finite_vector(self.encoder.run(None, {self.encoder.get_inputs()[0].name: history})[0],
                                   self.contract.encoder_output_size)
            actor = np.concatenate((latent, actor[self.contract.encoder_output_size:]))
            actions = finite_vector(self.policy.run(None, {self.policy.get_inputs()[0].name: actor})[0],
                                    len(self.contract.joints.action_names))
            alpha = np.float32(self.contract.ema.alpha)
            ema = (np.float32(1.) - alpha) * self.mem.vel_error_ema + alpha * velocity_error(state, commands, self.contract)
            self.mem = replace(self.mem, history=history.reshape(self.contract.history_length, -1),
                               last_action=actions.copy(), vel_error_ema=ema, encoder_output=latent)
            self.targets = action_targets(actions, self.contract)
            self.actor_input = actor
            self.last_policy_time = state.timestamp
        if self.targets is None:
            raise PolicyInputError("policy did not produce targets")
        return torque_preclip(self.targets, state, self.contract)

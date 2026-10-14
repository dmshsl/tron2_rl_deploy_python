# ruff: noqa: N999 - filename is the task's public controller entrypoint.
"""New contract-driven controller; vendor controllers are never imported.

Import as PolicyWheelfootController with this directory on sys.path, avoiding the
vendor controllers/__init__.py which eagerly imports SDK and GUI dependencies.
"""

from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort

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
from tron2_deploy.standup_mode import ScriptedStandUp

StandupMode = Literal["none", "scripted"]


class PolicyWheelfootController:
    """Mutable inference state; prepare always captures the latest measured pose."""

    def __init__(self, model_dir: Path, contract: DeploymentContract,
                 height_provider: HeightScanProvider | None = None,
                 standup: StandupMode = "none") -> None:
        self.contract = contract
        self.standup_mode: StandupMode = standup
        self.standup: ScriptedStandUp | None = None
        self.safety = SafetyMonitor(contract)
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
        if self.standup_mode == "scripted":
            # The reference clock owns the start; started_at is set again at handoff.
            self.started_at = None
            self.standup = ScriptedStandUp(self.contract, state)

    def stop(self) -> None:
        self.enabled = False
        self.started_at = None
        self.hold_pose = None
        self.mem = PolicyMemory.empty(self.contract)
        self.targets = None
        self.last_scan = None
        self.standup = None

    @property
    def standup_phase(self) -> int:
        """-1 without the scripted mode, reference phase 0..5 while rising, 6 after handoff."""
        return -1 if self.standup is None else self.standup.phase

    def handoff(self, state: SdkState) -> None:
        """DESIGN.md handoff rule: history pre-filled with the final measured stand-up state,
        vel_error_ema zero, then the unchanged preparation and policy (zero command 0.5 s)."""
        placeholder = None if self.height_provider is None else np.zeros(self.contract.scan.count, dtype=np.float32)
        _, history = build_actor_input(state, PolicyMemory.empty(self.contract), placeholder,
                                       np.zeros(3, dtype=np.float32), self.contract)
        self.mem = replace(PolicyMemory.empty(self.contract),
                           history=history.reshape(self.contract.history_length, -1))
        self.hold_pose = state.q.copy()
        self.started_at = state.timestamp
        self.last_policy_time = -np.inf
        self.targets = None

    def step(self, state: SdkState, commands: Vector) -> JointTargets:
        if self.safety.reason:
            return self.safety.damping()
        rising = self.standup is not None and not self.standup.complete
        self.safety.check_state(state, self.contract.standup.tilt_limit if rising else None)
        if self.safety.reason:
            return self.safety.damping()
        try:
            finite_vector(commands, 3)
            targets = self._step(state, commands)
            for value in (targets.q, targets.dq, targets.kp, targets.kd):
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
        if elapsed <= self.contract.control.power_on_seconds:
            return torque_preclip(power_on_targets(self.hold_pose, elapsed, self.contract, ramp), state, self.contract)
        if self.standup is not None and elapsed - self.contract.control.power_on_seconds < self.contract.standup.zero_command_seconds:
            commands = np.zeros_like(commands)
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

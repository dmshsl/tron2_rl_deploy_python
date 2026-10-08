# ruff: noqa: N999 - filename is the task's public controller entrypoint.
"""New contract-driven controller; vendor controllers are never imported.

Import as PolicyWheelfootController with this directory on sys.path, avoiding the
vendor controllers/__init__.py which eagerly imports SDK and GUI dependencies.
"""

from dataclasses import replace
from pathlib import Path

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
from tron2_deploy.policy_targets import (
    JointTargets,
    action_targets,
    power_on_targets,
    torque_preclip,
)


class PolicyWheelfootController:
    """Mutable inference state; prepare always captures the latest measured pose."""

    def __init__(self, model_dir: Path, contract: DeploymentContract,
                 height_provider: HeightScanProvider | None = None) -> None:
        self.contract = contract
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
        self.enabled = True
        self.started_at = state.timestamp
        self.hold_pose = state.q.copy()
        self.mem = PolicyMemory.empty(self.contract)
        self.last_policy_time = -np.inf
        self.targets = None
        self.last_scan = None

    def stop(self) -> None:
        self.enabled = False
        self.started_at = None
        self.hold_pose = None
        self.mem = PolicyMemory.empty(self.contract)
        self.targets = None
        self.last_scan = None

    def step(self, state: SdkState, commands: Vector) -> JointTargets:
        if not self.enabled:
            zero = np.zeros_like(state.q)
            return JointTargets(state.q.copy(), zero, zero.copy(), zero.copy())
        if self.hold_pose is None:
            self.hold_pose = state.q.copy()
        if self.started_at is None:
            return torque_preclip(power_on_targets(self.hold_pose, 0., self.contract), state, self.contract)
        elapsed = state.timestamp - self.started_at
        if elapsed <= self.contract.control.power_on_seconds:
            return torque_preclip(power_on_targets(self.hold_pose, elapsed, self.contract), state, self.contract)
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
